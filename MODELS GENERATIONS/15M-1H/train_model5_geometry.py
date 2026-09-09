# train_model5_geometry.py  ·  v1
# ─────────────────────────────────────────────────────────────────────
# MODEL 5 — CANDLESTICK GEOMETRY SPECIALIST
#
# Dataset    : model5_geometry_dataset.csv
# Features   : 6  (97_final_upper_wick_ratio, 99_final_lower_wick_ratio,
#                   71_intraday_variance_ratio, 104_body_size_to_atr_ratio,
#                   94_relative_body_to_range_ratio, 98_lower_wick_absorption_ratio)
# Target     : R_p ∈ [-1.0, +1.0]  — candlestick rejection polarity
#              -1.0 = dominant lower wick absorption (demand floor, buyers rejected sellers)
#               0.0 = balanced / no clear wick bias
#              +1.0 = heavy overhead supply rejection (supply ceiling, sellers rejected buyers)
# Preprocess : NONE — all features are geometry ratios in [0, 1].
#              These ratios are already scale-invariant by construction.
# Algorithm  : Gradient Boosted Trees (XGBoost + LightGBM ensemble)
#              Tree splits naturally capture the narrow ratio clustering
#              (~mean 0.24) without needing the deeper [128, 64, 32] NN
#              described for deep-learning fallback. Ensembles handle
#              the tight decision boundaries via many weak learners.
# ─────────────────────────────────────────────────────────────────────

import os, random, warnings
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    warnings.warn("xgboost not installed — XGB model will be skipped. "
                  "Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    warnings.warn("lightgbm not installed — LGB model will be skipped. "
                  "Install with: pip install lightgbm")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────
DATA_PATH        = "model5_geometry_dataset.csv"
META_PATH        = "model5_meta.pkl"
XGB_MODEL_PATH   = "model5_xgb.json"
LGB_MODEL_PATH   = "model5_lgb.txt"
ENSEMBLE_PATH    = "model5_ensemble.pkl"

VAL_RATIO  = 0.12
TARGET_MIN = -1.0
TARGET_MAX = +1.0

FEATURE_COLS = [
    "97_final_upper_wick_ratio",
    "99_final_lower_wick_ratio",
    "71_intraday_variance_ratio",
    "104_body_size_to_atr_ratio",
    "94_relative_body_to_range_ratio",
    "98_lower_wick_absorption_ratio",
]

# label_geometry: -1 = BUY (lower wick absorption / demand),
#                  0 = HOLD (balanced)
#                 +1 = SELL (upper wick rejection / supply)
TARGET_COL = "label_geometry"


# ─────────────────────────────────────────────────────────────────────
# TARGET CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────

def build_rejection_target(label_series, seed=SEED):
    """
    Convert integer label_geometry {-1, 0, +1} into a smooth continuous
    R_p ∈ [-1.0, +1.0] by adding calibrated Gaussian noise per zone.

    BUY  (-1) → R_p sampled from N(-0.72, 0.18) clipped to [-1.0, -0.10]
    HOLD ( 0) → R_p sampled from N( 0.00, 0.12) clipped to [-0.25, +0.25]
    SELL (+1) → R_p sampled from N(+0.72, 0.18) clipped to [+0.10, +1.0]

    The narrow HOLD band (-0.25, +0.25) forces the model to commit to a
    polarity signal outside the neutral zone.
    """
    rng = np.random.RandomState(seed)
    y   = np.zeros(len(label_series), dtype=np.float32)

    buy_mask  = label_series == -1
    hold_mask = label_series ==  0
    sell_mask = label_series ==  1

    y[buy_mask]  = np.clip(rng.normal(-0.72, 0.18, buy_mask.sum()),  -1.0, -0.10)
    y[hold_mask] = np.clip(rng.normal( 0.00, 0.12, hold_mask.sum()), -0.25, 0.25)
    y[sell_mask] = np.clip(rng.normal(+0.72, 0.18, sell_mask.sum()),  0.10, 1.0)
    return y


# ─────────────────────────────────────────────────────────────────────
# XGBoost MODEL  — expanded depth for tight ratio boundaries
# ─────────────────────────────────────────────────────────────────────

XGB_PARAMS = {
    "objective":        "reg:squarederror",
    "eval_metric":      "rmse",
    "n_estimators":     900,
    "learning_rate":    0.04,
    "max_depth":        7,        # deeper for subtle geometry splits
    "min_child_weight": 8,
    "subsample":        0.80,
    "colsample_bytree": 0.80,
    "reg_alpha":        0.05,
    "reg_lambda":       1.5,
    "random_state":     SEED,
    "n_jobs":           -1,
    "tree_method":      "hist",
}


# ─────────────────────────────────────────────────────────────────────
# LightGBM MODEL  — more leaves to capture narrow ratio clusters
# ─────────────────────────────────────────────────────────────────────

LGB_PARAMS = {
    "objective":          "regression",
    "metric":             "rmse",
    "n_estimators":       900,
    "learning_rate":      0.04,
    "max_depth":          7,
    "num_leaves":         64,     # wider leaf budget for tight boundaries
    "min_child_samples":  15,
    "subsample":          0.80,
    "colsample_bytree":   0.80,
    "reg_alpha":          0.05,
    "reg_lambda":         1.5,
    "random_state":       SEED,
    "n_jobs":             -1,
    "verbose":            -1,
}


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    # Polarity accuracy: sign of prediction matches sign of target
    # Exclude near-zero targets (HOLD zone) |y| <= 0.15
    mask = np.abs(y_true) > 0.15
    if mask.sum() > 0:
        pol_acc = (np.sign(y_pred[mask]) == np.sign(y_true[mask])).mean()
    else:
        pol_acc = float("nan")
    print(f"  {name:10s} | RMSE={rmse:.4f}  MAE={mae:.4f}  PolarityAcc={pol_acc:.4f}")
    return rmse


def main():
    if not XGB_AVAILABLE and not LGB_AVAILABLE:
        raise RuntimeError("Neither xgboost nor lightgbm is installed. "
                           "pip install xgboost lightgbm")

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded : {DATA_PATH}  shape={df.shape}")

    for c in FEATURE_COLS + [TARGET_COL]:
        if c not in df.columns:
            raise RuntimeError(f"Missing column '{c}'. Columns: {list(df.columns)}")

    X = df[FEATURE_COLS].fillna(0.0).values
    y_int = df[TARGET_COL].astype(int).values
    print(f"Features : {len(FEATURE_COLS)}  |  Samples : {len(df)}")
    print(f"Integer label dist: {dict(Counter(y_int.tolist()))}")

    # ── NO SCALER — geometry ratios are [0,1] by design ──────────────────
    joblib.dump({"feature_names": FEATURE_COLS, "scaler": None}, META_PATH)
    print(f"Meta    → {META_PATH}  (no scaler — geometry ratios pre-normalised)")

    # ── Build continuous R_p rejection target ─────────────────────────────
    y = build_rejection_target(y_int)
    print(f"R_p target : min={y.min():.3f}  max={y.max():.3f}  mean={y.mean():.3f}")

    # ── Train / val split ────────────────────────────────────────────────
    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=VAL_RATIO, random_state=SEED, shuffle=True
    )
    print(f"\nTrain: {len(Xtr):,}  |  Val: {len(Xva):,}")

    fitted = {}

    # ── XGBoost ──────────────────────────────────────────────────────────
    if XGB_AVAILABLE:
        print(f"\n{'─'*60}")
        print("Training XGBoost  (depth=7, expanded for tight wick boundaries)...")
        xgb_model = xgb.XGBRegressor(**XGB_PARAMS)
        xgb_model.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            verbose=100,
        )
        xgb_model.save_model(XGB_MODEL_PATH)
        va_pred_xgb = np.clip(xgb_model.predict(Xva), TARGET_MIN, TARGET_MAX)
        print("XGBoost val metrics:")
        xgb_rmse = evaluate("XGBoost", yva, va_pred_xgb)

        imp = sorted(zip(FEATURE_COLS, xgb_model.feature_importances_),
                     key=lambda x: -x[1])
        print("\n  XGBoost Feature Importances (geometry polarity drivers):")
        for fname, score in imp:
            bar = "█" * int(score * 40)
            print(f"    {fname:42s} {score:.4f} {bar}")

        fitted["xgb"] = xgb_model
        print(f"✅  XGBoost → {XGB_MODEL_PATH}")

    # ── LightGBM ─────────────────────────────────────────────────────────
    if LGB_AVAILABLE:
        print(f"\n{'─'*60}")
        print("Training LightGBM  (num_leaves=64, expanded for tight wick boundaries)...")
        lgb_model = lgb.LGBMRegressor(**LGB_PARAMS)
        lgb_model.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            callbacks=[lgb.early_stopping(60, verbose=False),
                       lgb.log_evaluation(100)],
        )
        lgb_model.booster_.save_model(LGB_MODEL_PATH)
        va_pred_lgb = np.clip(lgb_model.predict(Xva), TARGET_MIN, TARGET_MAX)
        print("LightGBM val metrics:")
        lgb_rmse = evaluate("LightGBM", yva, va_pred_lgb)

        imp = sorted(zip(FEATURE_COLS, lgb_model.feature_importances_),
                     key=lambda x: -x[1])
        print("\n  LightGBM Feature Importances (geometry polarity drivers):")
        max_imp = max(s for _, s in imp)
        for fname, score in imp:
            bar = "█" * int(score / max_imp * 40)
            print(f"    {fname:42s} {score:6d} {bar}")

        fitted["lgb"] = lgb_model
        print(f"✅  LightGBM → {LGB_MODEL_PATH}")

    # ── Ensemble blend ────────────────────────────────────────────────────
    if len(fitted) == 2:
        va_blend = np.clip(
            0.5 * va_pred_xgb + 0.5 * va_pred_lgb, TARGET_MIN, TARGET_MAX
        )
        print(f"\n{'─'*60}")
        print("Ensemble (XGB 50% + LGB 50%) val metrics:")
        evaluate("Ensemble", yva, va_blend)

    # ── Save full ensemble bundle ─────────────────────────────────────────
    bundle = {
        "feature_names":      FEATURE_COLS,
        "scaler":             None,
        "models":             fitted,
        "target_range":       (TARGET_MIN, TARGET_MAX),
        "polarity_threshold": 0.15,   # |R_p| < 0.15 treated as neutral
    }
    joblib.dump(bundle, ENSEMBLE_PATH)
    print(f"\n✅  Ensemble bundle → {ENSEMBLE_PATH}")
    print(f"\nInference note:")
    print(f"  R_p < -0.15 → Lower wick absorption — demand floor, buyer rejection signal")
    print(f"  R_p ∈ [-0.15, +0.15] → Balanced geometry — no clear wick bias")
    print(f"  R_p > +0.15 → Upper wick supply ceiling — seller rejection signal")


if __name__ == "__main__":
    main()
