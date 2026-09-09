# train_model4_volume.py  ·  v1
# ─────────────────────────────────────────────────────────────────────
# MODEL 4 — INSTITUTIONAL VOLUME & LIQUIDITY SPECIALIST
#
# Dataset    : model4_volume_dataset.csv
# Features   : 11 (90_money_flow_acceleration, 37_volume_trend_slope,
#                   92_volume_to_range_ratio_final, 93_mean_volume_to_range_ratio,
#                   86_volume_oscillator, 81_volume_rate_of_change,
#                   84_negative_volume_index, 80_relative_volume_z_score,
#                   89_volume_climax_filter, 82_force_index_final,
#                   88_range_volume_correlation)
# Target     : I_c ∈ [0.0, +5.0]  — institutional commitment intensity
#              0.0  = dead session, zero participation
#              2.5+ = high-conviction breakout confirmation
#              5.0  = extreme institutional influx (climax)
# Preprocess : MinMaxScaler REQUIRED — 81_volume_rate_of_change spikes to
#              +70,587% on institutional flow bars. StandardScaler would
#              squash normal rows into indistinguishable lines.
# Algorithm  : Gradient Boosted Trees (XGBoost + LightGBM ensemble)
#              Tree ensembles handle asymmetric spikes natively after
#              MinMax compression, and don't need a warm-up scheduler.
# ─────────────────────────────────────────────────────────────────────

import os, random, warnings
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.preprocessing import MinMaxScaler
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
DATA_PATH        = "model4_volume_dataset.csv"
SCALER_PATH      = "model4_scaler.pkl"
XGB_MODEL_PATH   = "model4_xgb.json"
LGB_MODEL_PATH   = "model4_lgb.txt"
ENSEMBLE_PATH    = "model4_ensemble.pkl"   # stores both models + scaler + meta

VAL_RATIO = 0.12
TARGET_MIN = 0.0
TARGET_MAX = 5.0

FEATURE_COLS = [
    "90_money_flow_acceleration",
    "37_volume_trend_slope",
    "92_volume_to_range_ratio_final",
    "93_mean_volume_to_range_ratio",
    "86_volume_oscillator",
    "81_volume_rate_of_change",
    "84_negative_volume_index",
    "80_relative_volume_z_score",
    "89_volume_climax_filter",
    "82_force_index_final",
    "88_range_volume_correlation",
]

# label_volume: 0 = HOLD (low commitment), 1 = BUY/SELL (high commitment)
# We convert this binary label into a continuous I_c target:
#   HOLD  → sampled from N(0.8, 0.3) clipped to [0.0, 1.8]   (low zone)
#   BREAK → sampled from N(3.2, 0.8) clipped to [1.5, 5.0]   (high zone)
# This gives the regression model a smooth intensity surface to learn.
TARGET_COL = "label_volume"


# ─────────────────────────────────────────────────────────────────────
# TARGET CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────

def build_intensity_target(label_series, seed=SEED):
    """
    Convert binary label_volume into a continuous I_c ∈ [0.0, 5.0].
    HOLD rows  → low intensity zone  [0.0, 1.8]
    BUY/SELL   → high intensity zone [1.5, 5.0]
    The overlap band [1.5, 1.8] represents ambiguous edge cases.
    """
    rng = np.random.RandomState(seed)
    y   = np.zeros(len(label_series), dtype=np.float32)
    hold_mask  = label_series == 0
    break_mask = label_series == 1

    y[hold_mask]  = np.clip(rng.normal(0.8, 0.35, hold_mask.sum()),  TARGET_MIN, 1.8)
    y[break_mask] = np.clip(rng.normal(3.2, 0.80, break_mask.sum()), 1.5, TARGET_MAX)
    return y


# ─────────────────────────────────────────────────────────────────────
# XGBoost MODEL
# ─────────────────────────────────────────────────────────────────────

XGB_PARAMS = {
    "objective":        "reg:squarederror",
    "eval_metric":      "rmse",
    "n_estimators":     800,
    "learning_rate":    0.05,
    "max_depth":        6,
    "min_child_weight": 5,
    "subsample":        0.80,
    "colsample_bytree": 0.75,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "random_state":     SEED,
    "n_jobs":           -1,
    "tree_method":      "hist",
}


# ─────────────────────────────────────────────────────────────────────
# LightGBM MODEL
# ─────────────────────────────────────────────────────────────────────

LGB_PARAMS = {
    "objective":          "regression",
    "metric":             "rmse",
    "n_estimators":       800,
    "learning_rate":      0.05,
    "max_depth":          6,
    "num_leaves":         48,
    "min_child_samples":  20,
    "subsample":          0.80,
    "colsample_bytree":   0.75,
    "reg_alpha":          0.1,
    "reg_lambda":         1.0,
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
    # Intensity-zone accuracy: correct low/high zone (threshold 1.8)
    z_true = (y_true >= 1.8).astype(int)
    z_pred = (y_pred >= 1.8).astype(int)
    zone_acc = (z_true == z_pred).mean()
    print(f"  {name:10s} | RMSE={rmse:.4f}  MAE={mae:.4f}  ZoneAcc={zone_acc:.4f}")
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

    X_raw = df[FEATURE_COLS].fillna(0.0).values
    y_bin = df[TARGET_COL].astype(int).values
    print(f"Features : {len(FEATURE_COLS)}  |  Samples : {len(df)}")
    print(f"Binary label dist: {dict(Counter(y_bin.tolist()))}")

    # ── Build continuous I_c target ──────────────────────────────────────
    y = build_intensity_target(y_bin)
    print(f"I_c target : min={y.min():.3f}  max={y.max():.3f}  mean={y.mean():.3f}")

    # ── MinMaxScaler (MANDATORY — volume_rate_of_change spikes to +70587%) ──
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    X = scaler.fit_transform(X_raw)
    joblib.dump({"feature_names": FEATURE_COLS, "scaler": scaler}, SCALER_PATH)
    print(f"Scaler  → {SCALER_PATH}  (MinMaxScaler [0,1] fitted on {len(df):,} rows)")

    # ── Train / val split ────────────────────────────────────────────────
    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=VAL_RATIO, random_state=SEED, shuffle=True
    )
    print(f"\nTrain: {len(Xtr):,}  |  Val: {len(Xva):,}")

    fitted = {}

    # ── XGBoost ──────────────────────────────────────────────────────────
    if XGB_AVAILABLE:
        print(f"\n{'─'*60}")
        print("Training XGBoost ...")
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

        # Feature importance
        imp = sorted(zip(FEATURE_COLS, xgb_model.feature_importances_),
                     key=lambda x: -x[1])
        print("\n  XGBoost Feature Importances:")
        for fname, score in imp:
            bar = "█" * int(score * 40)
            print(f"    {fname:42s} {score:.4f} {bar}")

        fitted["xgb"] = xgb_model
        print(f"✅  XGBoost → {XGB_MODEL_PATH}")

    # ── LightGBM ─────────────────────────────────────────────────────────
    if LGB_AVAILABLE:
        print(f"\n{'─'*60}")
        print("Training LightGBM ...")
        lgb_model = lgb.LGBMRegressor(**LGB_PARAMS)
        lgb_model.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(100)],
        )
        lgb_model.booster_.save_model(LGB_MODEL_PATH)
        va_pred_lgb = np.clip(lgb_model.predict(Xva), TARGET_MIN, TARGET_MAX)
        print("LightGBM val metrics:")
        lgb_rmse = evaluate("LightGBM", yva, va_pred_lgb)

        # Feature importance
        imp = sorted(zip(FEATURE_COLS, lgb_model.feature_importances_),
                     key=lambda x: -x[1])
        print("\n  LightGBM Feature Importances:")
        for fname, score in imp:
            bar = "█" * int(score / max(s for _, s in imp) * 40)
            print(f"    {fname:42s} {score:6d} {bar}")

        fitted["lgb"] = lgb_model
        print(f"✅  LightGBM → {LGB_MODEL_PATH}")

    # ── Ensemble blend (average if both available) ────────────────────────
    if len(fitted) == 2:
        va_blend = np.clip(
            0.5 * va_pred_xgb + 0.5 * va_pred_lgb, TARGET_MIN, TARGET_MAX
        )
        print(f"\n{'─'*60}")
        print("Ensemble (XGB 50% + LGB 50%) val metrics:")
        evaluate("Ensemble", yva, va_blend)

    # ── Save full ensemble bundle ─────────────────────────────────────────
    bundle = {
        "feature_names": FEATURE_COLS,
        "scaler":        scaler,
        "models":        fitted,
        "target_range":  (TARGET_MIN, TARGET_MAX),
        "intensity_threshold": 1.8,   # < 1.8 = low, >= 1.8 = institutional
    }
    joblib.dump(bundle, ENSEMBLE_PATH)
    print(f"\n✅  Ensemble bundle → {ENSEMBLE_PATH}")
    print(f"\nInference note:")
    print(f"  I_c < 1.8  → Low participation / retail trap risk")
    print(f"  I_c 1.8–2.5 → Moderate institutional interest")
    print(f"  I_c > 2.5  → High-conviction institutional flow confirmed")


if __name__ == "__main__":
    main()
