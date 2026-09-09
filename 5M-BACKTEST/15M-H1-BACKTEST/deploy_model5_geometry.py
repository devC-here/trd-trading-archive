# deploy_model5_geometry.py
# Live inference for the Candlestick Geometry Specialist
# Architecture and features match train_model5_geometry.py exactly
# Features : 6  |  Output : continuous scalar R_p ∈ [-1.0, +1.0]
# Ensemble  : XGBoost + LightGBM (average blend)
# ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import joblib

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    from utils import resource_path
except ImportError:
    def resource_path(p): return p

EPS        = 1e-12
LOOKBACK   = 60
WINDOW     = 16
TARGET_MIN = -1.0
TARGET_MAX = +1.0


def compute_geometry_features(df: pd.DataFrame) -> dict:
    df     = df.copy().reset_index(drop=True)
    window = df.tail(WINDOW)

    C = window["close"].values.astype(float)
    O = window["open"].values.astype(float)
    H = window["high"].values.astype(float)
    L = window["low"].values.astype(float)

    TR = np.array(
        [H[0] - L[0]] +
        [max(H[i] - L[i], abs(H[i] - C[i-1]), abs(L[i] - C[i-1]))
         for i in range(1, WINDOW)]
    )
    ATR    = TR.mean()
    bodies = np.abs(C - O)
    ranges = (H - L) + EPS
    total_h = (H[-1] - L[-1]) + EPS

    upper_wick_ratio   = float((H[-1] - max(O[-1], C[-1])) / total_h)
    lower_wick_ratio   = float((min(O[-1], C[-1]) - L[-1]) / total_h)
    var_ranges         = np.var(H - L)
    var_bodies         = np.var(bodies) + EPS
    intraday_var_ratio = float(var_ranges / var_bodies)
    body_to_atr        = float(bodies[-1] / (ATR + EPS))
    rel_body_to_range  = float(np.mean(bodies / ranges))
    lower_wicks        = np.minimum(O, C) - L
    lower_wick_abs     = float(np.mean(lower_wicks / ranges))

    return {
        "97_final_upper_wick_ratio":       upper_wick_ratio,
        "99_final_lower_wick_ratio":       lower_wick_ratio,
        "71_intraday_variance_ratio":      intraday_var_ratio,
        "104_body_size_to_atr_ratio":      body_to_atr,
        "94_relative_body_to_range_ratio": rel_body_to_range,
        "98_lower_wick_absorption_ratio":  lower_wick_abs,
    }


def interpret_rp(rp: float) -> str:
    if rp <= -0.60:   return "strong_demand_floor"
    elif rp <= -0.15: return "lower_wick_absorption"
    elif rp <   0.15: return "balanced_geometry"
    elif rp <   0.60: return "upper_wick_rejection"
    else:             return "strong_supply_ceiling"


class GeometryPredictor:
    def __init__(self, ensemble_path: str = "model5_ensemble.pkl"):
        ensemble_path = resource_path(ensemble_path)
        bundle        = joblib.load(ensemble_path)

        self.scaler        = bundle.get("scaler")
        self.feature_cols  = bundle["feature_names"]
        self.models        = bundle["models"]
        self.target_range  = bundle["target_range"]
        self.pol_threshold = bundle["polarity_threshold"]

        if not self.models:
            raise RuntimeError("No trained models found in ensemble bundle.")

        if MT5_AVAILABLE:
            if not mt5.initialize():
                raise RuntimeError("MT5 initialization failed")

    def predict(self, symbol: str, timeframe=None, start=1) -> dict:
        if not MT5_AVAILABLE:
            raise RuntimeError("MetaTrader5 not available")
        if timeframe is None:
            timeframe = mt5.TIMEFRAME_M15
        rates = mt5.copy_rates_from_pos(symbol, timeframe, start, LOOKBACK)
        if rates is None or len(rates) < WINDOW:
            return {"error": f"Insufficient bars: {0 if rates is None else len(rates)}"}
        df = pd.DataFrame(rates)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        return self.predict_from_df(df)

    def predict_from_df(self, df: pd.DataFrame) -> dict:
        feat_dict = compute_geometry_features(df)
        row       = np.array([[feat_dict[c] for c in self.feature_cols]], dtype=np.float32)

        preds = {}
        for name, model in self.models.items():
            raw = float(model.predict(row)[0])
            preds[name] = float(np.clip(raw, TARGET_MIN, TARGET_MAX))

        rp = float(np.clip(
            np.mean(list(preds.values())) if len(preds) > 1 else list(preds.values())[0],
            TARGET_MIN, TARGET_MAX
        ))

        if rp < -self.pol_threshold:
            polarity = "demand_absorption"
            signal   = "bullish_reversal_candidate"
        elif rp > self.pol_threshold:
            polarity = "supply_rejection"
            signal   = "bearish_reversal_candidate"
        else:
            polarity = "neutral"
            signal   = "no_clear_wick_bias"

        return {
            "rejection_polarity": {
                "R_p":              rp,
                "interpretation":   interpret_rp(rp),
                "polarity":         polarity,
                "individual_scores": preds,
            },
            "signal":   signal,
            "R_p":      rp,
            "features": feat_dict,
        }

    def shutdown(self):
        if MT5_AVAILABLE:
            mt5.shutdown()


if __name__ == "__main__":
    predictor = GeometryPredictor()
    result    = predictor.predict("EURUSD")
    print("\n── Model 5 · Candlestick Geometry ───────────────────")
    rp = result["rejection_polarity"]
    print(f"  R_p score   → {rp['R_p']:+.4f}  (range -1.0 to +1.0)")
    print(f"  state       → {rp['interpretation'].upper()}")
    print(f"  polarity    → {rp['polarity'].upper()}")
    print(f"  signal      → {result['signal'].upper()}")
    if len(rp["individual_scores"]) > 1:
        for m, s in rp["individual_scores"].items():
            print(f"  {m:<10} raw → {s:+.4f}")
    print(f"\n  Features used:")
    for k, v in result["features"].items():
        print(f"    {k:<42} {v:+.6f}")
    predictor.shutdown()
