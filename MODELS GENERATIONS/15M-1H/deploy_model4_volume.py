# deploy_model4_volume.py
# Live inference for the Institutional Volume & Liquidity Specialist
# Architecture and features match train_model4_volume.py exactly
# Features : 11  |  Output : continuous scalar I_c ∈ [0.0, +5.0]
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

EPS      = 1e-12
LOOKBACK = 60           # bars fetched from MT5 (16 used + warmup)
WINDOW   = 16

TARGET_MIN = 0.0
TARGET_MAX = 5.0


# ─────────────────────────────────────────────────────────────────────
# Feature computation  ·  matches model4_volume_generator.py
# ─────────────────────────────────────────────────────────────────────

def _ols_slope(y):
    n  = len(y); x = np.arange(n, dtype=float)
    mx, my = x.mean(), y.mean()
    return float(((x - mx) * (y - my)).sum() / (((x - mx) ** 2).sum() + EPS))


def _ema_val(series, period):
    k = 2.0 / (period + 1)
    e = float(series[0])
    for v in series[1:]: e = float(v) * k + e * (1 - k)
    return e


def _pearson_r(x, y):
    mx, my = x.mean(), y.mean()
    num = ((x - mx) * (y - my)).sum()
    den = np.sqrt(((x - mx) ** 2).sum() * ((y - my) ** 2).sum())
    return float(num / (den + EPS))


def compute_volume_features(df: pd.DataFrame) -> dict:
    """
    df : DataFrame with [open, high, low, close, volume]. Min 16 rows.
    Returns dict of 11 float features (raw — MinMaxScaler applied externally).
    """
    df     = df.copy().reset_index(drop=True)
    window = df.tail(WINDOW)

    C = window["close"].values.astype(float)
    H = window["high"].values.astype(float)
    L = window["low"].values.astype(float)
    V = window["volume"].values.astype(float)

    TP = (H + L + C) / 3.0

    # 90_money_flow_acceleration
    mf_accel = float(TP[-1] * V[-1] - TP[-2] * V[-2])

    # 37_volume_trend_slope
    vol_slope = _ols_slope(V)

    # 92_volume_to_range_ratio_final
    final_rng_pips = (H[-1] - L[-1]) * 10000
    v2r_final = float(V[-1] / (final_rng_pips + EPS))

    # 93_mean_volume_to_range_ratio
    v2r_all  = V / ((H - L) * 10000 + EPS)
    v2r_mean = float(v2r_all.mean())

    # 86_volume_oscillator
    e3 = _ema_val(V, 3); e8 = _ema_val(V, 8)
    vol_osc = float(((e3 - e8) / (e8 + EPS)) * 100)

    # 81_volume_rate_of_change
    vol_roc = float(((V[-1] - V[-4]) / (V[-4] + EPS)) * 100)

    # 84_negative_volume_index
    nvi = 1000.0
    for i in range(1, WINDOW):
        if V[i] < V[i - 1]:
            nvi += ((C[i] - C[i - 1]) / (C[i - 1] + EPS)) * nvi

    # 80_relative_volume_z_score
    vm, vs = V.mean(), max(V.std(), EPS)
    rv_z = float((V[-1] - vm) / vs)

    # 89_volume_climax_filter
    climax = int(V[-1] > vm + 2 * vs)

    # 82_force_index_final
    force = float((C[-1] - C[-2]) * V[-1])

    # 88_range_volume_correlation
    ranges  = H - L
    rv_corr = _pearson_r(ranges, V)

    return {
        "90_money_flow_acceleration":     mf_accel,
        "37_volume_trend_slope":          vol_slope,
        "92_volume_to_range_ratio_final": v2r_final,
        "93_mean_volume_to_range_ratio":  v2r_mean,
        "86_volume_oscillator":           vol_osc,
        "81_volume_rate_of_change":       vol_roc,
        "84_negative_volume_index":       nvi,
        "80_relative_volume_z_score":     rv_z,
        "89_volume_climax_filter":        float(climax),
        "82_force_index_final":           force,
        "88_range_volume_correlation":    rv_corr,
    }


# ─────────────────────────────────────────────────────────────────────
# Intensity interpretation
# ─────────────────────────────────────────────────────────────────────

def interpret_ic(ic: float) -> str:
    if ic < 0.80:   return "dead_session_no_flow"
    elif ic < 1.80: return "low_retail_activity"
    elif ic < 2.50: return "moderate_institutional_interest"
    elif ic < 3.50: return "high_conviction_flow"
    else:           return "extreme_institutional_influx"


# ─────────────────────────────────────────────────────────────────────
# Predictor
# ─────────────────────────────────────────────────────────────────────

class VolumePredictor:
    def __init__(self, ensemble_path: str = "model4_ensemble.pkl"):

        ensemble_path = resource_path(ensemble_path)
        bundle        = joblib.load(ensemble_path)

        self.scaler       = bundle["scaler"]          # MinMaxScaler — MANDATORY
        self.feature_cols = bundle["feature_names"]
        self.models       = bundle["models"]          # dict: {"xgb": ..., "lgb": ...}
        self.target_range = bundle["target_range"]    # (0.0, 5.0)
        self.threshold    = bundle["intensity_threshold"]  # 1.8

        if not self.models:
            raise RuntimeError("No trained models found in ensemble bundle.")

        if MT5_AVAILABLE:
            if not mt5.initialize():
                raise RuntimeError("MT5 initialization failed")

    def predict(self, symbol: str, timeframe=None) -> dict:
        if not MT5_AVAILABLE:
            raise RuntimeError("MetaTrader5 not available")
        if timeframe is None:
            timeframe = mt5.TIMEFRAME_M15
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, LOOKBACK)
        if rates is None or len(rates) < WINDOW:
            return {"error": f"Insufficient bars: {0 if rates is None else len(rates)}"}
        df = pd.DataFrame(rates)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        return self.predict_from_df(df)

    def predict_from_df(self, df: pd.DataFrame) -> dict:
        feat_dict = compute_volume_features(df)
        row = np.array([[feat_dict[c] for c in self.feature_cols]], dtype=np.float32)

        # MinMaxScaler applied — critical for vol_rate_of_change spike protection
        scaled = self.scaler.transform(row)
        # Wrap in DataFrame so LightGBM receives the feature names it was trained with
        scaled_df = pd.DataFrame(scaled, columns=self.feature_cols)

        preds = {}
        for name, model in self.models.items():
            raw = float(model.predict(scaled_df)[0])
            preds[name] = float(np.clip(raw, TARGET_MIN, TARGET_MAX))

        if len(preds) > 1:
            ic = float(np.mean(list(preds.values())))
        else:
            ic = list(preds.values())[0]

        ic = float(np.clip(ic, TARGET_MIN, TARGET_MAX))

        return {
            "volume_intensity": {
                "I_c":              ic,
                "interpretation":   interpret_ic(ic),
                "institutional":    bool(ic >= self.threshold),
                "individual_scores": preds,
            },
            "signal":          "confirmed" if ic >= self.threshold else "unconfirmed",
            "I_c":             ic,
            "trade_confirmed": bool(ic >= 2.50),   # high-conviction gate
            "features":        feat_dict,
        }

    def shutdown(self):
        if MT5_AVAILABLE:
            mt5.shutdown()


# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    predictor = VolumePredictor()
    result    = predictor.predict("EURUSD")
    print("\n── Model 4 · Institutional Volume Intensity ─────────")
    vi = result["volume_intensity"]
    print(f"  I_c score   → {vi['I_c']:.4f}  (range 0.0 – 5.0)")
    print(f"  state       → {vi['interpretation'].upper()}")
    print(f"  institutional→ {'YES' if vi['institutional'] else 'NO'} (threshold ≥ 1.80)")
    print(f"  trade gate  → {'CONFIRMED' if result['trade_confirmed'] else 'UNCONFIRMED (I_c < 2.50)'}")
    if len(vi["individual_scores"]) > 1:
        for m, s in vi["individual_scores"].items():
            print(f"  {m:<10} raw → {s:.4f}")
    print(f"\n  Features used (raw, pre-scale):")
    for k, v in result["features"].items():
        print(f"    {k:<42} {v:+.4f}")
    predictor.shutdown()
