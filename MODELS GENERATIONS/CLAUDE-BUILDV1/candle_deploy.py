# candle_deploy.py
# Live inference for the Candle Sequence model
# Computes 30 features EXACTLY matching synthetic_candle.py
# Outputs 6 head predictions + confidence dict
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import torch
import torch.nn as nn
import joblib

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

try:
    from utils import resource_path
except ImportError:
    def resource_path(p): return p

EPS  = 1e-12
WIN  = 10       # sequence window
LOOKBACK = 60   # bars fetched from MT5 (need rolling context)


# ─────────────────────────────────────────────────────────────────────
# Architecture  — must match train_candle.py exactly
# ─────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, dim, drop=0.25):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(drop),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.seq(x) + x)


class CandleNet(nn.Module):
    HEAD_SIZES = [6, 3, 3, 3, 4, 3]

    def __init__(self, input_dim, trunk=256, drop=0.25):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, trunk),
            nn.BatchNorm1d(trunk), nn.GELU(), nn.Dropout(drop)
        )
        self.res = nn.Sequential(
            ResidualBlock(trunk, drop),
            ResidualBlock(trunk, drop),
        )
        self.mid = nn.Sequential(
            nn.Linear(trunk, trunk // 2),
            nn.BatchNorm1d(trunk // 2), nn.GELU(), nn.Dropout(drop * 0.7)
        )
        hs = self.HEAD_SIZES
        h  = trunk // 2
        self.h0 = nn.Linear(h, hs[0])
        self.h1 = nn.Linear(h, hs[1])
        self.h2 = nn.Linear(h, hs[2])
        self.h3 = nn.Linear(h, hs[3])
        self.h4 = nn.Linear(h, hs[4])
        self.h5 = nn.Linear(h, hs[5])

    def forward(self, x):
        h = self.mid(self.res(self.stem(x)))
        return self.h0(h), self.h1(h), self.h2(h), self.h3(h), self.h4(h), self.h5(h)


# ─────────────────────────────────────────────────────────────────────
# Feature computation — identical to synthetic_candle.py
# ─────────────────────────────────────────────────────────────────────

def _ols_slope(arr):
    n = len(arr)
    if n < 2: return 0.0
    s = np.std(arr)
    if s < EPS: return 0.0
    x = np.arange(n, dtype=float)
    slope = np.polyfit(x, arr, 1)[0]
    return float(slope / (s + EPS))


def _safe_corr(a, b):
    if np.std(a) < EPS or np.std(b) < EPS: return 0.0
    try:
        r, _ = sp_stats.pearsonr(a, b)
        return float(r) if np.isfinite(r) else 0.0
    except Exception:
        return 0.0


def compute_candle_features(df: pd.DataFrame) -> dict:
    """
    Input : DataFrame with columns [open, high, low, close, volume]
            At least 50 rows recommended for stable rolling context.
    Output: dict of 30 features.
    """
    df = df.copy().reset_index(drop=True)

    # ── full-df base columns ─────────────────────────────────────────
    df["range"]      = df["high"] - df["low"]
    df["body"]       = df["close"] - df["open"]
    df["abs_body"]   = df["body"].abs()
    df["dir"]        = np.sign(df["body"]).fillna(0)
    df["upper_wick"] = df["high"] - df[["open","close"]].max(axis=1)
    df["lower_wick"] = df[["open","close"]].min(axis=1) - df["low"]
    df["atr14"]      = df["range"].rolling(14, min_periods=1).mean()
    df["abs_body_avg30"] = df["abs_body"].rolling(30, min_periods=1).mean()
    df["range_avg30"]    = df["range"].rolling(30,  min_periods=1).mean()
    df["vol_avg20"]      = df["volume"].rolling(20, min_periods=1).mean()
    df["body_ratio"]     = df["abs_body"] / (df["range"] + EPS)
    df["body_rat_avg5"]  = df["body_ratio"].rolling(5, min_periods=1).mean()
    df["close_grav"]     = (df["close"] - df["low"]) / (df["range"] + EPS)
    df["shadow_dom"]     = (df["lower_wick"] - df["upper_wick"]) / (df["range"] + EPS)
    bull = df["dir"] >= 0
    df["dir_eff"]        = np.where(bull,
                                     df["body_ratio"] * df["close_grav"],
                                     df["body_ratio"] * (1 - df["close_grav"]))
    df["vol_weight"]     = (df["volume"] / (df["vol_avg20"] + EPS)).clip(0, 4)

    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_open  = ha_close.ewm(span=3, adjust=False).mean()
    df["ha_body"] = (ha_close - ha_open) / (df["atr14"] + EPS)

    cons, cur, prev = [], 0, 0
    for d in df["dir"].fillna(0).tolist():
        if d == prev and d != 0:  cur += 1
        elif d != 0:               cur  = 1
        else:                      cur  = 0
        cons.append(cur)
        prev = d
    df["consec_len"] = cons

    # ── last WIN bars ────────────────────────────────────────────────
    w       = df.iloc[-WIN:].reset_index(drop=True)
    atr_val = float(df["atr14"].iloc[-1])
    f       = {}

    # GROUP A — last-bar anatomy (5)
    f["lb_close_grav"] = float(w["close_grav"].iloc[-1])
    f["lb_body_ratio"] = float(w["body_ratio"].iloc[-1])
    f["lb_shadow_dom"] = float(w["shadow_dom"].iloc[-1])
    f["lb_vol_weight"] = float(min(w["vol_weight"].iloc[-1], 4.0))
    f["lb_dir_eff"]    = float(w["dir_eff"].iloc[-1])

    # GROUP B — sequence narrative (8)
    f["seq_cg_slope"]    = _ols_slope(w["close_grav"].values)
    bm = float(w["body_ratio"].mean())
    bs = float(w["body_ratio"].std())
    f["body_entropy"]    = float(bs / (bm + EPS))
    f["wick_press_slope"]= _ols_slope(w["shadow_dom"].values)
    bar_conv = w["dir"] * w["body_ratio"] * w["vol_weight"]
    f["vol_narrative"]   = _ols_slope(bar_conv.values)
    f["seq_harmony"]     = _safe_corr(w["close_grav"].values, w["range"].values)
    early_dir  = np.sign(w["body"].iloc[:5].sum())
    late_dir   = np.sign(w["body"].iloc[5:].sum())
    f["handoff"]         = float(early_dir != late_dir)
    early_body = float(w["body_ratio"].iloc[:5].mean())
    late_body  = float(w["body_ratio"].iloc[5:].mean())
    f["handoff_str"]     = float((late_body - early_body) / (atr_val + EPS))
    full_mean  = float(w["body_ratio"].mean()) + EPS
    f["seq_accel"]       = float((late_body - early_body) / full_mean)

    # GROUP C — pattern physics (7)
    curr_body  = float(w["abs_body"].iloc[-1])
    prior_body = float(w["abs_body"].iloc[-2]) if len(w) >= 2 else curr_body
    curr_dir   = float(w["dir"].iloc[-1])
    f["engulf_qual"]     = float((curr_body - prior_body) / (atr_val + EPS) * (curr_dir or 1))

    inside_count = 0
    for i in range(1, len(w)):
        if w["high"].iloc[i] < w["high"].iloc[i-1] and w["low"].iloc[i] > w["low"].iloc[i-1]:
            inside_count += 1
        else:
            break
    mother_range = float(w["range"].iloc[0]) if inside_count > 0 else 0.0
    f["inside_energy"]   = float(inside_count * mother_range / (atr_val + EPS))

    f["ha_body_last"]    = float(w["ha_body"].iloc[-1])
    ha_arr = w["ha_body"].values
    f["ha_body_slope"]   = float((ha_arr[-1] - ha_arr[max(0, len(ha_arr)-4)]) / (atr_val + EPS))

    uw  = float(w["upper_wick"].iloc[-1])
    lw  = float(w["lower_wick"].iloc[-1])
    rng = float(w["range"].iloc[-1])
    br  = float(w["body_ratio"].iloc[-1])
    cg  = float(w["close_grav"].iloc[-1])
    body_small = 1.0 if br < 0.35 else 0.5
    f["pin_bull_qual"]   = float((lw / (rng + EPS)) * body_small * cg)
    f["pin_bear_qual"]   = float((uw / (rng + EPS)) * body_small * (1 - cg))

    closes  = w["close"].values
    c_min, c_max = float(closes.min()), float(closes.max())
    seq_pos = (closes[-1] - c_min) / (c_max - c_min + EPS)
    doji_score = float(max(0, 0.15 - br) / 0.15)
    f["doji_context"]    = float(doji_score * (seq_pos - 0.5))

    # GROUP D — relative sizing (3)
    f["body_vs_avg"]     = float(min(float(df["abs_body"].iloc[-1]) /
                                      (float(df["abs_body_avg30"].iloc[-1]) + EPS), 4.0))
    f["range_vs_avg"]    = float(min(float(df["range"].iloc[-1]) /
                                      (float(df["range_avg30"].iloc[-1]) + EPS), 4.0))
    br5_avg = float(df["body_rat_avg5"].iloc[-1])
    f["body_accel"]      = float(br - br5_avg)

    # GROUP E — fractal (2)
    br1  = float(w["body_ratio"].iloc[-1])
    br3  = float(w["body_ratio"].iloc[-3:].mean()) if len(w) >= 3 else br1
    br10 = float(w["body_ratio"].mean())
    price_up = closes[-1] > closes[0]
    f["fractal_bull"]    = float((br1 > br3 > br10) and price_up)
    f["fractal_bear"]    = float((br1 > br3 > br10) and not price_up)

    # GROUP F — supplementary (5)
    f["seq_dir_ratio"]   = float((w["dir"] > 0).mean())
    f["consec_len"]      = float(w["consec_len"].iloc[-1])
    vol_surge_w          = float(w["vol_weight"].mean())
    body_ratio_mean_w    = float(w["body_ratio"].mean()) + EPS
    f["vol_climax_div"]  = float(body_ratio_mean_w / (vol_surge_w + EPS))
    f["ha_trend_str"]    = float(w["ha_body"].ewm(span=5, adjust=False).mean().iloc[-1])
    f["seq_body_avg"]    = float(w["body_ratio"].mean())

    return f


# ─────────────────────────────────────────────────────────────────────
# Human-readable label maps
# ─────────────────────────────────────────────────────────────────────

HEAD_LABELS = {
    "pattern_type":     {0:"impulse_bull", 1:"impulse_bear", 2:"pin_bull",
                          3:"pin_bear",    4:"inside_coil",  5:"indecision"},
    "seq_dominance":    {0:"bear_dominant", 1:"contested",   2:"bull_dominant"},
    "conviction":       {0:"low",           1:"medium",      2:"high"},
    "wick_pressure":    {0:"upper",         1:"balanced",    2:"lower"},
    "seq_phase":        {0:"coiling",       1:"ignition",    2:"trending",   3:"exhausting"},
    "candle_signal":    {0:"avoid",         1:"wait",        2:"act"},
}


# ─────────────────────────────────────────────────────────────────────
# Predictor
# ─────────────────────────────────────────────────────────────────────

class CandlePredictor:
    def __init__(self,
                 model_path  = "candle_best.pth",
                 scaler_path = "candle_scaler.pkl"):

        model_path  = resource_path(model_path)
        scaler_path = resource_path(scaler_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        bundle = joblib.load(scaler_path)
        self.scaler       = bundle["scaler"]
        self.feature_cols = bundle["feature_names"]

        self.model = CandleNet(input_dim=len(self.feature_cols)).to(self.device)
        sd = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(sd)
        self.model.eval()

        if MT5_AVAILABLE:
            if not mt5.initialize():
                raise RuntimeError("MT5 initialization failed")

    def predict(self, symbol: str, timeframe=None):
        if not MT5_AVAILABLE:
            raise RuntimeError("MetaTrader5 not available")
        if timeframe is None:
            timeframe = mt5.TIMEFRAME_M15

        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, LOOKBACK + 5)
        if rates is None or len(rates) < WIN + 10:
            return {"error": f"Insufficient data: got {0 if rates is None else len(rates)} bars"}

        df = pd.DataFrame(rates)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        return self.predict_from_df(df)

    def predict_from_df(self, df: pd.DataFrame) -> dict:
        """
        df must have columns: open, high, low, close, volume
        Returns dict with all head predictions.
        """
        feat_dict = compute_candle_features(df)

        # Build ordered row matching training feature order
        row    = np.array([[feat_dict.get(c, 0.0) for c in self.feature_cols]], dtype=np.float32)
        scaled = self.scaler.transform(row)
        x      = torch.tensor(scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outs = self.model(x)

        head_names = list(HEAD_LABELS.keys())
        result     = {}
        for i, (name, out) in enumerate(zip(head_names, outs)):
            probs    = torch.softmax(out, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            lmap     = HEAD_LABELS[name]
            result[name] = {
                "class":      pred_idx,
                "label":      lmap[pred_idx],
                "confidence": float(np.max(probs)),
                "probs":      {lmap[k]: round(float(probs[k]), 4)
                               for k in range(len(probs))},
            }

        result["signal"]     = result["candle_signal"]["label"]
        result["confidence"] = result["candle_signal"]["confidence"]
        return result

    def shutdown(self):
        if MT5_AVAILABLE:
            mt5.shutdown()


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    predictor = CandlePredictor()
    result    = predictor.predict("EURUSD")

    print("\n── Candle Sequence Prediction ───────────────────")
    for head, v in result.items():
        if isinstance(v, dict) and "label" in v:
            print(f"  {head:<20} → {v['label']}  (conf={v['confidence']:.2%})")
    print(f"\n  SIGNAL: {result['signal'].upper()}  "
          f"(confidence={result['confidence']:.2%})")
    predictor.shutdown()
