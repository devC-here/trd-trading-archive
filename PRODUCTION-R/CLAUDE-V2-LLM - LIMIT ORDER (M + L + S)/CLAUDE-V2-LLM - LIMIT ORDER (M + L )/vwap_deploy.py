# vwap_deploy.py
# Live inference for the VWAP / Volume model
# Architecture and features match train_vwap.py + synthetic_vwap.py exactly
# Features : 32  |  Heads : 6  |  HEAD_SIZES : [3, 3, 4, 3, 3, 3]
# Rolling 50-bar VWAP — no session reset needed for FX
# ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from collections import defaultdict
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

EPS      = 1e-12
LOOKBACK = 110          # bars fetched from MT5 (100 used + warmup)
TRUNK    = 256
DROP     = 0.25


# ─────────────────────────────────────────────────────────────────────
# Architecture  ·  must be byte-identical to train_vwap.py
# ─────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, dim, drop=DROP):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(drop),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.seq(x) + x)


class VwapNet(nn.Module):
    # Heads: next_vwap_dir(3), next_vol_char(3), vwap_interaction(4),
    #        next_inst_signal(3), next_close_location(3), vwap_follow_through(3)
    HEAD_SIZES = [3, 3, 4, 3, 3, 3]

    def __init__(self, input_dim, trunk=TRUNK, drop=DROP):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, trunk),
            nn.BatchNorm1d(trunk), nn.GELU(), nn.Dropout(drop),
        )
        self.res = nn.Sequential(
            ResidualBlock(trunk, drop),
            ResidualBlock(trunk, drop),
        )
        self.mid = nn.Sequential(
            nn.Linear(trunk, trunk // 2),
            nn.BatchNorm1d(trunk // 2), nn.GELU(), nn.Dropout(drop * 0.7),
        )
        h = trunk // 2
        self.heads = nn.ModuleList([nn.Linear(h, s) for s in self.HEAD_SIZES])

    def forward(self, x):
        h = self.mid(self.res(self.stem(x)))
        return tuple(head(h) for head in self.heads)


# ─────────────────────────────────────────────────────────────────────
# Feature computation  ·  identical to synthetic_vwap.py
# ─────────────────────────────────────────────────────────────────────

def _rz(s, w):
    return (s - s.rolling(w).mean()) / (s.rolling(w).std() + EPS)


def compute_vwap_features(df: pd.DataFrame,
                           hour_vol_avg: dict = None) -> dict:
    """
    df : DataFrame with [open, high, low, close, volume].
         Optionally include 'hour' column (0-23) for session-adjusted volume.
         Min ~70 rows recommended.
    Returns dict of 32 float features (keys sorted to match training CSV).
    """
    df = df.copy().reset_index(drop=True)

    # ── base anatomy ────────────────────────────────────────────────
    df["range"]      = df["high"] - df["low"]
    df["body"]       = df["close"] - df["open"]
    df["abs_body"]   = df["body"].abs()
    df["dir"]        = np.sign(df["body"]).fillna(0)
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

    hl  = df["range"]
    hc  = (df["high"] - df["close"].shift(1)).abs().fillna(0)
    lc  = (df["low"]  - df["close"].shift(1)).abs().fillna(0)
    df["atr14"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14, min_periods=1).mean()

    # ── Rolling 50-bar VWAP ──────────────────────────────────────────
    tp        = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol    = tp * df["volume"]
    vol_50    = df["volume"].rolling(50, min_periods=1).sum()
    vwap_50   = tp_vol.rolling(50, min_periods=1).sum() / (vol_50 + EPS)
    df["vwap"] = vwap_50

    vwap_sq_dev = (tp - vwap_50) ** 2 * df["volume"]
    vwap_var    = vwap_sq_dev.rolling(50, min_periods=1).sum() / (vol_50 + EPS)
    df["vwap_std"] = np.sqrt(vwap_var.clip(lower=0)) / (df["atr14"] + EPS)

    # ── GROUP A ───────────────────────────────────────────────────────
    df["vwap_dev_atr"]  = (df["close"] - df["vwap"]) / (df["atr14"] + EPS)
    df["vwap_sigma"]    = df["vwap_dev_atr"] / (df["vwap_std"] + EPS)
    df["vwap_slope_z"]  = _rz((df["vwap"].diff(5) / (df["atr14"] + EPS)).fillna(0), 30).fillna(0.0)

    vol_avg20   = df["volume"].rolling(20, min_periods=1).mean()
    vol_z20     = _rz(df["volume"], 20).fillna(0.0)
    vol_z_cross = (vol_z20 + 1).clip(0, 3)
    cross_bull  = ((df["close"] > df["vwap"]) & (df["close"].shift(1) <= df["vwap"].shift(1))).astype(float)
    cross_bear  = ((df["close"] < df["vwap"]) & (df["close"].shift(1) >= df["vwap"].shift(1))).astype(float)
    df["vwap_reclaim_qual"]   = (df["abs_body"] / (df["atr14"] + EPS) * vol_z_cross * cross_bull).fillna(0).clip(0, 3)
    df["vwap_rejection_qual"] = (df["abs_body"] / (df["atr14"] + EPS) * vol_z_cross * cross_bear).fillna(0).clip(0, 3)
    near_vwap                 = (df["vwap_dev_atr"].abs() < 0.50).astype(int)
    df["vwap_touch_density"]  = near_vwap.rolling(20, min_periods=1).sum() / 20.0

    # ── GROUP B ───────────────────────────────────────────────────────
    close_grav       = (df["close"] - df["low"]) / (df["range"] + EPS)
    df["tick_delta"] = close_grav * 2 - 1
    delta_vol        = df["tick_delta"] * df["volume"]
    df["cum_delta_20z"] = _rz(delta_vol.rolling(20, min_periods=1).sum(), 50).fillna(0.0)
    up_vol           = df["volume"].where(df["dir"] > 0, 0.0)
    dn_vol           = df["volume"].where(df["dir"] < 0, 0.0)
    up_sum           = up_vol.rolling(20, min_periods=1).sum()
    dn_sum           = dn_vol.rolling(20, min_periods=1).sum()
    df["vol_profile_asym"] = ((up_sum - dn_sum) / (up_sum + dn_sum + EPS)).fillna(0.0)
    up_frac          = up_sum / (up_sum + dn_sum + EPS)
    df["buy_press_z"]     = _rz(up_frac, 30).fillna(0.0)
    delta_at_vwap         = delta_vol * near_vwap
    df["delta_at_vwap"]   = _rz(delta_at_vwap.rolling(20, min_periods=1).sum(), 30).fillna(0.0)

    # ── GROUP C ───────────────────────────────────────────────────────
    df["vol_z30"]     = _rz(df["volume"], 30).fillna(0.0)
    df["vol_z100"]    = _rz(df["volume"], min(100, len(df))).fillna(0.0)
    df["vol_trend_z"] = _rz(df["volume"].rolling(5, min_periods=1).mean() / (vol_avg20 + EPS), 30).fillna(0.0)

    if "hour" in df.columns and hour_vol_avg is not None:
        session_expected         = df["hour"].map(hour_vol_avg).fillna(vol_avg20)
        df["session_vol_factor"] = (df["volume"] / (session_expected + EPS)).clip(0, 4)
    else:
        df["session_vol_factor"] = (vol_z20.clip(-3, 3) + 1)

    price_dir        = np.sign(df["close"].diff(3).fillna(0))
    vol_dir          = np.sign(df["volume"].rolling(3, min_periods=1).mean().diff(1).fillna(0))
    df["vol_mom_div"]    = (price_dir * vol_dir).fillna(0.0)
    df["vol_efficiency"] = (df["abs_body"] / (df["atr14"] + EPS) /
                             (df["vol_z30"].abs() + 0.5)).clip(0, 3)

    # ── GROUP D ───────────────────────────────────────────────────────
    df["vol_at_vwap_z"]  = (vol_z20 * near_vwap).fillna(0.0)
    body_ratio           = df["abs_body"] / (df["range"] + EPS)
    comp_a               = (vol_z20 > 1.5).astype(int)
    comp_b               = (df["vwap_dev_atr"].abs() < 0.30).astype(int)
    comp_c               = (body_ratio < 0.25).astype(int)
    df["inst_fp"]        = (comp_a + comp_b + comp_c).astype(float)
    above_1sd            = (df["vwap_sigma"] - 1.0).clip(lower=0)
    df["vwap_band_walk"] = above_1sd.rolling(10, min_periods=1).sum()
    df["price_vol_div"]  = (df["abs_body"] / (df["atr14"] + EPS) /
                             (df["vol_z30"].abs() + 0.5)).clip(0, 3)
    df["absorption_lvl"] = (vol_z20 * (body_ratio < 0.25).astype(float)).clip(0, 3)

    # ── GROUP E ───────────────────────────────────────────────────────
    df["vwap_dist_pct"] = ((df["close"] - df["vwap"]) / (df["vwap"] + EPS) * 100).clip(-1, 1)
    above_streak        = []
    cur = 0
    for v in (df["close"] > df["vwap"]).tolist():
        cur = cur + 1 if v else 0
        above_streak.append(cur)
    df["above_vwap_streak"]   = above_streak
    df["vwap_mr_z"]           = _rz(df["vwap_dev_atr"].abs(), 50).fillna(0.0)
    vwap_side                 = np.sign(df["vwap_dev_atr"])
    delta_sign                = np.sign(df["tick_delta"])
    df["cum_delta_vwap_align"] = (vwap_side * delta_sign).fillna(0.0)
    far_vwap                  = (df["vwap_sigma"].abs() > 1.5).astype(float)
    df["vol_surge_extreme"]   = (vol_z20 * far_vwap).clip(0, 3)

    # ── GROUP F ───────────────────────────────────────────────────────
    df["body_ratio"]    = body_ratio
    df["close_grav_f"]  = close_grav
    df["vwap_std_feat"] = df["vwap_std"].clip(0, 5)
    df["atr14_z"]       = _rz(df["atr14"], 50).fillna(0.0)
    df["vol_surge_20"]  = vol_z20.clip(-3, 3)

    last = df.iloc[-1]
    # Keys sorted alphabetically — same order as synthetic_vwap_dataset.csv
    return {
        "above_vwap_streak":    float(last["above_vwap_streak"]),
        "absorption_lvl":       float(last["absorption_lvl"]),
        "atr14_z":              float(last["atr14_z"]),
        "body_ratio":           float(last["body_ratio"]),
        "buy_press_z":          float(last["buy_press_z"]),
        "close_grav_f":         float(last["close_grav_f"]),
        "cum_delta_20z":        float(last["cum_delta_20z"]),
        "cum_delta_vwap_align": float(last["cum_delta_vwap_align"]),
        "delta_at_vwap":        float(last["delta_at_vwap"]),
        "inst_fp":              float(last["inst_fp"]),
        "price_vol_div":        float(last["price_vol_div"]),
        "session_vol_factor":   float(last["session_vol_factor"]),
        "tick_delta":           float(last["tick_delta"]),
        "vol_at_vwap_z":        float(last["vol_at_vwap_z"]),
        "vol_efficiency":       float(last["vol_efficiency"]),
        "vol_mom_div":          float(last["vol_mom_div"]),
        "vol_profile_asym":     float(last["vol_profile_asym"]),
        "vol_surge_20":         float(last["vol_surge_20"]),
        "vol_surge_extreme":    float(last["vol_surge_extreme"]),
        "vol_trend_z":          float(last["vol_trend_z"]),
        "vol_z100":             float(last["vol_z100"]),
        "vol_z30":              float(last["vol_z30"]),
        "vwap_band_walk":       float(last["vwap_band_walk"]),
        "vwap_dev_atr":         float(last["vwap_dev_atr"]),
        "vwap_dist_pct":        float(last["vwap_dist_pct"]),
        "vwap_mr_z":            float(last["vwap_mr_z"]),
        "vwap_reclaim_qual":    float(last["vwap_reclaim_qual"]),
        "vwap_rejection_qual":  float(last["vwap_rejection_qual"]),
        "vwap_sigma":           float(last["vwap_sigma"]),
        "vwap_slope_z":         float(last["vwap_slope_z"]),
        "vwap_std_feat":        float(last["vwap_std_feat"]),
        "vwap_touch_density":   float(last["vwap_touch_density"]),
    }


# ─────────────────────────────────────────────────────────────────────
# Label maps  ·  sourced from synthetic_vwap.py label_maps dict
# ─────────────────────────────────────────────────────────────────────

HEAD_LABELS = {
    "next_vwap_dir":       {0: "below_away",    1: "oscillate_near", 2: "above_away"},
    "next_vol_char":       {0: "declining",     1: "stable",         2: "surge"},
    "vwap_interaction":    {0: "rejection",     1: "test_return",    2: "reclaim_hold",  3: "walk_extension"},
    "next_inst_signal":    {0: "distribution",  1: "neutral",        2: "accumulation"},
    "next_close_location": {0: "lower_third",   1: "mid",            2: "upper_third"},
    "vwap_follow_through": {0: "no",            1: "partial",        2: "confirmed"},
}


# ─────────────────────────────────────────────────────────────────────
# Rolling hour-volume tracker (updated on each live predict() call)
# ─────────────────────────────────────────────────────────────────────
_hour_vol_history: dict = defaultdict(list)


# ─────────────────────────────────────────────────────────────────────
# Predictor
# ─────────────────────────────────────────────────────────────────────

class VwapPredictor:
    def __init__(self,
                 model_path:  str = "vwap_best.pth",
                 scaler_path: str = "vwap_scaler.pkl"):

        model_path  = resource_path(model_path)
        scaler_path = resource_path(scaler_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        bundle            = joblib.load(scaler_path)
        self.scaler       = bundle["scaler"]
        self.feature_cols = bundle["feature_names"]

        self.model = VwapNet(input_dim=len(self.feature_cols)).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()

        if MT5_AVAILABLE:
            if not mt5.initialize():
                raise RuntimeError("MT5 initialization failed")

    def predict(self, symbol: str, timeframe=None) -> dict:
        if not MT5_AVAILABLE:
            raise RuntimeError("MetaTrader5 not available")
        if timeframe is None:
            timeframe = mt5.TIMEFRAME_M15
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 2, LOOKBACK + 5)
        if rates is None or len(rates) < 60:
            return {"error": f"Insufficient bars: {0 if rates is None else len(rates)}"}

        df = pd.DataFrame(rates)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)

        if "time" in df.columns:
            df["hour"] = pd.to_datetime(df["time"], unit="s").dt.hour
        else:
            df["hour"] = 0

        for _, row in df.iterrows():
            h = int(row["hour"])
            _hour_vol_history[h].append(float(row["volume"]))
            if len(_hour_vol_history[h]) > 400:
                _hour_vol_history[h] = _hour_vol_history[h][-400:]

        hour_vol_avg = {h: float(np.mean(v)) for h, v in _hour_vol_history.items()}
        return self.predict_from_df(df, hour_vol_avg=hour_vol_avg)

    def predict_from_df(self, df: pd.DataFrame,
                         hour_vol_avg: dict = None) -> dict:
        feat_dict = compute_vwap_features(df, hour_vol_avg=hour_vol_avg)
        row    = np.array([[feat_dict.get(c, 0.0) for c in self.feature_cols]], dtype=np.float32)
        scaled = self.scaler.transform(row)
        x      = torch.tensor(scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outs = self.model(x)

        result: dict = {}
        for name, out in zip(HEAD_LABELS.keys(), outs):
            probs    = torch.softmax(out, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            lmap     = HEAD_LABELS[name]
            result[name] = {
                "class":      pred_idx,
                "label":      lmap[pred_idx],
                "confidence": float(probs.max()),
                "probs":      {lmap[k]: round(float(probs[k]), 4) for k in range(len(probs))},
            }

        result["signal"]        = result["next_vwap_dir"]["label"]
        result["confidence"]    = result["next_vwap_dir"]["confidence"]
        result["follow_through"] = result["vwap_follow_through"]["label"]
        result["trade_active"]  = bool(
            result["next_vwap_dir"]["class"] != 1          # not oscillating
            and result["vwap_follow_through"]["class"] >= 1   # at least partial
        )
        return result

    def shutdown(self):
        if MT5_AVAILABLE:
            mt5.shutdown()


# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    predictor = VwapPredictor()
    result    = predictor.predict("EURUSD", mt5.TIMEFRAME_H1)
    print("\n── VWAP / Volume Prediction ──────────────────────────")
    for head, v in result.items():
        if isinstance(v, dict) and "label" in v:
            print(f"  {head:<22} → {v['label']:<16} conf={v['confidence']:.2%}")
    print(f"\n  vwap_dir      : {result['signal'].upper()}")
    print(f"  follow-through: {result['follow_through'].upper()}")
    print(f"  trade_active  : {result['trade_active']}  (conf={result['confidence']:.2%})")
    predictor.shutdown()
