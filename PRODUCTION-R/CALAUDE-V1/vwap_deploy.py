# vwap_deploy.py
# Live inference for the VWAP / Volume model
# Computes 32 features EXACTLY matching synthetic_vwap.py
# Rolling 50-bar VWAP — no session reset needed for FX
# NO ta library dependency — pure pandas/numpy
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
LOOKBACK = 110   # bars; 100 used for features, extra for warmup


# ─────────────────────────────────────────────────────────────────────
# Architecture — must match train_vwap.py exactly
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


class VwapNet(nn.Module):
    HEAD_SIZES = [5, 4, 3, 4, 3, 3]

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
# Feature computation — identical to synthetic_vwap.py
# ─────────────────────────────────────────────────────────────────────

def _rz(s, w):
    return (s - s.rolling(w).mean()) / (s.rolling(w).std() + EPS)


def compute_vwap_features(df: pd.DataFrame,
                           hour_vol_avg: dict = None) -> dict:
    """
    Input : DataFrame with [open, high, low, close, volume].
            Optionally include 'hour' column (0-23) for session-adjusted volume.
            Min ~70 bars recommended.
    Output: dict of 32 features (no _ context keys).
    """
    df = df.copy().reset_index(drop=True)

    # base
    df["range"]      = df["high"] - df["low"]
    df["body"]       = df["close"] - df["open"]
    df["abs_body"]   = df["body"].abs()
    df["dir"]        = np.sign(df["body"]).fillna(0)
    df["upper_wick"] = df["high"] - df[["open","close"]].max(axis=1)
    df["lower_wick"] = df[["open","close"]].min(axis=1) - df["low"]

    hl = df["range"]
    hc = (df["high"] - df["close"].shift(1)).abs().fillna(0)
    lc = (df["low"]  - df["close"].shift(1)).abs().fillna(0)
    df["atr14"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14, min_periods=1).mean()

    # Rolling 50-bar VWAP
    tp        = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol    = tp * df["volume"]
    vol_50    = df["volume"].rolling(50, min_periods=1).sum()
    vwap_50   = tp_vol.rolling(50, min_periods=1).sum() / (vol_50 + EPS)
    df["vwap"]= vwap_50

    vwap_sq_dev = (tp - vwap_50) ** 2 * df["volume"]
    vwap_var    = vwap_sq_dev.rolling(50, min_periods=1).sum() / (vol_50 + EPS)
    df["vwap_std"] = np.sqrt(vwap_var.clip(lower=0)) / (df["atr14"] + EPS)

    # GROUP A
    df["vwap_dev_atr"]      = (df["close"] - df["vwap"]) / (df["atr14"] + EPS)
    df["vwap_sigma"]        = df["vwap_dev_atr"] / (df["vwap_std"] + EPS)
    df["vwap_slope_z"]      = _rz((df["vwap"].diff(5) / (df["atr14"]+EPS)).fillna(0), 30).fillna(0.0)

    vol_avg20 = df["volume"].rolling(20, min_periods=1).mean()
    vol_z20   = _rz(df["volume"], 20).fillna(0.0)
    vol_z_cross = (vol_z20 + 1).clip(0, 3)
    cross_bull  = ((df["close"] > df["vwap"]) & (df["close"].shift(1) <= df["vwap"].shift(1))).astype(float)
    cross_bear  = ((df["close"] < df["vwap"]) & (df["close"].shift(1) >= df["vwap"].shift(1))).astype(float)
    df["vwap_reclaim_qual"]   = (df["abs_body"]/(df["atr14"]+EPS) * vol_z_cross * cross_bull).fillna(0).clip(0,3)
    df["vwap_rejection_qual"] = (df["abs_body"]/(df["atr14"]+EPS) * vol_z_cross * cross_bear).fillna(0).clip(0,3)
    near_vwap                 = (df["vwap_dev_atr"].abs() < 0.50).astype(int)
    df["vwap_touch_density"]  = near_vwap.rolling(20, min_periods=1).sum() / 20.0

    # GROUP B
    close_grav    = (df["close"] - df["low"]) / (df["range"] + EPS)
    df["tick_delta"] = close_grav * 2 - 1
    delta_vol     = df["tick_delta"] * df["volume"]
    df["cum_delta_20z"] = _rz(delta_vol.rolling(20, min_periods=1).sum(), 50).fillna(0.0)
    up_vol  = df["volume"].where(df["dir"] > 0, 0.0)
    dn_vol  = df["volume"].where(df["dir"] < 0, 0.0)
    up_sum  = up_vol.rolling(20, min_periods=1).sum()
    dn_sum  = dn_vol.rolling(20, min_periods=1).sum()
    df["vol_profile_asym"] = ((up_sum - dn_sum) / (up_sum + dn_sum + EPS)).fillna(0.0)
    up_frac = up_sum / (up_sum + dn_sum + EPS)
    df["buy_press_z"]      = _rz(up_frac, 30).fillna(0.0)
    delta_at_vwap = delta_vol * near_vwap
    df["delta_at_vwap"]    = _rz(delta_at_vwap.rolling(20, min_periods=1).sum(), 30).fillna(0.0)

    # GROUP C
    df["vol_z30"]       = _rz(df["volume"], 30).fillna(0.0)
    df["vol_z100"]      = _rz(df["volume"], min(100, len(df))).fillna(0.0)
    df["vol_trend_z"]   = _rz(df["volume"].rolling(5, min_periods=1).mean() / (vol_avg20 + EPS), 30).fillna(0.0)

    if "hour" in df.columns and hour_vol_avg is not None:
        session_expected = df["hour"].map(hour_vol_avg).fillna(vol_avg20)
        df["session_vol_factor"] = (df["volume"] / (session_expected + EPS)).clip(0, 4)
    else:
        # fallback: use vol_z20 shifted to [0,∞) as proxy
        df["session_vol_factor"] = (vol_z20.clip(-3, 3) + 1)

    price_dir = np.sign(df["close"].diff(3).fillna(0))
    vol_dir   = np.sign(df["volume"].rolling(3, min_periods=1).mean().diff(1).fillna(0))
    df["vol_mom_div"] = (price_dir * vol_dir).fillna(0.0)
    df["vol_efficiency"] = (df["abs_body"] / (df["atr14"] + EPS) /
                             (df["vol_z30"].abs() + 0.5)).clip(0, 3)

    # GROUP D
    df["vol_at_vwap_z"]   = (vol_z20 * near_vwap).fillna(0.0)
    body_ratio  = df["abs_body"] / (df["range"] + EPS)
    comp_a      = (vol_z20 > 1.5).astype(int)
    comp_b      = (df["vwap_dev_atr"].abs() < 0.30).astype(int)
    comp_c      = (body_ratio < 0.25).astype(int)
    df["inst_fp"] = (comp_a + comp_b + comp_c).astype(float)
    above_1sd   = (df["vwap_sigma"] - 1.0).clip(lower=0)
    df["vwap_band_walk"]  = above_1sd.rolling(10, min_periods=1).sum()
    df["price_vol_div"]   = (df["abs_body"] / (df["atr14"] + EPS) /
                              (df["vol_z30"].abs() + 0.5)).clip(0, 3)
    df["absorption_lvl"]  = (vol_z20 * (body_ratio < 0.25).astype(float)).clip(0, 3)

    # GROUP E
    df["vwap_dist_pct"]   = ((df["close"] - df["vwap"]) / (df["vwap"] + EPS) * 100).clip(-1, 1)
    above_streak = []
    cur = 0
    for v in (df["close"] > df["vwap"]).tolist():
        cur = cur + 1 if v else 0
        above_streak.append(cur)
    df["above_vwap_streak"] = above_streak
    df["vwap_mr_z"]         = _rz(df["vwap_dev_atr"].abs(), 50).fillna(0.0)
    vwap_side               = np.sign(df["vwap_dev_atr"])
    delta_sign              = np.sign(df["tick_delta"])
    df["cum_delta_vwap_align"] = (vwap_side * delta_sign).fillna(0.0)
    far_vwap                = (df["vwap_sigma"].abs() > 1.5).astype(float)
    df["vol_surge_extreme"] = (vol_z20 * far_vwap).clip(0, 3)

    # GROUP F
    df["body_ratio"]      = body_ratio
    df["close_grav_f"]    = close_grav
    df["vwap_std_feat"]   = df["vwap_std"].clip(0, 5)
    df["atr14_z"]         = _rz(df["atr14"], 50).fillna(0.0)
    df["vol_surge_20"]    = vol_z20.clip(-3, 3)

    last = df.iloc[-1]
    return {
        "vwap_dev_atr":        float(last["vwap_dev_atr"]),
        "vwap_sigma":          float(last["vwap_sigma"]),
        "vwap_slope_z":        float(last["vwap_slope_z"]),
        "vwap_reclaim_qual":   float(last["vwap_reclaim_qual"]),
        "vwap_rejection_qual": float(last["vwap_rejection_qual"]),
        "vwap_touch_density":  float(last["vwap_touch_density"]),
        "tick_delta":          float(last["tick_delta"]),
        "cum_delta_20z":       float(last["cum_delta_20z"]),
        "vol_profile_asym":    float(last["vol_profile_asym"]),
        "buy_press_z":         float(last["buy_press_z"]),
        "delta_at_vwap":       float(last["delta_at_vwap"]),
        "vol_z30":             float(last["vol_z30"]),
        "vol_z100":            float(last["vol_z100"]),
        "vol_trend_z":         float(last["vol_trend_z"]),
        "session_vol_factor":  float(last["session_vol_factor"]),
        "vol_mom_div":         float(last["vol_mom_div"]),
        "vol_efficiency":      float(last["vol_efficiency"]),
        "vol_at_vwap_z":       float(last["vol_at_vwap_z"]),
        "inst_fp":             float(last["inst_fp"]),
        "vwap_band_walk":      float(last["vwap_band_walk"]),
        "price_vol_div":       float(last["price_vol_div"]),
        "absorption_lvl":      float(last["absorption_lvl"]),
        "vwap_dist_pct":       float(last["vwap_dist_pct"]),
        "above_vwap_streak":   float(last["above_vwap_streak"]),
        "vwap_mr_z":           float(last["vwap_mr_z"]),
        "cum_delta_vwap_align":float(last["cum_delta_vwap_align"]),
        "vol_surge_extreme":   float(last["vol_surge_extreme"]),
        "body_ratio":          float(last["body_ratio"]),
        "close_grav_f":        float(last["close_grav_f"]),
        "vwap_std_feat":       float(last["vwap_std_feat"]),
        "atr14_z":             float(last["atr14_z"]),
        "vol_surge_20":        float(last["vol_surge_20"]),
    }


# ─────────────────────────────────────────────────────────────────────
# Label maps
# ─────────────────────────────────────────────────────────────────────

HEAD_LABELS = {
    "vwap_zone":         {0:"below_2sd",    1:"below_1sd",   2:"near_vwap",
                          3:"above_1sd",    4:"above_2sd"},
    "volume_character":  {0:"absence",      1:"normal",      2:"surge",       3:"climax_surge"},
    "inst_pressure":     {0:"selling",      1:"neutral",     2:"buying"},
    "vwap_interaction":  {0:"rejection",    1:"testing",     2:"reclaim",     3:"walk"},
    "volume_trend":      {0:"declining",    1:"stable",      2:"building"},
    "vwap_signal":       {0:"avoid",        1:"wait",        2:"act"},
}


# ─────────────────────────────────────────────────────────────────────
# Predictor
# ─────────────────────────────────────────────────────────────────────

# Rolling hour-volume average — updated per prediction call
_hour_vol_history: dict = defaultdict(list)


class VwapPredictor:
    def __init__(self,
                 model_path  = "vwap_best.pth",
                 scaler_path = "vwap_scaler.pkl"):

        model_path  = resource_path(model_path)
        scaler_path = resource_path(scaler_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        bundle = joblib.load(scaler_path)
        self.scaler       = bundle["scaler"]
        self.feature_cols = bundle["feature_names"]

        self.model = VwapNet(input_dim=len(self.feature_cols)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
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
        if rates is None or len(rates) < 60:
            return {"error": f"Insufficient data: {0 if rates is None else len(rates)} bars"}

        df = pd.DataFrame(rates)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)

        # Add hour for session-adjusted volume
        if "time" in df.columns:
            df["hour"] = pd.to_datetime(df["time"], unit="s").dt.hour
        else:
            df["hour"] = 0

        # Update hour_vol_history and build per-hour averages
        for _, row in df.iterrows():
            h = int(row["hour"])
            _hour_vol_history[h].append(float(row["volume"]))
            if len(_hour_vol_history[h]) > 400:   # keep last ~20 days of that hour
                _hour_vol_history[h] = _hour_vol_history[h][-400:]

        hour_vol_avg = {h: float(np.mean(v)) for h, v in _hour_vol_history.items()}
        return self.predict_from_df(df, hour_vol_avg=hour_vol_avg)

    def predict_from_df(self, df: pd.DataFrame,
                         hour_vol_avg: dict = None) -> dict:
        """
        df must have columns: open, high, low, close, volume.
        Optionally: hour (0-23).
        """
        feat_dict = compute_vwap_features(df, hour_vol_avg=hour_vol_avg)
        row       = np.array([[feat_dict.get(c, 0.0) for c in self.feature_cols]], dtype=np.float32)
        scaled    = self.scaler.transform(row)
        x         = torch.tensor(scaled, dtype=torch.float32).to(self.device)

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
                "probs":      {lmap[k]: round(float(probs[k]), 4) for k in range(len(probs))},
            }

        result["signal"]     = result["vwap_signal"]["label"]
        result["confidence"] = result["vwap_signal"]["confidence"]
        return result

    def shutdown(self):
        if MT5_AVAILABLE:
            mt5.shutdown()


# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    predictor = VwapPredictor()
    result    = predictor.predict("EURUSD")

    print("\n── VWAP / Volume Prediction ─────────────────────")
    for head, v in result.items():
        if isinstance(v, dict) and "label" in v:
            print(f"  {head:<24} → {v['label']}  (conf={v['confidence']:.2%})")
    print(f"\n  SIGNAL: {result['signal'].upper()}  "
          f"(confidence={result['confidence']:.2%})")
    predictor.shutdown()
