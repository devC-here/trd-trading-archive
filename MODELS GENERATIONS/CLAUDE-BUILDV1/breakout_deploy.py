# breakout_deploy.py
# Live inference for the Breakout / Squeeze model
# Computes 27 features EXACTLY matching synthetic_breakout.py
# NO ta library dependency — pure pandas/numpy
import numpy as np
import pandas as pd
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
LOOKBACK = 130   # bars to fetch; 120 used, extra for warmup


# ─────────────────────────────────────────────────────────────────────
# Architecture — must match train_breakout.py exactly
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


class BreakoutNet(nn.Module):
    HEAD_SIZES = [4, 3, 4, 4, 3, 3]

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
# Feature computation — identical to synthetic_breakout.py
# ─────────────────────────────────────────────────────────────────────

def _rz(s, w):
    return (s - s.rolling(w).mean()) / (s.rolling(w).std() + EPS)


def compute_breakout_features(df: pd.DataFrame) -> dict:
    """
    Input : DataFrame with [open, high, low, close, volume]. Min 100 rows.
    Output: dict of 27 features (no _ prefixed context keys).
    """
    df = df.copy().reset_index(drop=True)

    df["range"]      = df["high"] - df["low"]
    df["body"]       = df["close"] - df["open"]
    df["abs_body"]   = df["body"].abs()
    df["dir"]        = np.sign(df["body"]).fillna(0)
    df["upper_wick"] = df["high"] - df[["open","close"]].max(axis=1)
    df["lower_wick"] = df[["open","close"]].min(axis=1) - df["low"]
    df["atr14"]      = df["range"].rolling(14, min_periods=1).mean()
    df["atr5"]       = df["range"].rolling(5,  min_periods=1).mean()

    # Bollinger + Keltner for all 3 window sizes
    for w in [8, 14, 20]:
        mid   = df["close"].rolling(w, min_periods=1).mean()
        std   = df["close"].rolling(w, min_periods=1).std().fillna(0)
        atr_w = df["range"].rolling(w, min_periods=1).mean()
        df[f"bb_upper_{w}"] = mid + 2 * std
        df[f"bb_lower_{w}"] = mid - 2 * std
        df[f"bb_width_{w}"] = 4 * std
        df[f"kc_width_{w}"] = 3 * atr_w
        df[f"kc_upper_{w}"] = mid + 1.5 * atr_w
        df[f"kc_lower_{w}"] = mid - 1.5 * atr_w

    # GROUP A
    df["bb_width_z"]     = _rz(df["bb_width_14"], 100).fillna(0.0)
    df["kc_width_n"]     = df["kc_width_14"] / (df["atr14"] + EPS)

    sq8  = (df["bb_width_8"]  < df["kc_width_8"]).astype(int)
    sq14 = (df["bb_width_14"] < df["kc_width_14"]).astype(int)
    sq20 = (df["bb_width_20"] < df["kc_width_20"]).astype(int)
    df["squeeze_stack"]  = (sq8 + sq14 + sq20).astype(float)
    df["kb_ratio"]       = df["bb_width_14"] / (df["kc_width_14"] + EPS)
    df["kb_slope"]       = _rz(df["kb_ratio"].diff(1).rolling(10, min_periods=1).mean(), 20).fillna(0.0)

    sq_active_s = (df["kb_ratio"] < 1.0).astype(int)
    cons_sq     = []
    cur = 0
    for s in sq_active_s.tolist():
        cur = cur + 1 if s == 1 else 0
        cons_sq.append(cur)
    df["consec_squeeze"]    = cons_sq
    df["sq_active"]         = sq_active_s
    avg_sq = max(float(pd.Series(cons_sq).rolling(100, min_periods=5).mean().iloc[-1]), 1.0)
    df["squeeze_maturity"]  = (df["consec_squeeze"] / avg_sq).clip(0, 4)

    hv_pct = df["bb_width_14"].rank(pct=True).fillna(0.5)
    df["vol_potential_e"]   = ((1 - hv_pct) * (1 - df["kb_ratio"]).clip(0, 2)).clip(0, 2)

    don_high = df["high"].rolling(20, min_periods=1).max()
    don_low  = df["low"].rolling(20,  min_periods=1).min()
    don_mid  = (don_high + don_low) / 2
    don_width= (don_high - don_low) + EPS
    lean_raw = ((df["close"] - don_mid) / don_width).rolling(3, min_periods=1).mean()
    df["sq_lean"]           = (lean_raw * sq_active_s).fillna(0.0)

    # GROUP B
    df["atr_ratio"]      = df["atr5"] / (df["atr14"] + EPS)
    df["atr_ratio_z"]    = _rz(df["atr_ratio"], 50).fillna(0.0)
    df["exp_velocity"]   = _rz(df["bb_width_14"].diff(3).fillna(0), 50).fillna(0.0)
    above_upper = (df["close"] - df["bb_upper_14"]).clip(lower=0) / (df["atr14"] + EPS)
    below_lower = (df["bb_lower_14"] - df["close"]).clip(lower=0) / (df["atr14"] + EPS)
    df["band_walk_bull"] = above_upper.rolling(5, min_periods=1).sum()
    df["band_walk_bear"] = below_lower.rolling(5, min_periods=1).sum()
    df["hv_percentile"]  = hv_pct

    # GROUP C
    bb_range             = (df["bb_upper_14"] - df["bb_lower_14"]) + EPS
    df["bb_position"]    = (df["close"] - df["bb_lower_14"]) / bb_range
    df["bb_pos_z"]       = _rz(df["bb_position"], 50).fillna(0.0)
    df["kc_position"]    = (df["close"] - df["kc_lower_14"]) / (df["kc_width_14"] + EPS)
    mid_bb14             = df["close"].rolling(14, min_periods=1).mean()
    df["mr_dist"]        = ((df["close"] - mid_bb14) / (df["atr14"] + EPS)).clip(-3, 3)
    df["coil_tightness"] = (1 - df["kb_ratio"]).clip(-1, 1)

    # GROUP D
    body_avg20           = df["abs_body"].rolling(20, min_periods=1).mean()
    df["body_comp_r"]    = (df["abs_body"] / (body_avg20 + EPS)).clip(0, 3)
    df["close_grav"]     = (df["close"] - df["low"]) / (df["range"] + EPS)
    vol_avg20            = df["volume"].rolling(20, min_periods=1).mean()
    df["vol_z20"]        = _rz(df["volume"], 20).fillna(0.0)
    df["prebreak_qual"]  = (df["close_grav"] * df["body_comp_r"] *
                             df["vol_z20"].clip(0, 3)).clip(-3, 3)
    df["intrabar_fill"]  = df["abs_body"] / (df["range"] + EPS)
    df["vol_sq_div"]     = (df["vol_z20"] * (df["squeeze_stack"] / 3.0)).clip(-3, 3)
    close_above          = (df["close"] > df["bb_upper_14"]).astype(float)
    close_below          = (df["close"] < df["bb_lower_14"]).astype(float)
    df["bkout_confirm"]  = ((close_above - close_below) *
                             (df["vol_z20"] + 1).clip(0, 3)).clip(-3, 3)
    df["don_bias"]       = ((df["close"] - don_mid) / don_width).clip(-1, 1)

    # GROUP E
    df["vol_z30"]        = _rz(df["volume"], 30).fillna(0.0)
    df["vol_z100"]       = _rz(df["volume"], min(100, len(df))).fillna(0.0)
    df["vol_trend"]      = _rz(df["volume"].rolling(5, min_periods=1).mean() /
                                (vol_avg20 + EPS), 30).fillna(0.0)

    last = df.iloc[-1]
    return {
        "bb_width_z":       float(last["bb_width_z"]),
        "kc_width_n":       float(last["kc_width_n"]),
        "squeeze_stack":    float(last["squeeze_stack"]),
        "kb_ratio":         float(last["kb_ratio"]),
        "kb_slope":         float(last["kb_slope"]),
        "squeeze_maturity": float(last["squeeze_maturity"]),
        "vol_potential_e":  float(last["vol_potential_e"]),
        "sq_lean":          float(last["sq_lean"]),
        "atr_ratio_z":      float(last["atr_ratio_z"]),
        "exp_velocity":     float(last["exp_velocity"]),
        "band_walk_bull":   float(last["band_walk_bull"]),
        "band_walk_bear":   float(last["band_walk_bear"]),
        "hv_percentile":    float(last["hv_percentile"]),
        "bb_position":      float(last["bb_position"]),
        "bb_pos_z":         float(last["bb_pos_z"]),
        "kc_position":      float(last["kc_position"]),
        "mr_dist":          float(last["mr_dist"]),
        "coil_tightness":   float(last["coil_tightness"]),
        "body_comp_r":      float(last["body_comp_r"]),
        "prebreak_qual":    float(last["prebreak_qual"]),
        "intrabar_fill":    float(last["intrabar_fill"]),
        "vol_sq_div":       float(last["vol_sq_div"]),
        "bkout_confirm":    float(last["bkout_confirm"]),
        "don_bias":         float(last["don_bias"]),
        "vol_z30":          float(last["vol_z30"]),
        "vol_z100":         float(last["vol_z100"]),
        "vol_trend":        float(last["vol_trend"]),
    }


# ─────────────────────────────────────────────────────────────────────
# Label maps
# ─────────────────────────────────────────────────────────────────────

HEAD_LABELS = {
    "squeeze_state":     {0:"no_squeeze", 1:"forming", 2:"active",    3:"overdue"},
    "breakout_direction":{0:"bearish",    1:"neutral",  2:"bullish"},
    "breakout_quality":  {0:"no_breakout",1:"false",    2:"developing",3:"confirmed"},
    "volatility_phase":  {0:"deeply_compressing",1:"compressing",
                          2:"transitioning",     3:"expanding"},
    "band_structure":    {0:"lower_walk", 1:"mid_range", 2:"upper_walk"},
    "entry_timing":      {0:"avoid",      1:"wait",      2:"now"},
}


# ─────────────────────────────────────────────────────────────────────
# Predictor
# ─────────────────────────────────────────────────────────────────────

class BreakoutPredictor:
    def __init__(self,
                 model_path  = "breakout_best.pth",
                 scaler_path = "breakout_scaler.pkl"):

        model_path  = resource_path(model_path)
        scaler_path = resource_path(scaler_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        bundle = joblib.load(scaler_path)
        self.scaler       = bundle["scaler"]
        self.feature_cols = bundle["feature_names"]

        self.model = BreakoutNet(input_dim=len(self.feature_cols)).to(self.device)
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
        if rates is None or len(rates) < 100:
            return {"error": f"Insufficient data: {0 if rates is None else len(rates)} bars"}

        df = pd.DataFrame(rates)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        return self.predict_from_df(df)

    def predict_from_df(self, df: pd.DataFrame) -> dict:
        feat_dict = compute_breakout_features(df)
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

        result["signal"]     = result["entry_timing"]["label"]
        result["confidence"] = result["entry_timing"]["confidence"]
        return result

    def shutdown(self):
        if MT5_AVAILABLE:
            mt5.shutdown()


# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    predictor = BreakoutPredictor()
    result    = predictor.predict("EURUSD")

    print("\n── Breakout / Squeeze Prediction ────────────────")
    for head, v in result.items():
        if isinstance(v, dict) and "label" in v:
            print(f"  {head:<24} → {v['label']}  (conf={v['confidence']:.2%})")
    print(f"\n  SIGNAL: {result['signal'].upper()}  "
          f"(confidence={result['confidence']:.2%})")
    predictor.shutdown()
