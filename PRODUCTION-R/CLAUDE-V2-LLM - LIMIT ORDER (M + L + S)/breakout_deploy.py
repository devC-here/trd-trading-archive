# breakout_deploy.py
# Live inference for the Breakout / Squeeze model
# Architecture and features match train_breakout.py + synthetic_breakout.py exactly
# Features : 27  |  Heads : 6  |  HEAD_SIZES : [3, 3, 3, 3, 3, 3]
# ─────────────────────────────────────────────────────────────────────
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
LOOKBACK = 130          # bars fetched from MT5 (120 used + warmup)
TRUNK    = 256
DROP     = 0.25


# ─────────────────────────────────────────────────────────────────────
# Architecture  ·  must be byte-identical to train_breakout.py
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


class BreakoutNet(nn.Module):
    # Heads: next_direction(3), next_magnitude(3), next_pattern(3),
    #        next_volatility(3), next_close_loc(3), bk_follow_through(3)
    HEAD_SIZES = [3, 3, 3, 3, 3, 3]

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
# Feature computation  ·  identical to synthetic_breakout.py
# Output keys are sorted alphabetically — matching how the CSV is built.
# ─────────────────────────────────────────────────────────────────────

def _rz(s, w):
    return (s - s.rolling(w).mean()) / (s.rolling(w).std() + EPS)


def compute_breakout_features(df: pd.DataFrame) -> dict:
    """
    df : DataFrame with [open, high, low, close, volume]. Min 100 rows.
    Returns dict of 27 float features (keys sorted to match training CSV).
    """
    df = df.copy().reset_index(drop=True)

    df["range"]      = df["high"] - df["low"]
    df["body"]       = df["close"] - df["open"]
    df["abs_body"]   = df["body"].abs()
    df["dir"]        = np.sign(df["body"]).fillna(0)
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["atr14"]      = df["range"].rolling(14, min_periods=1).mean()
    df["atr5"]       = df["range"].rolling(5,  min_periods=1).mean()

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

    df["bb_width_z"] = _rz(df["bb_width_14"], 100).fillna(0.0)
    df["kc_width_n"] = df["kc_width_14"] / (df["atr14"] + EPS)

    sq8  = (df["bb_width_8"]  < df["kc_width_8"]).astype(int)
    sq14 = (df["bb_width_14"] < df["kc_width_14"]).astype(int)
    sq20 = (df["bb_width_20"] < df["kc_width_20"]).astype(int)
    df["squeeze_stack"] = (sq8 + sq14 + sq20).astype(float)
    df["kb_ratio"]      = df["bb_width_14"] / (df["kc_width_14"] + EPS)
    df["kb_slope"]      = _rz(df["kb_ratio"].diff(1).rolling(10, min_periods=1).mean(), 20).fillna(0.0)

    sq_active_s = (df["kb_ratio"] < 1.0).astype(int)
    cons_sq, cur = [], 0
    for s in sq_active_s.tolist():
        cur = cur + 1 if s == 1 else 0
        cons_sq.append(cur)
    df["consec_squeeze"]  = cons_sq
    avg_sq                = max(float(pd.Series(cons_sq).rolling(100, min_periods=5).mean().iloc[-1]), 1.0)
    df["squeeze_maturity"] = (df["consec_squeeze"] / avg_sq).clip(0, 4)

    hv_pct                = df["bb_width_14"].rank(pct=True).fillna(0.5)
    df["vol_potential_e"] = ((1 - hv_pct) * (1 - df["kb_ratio"]).clip(0, 2)).clip(0, 2)

    don_high  = df["high"].rolling(20, min_periods=1).max()
    don_low   = df["low"].rolling(20,  min_periods=1).min()
    don_mid   = (don_high + don_low) / 2
    don_width = (don_high - don_low) + EPS
    lean_raw  = ((df["close"] - don_mid) / don_width).rolling(3, min_periods=1).mean()
    df["sq_lean"] = (lean_raw * sq_active_s).fillna(0.0)

    df["atr_ratio"]   = df["atr5"] / (df["atr14"] + EPS)
    df["atr_ratio_z"] = _rz(df["atr_ratio"], 50).fillna(0.0)
    df["exp_velocity"] = _rz(df["bb_width_14"].diff(3).fillna(0), 50).fillna(0.0)
    above_upper         = (df["close"] - df["bb_upper_14"]).clip(lower=0) / (df["atr14"] + EPS)
    below_lower         = (df["bb_lower_14"] - df["close"]).clip(lower=0) / (df["atr14"] + EPS)
    df["band_walk_bull"] = above_upper.rolling(5, min_periods=1).sum()
    df["band_walk_bear"] = below_lower.rolling(5, min_periods=1).sum()
    df["hv_percentile"]  = hv_pct

    bb_range           = (df["bb_upper_14"] - df["bb_lower_14"]) + EPS
    df["bb_position"]  = (df["close"] - df["bb_lower_14"]) / bb_range
    df["bb_pos_z"]     = _rz(df["bb_position"], 50).fillna(0.0)
    df["kc_position"]  = (df["close"] - df["kc_lower_14"]) / (df["kc_width_14"] + EPS)
    mid_bb14           = df["close"].rolling(14, min_periods=1).mean()
    df["mr_dist"]      = ((df["close"] - mid_bb14) / (df["atr14"] + EPS)).clip(-3, 3)
    df["coil_tightness"] = (1 - df["kb_ratio"]).clip(-1, 1)

    body_avg20          = df["abs_body"].rolling(20, min_periods=1).mean()
    df["body_comp_r"]   = (df["abs_body"] / (body_avg20 + EPS)).clip(0, 3)
    df["close_grav"]    = (df["close"] - df["low"]) / (df["range"] + EPS)
    vol_avg20           = df["volume"].rolling(20, min_periods=1).mean()
    df["vol_z20"]       = _rz(df["volume"], 20).fillna(0.0)
    df["prebreak_qual"] = (df["close_grav"] * df["body_comp_r"] *
                           df["vol_z20"].clip(0, 3)).clip(-3, 3)
    df["intrabar_fill"] = df["abs_body"] / (df["range"] + EPS)
    df["vol_sq_div"]    = (df["vol_z20"] * (df["squeeze_stack"] / 3.0)).clip(-3, 3)
    close_above         = (df["close"] > df["bb_upper_14"]).astype(float)
    close_below         = (df["close"] < df["bb_lower_14"]).astype(float)
    df["bkout_confirm"] = ((close_above - close_below) *
                           (df["vol_z20"] + 1).clip(0, 3)).clip(-3, 3)
    df["don_bias"]      = ((df["close"] - don_mid) / don_width).clip(-1, 1)

    df["vol_z30"]   = _rz(df["volume"], 30).fillna(0.0)
    df["vol_z100"]  = _rz(df["volume"], min(100, len(df))).fillna(0.0)
    df["vol_trend"] = _rz(df["volume"].rolling(5, min_periods=1).mean() /
                          (vol_avg20 + EPS), 30).fillna(0.0)

    last = df.iloc[-1]
    # Keys sorted alphabetically — same order as synthetic_breakout_dataset.csv
    return {
        "atr_ratio_z":     float(last["atr_ratio_z"]),
        "band_walk_bear":  float(last["band_walk_bear"]),
        "band_walk_bull":  float(last["band_walk_bull"]),
        "bb_pos_z":        float(last["bb_pos_z"]),
        "bb_position":     float(last["bb_position"]),
        "bb_width_z":      float(last["bb_width_z"]),
        "bkout_confirm":   float(last["bkout_confirm"]),
        "body_comp_r":     float(last["body_comp_r"]),
        "coil_tightness":  float(last["coil_tightness"]),
        "don_bias":        float(last["don_bias"]),
        "exp_velocity":    float(last["exp_velocity"]),
        "hv_percentile":   float(last["hv_percentile"]),
        "intrabar_fill":   float(last["intrabar_fill"]),
        "kb_ratio":        float(last["kb_ratio"]),
        "kb_slope":        float(last["kb_slope"]),
        "kc_position":     float(last["kc_position"]),
        "kc_width_n":      float(last["kc_width_n"]),
        "mr_dist":         float(last["mr_dist"]),
        "prebreak_qual":   float(last["prebreak_qual"]),
        "sq_lean":         float(last["sq_lean"]),
        "squeeze_maturity": float(last["squeeze_maturity"]),
        "squeeze_stack":   float(last["squeeze_stack"]),
        "vol_potential_e": float(last["vol_potential_e"]),
        "vol_sq_div":      float(last["vol_sq_div"]),
        "vol_trend":       float(last["vol_trend"]),
        "vol_z100":        float(last["vol_z100"]),
        "vol_z30":         float(last["vol_z30"]),
    }


# ─────────────────────────────────────────────────────────────────────
# Label maps  ·  sourced from synthetic_breakout.py label_maps dict
# ─────────────────────────────────────────────────────────────────────

HEAD_LABELS = {
    "next_direction":    {0: "bearish",      1: "flat",     2: "bullish"},
    "next_magnitude":    {0: "small",        1: "medium",   2: "large"},
    "next_pattern":      {0: "continuation", 1: "reversal", 2: "indecision"},
    "next_volatility":   {0: "contracting",  1: "normal",   2: "expanding"},
    "next_close_loc":    {0: "lower_third",  1: "mid",      2: "upper_third"},
    "bk_follow_through": {0: "no",           1: "partial",  2: "yes"},
}


# ─────────────────────────────────────────────────────────────────────
# Predictor
# ─────────────────────────────────────────────────────────────────────

class BreakoutPredictor:
    def __init__(self,
                 model_path:  str = "breakout_best.pth",
                 scaler_path: str = "breakout_scaler.pkl"):

        model_path  = resource_path(model_path)
        scaler_path = resource_path(scaler_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        bundle            = joblib.load(scaler_path)
        self.scaler       = bundle["scaler"]
        self.feature_cols = bundle["feature_names"]   # preserves training order

        self.model = BreakoutNet(input_dim=len(self.feature_cols)).to(self.device)
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
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, LOOKBACK + 5)
        if rates is None or len(rates) < 100:
            return {"error": f"Insufficient bars: {0 if rates is None else len(rates)}"}
        df = pd.DataFrame(rates)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        return self.predict_from_df(df)

    def predict_from_df(self, df: pd.DataFrame) -> dict:
        feat_dict = compute_breakout_features(df)
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

        result["signal"]         = result["next_direction"]["label"]
        result["confidence"]     = result["next_direction"]["confidence"]
        result["follow_through"] = result["bk_follow_through"]["label"]
        result["trade_active"]   = bool(
            result["next_direction"]["class"] != 1
            and result["bk_follow_through"]["class"] >= 1
        )
        return result

    def shutdown(self):
        if MT5_AVAILABLE:
            mt5.shutdown()


# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    predictor = BreakoutPredictor()
    result    = predictor.predict("EURUSD", mt5.TIMEFRAME_M15)
    print("\n── Breakout Prediction ──────────────────────────────")
    for head, v in result.items():
        if isinstance(v, dict) and "label" in v:
            print(f"  {head:<20} → {v['label']:<14} conf={v['confidence']:.2%}")
    print(f"\n  direction     : {result['signal'].upper()}")
    print(f"  follow-through: {result['follow_through'].upper()}")
    print(f"  trade_active  : {result['trade_active']}  (conf={result['confidence']:.2%})")
    predictor.shutdown()
