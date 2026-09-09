# microstructure_deploy.py
# Live inference for the Microstructure model
# Architecture and features match train_microstructure.py + synthetic_microstructure.py exactly
# Features : 32  |  Heads : 6  |  HEAD_SIZES : [3, 4, 3, 3, 3, 3]
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
LOOKBACK = 60           # bars fetched from MT5
TRUNK    = 256
DROP     = 0.25


# ─────────────────────────────────────────────────────────────────────
# Architecture  ·  must be byte-identical to train_microstructure.py
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


class MicroNet(nn.Module):
    # Heads: next_flow_dir(3), next_bar_intent(4), next_magnitude(3),
    #        next_rejection(3), momentum_follow(3), next_range_type(3)
    HEAD_SIZES = [3, 4, 3, 3, 3, 3]

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
# Feature computation  ·  identical to synthetic_microstructure.py
# ─────────────────────────────────────────────────────────────────────

def _rolling_z(s, w):
    return (s - s.rolling(w).mean()) / (s.rolling(w).std() + EPS)


def compute_microstructure_features(df: pd.DataFrame) -> dict:
    """
    df : DataFrame with [open, high, low, close, volume]. Min 40 rows.
    Returns dict of 32 float features (keys sorted to match training CSV).
    """
    df = df.copy().reset_index(drop=True)

    # ── base anatomy ────────────────────────────────────────────────
    df["range"]      = df["high"] - df["low"]
    df["body"]       = df["close"] - df["open"]
    df["abs_body"]   = df["body"].abs()
    df["dir"]        = np.sign(df["body"]).fillna(0).astype(int)
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

    # ── 23 base features ────────────────────────────────────────────
    df["body_ratio"]     = df["abs_body"]   / (df["range"] + EPS)
    df["upper_ratio"]    = df["upper_wick"] / (df["range"] + EPS)
    df["lower_ratio"]    = df["lower_wick"] / (df["range"] + EPS)
    df["body_ratio_s5"]  = df["body_ratio"].rolling(5, min_periods=1).mean()
    df["upper_ratio_s5"] = df["upper_ratio"].rolling(5, min_periods=1).mean()
    df["lower_ratio_s5"] = df["lower_ratio"].rolling(5, min_periods=1).mean()

    df["body_z"]       = _rolling_z(df["abs_body"], 40).fillna(0.0)
    df["range_z"]      = _rolling_z(df["range"],    40).fillna(0.0)
    df["tail_asym"]    = (df["upper_wick"] - df["lower_wick"]) / (df["range"] + EPS)
    df["tail_to_body"] = (df["upper_wick"] + df["lower_wick"]) / (df["abs_body"] + EPS)

    df["signed_vol"]    = df["volume"] * df["dir"]
    df["orderflow_5"]   = df["signed_vol"].rolling(5,  min_periods=1).sum()
    df["orderflow_10"]  = df["signed_vol"].rolling(10, min_periods=1).sum()

    df["small_body_flag"] = (df["abs_body"] < df["abs_body"].rolling(5, min_periods=1).mean() * 0.6).astype(int)
    df["opp_wick"]        = 0.0
    df.loc[df["dir"] >  0, "opp_wick"] = df["lower_wick"]
    df.loc[df["dir"] <  0, "opp_wick"] = df["upper_wick"]
    df["absorption_raw"]  = (df["small_body_flag"] * df["opp_wick"]).rolling(5, min_periods=1).sum()

    df["atr5"]           = df["range"].rolling(5,  min_periods=1).mean()
    df["atr10"]          = df["range"].rolling(10, min_periods=1).mean()
    df["pullback_depth"] = df["range"] / (df["atr5"] + EPS)
    df["pullback_z"]     = _rolling_z(df["pullback_depth"], 20).fillna(0.0)
    df["exhaustion"]     = ((df["upper_wick"] + df["lower_wick"]) / (df["abs_body"] + EPS)).fillna(0.0)
    df["expected_move_raw"] = df["atr5"] * (
        1 + df["orderflow_5"].abs() / (df["volume"].rolling(5, min_periods=1).mean() + EPS))
    df["wick_body_inter"]   = ((df["upper_wick"] - df["lower_wick"]) * df["abs_body"]).clip(-2.0, 2.0)
    df["body_vol_inter"]    = (df["abs_body"] * df["volume"]).clip(-1e6, 1e6)

    cons, cur, prev = [], 0, 0
    for d in df["dir"].fillna(0).tolist():
        if d == prev and d != 0:  cur += 1
        elif d != 0:               cur  = 1
        else:                      cur  = 0
        cons.append(cur)
        prev = d
    df["consec_len"] = cons

    # ── 9 physics features ───────────────────────────────────────────
    df["close_gravity"]  = (df["close"] - df["low"]) / (df["range"] + EPS)
    df["body_mid_disp"]  = (((df["open"] + df["close"]) / 2) -
                              ((df["high"] + df["low"]) / 2)) / (df["range"] + EPS)
    df["tick_delta"]     = df["close_gravity"] * 2 - 1

    delta_vol            = df["tick_delta"] * df["volume"]
    df["cum_delta_5z"]   = _rolling_z(delta_vol.rolling(5, min_periods=1).sum(), 20).fillna(0.0)

    vol_avg10            = df["volume"].rolling(10, min_periods=1).mean()
    dom_wick             = df[["upper_wick", "lower_wick"]].max(axis=1)
    rej_dir              = np.where(df["lower_wick"] > df["upper_wick"], 1, -1)
    df["rejection_qual"] = ((dom_wick / (df["range"] + EPS)) *
                             (df["volume"] / (vol_avg10 + EPS)) * rej_dir).clip(-4, 4)

    comp_a               = (df["body_ratio"] < 0.25).astype(int)
    comp_b               = (df["range"] / (df["atr10"] + EPS) > 0.80).astype(int)
    comp_c               = (df["volume"] / (vol_avg10 + EPS) > 1.20).astype(int)
    df["absorption_fp"]  = (comp_a + comp_b + comp_c).astype(float)

    bull_mask            = df["dir"] >= 0
    df["dir_efficiency"] = np.where(bull_mask,
                                     df["body_ratio"] * df["close_gravity"],
                                     df["body_ratio"] * (1 - df["close_gravity"]))

    vol_avg5             = df["volume"].rolling(5, min_periods=1).mean()
    df["bar_energy"]     = (df["range"] / (df["atr5"] + EPS) *
                             df["volume"] / (vol_avg5 + EPS)).clip(0, 6)

    df["gap_pressure"]   = ((df["open"] - df["close"].shift(1).fillna(df["open"])) /
                              (df["atr5"] + EPS)).clip(-3, 3)

    last = df.iloc[-1]
    # Keys sorted alphabetically — same order as synthetic_microstructure_dataset.csv
    return {
        "abs_body":        float(last["abs_body"]),
        "absorption_fp":   float(last["absorption_fp"]),
        "absorption_raw":  float(last["absorption_raw"]),
        "bar_energy":      float(last["bar_energy"]),
        "body_mid_disp":   float(last["body_mid_disp"]),
        "body_ratio":      float(last["body_ratio"]),
        "body_ratio_s5":   float(last["body_ratio_s5"]),
        "body_vol_inter":  float(last["body_vol_inter"]),
        "body_z":          float(last["body_z"]),
        "close_gravity":   float(last["close_gravity"]),
        "consec_len":      float(last["consec_len"]),
        "cum_delta_5z":    float(last["cum_delta_5z"]),
        "dir_efficiency":  float(last["dir_efficiency"]),
        "exhaustion":      float(last["exhaustion"]),
        "expected_move_raw": float(last["expected_move_raw"]),
        "gap_pressure":    float(last["gap_pressure"]),
        "lower_ratio":     float(last["lower_ratio"]),
        "lower_ratio_s5":  float(last["lower_ratio_s5"]),
        "lower_wick":      float(last["lower_wick"]),
        "orderflow_10":    float(last["orderflow_10"]),
        "orderflow_5":     float(last["orderflow_5"]),
        "pullback_z":      float(last["pullback_z"]),
        "range":           float(last["range"]),
        "range_z":         float(last["range_z"]),
        "rejection_qual":  float(last["rejection_qual"]),
        "tail_asym":       float(last["tail_asym"]),
        "tail_to_body":    float(last["tail_to_body"]),
        "tick_delta":      float(last["tick_delta"]),
        "upper_ratio":     float(last["upper_ratio"]),
        "upper_ratio_s5":  float(last["upper_ratio_s5"]),
        "upper_wick":      float(last["upper_wick"]),
        "wick_body_inter": float(last["wick_body_inter"]),
    }


# ─────────────────────────────────────────────────────────────────────
# Label maps  ·  sourced from synthetic_microstructure.py label_maps dict
# ─────────────────────────────────────────────────────────────────────

HEAD_LABELS = {
    "next_flow_dir":   {0: "bearish",        1: "contested",  2: "bullish"},
    "next_bar_intent": {0: "absorption",     1: "rejection",  2: "impulse",  3: "fade"},
    "next_magnitude":  {0: "thin",           1: "normal",     2: "strong"},
    "next_rejection":  {0: "none",           1: "partial",    2: "strong"},
    "momentum_follow": {0: "failed",         1: "partial",    2: "confirmed"},
    "next_range_type": {0: "inside_compress", 1: "normal",    2: "outside_expand"},
}


# ─────────────────────────────────────────────────────────────────────
# Predictor
# ─────────────────────────────────────────────────────────────────────

class MicrostructurePredictor:
    def __init__(self,
                 model_path:  str = "microstructure_best.pth",
                 scaler_path: str = "microstructure_scaler.pkl",
                 min_bars:    int = 50):

        model_path  = resource_path(model_path)
        scaler_path = resource_path(scaler_path)

        self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_bars = min_bars

        bundle            = joblib.load(scaler_path)
        self.scaler       = bundle["scaler"]
        self.feature_cols = bundle["feature_names"]

        self.model = MicroNet(input_dim=len(self.feature_cols)).to(self.device)
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
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 9, LOOKBACK + 10)
        if rates is None or len(rates) < self.min_bars:
            return {"error": f"Insufficient bars: {0 if rates is None else len(rates)}"}
        df = pd.DataFrame(rates)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        return self.predict_from_df(df)

    def predict_from_df(self, df: pd.DataFrame) -> dict:
        feat_dict = compute_microstructure_features(df)
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

        result["signal"]       = result["next_flow_dir"]["label"]
        result["confidence"]   = result["next_flow_dir"]["confidence"]
        result["follow"]       = result["momentum_follow"]["label"]
        result["trade_active"] = bool(
            result["next_flow_dir"]["class"] != 1        # not contested
            and result["momentum_follow"]["class"] >= 1  # at least partial
        )
        return result

    def shutdown(self):
        if MT5_AVAILABLE:
            mt5.shutdown()


# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    predictor = MicrostructurePredictor()
    result    = predictor.predict("EURUSD", mt5.TIMEFRAME_H1)
    print("\n── Microstructure Prediction ─────────────────────────")
    for head, v in result.items():
        if isinstance(v, dict) and "label" in v:
            print(f"  {head:<20} → {v['label']:<16} conf={v['confidence']:.2%}")
    print(f"\n  flow_dir    : {result['signal'].upper()}")
    print(f"  follow      : {result['follow'].upper()}")
    print(f"  trade_active: {result['trade_active']}  (conf={result['confidence']:.2%})")
    predictor.shutdown()
