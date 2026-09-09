# deploy_model3_volatility.py
# Live inference for the Volatility Squeeze Specialist
# Architecture and features match train_model3_volatility.py exactly
# Features : 4  |  Head : 1 binary (Chop / Breakout) — NON-DIRECTIONAL
# ─────────────────────────────────────────────────────────────────────
import math
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
LOOKBACK = 80
TRUNK    = 64
DROP     = 0.20
WINDOW   = 16


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


class VolatilityNet(nn.Module):
    HEAD_SIZE = 2

    def __init__(self, input_dim, trunk=TRUNK, drop=DROP):
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
        self.head = nn.Linear(trunk // 2, self.HEAD_SIZE)

    def forward(self, x):
        h = self.mid(self.res(self.stem(x)))
        return self.head(h)


def compute_volatility_features(df: pd.DataFrame) -> dict:
    df     = df.copy().reset_index(drop=True)
    window = df.tail(WINDOW)

    C = window["close"].values.astype(float)
    H = window["high"].values.astype(float)
    L = window["low"].values.astype(float)

    H_max  = H.max()
    L_min  = L.min()
    mean_C = C.mean()
    std_C  = max(C.std(), EPS)

    TR = np.array(
        [H[0] - L[0]] +
        [max(H[i] - L[i], abs(H[i] - C[i-1]), abs(L[i] - C[i-1]))
         for i in range(1, WINDOW)]
    )
    ATR = TR.mean()

    bb_bw = (4.0 * std_C) / (mean_C + EPS)
    sub_bws = np.array([
        (4.0 * max(C[:i].std(), EPS)) / (C[:i].mean() + EPS)
        for i in range(4, WINDOW + 1)
    ])
    sub_mean = sub_bws.mean()
    sub_std  = max(sub_bws.std(), EPS)
    bb_z = float((bb_bw - sub_mean) / sub_std)

    kc_bw = (2.0 * ATR) / (mean_C + EPS)
    kc_sq = float(bb_bw / (kc_bw + EPS))

    tr_mean = TR.mean()
    tr_std  = max(TR.std(), EPS)
    tr_z    = float((TR[-1] - tr_mean) / tr_std)

    path_len  = float(np.abs(np.diff(C)).sum())
    total_rng = float(H_max - L_min) + EPS
    fdi = float(1.0 + math.log(path_len / total_rng + EPS) / math.log(WINDOW))

    return {
        "62_bollinger_bandwidth_z_score": bb_z,
        "63_keltner_channel_squeeze":     kc_sq,
        "72_true_range_z_score":          tr_z,
        "65_fractal_dimension_index":     fdi,
    }


HEAD_LABELS = {0: "chop_sideways", 1: "breakout_expansion"}


def interpret_regime(p_breakout: float) -> str:
    if p_breakout >= 0.75:   return "strong_expansion"
    elif p_breakout >= 0.55: return "likely_expansion"
    elif p_breakout >= 0.45: return "neutral_edge"
    elif p_breakout >= 0.25: return "likely_chop"
    else:                    return "strong_squeeze"


class VolatilityPredictor:
    def __init__(self,
                 model_path: str = "model3_best.pth",
                 meta_path:  str = "model3_meta.pkl"):

        model_path = resource_path(model_path)
        meta_path  = resource_path(meta_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        bundle            = joblib.load(meta_path)
        self.scaler       = bundle.get("scaler")
        self.feature_cols = bundle["feature_names"]

        self.model = VolatilityNet(input_dim=len(self.feature_cols)).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()

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
        feat_dict = compute_volatility_features(df)
        row = np.array([[feat_dict[c] for c in self.feature_cols]], dtype=np.float32)
        x   = torch.tensor(row, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(x)

        probs      = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx   = int(np.argmax(probs))
        p_breakout = float(probs[1])

        return {
            "regime": {
                "class":          pred_idx,
                "label":          HEAD_LABELS[pred_idx],
                "p_chop":         round(float(probs[0]), 4),
                "p_breakout":     round(p_breakout, 4),
                "interpretation": interpret_regime(p_breakout),
            },
            "signal":        HEAD_LABELS[pred_idx],
            "confidence":    float(probs.max()),
            "trade_enabled": bool(p_breakout >= 0.55),
            "features":      feat_dict,
        }

    def shutdown(self):
        if MT5_AVAILABLE:
            mt5.shutdown()


if __name__ == "__main__":
    predictor = VolatilityPredictor()
    result    = predictor.predict("EURUSD")
    print("\n── Model 3 · Volatility Regime ──────────────────────")
    r = result["regime"]
    print(f"  regime      → {r['label'].upper()}")
    print(f"  p_chop      → {r['p_chop']:.4f}")
    print(f"  p_breakout  → {r['p_breakout']:.4f}")
    print(f"  state       → {r['interpretation'].upper()}")
    print(f"  trade gate  → {'OPEN' if result['trade_enabled'] else 'BLOCKED (chop detected)'}")
    print(f"\n  Features used:")
    for k, v in result["features"].items():
        print(f"    {k:<40} {v:+.6f}")
    predictor.shutdown()
