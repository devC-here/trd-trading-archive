# deploy_model1_momentum.py
# Live inference for the Directional Momentum Specialist
# Architecture and features match train_model1_momentum.py exactly
# Features : 6  |  Head : 1 binary (Bearish / Bullish)
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
LOOKBACK = 60
TRUNK    = 128
DROP     = 0.20


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


class MomentumNet(nn.Module):
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


def _ols_slope_r2(y):
    n  = len(y)
    x  = np.arange(n, dtype=float)
    mx, my = x.mean(), y.mean()
    ss_xy  = ((x - mx) * (y - my)).sum()
    ss_xx  = ((x - mx) ** 2).sum() + EPS
    ss_yy  = ((y - my) ** 2).sum() + EPS
    slope  = ss_xy / ss_xx
    r2     = (ss_xy ** 2) / (ss_xx * ss_yy)
    return float(slope), float(r2)


def compute_momentum_features(df: pd.DataFrame) -> dict:
    df = df.copy().reset_index(drop=True)
    C  = df["close"].values.astype(float)

    roc_4 = float((C[-1] - C[-5]) / (C[-5] + EPS) * 100) if len(C) >= 5 else 0.0

    window = min(20, len(C))
    rets   = np.diff(np.log(C[-window:] + EPS))
    if len(rets) >= 3:
        mean_r, std_r = rets.mean(), max(rets.std(), EPS)
        skew = float(((rets - mean_r) ** 3).mean() / (std_r ** 3))
    else:
        skew = 0.0

    counter = 0
    for i in range(len(C) - 1, 0, -1):
        if C[i] > C[i - 1]:
            if counter >= 0: counter += 1
            else: break
        elif C[i] < C[i - 1]:
            if counter <= 0: counter -= 1
            else: break
        else:
            break

    o_last, c_last = float(df["open"].iloc[-1]), float(df["close"].iloc[-1])
    candle_color = 1 if c_last > o_last else (-1 if c_last < o_last else 0)

    roc_4_prev = float((C[-2] - C[-6]) / (C[-6] + EPS) * 100) if len(C) >= 6 else roc_4
    roc_accel  = roc_4 - roc_4_prev

    _, r2 = _ols_slope_r2(C[-min(20, len(C)):])

    return {
        "13_roc_4":                          roc_4,
        "58_skewness_returns":               skew,
        "101_consecutive_direction_counter": float(counter),
        "103_final_candle_color":            float(candle_color),
        "15_roc_acceleration":               roc_accel,
        "36_r_squared":                      r2,
    }


HEAD_LABELS = {1: "bearish", 0: "bullish"}


class MomentumPredictor:
    def __init__(self,
                 model_path: str = "model1_best.pth",
                 meta_path:  str = "model1_meta.pkl"):

        model_path = resource_path(model_path)
        meta_path  = resource_path(meta_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        bundle            = joblib.load(meta_path)
        self.scaler       = bundle.get("scaler")
        self.feature_cols = bundle["feature_names"]

        self.model = MomentumNet(input_dim=len(self.feature_cols)).to(self.device)
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
        if rates is None or len(rates) < 20:
            return {"error": f"Insufficient bars: {0 if rates is None else len(rates)}"}
        df = pd.DataFrame(rates)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        return self.predict_from_df(df)

    def predict_from_df(self, df: pd.DataFrame) -> dict:
        feat_dict = compute_momentum_features(df)
        row = np.array([[feat_dict[c] for c in self.feature_cols]], dtype=np.float32)
        x = torch.tensor(row, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(x)

        probs    = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))

        return {
            "direction": {
                "class":      pred_idx,
                "label":      HEAD_LABELS[pred_idx],
                "confidence": float(probs.max()),
                "probs":      {HEAD_LABELS[k]: round(float(probs[k]), 4) for k in range(2)},
            },
            "signal":     HEAD_LABELS[pred_idx],
            "confidence": float(probs.max()),
            "features":   feat_dict,
        }

    def shutdown(self):
        if MT5_AVAILABLE:
            mt5.shutdown()


if __name__ == "__main__":
    predictor = MomentumPredictor()
    result    = predictor.predict("EURUSD")
    print("\n── Model 1 · Momentum Prediction ───────────────────")
    d = result["direction"]
    print(f"  direction   → {d['label'].upper():<10} conf={d['confidence']:.2%}")
    print(f"  probs       → bearish={d['probs']['bearish']:.4f}  bullish={d['probs']['bullish']:.4f}")
    print(f"\n  Features used:")
    for k, v in result["features"].items():
        print(f"    {k:<40} {v:+.6f}")
    predictor.shutdown()
