# ─────────────────────────────────────────────────────────────
# Candle Classifier — MT5 LIVE (ONE-SHOT CLEAN VERSION)
# ─────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# CONFIG (SET HERE)
# ─────────────────────────────────────────────
SYMBOL    = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_H1   # change here if needed
LOOKBACK  = 10
EPS       = 1e-10


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class CandleNet(nn.Module):
    def __init__(self, input_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.net(x)


LABELS = {
    0: "bearish",
    1: "neutral",
    2: "bullish"
}


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
def compute_features(df: pd.DataFrame):
    last = df.iloc[-1]

    o, h, l, c = last["open"], last["high"], last["low"], last["close"]

    whole = max(h - l, EPS)

    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    body       = abs(c - o)

    wick_ratio = upper_wick / whole
    tail_ratio = lower_wick / whole
    body_ratio = body / whole

    return np.array([
        wick_ratio,
        body_ratio,
        tail_ratio,
        (upper_wick - lower_wick) / whole,
        body_ratio - (wick_ratio + tail_ratio)
    ], dtype=np.float32)


# ─────────────────────────────────────────────
# PREDICTOR
# ─────────────────────────────────────────────
class CandlePredictor:

    def __init__(self,
                 model_path="candle_model.pth",
                 scaler_path="candle_scaler.pkl"):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        bundle = joblib.load(scaler_path)
        self.scaler = bundle["scaler"]
        self.feature_cols = bundle["features"]

        self.model = CandleNet(len(self.feature_cols)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        if MT5_AVAILABLE:
            if not mt5.initialize():
                raise RuntimeError("MT5 initialization failed")

    # ─────────────────────────────────────────
    # SINGLE RUN
    # ─────────────────────────────────────────
    def predict_once(self):

        if not MT5_AVAILABLE:
            return {"error": "MT5 not available"}

        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 1, LOOKBACK)

        if rates is None or len(rates) < 3:
            return {"error": f"Insufficient bars: {0 if rates is None else len(rates)}"}

        df = pd.DataFrame(rates)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)

        feat = compute_features(df).reshape(1, -1)
        feat = self.scaler.transform(feat)

        x = torch.tensor(feat, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(x)

        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred  = int(np.argmax(probs))

        return {
            "symbol": SYMBOL,
            "timeframe": TIMEFRAME,
            "signal": LABELS[pred],
            "confidence": float(probs[pred]),
            "probs": {
                LABELS[i]: float(probs[i]) for i in range(3)
            }
        }

    def shutdown(self):
        if MT5_AVAILABLE:
            mt5.shutdown()


# ─────────────────────────────────────────────
# MAIN (ONE RUN ONLY)
# ─────────────────────────────────────────────
if __name__ == "__main__":

    predictor = CandlePredictor()

    result = predictor.predict_once()

    print("\n── Candle Prediction ────────────")

    if "error" in result:
        print("⚠️", result["error"])
    else:
        print(f"Symbol     : {result['symbol']}")
        print(f"Timeframe  : {result['timeframe']}")
        print(f"Signal     : {result['signal'].upper()}")
        print(f"Confidence : {result['confidence']:.2%}")
        print(f"Probs      : {result['probs']}")

    predictor.shutdown()