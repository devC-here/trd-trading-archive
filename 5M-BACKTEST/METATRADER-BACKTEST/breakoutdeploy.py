import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator
from utils import resource_path # ADDED THIS LINE

# --- Model ---
class BreakoutNet(nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.heads = nn.ModuleList([nn.Linear(64, 3) for _ in range(3)])

    def forward(self, x):
        x = self.shared(x)
        return [head(x) for head in self.heads]

# --- Feature Engineering ---
def compute_features(df):
    # --- Indicators tuned for EURUSD M15 ---
    # Bollinger Bands
    bb = BollingerBands(df["close"], window=20, window_dev=2)

    # Standard MACD for M15 trend detection
    macd = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)

    # RSI
    rsi = RSIIndicator(df["close"], window=14)

    # -----------------------------
    # 🔵 Bollinger Band Features
    # -----------------------------
    df["boll_upper"] = bb.bollinger_hband()
    df["boll_lower"] = bb.bollinger_lband()
    df["boll_middle"] = bb.bollinger_mavg()

    # Bandwidth should be ATR-normalized for M15
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    df["boll_bandwidth"] = (df["boll_upper"] - df["boll_lower"]) / (df["atr"] + 1e-9)

    # Position inside bands (normalized)
    rng = (df["boll_upper"] - df["boll_lower"]).replace(0, 1e-9)
    df["boll_position"] = (df["close"] - df["boll_lower"]) / (rng + 1e-9)

    # -----------------------------
    # 🔵 MACD Features
    # -----------------------------
    macd_line = macd.macd()
    macd_signal = macd.macd_signal()
    macd_hist = macd.macd_diff()

    df["macd_hist"] = macd_hist

    # Cross flags: 0=Bearish cross, 1=No cross, 2=Bullish cross
    df["macd_cross_flag"] = 1
    up_cross = (macd_line > macd_signal) & (macd_line.shift(1) <= macd_signal.shift(1))
    down_cross = (macd_line < macd_signal) & (macd_line.shift(1) >= macd_signal.shift(1))

    df.loc[up_cross, "macd_cross_flag"] = 2
    df.loc[down_cross, "macd_cross_flag"] = 0

    # -----------------------------
    # 🔵 RSI Features
    # -----------------------------
    df["rsi_14"] = rsi.rsi()

    # EURUSD M15 requires longer zscore windows (smooth & stable)
    zwin = 40
    df["rsi_zscore"] = (df["rsi_14"] - df["rsi_14"].rolling(zwin).mean()) / \
                       (df["rsi_14"].rolling(zwin).std() + 1e-9)

    # -----------------------------
    # 🔵 Volatility / Compression
    # -----------------------------
    df["range"] = df["high"] - df["low"]

    # Short-term range → 3-candle (45 minutes)
    df["range_3"] = df["range"].rolling(3).mean()

    # Medium-term range → 12-candle (3 hours)
    df["range_12"] = df["range"].rolling(12).mean()

    # Compression = short-term volatility relative to medium-term
    df["compression_ratio"] = df["range_3"] / (df["range_12"] + 1e-9)

    # BB squeeze zscore (use 40 window for stability)
    df["squeeze_zscore"] = (df["boll_bandwidth"] - df["boll_bandwidth"].rolling(40).mean()) / \
                           (df["boll_bandwidth"].rolling(40).std() + 1e-9)

    # -----------------------------
    # 🔵 Interaction Terms
    # -----------------------------
    # These should be clipped to avoid exploding model logits
    df["boll_macd"] = np.clip(df["boll_position"] * df["macd_hist"], -3, 3)
    df["boll_rsi"] = np.clip(df["boll_position"] * df["rsi_zscore"], -3, 3)

    return df

# --- Inference Wrapper ---
class BreakoutPredictor:
    def __init__(self, model_path="breakout_model.pth", scaler_path="breakout_scaler.pkl"):
        # Corrected paths using resource_path
        model_path = resource_path(model_path) # MODIFIED LINE
        scaler_path = resource_path(scaler_path) # MODIFIED LINE

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_cols = [
            "boll_bandwidth", "boll_position", "compression_ratio", "squeeze_zscore",
            "macd_hist", "macd_cross_flag", "rsi_14", "rsi_zscore",
            "boll_macd", "boll_rsi"
        ]
        self.model = BreakoutNet(input_dim=len(self.feature_cols)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.scaler = joblib.load(scaler_path)

        if not mt5.initialize():
            raise RuntimeError("MT5 initialization failed")

    def predict(self, symbol, timeframe, start_candle=1, count=100):
        rates = mt5.copy_rates_from_pos(symbol, timeframe, start_candle, count)
        if rates is None or len(rates) < 50:
            return {"error": "Insufficient data"}

        df = pd.DataFrame(rates)
        df = compute_features(df).dropna()
        latest = df.iloc[-1:][self.feature_cols]
        scaled = self.scaler.transform(latest)

        x = torch.tensor(scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(x)

        preds = [int(torch.argmax(out).cpu()) for out in outputs]
        probs = [torch.softmax(out, dim=1).cpu().numpy()[0] for out in outputs]
        confidences = [float(np.max(p)) for p in probs]

        return {
            "squeeze_state": preds[0],
            "breakout_bias": preds[1],
            "entry_viability": preds[2],
            "confidence": confidences[2]
        }

    def shutdown(self):
        mt5.shutdown()

# --- Example Usage ---
if __name__ == "__main__":
    predictor = BreakoutPredictor()
    result = predictor.predict("BTCUSD", mt5.TIMEFRAME_M15)
    print("📊 Breakout–Compression Prediction:", result)
    predictor.shutdown()