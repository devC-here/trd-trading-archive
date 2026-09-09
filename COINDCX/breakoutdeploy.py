import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator
from utils import resource_path
import cdx  # your CoinDCX library

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
    bb = BollingerBands(df["close"], window=20, window_dev=2)
    macd = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    rsi = RSIIndicator(df["close"], window=14)

    df["boll_upper"] = bb.bollinger_hband()
    df["boll_lower"] = bb.bollinger_lband()
    df["boll_middle"] = bb.bollinger_mavg()
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    df["boll_bandwidth"] = (df["boll_upper"] - df["boll_lower"]) / (df["atr"] + 1e-9)
    rng = (df["boll_upper"] - df["boll_lower"]).replace(0, 1e-9)
    df["boll_position"] = (df["close"] - df["boll_lower"]) / (rng + 1e-9)

    macd_line = macd.macd()
    macd_signal = macd.macd_signal()
    macd_hist = macd.macd_diff()
    df["macd_hist"] = macd_hist
    df["macd_cross_flag"] = 1
    up_cross = (macd_line > macd_signal) & (macd_line.shift(1) <= macd_signal.shift(1))
    down_cross = (macd_line < macd_signal) & (macd_line.shift(1) >= macd_signal.shift(1))
    df.loc[up_cross, "macd_cross_flag"] = 2
    df.loc[down_cross, "macd_cross_flag"] = 0

    df["rsi_14"] = rsi.rsi()
    zwin = 40
    df["rsi_zscore"] = (df["rsi_14"] - df["rsi_14"].rolling(zwin).mean()) / \
                       (df["rsi_14"].rolling(zwin).std() + 1e-9)

    df["range"] = df["high"] - df["low"]
    df["range_3"] = df["range"].rolling(3).mean()
    df["range_12"] = df["range"].rolling(12).mean()
    df["compression_ratio"] = df["range_3"] / (df["range_12"] + 1e-9)
    df["squeeze_zscore"] = (df["boll_bandwidth"] - df["boll_bandwidth"].rolling(40).mean()) / \
                           (df["boll_bandwidth"].rolling(40).std() + 1e-9)

    df["boll_macd"] = np.clip(df["boll_position"] * df["macd_hist"], -3, 3)
    df["boll_rsi"] = np.clip(df["boll_position"] * df["rsi_zscore"], -3, 3)

    return df

# --- Breakout Predictor using CDX ---
class BreakoutPredictor:
    def __init__(self, model_path="breakout_model.pth", scaler_path="breakout_scaler.pkl"):
        model_path = resource_path(model_path)
        scaler_path = resource_path(scaler_path)

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

    def predict(self, symbol="B-BTC_USDT", timeframe="15", debug=False):
        # Fetch 100 candles starting from second-to-last
        df = cdx.fetch_candles(symbol=symbol, timeframe=timeframe, start_index=1, num_candles=100)

        if debug:
            print("📊 Raw candles fetched:")
            print(df.head(10))
            print(f"Total candles fetched: {len(df)}")

        if df is None or len(df) < 50:
            return {"error": "Insufficient data"}

        df = compute_features(df).dropna()
        latest = df.iloc[-1:][self.feature_cols]
        scaled = self.scaler.transform(latest)

        x = torch.tensor(scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(x)

        preds = [int(torch.argmax(out).cpu()) for out in outputs]
        probs = [torch.softmax(out, dim=1).cpu().numpy()[0] for out in outputs]
        confidences = [float(np.max(p)) for p in probs]

        if debug:
            print("📊 Latest features used for prediction:")
            print(latest)

        return {
            "squeeze_state": preds[0],
            "breakout_bias": preds[1],
            "entry_viability": preds[2],
            "confidence": confidences[2]
        }

# --- Example Usage ---
if __name__ == "__main__":
    predictor = BreakoutPredictor()
    result = predictor.predict(debug=False)
    print("📊 Breakout–Compression Prediction:", result)
