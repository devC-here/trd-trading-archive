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
    bb = BollingerBands(df["close"], window=20, window_dev=2)
    macd = MACD(df["close"])
    rsi = RSIIndicator(df["close"])

    df["boll_upper"] = bb.bollinger_hband()
    df["boll_lower"] = bb.bollinger_lband()
    df["boll_middle"] = bb.bollinger_mavg()
    df["boll_bandwidth"] = (df["boll_upper"] - df["boll_lower"]) / df["boll_middle"]
    df["boll_position"] = (df["close"] - df["boll_lower"]) / (df["boll_upper"] - df["boll_lower"])

    df["macd_hist"] = macd.macd_diff()
    df["macd_cross_flag"] = 1
    up = (macd.macd() > macd.macd_signal()) & (macd.macd().shift(1) <= macd.macd_signal().shift(1))
    down = (macd.macd() < macd.macd_signal()) & (macd.macd().shift(1) >= macd.macd_signal().shift(1))
    df.loc[up, "macd_cross_flag"] = 2
    df.loc[down, "macd_cross_flag"] = 0

    df["rsi_14"] = rsi.rsi()
    df["rsi_zscore"] = (df["rsi_14"] - df["rsi_14"].rolling(20).mean()) / df["rsi_14"].rolling(20).std()

    df["range"] = df["high"] - df["low"]
    df["range_5"] = df["range"].rolling(5).mean()
    df["range_20"] = df["range"].rolling(20).mean()
    df["compression_ratio"] = df["range_5"] / df["range_20"]
    df["squeeze_zscore"] = (df["boll_bandwidth"] - df["boll_bandwidth"].rolling(20).mean()) / df["boll_bandwidth"].rolling(20).std()

    df["boll_macd"] = df["boll_position"] * df["macd_hist"]
    df["boll_rsi"] = df["boll_position"] * df["rsi_zscore"]

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

    def predict(self, symbol, timeframe):
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 100)
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
    result = predictor.predict("XAUUSDm", mt5.TIMEFRAME_M30)
    print("📊 Breakout–Compression Prediction:", result)
    predictor.shutdown()