import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from utils import resource_path # ADDED THIS LINE

# --- Model ---
class CandlestickSequenceNet(nn.Module):
    def __init__(self, input_dim=35):  # Adjusted to match training shape
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.heads = nn.ModuleList([nn.Linear(64, 5), nn.Linear(64, 3), nn.Linear(64, 3)])  # updated here

    def forward(self, x):
        x = self.shared(x)
        return [head(x) for head in self.heads]

# --- Feature Engineering ---
def engineer_candle_features(df):
    df["range"] = df["high"] - df["low"]
    df["body"] = abs(df["close"] - df["open"])
    df["wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["tail"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["body_dir"] = np.sign(df["close"] - df["open"])
    df["body_ratio"] = df["body"] / (df["range"] + 1e-6)
    df["tail_ratio"] = df["tail"] / (df["range"] + 1e-6)
    df["wick_ratio"] = df["wick"] / (df["range"] + 1e-6)
    return df

def get_sequence_features(df, seq_len=5):
    out = pd.DataFrame()
    for i in range(seq_len):
        shifted = df.shift(i).add_suffix(f"_t-{i}")
        out = pd.concat([out, shifted], axis=1)
    return out.dropna()

# --- Predictor ---
class CandlestickSequencePredictor:
    def __init__(self, model_path="candle_seq_model.pth", scaler_path="candle_seq_scaler.pkl"):
        # Corrected paths using resource_path
        model_path = resource_path(model_path) # MODIFIED LINE
        scaler_path = resource_path(scaler_path) # MODIFIED LINE

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CandlestickSequenceNet(input_dim=35).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.scaler = joblib.load(scaler_path)
        self.feature_cols = [
            f"{feat}_t-{i}" for i in range(5)
            for feat in ["body", "wick", "tail", "body_ratio", "tail_ratio", "wick_ratio", "body_dir"]
        ]

        if not mt5.initialize():
            raise RuntimeError("MT5 initialization failed")

    def predict(self, symbol, timeframe):
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 100)
        if rates is None or len(rates) < 10:
            return {"error": "Insufficient data"}

        df = pd.DataFrame(rates)
        df = engineer_candle_features(df)
        seq_df = get_sequence_features(df).dropna()
        if seq_df.empty:
            return {"error": "Not enough valid sequence"}

        # --- Fix: Enforce training-time feature order
        latest = seq_df.iloc[-1:][self.scaler.feature_names_in_]
        scaled = self.scaler.transform(latest)
        x = torch.tensor(scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(x)

        preds = [int(torch.argmax(out).cpu()) for out in outputs]
        probs = [torch.softmax(out, dim=1).cpu().numpy()[0] for out in outputs]
        confidences = [float(np.max(p)) for p in probs]

        return {
            "sequence_type": preds[0],
            "candle_dominance": preds[1],
            "pattern_strength": preds[2],
            "confidence": confidences[2]
        }

    def shutdown(self):
        mt5.shutdown()

# --- Example Use ---
if __name__ == "__main__":
    predictor = CandlestickSequencePredictor()
    result = predictor.predict("XAUUSDm", mt5.TIMEFRAME_M30)
    print("\U0001F4CA Candlestick Sequence Prediction:", result)
    predictor.shutdown()