import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from utils import resource_path
import cdx  # your CoinDCX library

# --- Model ---
class MicrostructureNet(nn.Module):
    def __init__(self, input_dim=15):
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

# --- Feature Computation ---
def compute_features(df):
    df["range"] = df["high"] - df["low"] + 1e-9
    df["body"] = (df["close"] - df["open"]).abs()
    df["wick"] = df["high"] - df[["close", "open"]].max(axis=1)
    df["tail"] = df[["close", "open"]].min(axis=1) - df["low"]

    df["body_ratio"] = (df["body"] / df["range"]) ** 0.9
    df["wick_ratio"] = (df["wick"] / df["range"]) ** 1.2
    df["tail_ratio"] = (df["tail"] / df["range"]) ** 1.2

    df["body_direction"] = np.sign(df["close"] - df["open"])
    df["body_dir"] = df["body"] * df["body_direction"]
    df["tail_dir"] = df["tail"] * df["body_direction"]
    df["wick_dir"] = df["wick"] * df["body_direction"]

    df["ret1"] = df["close"].pct_change()
    df["ret1_zscore_10"] = (df["ret1"] - df["ret1"].rolling(5).mean()) / (df["ret1"].rolling(5).std() + 1e-9)
    df["range_zscore_10"] = (df["range"] - df["range"].rolling(5).mean()) / (df["range"].rolling(5).std() + 1e-9)
    df["ret1_body_ratio"] = df["ret1"] * df["body_ratio"]

    return df

# --- Predictor ---
class MicrostructurePredictor:
    def __init__(self, model_path="microstructure_model.pth", scaler_path="microstructure_scaler.pkl"):
        model_path = resource_path(model_path)
        scaler_path = resource_path(scaler_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_cols = [
            "body", "wick", "tail", "body_ratio", "wick_ratio", "tail_ratio",
            "body_direction", "body_dir", "tail_dir", "wick_dir",
            "ret1", "ret1_zscore_10", "range", "range_zscore_10", "ret1_body_ratio"
        ]
        self.model = MicrostructureNet(input_dim=len(self.feature_cols)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.scaler = joblib.load(scaler_path)

    def predict(self, symbol="B-BTC_USDT", timeframe="15", debug=False):
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
            "burst_type": preds[0],
            "tail_pressure_direction": preds[1],
            "micro_risk_zone": preds[2],
            "confidence": confidences[2]
        }

# --- Example Use ---
if __name__ == "__main__":
    predictor = MicrostructurePredictor()
    result = predictor.predict(debug=False)
    print("📊 Microstructure Prediction:", result)
