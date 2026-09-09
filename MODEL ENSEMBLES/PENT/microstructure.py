import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from utils import resource_path # ADDED THIS LINE

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
    df["range"] = df["high"] - df["low"]
    df["body"] = (df["close"] - df["open"]).abs()
    df["wick"] = df["high"] - df[["close", "open"]].max(axis=1)
    df["tail"] = df[["close", "open"]].min(axis=1) - df["low"]

    df["body_ratio"] = df["body"] / (df["range"] + 1e-6)
    df["wick_ratio"] = df["wick"] / (df["range"] + 1e-6)
    df["tail_ratio"] = df["tail"] / (df["range"] + 1e-6)

    df["body_direction"] = np.sign(df["close"] - df["open"])
    df["body_dir"] = df["body"] * df["body_direction"]
    df["tail_dir"] = df["tail"] * df["body_direction"]
    df["wick_dir"] = df["wick"] * df["body_direction"]

    df["ret1"] = df["close"].pct_change()
    df["ret1_zscore_10"] = (df["ret1"] - df["ret1"].rolling(10).mean()) / df["ret1"].rolling(10).std()
    df["range_zscore_10"] = (df["range"] - df["range"].rolling(10).mean()) / df["range"].rolling(10).std()
    df["ret1_body_ratio"] = df["ret1"] * df["body_ratio"]

    return df

# --- Inference Wrapper ---
class MicrostructurePredictor:
    def __init__(self, model_path="microstructure_model.pth", scaler_path="microstructure_scaler.pkl"):
        # Corrected paths using resource_path
        model_path = resource_path(model_path) # MODIFIED LINE
        scaler_path = resource_path(scaler_path) # MODIFIED LINE

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
            "burst_type": preds[0],
            "tail_pressure_direction": preds[1],
            "micro_risk_zone": preds[2],
            "confidence": confidences[2]
        }

    def shutdown(self):
        mt5.shutdown()

# --- Example Use ---
if __name__ == "__main__":
    predictor = MicrostructurePredictor()
    result = predictor.predict("XAUUSDm", mt5.TIMEFRAME_M30)
    print("📊 Microstructure Prediction:", result)
    predictor.shutdown()