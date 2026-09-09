import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from utils import resource_path
import cdx  # your CoinDCX library

# --- Model ---
class VwapVolumeNet(nn.Module):
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
    if "tick_volume" in df.columns and "volume" not in df.columns:
        df["volume"] = df["tick_volume"]

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    session_id = pd.to_datetime(df["time"], unit="s").dt.date

    df["cum_tp_vol"] = typical_price * df["volume"]
    df["cum_vol"] = df["volume"]
    df["cum_tp_vol"] = df.groupby(session_id)["cum_tp_vol"].cumsum()
    df["cum_vol"] = df.groupby(session_id)["cum_vol"].cumsum()
    df["vwap"] = df["cum_tp_vol"] / (df["cum_vol"] + 1e-9)

    df["close_minus_vwap"] = df["close"] - df["vwap"]
    df["is_above_vwap"] = (df["close"] > df["vwap"]).astype(int)
    df["vwap_distance_ratio"] = np.abs(df["close"] - df["vwap"]) / (df["close"] + 1e-9)

    df["volume_z"] = (df["volume"] - df["volume"].rolling(10).mean()) / (df["volume"].rolling(10).std() + 1e-9)
    df["volume_rolling_change"] = df["volume"].diff(1)
    df["volatility"] = df["close"].pct_change().rolling(3).std()

    df["vwap_x_volume"] = df["vwap"] * df["volume"]
    df["vwap_vol_displacement"] = df["close_minus_vwap"] * df["volume"]
    df["volatility_x_volume"] = df["volatility"] * df["volume"]

    return df

# --- Predictor Class ---
class VwapVolumePredictor:
    def __init__(self, model_path="vwap_volume_model.pth", scaler_path="vwap_volume_scaler.pkl"):
        model_path = resource_path(model_path)
        scaler_path = resource_path(scaler_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_cols = [
            "close_minus_vwap", "is_above_vwap", "vwap_distance_ratio",
            "volume", "volume_z", "volume_rolling_change", "volatility",
            "vwap_x_volume", "vwap_vol_displacement", "volatility_x_volume"
        ]
        self.model = VwapVolumeNet(input_dim=len(self.feature_cols)).to(self.device)
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
            "vwap_zone": preds[0],
            "volume_pressure": preds[1],
            "liquidity_structure": preds[2],
            "confidence": confidences[2]
        }

# --- Usage Example ---
if __name__ == "__main__":
    predictor = VwapVolumePredictor()
    result = predictor.predict(debug=False)
    print("📊 VWAP–Volume Prediction:", result)
