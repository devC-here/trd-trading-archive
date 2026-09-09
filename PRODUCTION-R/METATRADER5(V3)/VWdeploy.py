import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from utils import resource_path # ADDED THIS LINE

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
    # 1. Standardize Volume Input
    if "tick_volume" in df.columns and "volume" not in df.columns:
        df["volume"] = df["tick_volume"]

    # 2. Session VWAP (Daily Reset)
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    session_id = pd.to_datetime(df["time"], unit="s").dt.date
    
    # Calculate cumulative sums for the session
    cum_tp_vol = (typical_price * df["volume"]).groupby(session_id).cumsum()
    cum_vol = df["volume"].groupby(session_id).cumsum()
    df["vwap"] = cum_tp_vol / (cum_vol + 1e-9)

    # 3. Calculate ATR for Normalization (Needed to keep price-scale agnostic)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(14).mean() + 1e-9

    # --- RETAINING ORIGINAL NAMES BUT WITH NORMALIZED MATH ---

    # Instead of raw distance, we use ATR-relative distance
    df["close_minus_vwap"] = (df["close"] - df["vwap"]) / atr
    
    # Binary stays the same (it's already 0 or 1)
    df["is_above_vwap"] = (df["close"] > df["vwap"]).astype(int)
    
    # Distance ratio (using ATR makes this scale-agnostic)
    df["vwap_distance_ratio"] = np.abs(df["close"] - df["vwap"]) / atr

    # Volume Z-Score (Scale-agnostic volume)
    vol_rolling = df["volume"].rolling(10)
    df["volume_z"] = (df["volume"] - vol_rolling.mean()) / (vol_rolling.std() + 1e-9)
    
    # Volume change relative to its own standard deviation
    df["volume_rolling_change"] = df["volume"].diff(1) / (vol_rolling.std() + 1e-9)

    # Volatility (Percentage change based is already scale-agnostic)
    df["volatility"] = df["close"].pct_change().rolling(3).std()

    # --- Interaction terms (Now using normalized components) ---
    # vwap_x_volume: (ATR distance) * (Volume Z-score)
    df["vwap_x_volume"] = df["close_minus_vwap"] * df["volume_z"]
    
    # vwap_vol_displacement: (Distance) * (Volume Z-score)
    df["vwap_vol_displacement"] = df["close_minus_vwap"] * df["volume_z"]
    
    # volatility_x_volume: (Volatility %) * (Volume Z-score)
    df["volatility_x_volume"] = df["volatility"] * df["volume_z"]

    return df.dropna()
# --- Predictor Class ---
class VwapVolumePredictor:
    def __init__(self, model_path="vwap_volume_model.pth", scaler_path="vwap_volume_scaler.pkl"):
        # Corrected paths using resource_path
        model_path = resource_path(model_path) # MODIFIED LINE
        scaler_path = resource_path(scaler_path) # MODIFIED LINE

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

        if not mt5.initialize():
            raise RuntimeError("MT5 initialization failed")

    def predict(self, symbol, timeframe):
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, 100)
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
            "vwap_zone": preds[0],
            "volume_pressure": preds[1],
            "liquidity_structure": preds[2],
            "confidence": confidences[2]
        }

    def shutdown(self):
        mt5.shutdown()

# --- Example Use ---
if __name__ == "__main__":
    predictor = VwapVolumePredictor()
    result = predictor.predict("EURUSD.e", mt5.TIMEFRAME_M15)
    print("📊 VWAP–Volume Prediction:", result)
    predictor.shutdown()