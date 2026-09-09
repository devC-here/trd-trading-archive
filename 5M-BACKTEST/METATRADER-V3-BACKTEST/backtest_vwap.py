import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from utils import resource_path

# --- Model (identical to deploy) ---
class VwapVolumeNet(nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2)
        )
        self.heads = nn.ModuleList([nn.Linear(64, 3) for _ in range(3)])

    def forward(self, x):
        x = self.shared(x)
        return [head(x) for head in self.heads]

# --- Feature Engineering (identical to deploy) ---
def compute_features(df):
    if "tick_volume" in df.columns and "volume" not in df.columns:
        df["volume"] = df["tick_volume"]

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    session_id = pd.to_datetime(df["time"], unit="s").dt.date
    cum_tp_vol = (typical_price * df["volume"]).groupby(session_id).cumsum()
    cum_vol = df["volume"].groupby(session_id).cumsum()
    df["vwap"] = cum_tp_vol / (cum_vol + 1e-9)

    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(14).mean() + 1e-9

    df["close_minus_vwap"] = (df["close"] - df["vwap"]) / atr
    df["is_above_vwap"] = (df["close"] > df["vwap"]).astype(int)
    df["vwap_distance_ratio"] = np.abs(df["close"] - df["vwap"]) / atr

    vol_rolling = df["volume"].rolling(10)
    df["volume_z"] = (df["volume"] - vol_rolling.mean()) / (vol_rolling.std() + 1e-9)
    df["volume_rolling_change"] = df["volume"].diff(1) / (vol_rolling.std() + 1e-9)
    df["volatility"] = df["close"].pct_change().rolling(3).std()

    df["vwap_x_volume"] = df["close_minus_vwap"] * df["volume_z"]
    df["vwap_vol_displacement"] = df["close_minus_vwap"] * df["volume_z"]
    df["volatility_x_volume"] = df["volatility"] * df["volume_z"]

    return df.dropna()

# --- Backtest Runner ---
class VwapBacktest:
    def __init__(self, model_path="vwap_volume_model.pth", scaler_path="vwap_volume_scaler.pkl"):
        model_path = resource_path(model_path)
        scaler_path = resource_path(scaler_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_cols = [
            "close_minus_vwap", "is_above_vwap", "vwap_distance_ratio",
            "volume", "volume_z", "volume_rolling_change", "volatility",
            "vwap_x_volume", "vwap_vol_displacement", "volatility_x_volume"
        ]
        # liquidity_structure head (index 2): 0=sell, 1=neutral, 2=buy
        self.direction_map = {0: "SELL", 1: "NEUTRAL", 2: "BUY"}

        self.model = VwapVolumeNet(input_dim=len(self.feature_cols)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.scaler = joblib.load(scaler_path)

        if not mt5.initialize():
            raise RuntimeError("MT5 initialization failed")

    def run(self, symbol, timeframe, n_candles=500):
        fetch_count = n_candles + 50
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, fetch_count)
        if rates is None or len(rates) < 50:
            raise RuntimeError(f"Insufficient data for {symbol}")

        df = pd.DataFrame(rates)
        # Keep raw time col for VWAP session groupby, then convert
        df = compute_features(df)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        feature_df = df.reset_index(drop=True)

        results = []
        for i in range(len(feature_df) - 1):
            row = feature_df.iloc[i:i+1][self.feature_cols]
            scaled = self.scaler.transform(row)
            x = torch.tensor(scaled, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                outputs = self.model(x)

            preds = [int(torch.argmax(out).cpu()) for out in outputs]
            probs = [torch.softmax(out, dim=1).cpu().numpy()[0] for out in outputs]
            confidences = [float(np.max(p)) for p in probs]

            pred_direction = self.direction_map[preds[2]]

            curr_close = feature_df.iloc[i]["close"]
            next_close = feature_df.iloc[i + 1]["close"]
            actual_direction = "BUY" if next_close > curr_close else "SELL"

            correct = None if pred_direction == "NEUTRAL" else (pred_direction == actual_direction)

            results.append({
                "time":                feature_df.iloc[i]["time"],
                "open":                feature_df.iloc[i]["open"],
                "high":                feature_df.iloc[i]["high"],
                "low":                 feature_df.iloc[i]["low"],
                "close":               curr_close,
                "next_close":          next_close,
                "vwap_zone":           preds[0],
                "volume_pressure":     preds[1],
                "liquidity_structure": preds[2],
                "pred_direction":      pred_direction,
                "actual_direction":    actual_direction,
                "confidence":          round(confidences[2], 4),
                "correct":             correct,
            })

        return pd.DataFrame(results)

    def shutdown(self):
        mt5.shutdown()
