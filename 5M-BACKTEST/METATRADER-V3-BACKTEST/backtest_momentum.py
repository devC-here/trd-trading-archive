import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from ta.momentum import RSIIndicator
from ta.trend import MACD
from utils import resource_path

# --- Model (identical to deploy) ---
class MomentumOscillatorNet(nn.Module):
    def __init__(self, input_dim=14):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
        )
        self.heads = nn.ModuleList([nn.Linear(64, 3) for _ in range(3)])

    def forward(self, x):
        x = self.shared(x)
        return [head(x) for head in self.heads]

# --- Feature Engineering (identical to deploy) ---
def compute_features(df):
    rsi = RSIIndicator(df["close"], window=14)
    macd = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)

    df["rsi_14"] = rsi.rsi()
    df["rsi_zscore_20"] = (df["rsi_14"] - df["rsi_14"].rolling(10).mean()) / (df["rsi_14"].rolling(10).std() + 1e-9)

    df["roc_3"] = df["close"].pct_change(3)
    df["roc_5"] = df["close"].pct_change(5)
    df["roc_slope_3"] = df["roc_3"].diff(2)

    df["macd_line"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    df["macd_hist_slope"] = df["macd_hist"].diff(1)

    df["return_3"] = df["close"].pct_change(3)
    df["return_zscore_20"] = (df["return_3"] - df["return_3"].rolling(10).mean()) / (df["return_3"].rolling(10).std() + 1e-9)

    df["rsi_x_roc"] = np.clip(df["rsi_14"] * df["roc_3"], -2, 2)
    df["rsi_x_macd_hist"] = np.clip(df["rsi_14"] * df["macd_hist"], -5, 5)
    df["macd_hist_x_roc"] = np.clip(df["macd_hist"] * df["roc_3"], -5, 5)

    return df

# --- Backtest Runner ---
class MomentumBacktest:
    def __init__(self, model_path="momentum_model.pth", scaler_path="momentum_scaler.pkl"):
        model_path = resource_path(model_path)
        scaler_path = resource_path(scaler_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_cols = [
            "rsi_14", "rsi_zscore_20", "roc_3", "roc_5", "roc_slope_3",
            "macd_line", "macd_signal", "macd_hist", "macd_hist_slope",
            "return_3", "return_zscore_20",
            "rsi_x_roc", "rsi_x_macd_hist", "macd_hist_x_roc"
        ]
        # entry_bias head (index 2): 0=sell, 1=neutral, 2=buy
        self.direction_map = {0: "SELL", 1: "NEUTRAL", 2: "BUY"}

        self.model = MomentumOscillatorNet(input_dim=len(self.feature_cols)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.scaler = joblib.load(scaler_path)

        if not mt5.initialize():
            raise RuntimeError("MT5 initialization failed")

    def run(self, symbol, timeframe, n_candles=500):
        fetch_count = n_candles + 100
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, fetch_count)
        if rates is None or len(rates) < 50:
            raise RuntimeError(f"Insufficient data for {symbol} {timeframe}")

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = compute_features(df)
        feature_df = df.dropna().reset_index(drop=True)

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
                "time":               feature_df.iloc[i]["time"],
                "open":               feature_df.iloc[i]["open"],
                "high":               feature_df.iloc[i]["high"],
                "low":                feature_df.iloc[i]["low"],
                "close":              curr_close,
                "next_close":         next_close,
                "momentum_state":     preds[0],
                "momentum_slope_zone":preds[1],
                "entry_bias":         preds[2],
                "pred_direction":     pred_direction,
                "actual_direction":   actual_direction,
                "confidence":         round(confidences[2], 4),
                "correct":            correct,
            })

        return pd.DataFrame(results)

    def shutdown(self):
        mt5.shutdown()
