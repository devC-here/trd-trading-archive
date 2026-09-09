import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from utils import resource_path

# --- Model (identical to deploy) ---
class MicrostructureNet(nn.Module):
    def __init__(self, input_dim=15):
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

# --- Backtest Runner ---
class MicrostructureBacktest:
    def __init__(self, model_path="microstructure_model.pth", scaler_path="microstructure_scaler.pkl"):
        model_path = resource_path(model_path)
        scaler_path = resource_path(scaler_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_cols = [
            "body", "wick", "tail", "body_ratio", "wick_ratio", "tail_ratio",
            "body_direction", "body_dir", "tail_dir", "wick_dir",
            "ret1", "ret1_zscore_10", "range", "range_zscore_10", "ret1_body_ratio"
        ]
        # micro_risk_zone head (index 2): 0=sell, 1=neutral, 2=buy
        self.direction_map = {0: "SELL", 1: "NEUTRAL", 2: "BUY"}

        self.model = MicrostructureNet(input_dim=len(self.feature_cols)).to(self.device)
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
                "time":                    feature_df.iloc[i]["time"],
                "open":                    feature_df.iloc[i]["open"],
                "high":                    feature_df.iloc[i]["high"],
                "low":                     feature_df.iloc[i]["low"],
                "close":                   curr_close,
                "next_close":              next_close,
                "burst_type":              preds[0],
                "tail_pressure_direction": preds[1],
                "micro_risk_zone":         preds[2],
                "pred_direction":          pred_direction,
                "actual_direction":        actual_direction,
                "confidence":              round(confidences[2], 4),
                "correct":                 correct,
            })

        return pd.DataFrame(results)

    def shutdown(self):
        mt5.shutdown()
