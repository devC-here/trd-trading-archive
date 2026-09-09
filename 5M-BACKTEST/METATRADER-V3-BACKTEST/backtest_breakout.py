import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator
from utils import resource_path

# --- Model (identical to deploy) ---
class BreakoutNet(nn.Module):
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
    bb = BollingerBands(df["close"], window=14, window_dev=2)
    macd = MACD(df["close"], window_slow=18, window_fast=6, window_sign=5)
    rsi = RSIIndicator(df["close"], window=9)

    df["boll_upper"] = bb.bollinger_hband()
    df["boll_lower"] = bb.bollinger_lband()
    df["boll_middle"] = bb.bollinger_mavg()
    df["atr"] = (df["high"] - df["low"]).rolling(10).mean()
    df["boll_bandwidth"] = (df["boll_upper"] - df["boll_lower"]) / (df["atr"] + 1e-9)

    rng = (df["boll_upper"] - df["boll_lower"]).replace(0, 1e-9)
    df["boll_position"] = (df["close"] - df["boll_lower"]) / (rng + 1e-9)

    macd_line = macd.macd()
    macd_signal = macd.macd_signal()
    macd_hist = macd.macd_diff()
    df["macd_hist"] = macd_hist * 1.5
    df["macd_cross_flag"] = 1
    up_cross = (macd_line > macd_signal) & (macd_line.shift(1) <= macd_signal.shift(1))
    down_cross = (macd_line < macd_signal) & (macd_line.shift(1) >= macd_signal.shift(1))
    df.loc[up_cross, "macd_cross_flag"] = 2
    df.loc[down_cross, "macd_cross_flag"] = 0

    df["rsi_14"] = rsi.rsi()
    zwin = 20
    df["rsi_zscore"] = (df["rsi_14"] - df["rsi_14"].rolling(zwin).mean()) / (df["rsi_14"].rolling(zwin).std() + 1e-9)

    df["range"] = df["high"] - df["low"]
    df["range_2"] = df["range"].rolling(2).mean()
    df["range_8"] = df["range"].rolling(8).mean()
    df["compression_ratio"] = df["range_2"] / (df["range_8"] + 1e-9)
    df["squeeze_zscore"] = (df["boll_bandwidth"] - df["boll_bandwidth"].rolling(20).mean()) / (df["boll_bandwidth"].rolling(20).std() + 1e-9)

    df["boll_macd"] = np.clip(df["boll_position"] * df["macd_hist"], -3, 3)
    df["boll_rsi"] = np.clip(df["boll_position"] * df["rsi_zscore"], -3, 3)

    return df

# --- Backtest Runner ---
class BreakoutBacktest:
    def __init__(self, model_path="breakout_model.pth", scaler_path="breakout_scaler.pkl"):
        model_path = resource_path(model_path)
        scaler_path = resource_path(scaler_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_cols = [
            "boll_bandwidth", "boll_position", "compression_ratio", "squeeze_zscore",
            "macd_hist", "macd_cross_flag", "rsi_14", "rsi_zscore",
            "boll_macd", "boll_rsi"
        ]
        # Map label → direction string
        # entry_viability head (index 2): 0=sell, 1=neutral, 2=buy
        self.direction_map = {0: "SELL", 1: "NEUTRAL", 2: "BUY"}

        self.model = BreakoutNet(input_dim=len(self.feature_cols)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.scaler = joblib.load(scaler_path)

        if not mt5.initialize():
            raise RuntimeError("MT5 initialization failed")

    def run(self, symbol, timeframe, n_candles=500):
        """
        Fetch n_candles of history, run prediction on each bar (rolling window),
        and evaluate direction vs next-bar actual move.
        """
        # Fetch enough history: n_candles + warm-up buffer for indicators
        fetch_count = n_candles + 150
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, fetch_count)
        if rates is None or len(rates) < 100:
            raise RuntimeError(f"Insufficient data for {symbol}")

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = compute_features(df)

        feature_df = df.dropna().reset_index(drop=True)

        # Need at least 1 future candle to check direction
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

            # entry_viability is head[2]: 0=sell, 1=neutral, 2=buy
            pred_direction = self.direction_map[preds[2]]

            # Actual next-bar direction
            curr_close = feature_df.iloc[i]["close"]
            next_close = feature_df.iloc[i + 1]["close"]
            actual_direction = "BUY" if next_close > curr_close else "SELL"

            # Correctness (neutral predictions are excluded from accuracy calc)
            if pred_direction == "NEUTRAL":
                correct = None
            else:
                correct = (pred_direction == actual_direction)

            results.append({
                "time":               feature_df.iloc[i]["time"],
                "open":               feature_df.iloc[i]["open"],
                "high":               feature_df.iloc[i]["high"],
                "low":                feature_df.iloc[i]["low"],
                "close":              curr_close,
                "next_close":         next_close,
                "squeeze_state":      preds[0],
                "breakout_bias":      preds[1],
                "entry_viability":    preds[2],
                "pred_direction":     pred_direction,
                "actual_direction":   actual_direction,
                "confidence":         round(confidences[2], 4),
                "correct":            correct,
            })

        return pd.DataFrame(results)

    def shutdown(self):
        mt5.shutdown()
