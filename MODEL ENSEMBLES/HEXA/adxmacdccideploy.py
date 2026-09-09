import MetaTrader5 as mt5
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
from ta.trend import ADXIndicator, MACD
from ta.volatility import AverageTrueRange

def compute_cci(high, low, close, window=20, constant=0.015):
    tp = (high + low + close) / 3
    sma = tp.rolling(window).mean()
    mad = tp.rolling(window).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci = (tp - sma) / (constant * mad)
    return cci

class TriSignalModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, 3),
            nn.Linear(hidden_dim, 3),
            nn.Linear(hidden_dim, 3)
        ])

    def forward(self, x):
        x = self.shared(x)
        return [head(x) for head in self.heads]

class MarketPredictor:
    def __init__(self, symbols=None):
        self.symbols = symbols or ["XAUUSDm"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.scaler = joblib.load("adx_cci_macd_asset_scaler.pkl")
        self.feature_columns = [
            'adx', 'cci', 'macd', 'macd_signal', 'macd_hist',
            'adx_delta_1', 'adx_slope_3', 'adx_avg_5', 'cci_zscore_20',
            'cci_trend_change', 'macd_hist_slope', 'macd_cross_up',
            'macd_cross_down', 'adx_x_macd_hist', 'cci_x_macd',
            'cci_x_ret3', 'roc_3', 'atr_14', 'atr_norm',
            'body', 'wick', 'tail', 'body_ratio', 'wick_ratio', 'tail_ratio'
        ]

        self.model = TriSignalModel(input_dim=len(self.feature_columns)).to(self.device)
        self.model.load_state_dict(torch.load("adx_cci_macd_asset_model.pth", map_location=self.device))
        self.model.eval()

        if not mt5.initialize():
            raise RuntimeError("❌ MT5 Initialization Failed!")

    def calculate_features(self, df):
        df["adx"] = ADXIndicator(df["high"], df["low"], df["close"], window=14).adx()
        macd = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()
        df["atr_14"] = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
        df["atr_norm"] = df["atr_14"] / df["close"]
        df["cci"] = compute_cci(df["high"], df["low"], df["close"])

        df["adx_delta_1"] = df["adx"].diff()
        df["adx_slope_3"] = df["adx"].diff(3)
        df["adx_avg_5"] = df["adx"].rolling(5).mean()
        df["cci_zscore_20"] = (df["cci"] - df["cci"].rolling(20).mean()) / df["cci"].rolling(20).std()
        df["cci_trend_change"] = df["cci"].diff().apply(np.sign).diff().fillna(0)

        df["macd_hist_slope"] = df["macd_hist"].diff()
        df["macd_cross_up"] = ((df["macd"] > df["macd_signal"]) & (df["macd"].shift(1) <= df["macd_signal"].shift(1))).astype(int)
        df["macd_cross_down"] = ((df["macd"] < df["macd_signal"]) & (df["macd"].shift(1) >= df["macd_signal"].shift(1))).astype(int)

        df["adx_x_macd_hist"] = df["adx"] * df["macd_hist"]
        df["cci_x_macd"] = df["cci"] * df["macd"]
        df["cci_x_ret3"] = df["cci"] * df["close"].pct_change(3)
        df["roc_3"] = df["close"].pct_change(3)

        df["body"] = (df["close"] - df["open"]).abs()
        df["wick"] = df["high"] - df[["close", "open"]].max(axis=1)
        df["tail"] = df[["close", "open"]].min(axis=1) - df["low"]

        price_range = df["high"] - df["low"] + 1e-6
        df["body_ratio"] = df["body"] / price_range
        df["wick_ratio"] = df["wick"] / price_range
        df["tail_ratio"] = df["tail"] / price_range

        return df

    def get_prediction(self, symbol, timeframe):
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, 100)
            if rates is None or len(rates) < 50:
                return {"symbol": symbol, "direction": "No Trade", "confidence": 0.0}

            df = pd.DataFrame(rates)
            df.rename(columns={"time": "datetime"}, inplace=True)
            df["datetime"] = pd.to_datetime(df["datetime"], unit="s")
            df = self.calculate_features(df)
            df.dropna(inplace=True)

            latest = df.iloc[-1:][self.feature_columns]
            scaled = self.scaler.transform(latest)

            inputs = torch.tensor(scaled, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs)

            results = [torch.softmax(out, dim=1).cpu().numpy()[0] for out in outputs]
            preds = [int(np.argmax(r)) for r in results]
            confidences = [float(np.max(r)) for r in results]

            direction_map = {0: "Sell", 1: "Neutral", 2: "Buy"}

            return {
                "symbol": symbol,
                "adx_regime": preds[0],
                "cci_zone": preds[1],
                "macd_cross_flag": preds[2],
                "direction": direction_map.get(preds[2], "No Trade"),
                "confidence": confidences[2]
            }

        except Exception as e:
            return {
                "symbol": symbol,
                "direction": "No Trade",
                "confidence": 0.0,
                "error": str(e)
            }

    def shutdown(self):
        mt5.shutdown()
