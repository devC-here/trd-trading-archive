import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import joblib
from ta.trend import ADXIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange


def kama(series, window=10, fast=2, slow=30):
    n = len(series)
    kama_values = np.zeros(n)
    sc_fast = 2 / (fast + 1)
    sc_slow = 2 / (slow + 1)
    kama_values[:window] = series.iloc[:window]
    for i in range(window, n):
        change = abs(series.iloc[i] - series.iloc[i - window])
        volatility = series.iloc[i - window + 1:i + 1].diff().abs().sum()
        er = change / volatility if volatility != 0 else 0
        sc = (er * (sc_fast - sc_slow) + sc_slow) ** 2
        kama_values[i] = kama_values[i - 1] + sc * (series.iloc[i] - kama_values[i - 1])
    return pd.Series(kama_values, index=series.index)


class BollingerPredictor:
    # Manual mapping for timeframe constants to string names
    TIMEFRAME_NAMES = {
        mt5.TIMEFRAME_M1: "M1",
        mt5.TIMEFRAME_M5: "M5",
        mt5.TIMEFRAME_M15: "M15",
        mt5.TIMEFRAME_M30: "M30",
        mt5.TIMEFRAME_H1: "H1",
        mt5.TIMEFRAME_H4: "H4",
        mt5.TIMEFRAME_D1: "D1",
        mt5.TIMEFRAME_W1: "W1",
        mt5.TIMEFRAME_MN1: "MN1",
    }

    def __init__(self):
        self.symbol = "XAUUSDm"
        self.scaler = joblib.load("bollinger_advanced_scaler.pkl")
        if not mt5.initialize():
            raise RuntimeError("MT5 initialization failed!")

    def fetch_data(self, timeframe_code, bars=500):
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe_code, 0, bars)
        if rates is None or len(rates) < 100:
            return None 
        return pd.DataFrame(rates)

    def engineer_features(self, df):
        df["bb_upper"] = BollingerBands(close=df["close"], window=20).bollinger_hband()
        df["bb_lower"] = BollingerBands(close=df["close"], window=20).bollinger_lband()
        df["bb_middle"] = BollingerBands(close=df["close"], window=20).bollinger_mavg()
        df["bandwidth"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_middle"] + 1e-6)
        df["band_squeeze"] = (df["bandwidth"] - df["bandwidth"].rolling(20).mean()) / (df["bandwidth"].rolling(20).std() + 1e-6)
        df["lb"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-6)
        df["ub"] = 1 - df["lb"]
        df["pi"] = 0.5
        for i in range(20, len(df)):
            window = df.iloc[i-20:i]
            if (window["close"] >= window["bb_upper"]).any():
                df.at[i, "pi"] = -1.0
            elif (window["close"] <= window["bb_lower"]).any():
                df.at[i, "pi"] = 1.0
        df["adx"] = ADXIndicator(df["high"], df["low"], df["close"]).adx()
        df["adx_slope_z"] = (df["adx"].diff() - df["adx"].diff().rolling(20).mean()) / (df["adx"].diff().rolling(20).std() + 1e-6)
        macd = MACD(df["close"])
        df["macd_hist"] = macd.macd_diff()
        df["macd_hist_slope"] = df["macd_hist"].diff()
        df["macd_line"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_cross"] = 0
        df.loc[(df["macd_line"] > df["macd_signal"]) & (df["macd_line"].shift(1) <= df["macd_signal"].shift(1)), "macd_cross"] = 1
        df.loc[(df["macd_line"] < df["macd_signal"]) & (df["macd_line"].shift(1) >= df["macd_signal"].shift(1)), "macd_cross"] = -1
        rsi = RSIIndicator(df["close"])
        df["rsi"] = rsi.rsi()
        df["rsi_z"] = (df["rsi"] - df["rsi"].rolling(20).mean()) / (df["rsi"].rolling(20).std() + 1e-6)
        atr = AverageTrueRange(df["high"], df["low"], df["close"])
        df["atr_norm"] = atr.average_true_range() / df["close"]
        df["volatility_z"] = df["close"].rolling(20).std() / df["close"]
        df["range"] = df["high"] - df["low"]
        df["range_compression"] = df["range"].rolling(10).std() / (df["range"].rolling(50).std() + 1e-6)
        median_price = (df["high"] + df["low"]) / 2
        df["ao"] = median_price.rolling(5).mean() - median_price.rolling(34).mean()
        df["ao_slope"] = df["ao"].diff()
        df["kama"] = kama(df["close"])
        df["kama_slope"] = df["kama"].diff()
        df["kama_slope_z"] = (df["kama_slope"] - df["kama_slope"].rolling(20).mean()) / (df["kama_slope"].rolling(20).std() + 1e-6)
        df["adx_x_macd_hist"] = df["adx"] * df["macd_hist"]
        df["rsi_x_macd"] = df["rsi"] * df["macd_hist"]
        df["rsi_x_pi"] = df["rsi"] * df["pi"]
        df.dropna(inplace=True)
        return df

    def classify(self, df):
        features = [
            "lb", "ub", "pi", "bandwidth", "band_squeeze", "adx", "adx_slope_z",
            "macd_hist", "macd_hist_slope", "macd_cross", "rsi", "rsi_z", "atr_norm", "volatility_z",
            "range_compression", "ao", "ao_slope", "kama", "kama_slope", "kama_slope_z",
            "adx_x_macd_hist", "rsi_x_macd", "rsi_x_pi"
        ]
        latest = df.iloc[-1:][features]
        scaled = self.scaler.transform(latest)
        pred = self.interpret_outputs(scaled)
        return pred

    def interpret_outputs(self, scaled_features):
        data = scaled_features.flatten()
        pred = {
            "boll_position_class": self.decode_bin(data[0], [-0.15, 0.3, 0.7, 1.15], ["lower-out", "lower-in", "mean", "upper-in", "upper-out"]),
            "squeeze_state": self.decode_bin(data[4], [-1.0, 1.0], ["squeeze", "stable", "expanding"]),
            "trend_regime": self.decode_bin(data[5], [15, 25], ["no_trend", "weak_trend", "strong_trend"]),
            "momentum_zone": self.decode_bin(data[10], [40, 60], ["bearish", "neutral", "bullish"]),
            # Reverted to original string labels based on user feedback
            "reversal_signal": "reversal" if data[9] != 0 and data[2] != 0.5 else "none",
            "compression_event": "compression" if data[14] < -0.2 else "no_compression"
        }
        return pred

    def decode_bin(self, value, bins, labels):
        return pd.cut([value], bins=[-np.inf] + bins + [np.inf], labels=labels)[0]

    def predict(self, timeframe_code):
        timeframe_name = self.TIMEFRAME_NAMES.get(timeframe_code, str(timeframe_code))
        df = self.fetch_data(timeframe_code)
        
        if df is None:
            return {"timeframe": timeframe_name, "error": f"Insufficient data for {timeframe_name}"}
        
        try:
            df = self.engineer_features(df)
            prediction = self.classify(df)
            return {"timeframe": timeframe_name, "predictions": prediction}
        except Exception as e:
            return {"timeframe": timeframe_name, "error": f"Prediction failure for {timeframe_name}: {e}"}

    def shutdown(self):
        mt5.shutdown()


if __name__ == "__main__":
    predictor = BollingerPredictor()
    timeframes = {
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1
    }

    print("--- Position Bollinger Predictions ---")
    for name, tf_code in timeframes.items():
        result = predictor.predict(tf_code)
        if "error" not in result:
            print(f"{name}: {result['predictions']}")
        else:
            print(f"{name}: {result['error']}")

    predictor.shutdown()