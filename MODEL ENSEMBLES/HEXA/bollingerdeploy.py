import MetaTrader5 as mt5
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
from ta.trend import ADXIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

# ---------------------- Model ----------------------

class MultiOutputBollingerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, 5),
            nn.Linear(hidden_dim, 3),
            nn.Linear(hidden_dim, 3),
            nn.Linear(hidden_dim, 3),
            nn.Linear(hidden_dim, 2),
            nn.Linear(hidden_dim, 2),
            nn.Linear(hidden_dim, 2)
        ])

    def forward(self, x):
        x = self.shared(x)
        return [head(x) for head in self.heads]

# ---------------------- Predictor ----------------------

class TradingPredictor:
    def __init__(self):
        self.symbol = "XAUUSDm"
        self.timeframes = {
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1
        }

        self.scaler = joblib.load("bollinger_asset_scaler.pkl")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_columns = [
            "lb", "ub", "pi", "bandwidth", "band_squeeze",
            "adx", "adx_slope_z",
            "macd_hist", "macd_hist_slope", "macd_cross",
            "rsi", "rsi_z",
            "atr_norm", "volatility_z",
            "body_ratio", "wick_ratio", "tail_ratio",
            "range_compression",
            "adx_x_macd_hist", "rsi_x_macd", "rsi_x_pi"
        ]

        self.model = MultiOutputBollingerModel(input_dim=len(self.feature_columns)).to(self.device)
        self.model.load_state_dict(torch.load("bollinger_asset_model.pth", map_location=self.device))
        self.model.eval()

        if not mt5.initialize():
            raise RuntimeError("MT5 Initialization Failed!")

    def extract_features(self, df):
        bb = BollingerBands(close=df["close"], window=20, window_dev=2)
        adx = ADXIndicator(df["high"], df["low"], df["close"])
        macd = MACD(df["close"])
        rsi = RSIIndicator(df["close"])
        atr = AverageTrueRange(df["high"], df["low"], df["close"])

        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_middle"] = bb.bollinger_mavg()

        df["bandwidth"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_middle"] + 1e-6)
        df["band_squeeze"] = (df["bandwidth"] - df["bandwidth"].rolling(20).mean()) / (df["bandwidth"].rolling(20).std() + 1e-6)
        df["lb"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-6)
        df["ub"] = 1 - df["lb"]

        df["pi"] = 0.5
        recent = df.iloc[-20:]
        for i in range(len(recent) - 1, -1, -1):
            row = recent.iloc[i]
            if row["close"] >= row["bb_upper"]:
                df.at[recent.index[-1], "pi"] = -1.0
                break
            elif row["close"] <= row["bb_lower"]:
                df.at[recent.index[-1], "pi"] = 1.0
                break

        df["adx"] = adx.adx()
        df["adx_slope_z"] = (df["adx"].diff() - df["adx"].diff().rolling(20).mean()) / (df["adx"].diff().rolling(20).std() + 1e-6)
        df["macd_hist"] = macd.macd_diff()
        df["macd_hist_slope"] = df["macd_hist"].diff()
        df["macd_cross"] = ((macd.macd() > macd.macd_signal()) & (macd.macd().shift(1) <= macd.macd_signal().shift(1))).astype(int) - \
                           ((macd.macd() < macd.macd_signal()) & (macd.macd().shift(1) >= macd.macd_signal().shift(1))).astype(int)

        df["rsi"] = rsi.rsi()
        df["rsi_z"] = (df["rsi"] - df["rsi"].rolling(20).mean()) / (df["rsi"].rolling(20).std() + 1e-6)
        df["atr_norm"] = atr.average_true_range() / df["close"]
        df["volatility_z"] = df["close"].rolling(20).std() / df["close"]

        df["body"] = (df["close"] - df["open"]).abs()
        df["wick"] = df["high"] - df[["close", "open"]].max(axis=1)
        df["tail"] = df[["close", "open"]].min(axis=1) - df["low"]
        total = df["wick"] + df["body"] + df["tail"] + 1e-6
        df["body_ratio"] = df["body"] / total
        df["wick_ratio"] = df["wick"] / total
        df["tail_ratio"] = df["tail"] / total

        df["range"] = df["high"] - df["low"]
        df["range_compression"] = df["range"].rolling(10).std() / (df["range"].rolling(50).std() + 1e-6)

        df["adx_x_macd_hist"] = df["adx"] * df["macd_hist"]
        df["rsi_x_macd"] = df["rsi"] * df["macd_hist"]
        df["rsi_x_pi"] = df["rsi"] * df["pi"]

        df.dropna(inplace=True)
        return df

    def predict_timeframe(self, timeframe_name, timeframe_code):
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe_code, 0, 100)
        if rates is None or len(rates) < 60:
            return {"timeframe": timeframe_name, "error": "Not enough data."}

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = self.extract_features(df)

        if df.empty or len(df) < 1:
            return {"timeframe": timeframe_name, "error": "Insufficient features."}

        latest = df.iloc[-1:][self.feature_columns]
        scaled = self.scaler.transform(latest)
        X = torch.tensor(scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(X)

        prediction = {}
        head_names = [
            "boll_position_class", "squeeze_state", "trend_regime",
            "momentum_zone", "reversal_signal", "rejection_flag", "compression_event"
        ]
        for name, out in zip(head_names, outputs):
            prob = torch.softmax(out, dim=1).cpu().numpy()[0]
            prediction[name] = {
                "class": int(np.argmax(prob)),
                "confidence": float(np.max(prob))
            }

        return {"timeframe": timeframe_name, "predictions": prediction}

    def shutdown(self):
        try:
            mt5.shutdown()
        except Exception as e:
            print(f"⚠ Anomaly Detected during shutdown: {e}")

# ---------------------- Run ----------------------

if __name__ == "__main__":
    predictor = TradingPredictor()
    results = {}

    for tf_name, tf_code in predictor.timeframes.items():
        result = predictor.predict_timeframe(tf_name, tf_code)
        results[tf_name] = result

    for tf, out in results.items():
        if "error" in out:
            print(f"⚠ Anomaly [{tf}]: {out['error']}")
        else:
            print(f"{tf}: {out['predictions']}")

    predictor.shutdown()
