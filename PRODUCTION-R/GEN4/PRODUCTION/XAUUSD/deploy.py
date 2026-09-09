import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
import MetaTrader5 as mt5
import os
import joblib

# Safe division function
def safe_divide(a, b, eps=1e-6):
    return np.where(np.abs(b) > eps, a / b, 0)

# Define the Neural Network Model
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim=9):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 3)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn3(self.fc3(x)))
        x = self.leaky_relu(self.fc4(x))
        x = self.fc5(x)
        return x

class TradingPredictor:
    def __init__(self, model_path="xauusd_model_best.pth"):
        if not mt5.initialize():
            raise RuntimeError("MT5 initialization failed")
        self.model = self._load_model(model_path)
        self.scaler = self._load_scaler()
        self.model.eval()

    def _load_model(self, model_path):
        input_dim = 9  # Match trained model
        model = NeuralNetwork(input_dim)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        return model

    def _load_scaler(self):
        scaler_path = "scaler.pkl"
        if os.path.exists(scaler_path):
            return joblib.load(scaler_path)
        return RobustScaler()

    def _fetch_mt5_data(self, symbol, timeframe, count=5):
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None:
            return None
        return pd.DataFrame(rates)

    def _compute_features(self, df):
        df_feat = df[["high", "low", "tick_volume"]].copy()
        df_feat["high_low_high_close_ratio"] = safe_divide(df["high"] - df["low"], df["high"] - df["close"])
        df_feat["high_close_low_close_ratio"] = safe_divide(df["high"] - df["close"], df["low"] - df["close"])
        df_feat["open_low_high_close_ratio"] = safe_divide(df["open"] - df["low"], df["high"] - df["close"])
        df_feat["high_low_low_close_ratio"] = safe_divide(df["high"] - df["low"], df["low"] - df["close"])
        df_feat["open_close_low_close_ratio"] = safe_divide(df["open"] - df["close"], df["low"] - df["close"])
        df_feat["wick_to_length_ratio"] = safe_divide(df["high"] - df[["open", "close"]].max(axis=1), df["high"] - df["low"])
        df_feat["body_to_length_ratio"] = safe_divide(np.abs(df["close"] - df["open"]), df["high"] - df["low"])
        df_feat["tail_to_length_ratio"] = safe_divide(df[["open", "close"]].min(axis=1) - df["low"], df["high"] - df["low"])
        
        features = ["high_low_high_close_ratio", "high_close_low_close_ratio", "open_low_high_close_ratio",
                    "high_low_low_close_ratio", "open_close_low_close_ratio", "wick_to_length_ratio",
                    "body_to_length_ratio", "tail_to_length_ratio", "tick_volume"]
        
        return df_feat[features]

    def _preprocess_data(self, df):
        df_features = self._compute_features(df)
        scaled_features = self.scaler.transform(df_features.values)  # Convert to NumPy array
        return scaled_features

    def _predict_trade_decision(self, data):
        data_tensor = torch.tensor(data, dtype=torch.float32)
        with torch.no_grad():
            output = self.model(data_tensor)
            prediction = torch.argmax(output, dim=1).numpy()
        return prediction[-1]

    def get_prediction(self, symbol, timeframe):
        df = self._fetch_mt5_data(symbol, timeframe, count=5)
        if df is None or len(df) < 2:
            return None
        df_recent = df.tail(2)
        data = self._preprocess_data(df_recent)
        prediction = self._predict_trade_decision(data)
        decision_map = {0: "Sell", 1: "No Trade", 2: "Buy"}
        return {"decision": decision_map[prediction]}

    def get_predictions_1H(self):
        return self.get_prediction("XAUUSD", mt5.TIMEFRAME_H1)

    def get_predictions_30M(self):
        return self.get_prediction("XAUUSD", mt5.TIMEFRAME_M30)
    
    def get_predictions_15M(self):
        return self.get_prediction("XAUUSD", mt5.TIMEFRAME_M15)
