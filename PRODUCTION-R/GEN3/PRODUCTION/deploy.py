import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import MetaTrader5 as mt5
import os
import re
import joblib
from datetime import datetime

# Define a safe division function
def safe_divide(a, b, eps=1e-6):
    return np.where(np.abs(b) > eps, a / b, 0)

# Our best model architecture: a fully-connected network
class ParameterizedComplexNeuralNetwork(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.2):
        super(ParameterizedComplexNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.fc6 = nn.Linear(64, 32)
        self.bn6 = nn.BatchNorm1d(32)
        self.fc7 = nn.Linear(32, 3)  # 3 output classes: Sell, No Trade, Buy
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn5(self.fc5(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn6(self.fc6(x)))
        x = self.dropout(x)
        x = self.fc7(x)
        return x

class TradingPredictor:
    def __init__(self):
        """Initialize MT5 connection, model, and scaler."""
        if not mt5.initialize():
            raise RuntimeError("MT5 initialization failed")
        self.model = self._load_model()
        self.scaler = self._load_scaler()

    def _load_model(self):
        """Loads our best trained model from file."""
        model_path = "xauusd_model_best.pth"
        print(f"Loading model: {model_path}")
        # Our model expects 11 features
        model = ParameterizedComplexNeuralNetwork(input_dim=11, dropout_rate=0.2)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model

    def _load_scaler(self):
        """Loads the scaler used during training."""
        scaler_path = "scaler.pkl"
        if os.path.exists(scaler_path):
            return joblib.load(scaler_path)
        return StandardScaler()

    def _fetch_mt5_data(self, symbol, timeframe, count=5):
        """Fetches market data from MT5."""
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None:
            return None
        df = pd.DataFrame(rates)
        df["volume"] = df["tick_volume"]
        return df

    def _compute_features(self, df):
        """
        Compute ratio and candlestick features from raw OHLCV data.
        Expected input columns: open, high, low, close, volume.
        Returns a DataFrame with 11 features.
        """
        df_feat = df[["open", "high", "low", "close", "volume"]].copy()
        # Ratio features
        df_feat["high_low_high_close_ratio"] = safe_divide(df_feat["high"] - df_feat["low"], df_feat["high"] - df_feat["close"])
        df_feat["high_close_low_close_ratio"] = safe_divide(df_feat["high"] - df_feat["close"], df_feat["low"] - df_feat["close"])
        df_feat["open_low_high_close_ratio"] = safe_divide(df_feat["open"] - df_feat["low"], df_feat["high"] - df_feat["close"])
        df_feat["high_low_low_close_ratio"] = safe_divide(df_feat["high"] - df_feat["low"], df_feat["low"] - df_feat["close"])
        df_feat["open_close_low_close_ratio"] = safe_divide(df_feat["open"] - df_feat["close"], df_feat["low"] - df_feat["close"])
        # Candlestick shape features
        df_feat["wick_to_length_ratio"] = safe_divide(df_feat["high"] - df_feat[["open", "close"]].max(axis=1), df_feat["high"] - df_feat["low"])
        df_feat["body_to_length_ratio"] = safe_divide(np.abs(df_feat["close"] - df_feat["open"]), df_feat["high"] - df_feat["low"])
        df_feat["tail_to_length_ratio"] = safe_divide(df_feat[["open", "close"]].min(axis=1) - df_feat["low"], df_feat["high"] - df_feat["low"])
        # Select features in the order expected by the model
        features = ['open', 'close', 
                    'high_low_high_close_ratio', 'high_close_low_close_ratio', 'open_low_high_close_ratio',
                    'high_low_low_close_ratio', 'open_close_low_close_ratio',
                    'wick_to_length_ratio', 'body_to_length_ratio', 'tail_to_length_ratio',
                    'volume']
        return df_feat[features]

    def _preprocess_data(self, df):
        """Prepares data for model input by computing features and scaling them."""
        df_features = self._compute_features(df)
        scaled_features = self.scaler.transform(df_features)
        # Our model is fully-connected, so the input shape is (n_samples, 11)
        return scaled_features

    def _predict_trade_decision(self, data):
        """Makes a trade prediction using the best model."""
        data_tensor = torch.tensor(data, dtype=torch.float32)
        with torch.no_grad():
            output = self.model(data_tensor)
            prediction = torch.argmax(output, dim=1).numpy()
        return prediction[0]

    def get_prediction(self, symbol, timeframe):
        """Fetch market data for the given symbol/timeframe, process it, and return a trade prediction."""
        df = self._fetch_mt5_data(symbol, timeframe, count=5)
        if df is None or len(df) < 2:
            return None
        # Use the last 2 candles for prediction
        df_recent = df.tail(2)
        data = self._preprocess_data(df_recent)
        prediction = self._predict_trade_decision(data)
        decision_map = {0: "Sell", 1: "No Trade", 2: "Buy"}
        return {"decision": decision_map[prediction]}

    def get_predictions_1H(self):
        """Fetches 1-hour timeframe data and returns the trade prediction."""
        return self.get_prediction("XAUUSD", mt5.TIMEFRAME_H1)

    def get_predictions_30M(self):
        """Fetches 30-minute timeframe data and returns the trade prediction."""
        return self.get_prediction("XAUUSD", mt5.TIMEFRAME_M30)

