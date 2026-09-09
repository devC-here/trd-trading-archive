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

class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 3)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)  # Reshape for LSTM
        x, _ = self.lstm(x)
        x = self.leaky_relu(self.fc1(x[:, -1, :]))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TradingPredictor:
    def __init__(self):
        """Initialize MT5 connection, model, and scaler."""
        if not mt5.initialize():
            raise RuntimeError("MT5 initialization failed")

        self.model = self._load_model()
        self.scaler = self._load_scaler()

    def _load_model(self):
        """Loads the latest trained model from the directory."""
        model_files = [f for f in os.listdir() if re.match(r"cnn_lstm_xauusd_model\d*\.pth$", f)]
        model_path = model_files[-1] if model_files else "cnn_lstm_xauusd_model.pth"

        print(f"Loading model: {model_path}")
        model = CNN_LSTM()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model

    def _load_scaler(self):
        """Loads the scaler used during training."""
        scaler_path = "scaler.pkl"
        if os.path.exists(scaler_path):
            return joblib.load(scaler_path)
        return StandardScaler()

    def _fetch_mt5_data(self, symbol, timeframe, count=2):
        """Fetches market data from MT5."""
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None:
            return None
        df = pd.DataFrame(rates)
        df["volume"] = df["tick_volume"]
        return df

    def _preprocess_data(self, df):
        """Prepares data for model input."""
        df["time"] = pd.to_datetime(df["time"], unit="s", errors='coerce')
        df = df[["open", "high", "low", "close", "volume"]]
        df = self.scaler.transform(df)
        df = df.reshape((df.shape[0], 1, df.shape[1]))  # Reshape for CNN-LSTM
        return df

    def _predict_trade_decision(self, data):
        """Makes a trade prediction using the model."""
        data_tensor = torch.tensor(data, dtype=torch.float32)
        with torch.no_grad():
            output = self.model(data_tensor)
            prediction = torch.argmax(output, dim=1).numpy()
        return prediction[0]

    def get_predictions_1H(self):
        """Fetches 1-hour timeframe data and returns trade prediction."""
        df = self._fetch_mt5_data("XAUUSD", mt5.TIMEFRAME_H1, count=3)
        if df is None or len(df) <= 2:
            return None  

        df = self._preprocess_data(df[-2:])
        prediction = self._predict_trade_decision(df)
        return {"decision": ["Sell", "No Trade", "Buy"][prediction]}

    def get_predictions_30M(self):
        """Fetches 30-minute timeframe data and returns trade prediction."""
        df = self._fetch_mt5_data("XAUUSD", mt5.TIMEFRAME_M30, count=3)
        if df is None or len(df) <= 2:
            return None  

        df = self._preprocess_data(df[-2:])
        prediction = self._predict_trade_decision(df)
        return {"decision": ["Sell", "No Trade", "Buy"][prediction]}
