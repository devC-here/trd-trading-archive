import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import MetaTrader5 as mt5
import os
import re
import joblib

class TradingPredictor:
    def __init__(self):
        """Initialize MT5 connection, model, and scaler."""
        if not mt5.initialize():
            raise RuntimeError("MT5 initialization failed")

        self.model = self._load_model()
        self.scaler = self._load_scaler()

    def _load_model(self):
        """Loads the trained model from the latest available file."""
        model_files = [f for f in os.listdir() if re.match(r"xauusd_model\d*\.pth$", f)]
        model_path = model_files[-1] if model_files else "xauusd_model.pth"

        print(f"Loading model: {model_path}")
        model = NeuralNetwork()  # Uses the trained architecture
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model

    def _load_scaler(self):
        """Loads the scaler used during training."""
        scaler_path = "scaler.pkl"
        if os.path.exists(scaler_path):
            return joblib.load(scaler_path)
        raise FileNotFoundError("Scaler file 'scaler.pkl' not found. Ensure it exists.")

    def _fetch_mt5_data(self, symbol, timeframe, count=2):
        """Fetches market data from MT5."""
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None:
            return None
        df = pd.DataFrame(rates)
        df["volume"] = df["tick_volume"]  # Adjust volume field for consistency
        return df

    def _preprocess_data(self, df):
        """Prepares data for model input."""
        df["time"] = pd.to_datetime(df["time"], unit="s", errors='coerce')
        return df[["open", "high", "low", "close", "volume"]]

    def _predict_trade_decision(self, data):
        """Makes a trade prediction using the trained model."""
        data = self.scaler.transform(data)  # Apply the same scaling used during training
        data_tensor = torch.tensor(data, dtype=torch.float32)

        with torch.no_grad():
            output = self.model(data_tensor)
            prediction = torch.argmax(output, dim=1).numpy()
        return prediction[0]

    def get_predictions(self, timeframe, symbol="XAUUSD"):
        """Fetches market data and returns a trade prediction."""
        df = self._fetch_mt5_data(symbol, timeframe, count=3)
        if df is None or len(df) <= 2:
            return None  # Not enough data for prediction

        df = self._preprocess_data(df).iloc[-2].to_frame().T
        prediction = self._predict_trade_decision(df.values)
        return {"decision": ["Sell", "No Trade", "Buy"][prediction]}

# Define Neural Network Model (Same as the one used in training)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(5, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 3)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.2)  # Keep the dropout value consistent with training

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn3(self.fc3(x)))
        x = self.leaky_relu(self.fc4(x))
        x = self.fc5(x)
        return x
