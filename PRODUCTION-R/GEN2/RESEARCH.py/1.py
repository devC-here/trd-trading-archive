import MetaTrader5 as mt5
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# Connect to MT5
if not mt5.initialize():
    print("MT5 Initialization Failed")
    exit()

# Function to fetch last completed 1-hour candle's OHLCV data
def get_last_completed_candle(symbol, timeframe=mt5.TIMEFRAME_H1):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, 1)  # Fetch last completed candle
    if rates is None or len(rates) == 0:
        print("Failed to fetch candle data")
        return None
    candle = rates[0]
    return np.array([candle['open'], candle['high'], candle['low'], candle['close'], candle['tick_volume']], dtype=np.float32)

# Define the Neural Network Model (Updated to match saved model)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(5, 512)  # Input: 5 features (OHLCV)
        self.bn1 = nn.BatchNorm1d(512)  # Batch Normalization
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 3)   # Output: 3 classes (Sell, No Trade, Buy)

        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.fc1(x)))  # Added BatchNorm1d
        x = self.dropout(x)
        x = self.leaky_relu(self.bn2(self.fc2(x)))  # Added BatchNorm1d
        x = self.dropout(x)
        x = self.leaky_relu(self.bn3(self.fc3(x)))  # Added BatchNorm1d
        x = self.leaky_relu(self.fc4(x))
        x = self.fc5(x)  # No activation (raw logits)
        return x

# Load the trained model (Now matches saved structure)
model = NeuralNetwork()
model.load_state_dict(torch.load("xauusd_model.pth"))
model.eval()  # Set model to evaluation mode

# Fetch the last completed candle's data
candle_data = get_last_completed_candle("XAUUSD")

if candle_data is not None:
    # Print the OHLCV data before prediction
    print("\n📊 Last Completed Candle Data (OHLCV):")
    print(f"Open:  {candle_data[0]}")
    print(f"High:  {candle_data[1]}")
    print(f"Low:   {candle_data[2]}")
    print(f"Close: {candle_data[3]}")
    print(f"Volume: {candle_data[4]}\n")

    # Convert to tensor and reshape for model
    data_tensor = torch.tensor(candle_data, dtype=torch.float32).unsqueeze(0)

    # Get prediction
    with torch.no_grad():
        output = model(data_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    # Interpret prediction
    trade_decision = {0: "Sell", 1: "No Trade", 2: "Buy"}[predicted_class]
    
    print(f"🟢 Prediction for last completed candle: {trade_decision}")

# Shutdown MT5 connection
mt5.shutdown()
