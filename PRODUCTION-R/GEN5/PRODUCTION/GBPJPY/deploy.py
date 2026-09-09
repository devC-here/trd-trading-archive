import MetaTrader5 as mt5
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd

class TradingPredictor:
    def __init__(self, symbol="GBPJPY"):
        self.symbol = symbol

        # Load preprocessing objects
        self.scaler = joblib.load("scaler.pkl")
        self.label_encoder = joblib.load("label_encoder.pkl")

        # Load trained model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TradeDecisionNN(input_dim=3).to(self.device)
        self.model.load_state_dict(torch.load("gbpjpy_model_best.pth", map_location=self.device))
        self.model.eval()

        # Initialize MT5
        if not mt5.initialize():
            raise RuntimeError("MT5 Initialization Failed!")

    def calculate_ratios(self, open_price, high_price, low_price, close_price):
        """Calculate wick, body, and tail length ratios."""
        candle_length = max(high_price - low_price, 1e-6)  # Avoid division by zero
        wick_length = high_price - max(open_price, close_price)
        body_length = abs(open_price - close_price)
        tail_length = min(open_price, close_price) - low_price

        return (
            wick_length / candle_length,
            body_length / candle_length,
            tail_length / candle_length,
        )

    def get_prediction(self, symbol, timeframe):
        """Fetch last completed candle and return trade decision."""
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, 1)  # Fetch last completed candle
        if rates is None or len(rates) < 1:
            return {"decision": "No Trade"}

        rates_df = pd.DataFrame(rates)
        open_price, high_price, low_price, close_price = (
            rates_df.iloc[0]["open"],
            rates_df.iloc[0]["high"],
            rates_df.iloc[0]["low"],
            rates_df.iloc[0]["close"],
        )

        # Compute ratios
        ratios = self.calculate_ratios(open_price, high_price, low_price, close_price)
        features_df = pd.DataFrame([ratios], columns=["wick_to_length_ratio", "body_to_length_ratio", "tail_to_length_ratio"])
        features_scaled = self.scaler.transform(features_df)
        X_input = torch.tensor(features_scaled, dtype=torch.float32).to(self.device)

        # Make prediction
        with torch.no_grad():
            output = self.model(X_input)
            predicted_class = torch.argmax(output, dim=1).cpu().numpy()[0]

        decision = self.label_encoder.inverse_transform([predicted_class])[0]
        return {"decision": decision}

    def get_predictions_1H(self):
        return self.get_prediction(self.symbol, mt5.TIMEFRAME_H1)

    def get_predictions_30M(self):
        return self.get_prediction(self.symbol, mt5.TIMEFRAME_M30)

    def get_predictions_15M(self):
        return self.get_prediction(self.symbol, mt5.TIMEFRAME_M15)

# Define Neural Network
class TradeDecisionNN(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.3):
        super(TradeDecisionNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 3)  # 3 classes: Buy, Sell, No Trade
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x
