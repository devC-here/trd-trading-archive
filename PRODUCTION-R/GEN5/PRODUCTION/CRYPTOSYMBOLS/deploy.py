import MetaTrader5 as mt5
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class TradingPredictor:
    def __init__(self, symbols=None):
        self.symbols = symbols or ["BTCUSDm", "ETHUSDm", "BTCXAUm", "BTCXAGm"]

        # Load preprocessing objects
        try:
            self.scaler = joblib.load("scaler.pkl")
            self.label_encoder = joblib.load("label_encoder.pkl")
            logging.info("Scaler and Label Encoder loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load scaler or label encoder: {e}")

        # Load trained model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model = TradeDecisionNN(input_dim=3).to(self.device)
            self.model.load_state_dict(torch.load("xauusd_model_best.pth", map_location=self.device))
            self.model.eval()
            logging.info("Model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

        # Initialize MT5
        if not mt5.initialize():
            raise RuntimeError("MT5 Initialization Failed!")

    def get_prediction(self, symbol, timeframe):
        """Fetch last completed candle and return trade decision with confidence."""
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, 1)
            if rates is None or len(rates) == 0:
                logging.warning(f"No data retrieved for {symbol} on timeframe {timeframe}")
                return {"symbol": symbol, "decision": "No Trade", "confidence": 0.0}

            rates_df = pd.DataFrame(rates)
            open_price, high_price, low_price, close_price = (
                rates_df.iloc[0]["open"],
                rates_df.iloc[0]["high"],
                rates_df.iloc[0]["low"],
                rates_df.iloc[0]["close"],
            )

            # Compute ratios
            wick_length = high_price - max(open_price, close_price)
            body_length = abs(open_price - close_price)
            tail_length = min(open_price, close_price) - low_price
            candle_length = max(high_price - low_price, 1e-6)  # Avoid division by zero

            features_df = pd.DataFrame(
                [[wick_length / candle_length, body_length / candle_length, tail_length / candle_length]],
                columns=["wick_to_length_ratio", "body_to_length_ratio", "tail_to_length_ratio"],
            )

            features_scaled = self.scaler.transform(features_df)
            X_input = torch.tensor(features_scaled, dtype=torch.float32).to(self.device)

            # Make prediction
            with torch.no_grad():
                output = self.model(X_input)
                probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]

                predicted_class = np.argmax(probabilities)
                confidence = probabilities[predicted_class] if probabilities.size > 0 else 0.0

            decision = self.label_encoder.inverse_transform([predicted_class])[0]

            logging.info(f"Prediction for {symbol}: {decision} with {confidence:.2%} confidence")

            return {"symbol": symbol, "decision": decision, "confidence": confidence}

        except Exception as e:
            logging.error(f"Error in get_prediction for {symbol}: {e}")
            return {"symbol": symbol, "decision": "No Trade", "confidence": 0.0}

    def get_most_confident_predictions(self, timeframe, top_n=4):
        """Get predictions for all symbols, filter out 'No Trade' signals,
        and return the top N most confident trades among the remaining ones."""

        predictions = [self.get_prediction(symbol, timeframe) for symbol in self.symbols]

        # Filter out "No Trade" signals
        valid_predictions = [p for p in predictions if p["decision"] != "No Trade"]

        # Sort predictions by confidence (descending)
        valid_predictions.sort(key=lambda x: x["confidence"], reverse=True)

        return valid_predictions[:top_n]  # Return top 4 (or fewer if not enough signals)

    def shutdown(self):
        """Shutdown MT5 connection."""
        mt5.shutdown()
        logging.info("MT5 connection closed.")

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

