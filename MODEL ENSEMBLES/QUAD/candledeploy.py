import MetaTrader5 as mt5
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------- Model Definition ----------------------
class CandleSequenceModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, out_dim=64, pattern_classes=7, direction_classes=3):
        super().__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.lstm = nn.LSTM(2 * hidden_dim, hidden_dim, batch_first=True)
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.heads = nn.ModuleList([
            nn.Linear(out_dim, pattern_classes),
            nn.Linear(out_dim, direction_classes)
        ])

    def forward(self, x):
        bilstm_out, _ = self.bilstm(x)
        lstm_out, _ = self.lstm(bilstm_out)
        weights = self.attention_pool(lstm_out)
        context = torch.sum(weights * lstm_out, dim=1)
        shared = self.shared(context)
        return [head(shared) for head in self.heads]

# ---------------------- Predictor ----------------------
class CandleSequencePredictor:
    def __init__(self, symbols=None):
        self.symbols = symbols or ["XAUUSDm"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.scaler = joblib.load("candle_scaler.pkl")
            logging.info("Scaler loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load scaler: {e}")

        try:
            self.model = CandleSequenceModel().to(self.device)
            self.model.load_state_dict(torch.load("candle_model.pth", map_location=self.device))
            self.model.eval()
            logging.info("Model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

        if not mt5.initialize():
            raise RuntimeError("MT5 Initialization Failed!")

    def extract_features(self, candles):
        rows = []
        for i in range(len(candles)):
            o, h, l, c = candles[i]
            wick = h - max(o, c)
            body = abs(c - o)
            tail = min(o, c) - l
            rows.append([wick, body, tail])
        return pd.DataFrame(rows, columns=["wick", "body", "tail"])

    def get_prediction(self, symbol, timeframe):
        try:
            candles = mt5.copy_rates_from_pos(symbol, timeframe, 1, 5)
            if candles is None or len(candles) < 5:
                logging.warning(f"Not enough data for {symbol}")
                return {"symbol": symbol, "pattern": "N/A", "direction": "No Trade", "confidence": 0.0}

            candle_data = [[row["open"], row["high"], row["low"], row["close"]] for row in candles]
            features_df = self.extract_features(candle_data)

            # Flatten sequence to scale
            flattened = features_df.values.flatten().reshape(1, -1)
            scaled = self.scaler.transform(flattened)

            # Reshape back to sequence shape (1, 5, 3)
            sequence_input = torch.tensor(scaled.reshape(1, 5, 3), dtype=torch.float32).to(self.device)

            with torch.no_grad():
                pattern_logits, direction_logits = self.model(sequence_input)
                pattern_probs = torch.softmax(pattern_logits, dim=1).cpu().numpy()[0]
                direction_probs = torch.softmax(direction_logits, dim=1).cpu().numpy()[0]

                pattern_pred = int(np.argmax(pattern_probs))
                direction_pred = int(np.argmax(direction_probs))
                confidence = float(np.max(direction_probs))

            logging.info(f"{symbol} | Pattern: {pattern_pred} | Direction: {direction_pred} | Confidence: {confidence:.2%}")
            return {
                "symbol": symbol,
                "pattern": pattern_pred,
                "direction": ["Sell", "No Trade", "Buy"][direction_pred],
                "confidence": confidence
            }

        except Exception as e:
            logging.error(f"Prediction error for {symbol}: {e}")
            return {"symbol": symbol, "pattern": "N/A", "direction": "No Trade", "confidence": 0.0}

    def get_most_confident_trade(self, timeframe):
        predictions = [self.get_prediction(symbol, timeframe) for symbol in self.symbols]
        filtered = [p for p in predictions if p["direction"] != "No Trade"]
        return max(filtered, key=lambda x: x["confidence"], default={"symbol": None, "direction": "No Trade", "confidence": 0.0})

    def shutdown(self):
        mt5.shutdown()
        logging.info("MT5 connection closed.")

# ---------------------- Run ----------------------
if __name__ == "__main__":
    predictor = CandleSequencePredictor(symbols=["XAUUSDm"])

    timeframes = {
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1
    }

    for label, tf in timeframes.items():
        print(f"\n📈 {label} timeframe prediction:")
        result = predictor.get_most_confident_trade(tf)
        print(f"🔹 Symbol: {result['symbol']} | Pattern: {result['pattern']} | Direction: {result['direction']} | Confidence: {result['confidence']:.2%}")

    predictor.shutdown()
