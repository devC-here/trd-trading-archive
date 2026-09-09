import MetaTrader5 as mt5
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd

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
        except Exception as e:
            print(f"⚠ Anomaly Detected: Failed to load scaler → {e}")
            raise

        try:
            self.model = CandleSequenceModel().to(self.device)
            self.model.load_state_dict(torch.load("candle_model.pth", map_location=self.device))
            self.model.eval()
        except Exception as e:
            print(f"⚠ Anomaly Detected: Failed to load model → {e}")
            raise

        if not mt5.initialize():
            print("⚠ Anomaly Detected: MT5 Initialization Failed!")
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
                return {"symbol": symbol, "pattern": "N/A", "direction": "No Trade", "confidence": 0.0}

            candle_data = [[row["open"], row["high"], row["low"], row["close"]] for row in candles]
            features_df = self.extract_features(candle_data)

            # Flatten sequence to scale
            flattened = features_df.values.flatten().reshape(1, -1)

            # Use feature names to prevent warning
            input_df = pd.DataFrame(flattened, columns=self.scaler.feature_names_in_)
            scaled = self.scaler.transform(input_df)

            # Reshape to (1, 5, 3)
            sequence_input = torch.tensor(scaled.reshape(1, 5, 3), dtype=torch.float32).to(self.device)

            with torch.no_grad():
                pattern_logits, direction_logits = self.model(sequence_input)
                pattern_probs = torch.softmax(pattern_logits, dim=1).cpu().numpy()[0]
                direction_probs = torch.softmax(direction_logits, dim=1).cpu().numpy()[0]

                pattern_pred = int(np.argmax(pattern_probs))
                direction_pred = int(np.argmax(direction_probs))
                confidence = float(np.max(direction_probs))

            return {
                "symbol": symbol,
                "pattern": pattern_pred,
                "direction": ["Sell", "No Trade", "Buy"][direction_pred],
                "confidence": confidence
            }

        except Exception as e:
            print(f"⚠ Anomaly Detected: Prediction failure for {symbol} → {e}")
            return {"symbol": symbol, "pattern": "N/A", "direction": "No Trade", "confidence": 0.0}

    def get_most_confident_trade(self, timeframe):
        predictions = [self.get_prediction(symbol, timeframe) for symbol in self.symbols]
        filtered = [p for p in predictions if p["direction"] != "No Trade"]
        return max(
            filtered,
            key=lambda x: x["confidence"],
            default={"symbol": None, "pattern": "N/A", "direction": "No Trade", "confidence": 0.0}
        )

    def shutdown(self):
        mt5.shutdown()

# ---------------------- Run ----------------------
if __name__ == "__main__":
    predictor = CandleSequencePredictor(symbols=["XAUUSDm"])

    timeframes = {
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1
    }

    for label, tf in timeframes.items():
        result = predictor.get_most_confident_trade(tf)
        print(f"{label}: {result['direction']}")

    predictor.shutdown()
