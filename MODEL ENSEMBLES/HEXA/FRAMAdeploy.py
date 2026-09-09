import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import joblib
import MetaTrader5 as mt5

# ---------------------- Model Definition ----------------------

class HybridFRAMAModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, out_dim=64):
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
            nn.Linear(out_dim, 3),  # frama_trend
            nn.Linear(out_dim, 2),  # frama_strength
            nn.Linear(out_dim, 3),  # volatility_bucket
            nn.Linear(out_dim, 3),  # position_relative
            nn.Linear(out_dim, 3),  # future_sim_direction
        ])

    def forward(self, x):
        bilstm_out, _ = self.bilstm(x)
        lstm_out, _ = self.lstm(bilstm_out)
        weights = self.attention_pool(lstm_out)
        context = torch.sum(weights * lstm_out, dim=1)
        shared = self.shared(context)
        return [head(shared) for head in self.heads]

# ---------------------- Predictor ----------------------

class FRAMAPredictor:
    def __init__(self, model_path, scaler_path):
        self.sequence_length = 20
        self.features = ["frama", "slope", "deviation", "volatility"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = HybridFRAMAModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.seq_scaler = joblib.load(scaler_path)

        self.symbol = "XAUUSDm"
        if not mt5.initialize():
            raise RuntimeError("MT5 initialization failed")

    def get_mt5_data(self, timeframe):
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, 100)
        if rates is None or len(rates) < 30:
            raise ValueError("Insufficient MT5 data")
        df = pd.DataFrame(rates)
        df['mid'] = (df['high'] + df['low']) / 2
        df['frama'] = df['mid'].ewm(span=10).mean()
        df['slope'] = df['frama'].diff()
        df['deviation'] = df['mid'] - df['frama']
        df['volatility'] = df['deviation'].rolling(window=5).std()
        return df.dropna()

    def prepare_sequence(self, df):
        recent = df[self.features].tail(self.sequence_length)
        if len(recent) < self.sequence_length:
            raise ValueError("Not enough data for sequence")

        flat = recent.values.reshape(1, -1)
        scaled_flat = self.seq_scaler.transform(flat)
        scaled = scaled_flat.reshape(1, self.sequence_length, len(self.features))

        return torch.tensor(scaled, dtype=torch.float32).to(self.device)

    def predict(self, timeframe):
        try:
            df = self.get_mt5_data(timeframe)
            seq_tensor = self.prepare_sequence(df)
            with torch.no_grad():
                outputs = self.model(seq_tensor)
                predictions = [torch.argmax(out, dim=1).item() for out in outputs]
            return {
                "predictions": {
                    "frama_trend": predictions[0],
                    "frama_strength": predictions[1],
                    "volatility_bucket": predictions[2],
                    "position_relative": predictions[3],
                    "future_sim_direction": predictions[4]
                }
            }
        except Exception as e:
            return {"error": str(e)}

    def shutdown(self):
        mt5.shutdown()

# ---------------------- Usage ----------------------

if __name__ == "__main__":
    predictor = FRAMAPredictor("frama_multi_target_model.pth", "frama_seq_scaler.pkl")

    timeframes = {
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1
    }

    print("--- FRAMA Multi-Head Predictions ---")
    for name, tf_code in timeframes.items():
        result = predictor.predict(tf_code)
        if "error" not in result:
            print(f"{name}: {result['predictions']}")
        else:
            print(f"{name}: {result['error']}")

    predictor.shutdown()
