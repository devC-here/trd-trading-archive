import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import joblib
import MetaTrader5 as mt5

# ---------------------- Model Definition ----------------------

class HybridVWRSIModel(nn.Module):
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
            nn.Linear(out_dim, 3),  # vwrsi_trend
            nn.Linear(out_dim, 2),  # vwrsi_strength
            nn.Linear(out_dim, 4),  # volatility_bucket
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

# ---------------------- Predictor Class ----------------------

class VWRSIPredictor:
    # Manual mapping for timeframe constants to string names
    TIMEFRAME_NAMES = {
        mt5.TIMEFRAME_M1: "M1",
        mt5.TIMEFRAME_M5: "M5",
        mt5.TIMEFRAME_M15: "M15",
        mt5.TIMEFRAME_M30: "M30",
        mt5.TIMEFRAME_H1: "H1",
        mt5.TIMEFRAME_H4: "H4",
        mt5.TIMEFRAME_D1: "D1",
        mt5.TIMEFRAME_W1: "W1",
        mt5.TIMEFRAME_MN1: "MN1",
    }

    def __init__(self, model_path, scaler_path):
        self.sequence_length = 20
        self.features = ["vw_rsi", "slope", "deviation", "volatility"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = HybridVWRSIModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.seq_scaler = joblib.load(scaler_path)

        self.symbol = "XAUUSDm"
        if not mt5.initialize():
            raise RuntimeError("MT5 initialization failed")

    def get_mt5_data(self, timeframe):
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, 100)
        if rates is None or len(rates) < 30:
            raise ValueError(f"Insufficient MT5 data for timeframe {self.TIMEFRAME_NAMES.get(timeframe, str(timeframe))}")
        df = pd.DataFrame(rates)

        df['typical'] = (df['high'] + df['low'] + df['close']) / 3
        df['volume_weight'] = df['tick_volume'] * df['typical']
        df['vw_price'] = df['volume_weight'].rolling(window=14).sum() / df['tick_volume'].rolling(window=14).sum()
        
        change_pos = df['close'].pct_change().apply(lambda x: max(x, 0))
        change_neg = df['close'].pct_change().apply(lambda x: abs(min(x, 0)))
        
        avg_gain = change_pos.rolling(14).mean()
        avg_loss = change_neg.rolling(14).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['vw_rsi'] = df['rsi'] * (df['close'] / df['vw_price'])

        df['slope'] = df['vw_rsi'].diff()
        df['deviation'] = df['vw_rsi'] - df['vw_rsi'].rolling(window=5).mean()
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
        df = self.get_mt5_data(timeframe)
        seq_tensor = self.prepare_sequence(df)
        with torch.no_grad():
            outputs = self.model(seq_tensor)
            predictions = [torch.argmax(out, dim=1).item() for out in outputs]
        return {
            "timeframe": self.TIMEFRAME_NAMES.get(timeframe, str(timeframe)), # Corrected line
            "vwrsi_trend": predictions[0],
            "vwrsi_strength": predictions[1],
            "volatility_bucket": predictions[2],
            "position_relative": predictions[3],
            "future_sim_direction": predictions[4]
        }

    def shutdown(self):
        mt5.shutdown()

# ---------------------- Minimal Usage ----------------------

if __name__ == "__main__":
    try:
        predictor = VWRSIPredictor("vwrsi_multi_target_model.pth", "vwrsi_seq_scaler.pkl")
        timeframes = {
           "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1
        }
        print("M30 Prediction:")
        result_m30 = predictor.predict(timeframes["M30"])
        print(result_m30)

        print("\nH1 Prediction:")
        result_h1 = predictor.predict(timeframes["H1"])
        print(result_h1)

    except Exception as e:
        print(f"⚠ Anomaly Detected: {e}")
    finally:
        predictor.shutdown()