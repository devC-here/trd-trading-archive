# breakout_deploy.py
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

# -----------------------------
# Model Definition (must match training)
# -----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.block(x) + x)

class BreakoutNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(512, dropout=0.3),
            ResidualBlock(512, dropout=0.3),
            ResidualBlock(512, dropout=0.3)
        )
        self.mid_layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.25)
        )
        self.heads = nn.ModuleList([
            nn.Linear(256, 3),  # squeeze_state
            nn.Linear(256, 3),  # breakout_bias
            nn.Linear(256, 2),  # entry_viability
            nn.Linear(256, 3),  # breakout_strength
            nn.Linear(256, 3),  # false_breakout_risk
            nn.Linear(256, 3),  # post_vol_regime
        ])

    def forward(self, x):
        h = self.input_layer(x)
        h = self.res_blocks(h)
        h = self.mid_layer(h)
        return [head(h) for head in self.heads]

# -----------------------------
# Predictor
# -----------------------------
class BreakoutPredictor:
    def __init__(self,
                 model_path="breakout_model_best.pth",
                 scaler_path="breakout_scaler.pkl"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        bundle = joblib.load(scaler_path)
        self.scaler = bundle["scaler"]
        self.feature_cols = bundle["feature_names"]

        self.model = BreakoutNet(input_dim=len(self.feature_cols)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        if not mt5.initialize():
            raise RuntimeError("MT5 initialization failed")

    # --- Feature Engineering (standalone) ---
    def _compute_features(self, df, seq_len=20):
        df = df.copy()

        # Candle basics
        df["range"] = df["high"] - df["low"]
        df["body"] = (df["close"] - df["open"]).abs()
        df["dir"] = np.sign(df["close"] - df["open"])
        df["wick"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["tail"] = df[["open", "close"]].min(axis=1) - df["low"]

        df["body_ratio"] = df["body"] / (df["range"] + 1e-6)
        df["tail_ratio"] = df["tail"] / (df["range"] + 1e-6)
        df["wick_ratio"] = df["wick"] / (df["range"] + 1e-6)

        # ATR
        df["atr5"] = df["range"].rolling(5).mean()
        df["atr20"] = df["range"].rolling(20).mean()
        df["atr_ratio"] = df["atr5"] / (df["atr20"] + 1e-6)

        # Bollinger
        ma20 = df["close"].rolling(20).mean()
        std20 = df["close"].rolling(20).std()
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20
        df["boll_bw"] = (upper - lower) / (ma20 + 1e-6)
        df["boll_pos"] = (df["close"] - lower) / ((upper - lower) + 1e-6)

        # RSI
        delta = df["close"].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        rs = up.rolling(14).mean() / (down.rolling(14).mean() + 1e-6)
        df["rsi"] = 100 - (100 / (1 + rs))
        df["rsi_z"] = (df["rsi"] - df["rsi"].rolling(20).mean()) / (df["rsi"].rolling(20).std() + 1e-6)

        # MACD
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        df["macd"] = macd
        df["macd_hist"] = macd - macd_signal

        # Interactions
        df["boll_rsi"] = df["boll_pos"] * df["rsi_z"]
        df["boll_macd"] = df["boll_pos"] * df["macd_hist"]
        df["rsi_macd"] = df["rsi_z"] * df["macd_hist"]

        # Compression
        df["range_std5"] = df["range"].rolling(5).std()
        df["range_mean5"] = df["range"].rolling(5).mean()
        df["compression_ratio"] = df["range_mean5"] / (df["atr20"] + 1e-6)

        # Select last seq_len bars
        df = df.iloc[-seq_len:].fillna(0)

        # Flatten sequence features
        flat = {}
        for i in range(seq_len):
            for col in ["body", "wick", "tail", "body_ratio",
                        "tail_ratio", "wick_ratio", "dir"]:
                flat[f"{col}_t-{i}"] = df.iloc[i][col]

        # Aggregate features
        for col in ["atr5","atr20","atr_ratio","boll_bw","boll_pos",
                    "rsi","rsi_z","macd","macd_hist",
                    "boll_rsi","boll_macd","rsi_macd","compression_ratio"]:
            flat[col] = df.iloc[-1][col]

        return pd.DataFrame([flat])

    def predict(self, symbol="EURUSDm", timeframe=mt5.TIMEFRAME_M15, n_bars=200):
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
        if rates is None or len(rates) < 30:
            return {"error": "Insufficient data"}

        df = pd.DataFrame(rates)
        feat_df = self._compute_features(df)

        # Ensure column order
        feat_df = feat_df.reindex(columns=self.feature_cols, fill_value=0)

        scaled = self.scaler.transform(feat_df)
        x = torch.tensor(scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(x)

        preds = [int(torch.argmax(out, dim=1).cpu()) for out in outputs]
        probs = [torch.softmax(out, dim=1).cpu().numpy()[0] for out in outputs]
        confidences = [float(np.max(p)) for p in probs]

        return {
            "squeeze_state": preds[0],
            "breakout_bias": preds[1],
            "entry_viability": preds[2],
            "breakout_strength": preds[3],
            "false_breakout_risk": preds[4],
            "post_vol_regime": preds[5],
            "confidences": {
                "squeeze_state": confidences[0],
                "breakout_bias": confidences[1],
                "entry_viability": confidences[2],
                "breakout_strength": confidences[3],
                "false_breakout_risk": confidences[4],
                "post_vol_regime": confidences[5],
            }
        }

    def shutdown(self):
        mt5.shutdown()

# --- Example ---
if __name__ == "__main__":
    predictor = BreakoutPredictor()
    result = predictor.predict("EURUSDm", mt5.TIMEFRAME_M15)
    print("📊 Breakout–Compression Prediction:", result)
    predictor.shutdown()
