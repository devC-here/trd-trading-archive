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

class StructureNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.25)
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(256, dropout=0.25),
            ResidualBlock(256, dropout=0.25)
        )
        self.mid_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        self.heads = nn.ModuleList([
            nn.Linear(128, 5),  # structure_state
            nn.Linear(128, 4),  # bos_choc_flag
            nn.Linear(128, 3)   # breakout_potential
        ])

    def forward(self, x):
        h = self.input_layer(x)
        h = self.res_blocks(h)
        h = self.mid_layer(h)
        return [head(h) for head in self.heads]

# -----------------------------
# Predictor
# -----------------------------
class StructurePredictor:
    def __init__(self,
                 model_path="structure_model_best.pth",
                 scaler_path="structure_scaler.pkl"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        bundle = joblib.load(scaler_path)
        self.scaler = bundle["scaler"]
        self.feature_cols = bundle["feature_names"]

        self.model = StructureNet(input_dim=len(self.feature_cols)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        if not mt5.initialize():
            raise RuntimeError("MT5 initialization failed")

    # --- Feature Engineering (standalone, no imports from elsewhere)
    def _compute_features(self, df, seq_len=20):
        df = df.copy()

        # Core candle features
        df["range"] = df["high"] - df["low"]
        df["body"] = (df["close"] - df["open"]).abs()
        df["dir"] = np.sign(df["close"] - df["open"])
        df["wick"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["tail"] = df[["open", "close"]].min(axis=1) - df["low"]

        df["body_ratio"] = df["body"] / (df["range"] + 1e-6)
        df["tail_ratio"] = df["tail"] / (df["range"] + 1e-6)
        df["wick_ratio"] = df["wick"] / (df["range"] + 1e-6)

        # Rolling ATRs
        df["atr5"] = df["range"].rolling(5).mean()
        df["atr14"] = df["range"].rolling(14).mean()
        df["atr20"] = df["range"].rolling(20).mean()
        df["atr_ratio"] = df["atr5"] / (df["atr20"] + 1e-6)

        # Bollinger compression
        df["bandwidth"] = (
            df["close"].rolling(20).std() /
            (df["close"].rolling(20).mean() + 1e-6)
        )

        # Directional entropy proxy
        df["dir_entropy"] = (
            df["dir"].rolling(20).apply(lambda x: (x != 0).mean(), raw=True)
        )

        # Keep last N bars
        df = df.iloc[-seq_len:].fillna(0)

        # Flatten sequence features
        flat = {}
        for i in range(seq_len):
            for col in ["body", "wick", "tail", "body_ratio",
                        "tail_ratio", "wick_ratio", "dir"]:
                flat[f"{col}_t-{i}"] = df.iloc[i][col]

        # Aggregate features
        flat["atr5"] = df["atr5"].iloc[-1]
        flat["atr14"] = df["atr14"].iloc[-1]
        flat["atr20"] = df["atr20"].iloc[-1]
        flat["atr_ratio"] = df["atr_ratio"].iloc[-1]
        flat["bandwidth"] = df["bandwidth"].iloc[-1]
        flat["dir_entropy"] = df["dir_entropy"].iloc[-1]

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
            "structure_state": preds[0],
            "bos_choc_flag": preds[1],
            "breakout_potential": preds[2],
            "confidences": {
                "structure_state": confidences[0],
                "bos_choc_flag": confidences[1],
                "breakout_potential": confidences[2],
            }
        }

    def shutdown(self):
        mt5.shutdown()

# --- Example ---
if __name__ == "__main__":
    predictor = StructurePredictor()
    result = predictor.predict("EURUSDm", mt5.TIMEFRAME_M15)
    print("📊 Structure Prediction:", result)
    predictor.shutdown()
