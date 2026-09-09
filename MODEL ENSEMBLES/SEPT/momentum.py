# momentum_deploy.py
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

class MomentumNet(nn.Module):
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
            ResidualBlock(512, dropout=0.3)
        )
        self.mid_layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.25)
        )
        # Heads: momentum_dir(3), sustainability(3), exhaustion(2), ignition_quality(3)
        self.heads = nn.ModuleList([
            nn.Linear(256, 3),
            nn.Linear(256, 3),
            nn.Linear(256, 2),
            nn.Linear(256, 3),
        ])

    def forward(self, x):
        h = self.input_layer(x)
        h = self.res_blocks(h)
        h = self.mid_layer(h)
        return [head(h) for head in self.heads]


# -----------------------------
# Predictor
# -----------------------------
class MomentumPredictor:
    def __init__(self,
                 model_path="momentum_model_best.pth",
                 scaler_path="momentum_scaler.pkl",
                 seq_len: int = 20):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = seq_len

        bundle = joblib.load(scaler_path)
        self.scaler = bundle["scaler"]
        self.feature_cols = bundle["feature_names"]

        self.model = MomentumNet(input_dim=len(self.feature_cols)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        if not mt5.initialize():
            raise RuntimeError("MT5 initialization failed")

    # --- Feature Engineering (standalone) ---
    def _compute_features(self, df):
        """
        Build identical features used in synthetic generator:
        - sequence flattening of body/wick/tail/... for last seq_len bars
        - aggregated momentum features from last bar
        """
        df = df.copy().reset_index(drop=True)

        # Basic candle features
        df["range"] = df["high"] - df["low"]
        df["body"] = (df["close"] - df["open"]).abs()
        df["dir"] = np.sign(df["close"] - df["open"])
        df["wick"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["tail"] = df[["open", "close"]].min(axis=1) - df["low"]

        df["body_ratio"] = df["body"] / (df["range"] + 1e-6)
        df["wick_ratio"] = df["wick"] / (df["range"] + 1e-6)
        df["tail_ratio"] = df["tail"] / (df["range"] + 1e-6)

        # ROC + smoothing
        df["roc"] = df["close"].pct_change().fillna(0.0)
        df["roc_smooth"] = df["roc"].rolling(5, min_periods=1).mean()

        # RSI
        delta = df["close"].diff().fillna(0.0)
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        # use rolling mean of ups/downs (matches generator)
        up_avg = up.rolling(14, min_periods=1).mean()
        down_avg = down.rolling(14, min_periods=1).mean()
        rs = up_avg / (down_avg + 1e-6)
        df["rsi"] = 100 - (100 / (1 + rs))
        df["rsi_slope"] = df["rsi"].diff().fillna(0.0)

        # MACD and hist + slope
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        df["macd_slope"] = df["macd_hist"].diff().fillna(0.0)

        # Candle velocity (rolling avg abs(body))
        df["velocity"] = df["body"].rolling(5, min_periods=1).mean()

        # Exhaustion features
        df["body_shrink"] = df["body"] / (df["body"].rolling(5, min_periods=1).mean() + 1e-6)
        df["wick_tail_asym"] = (df["wick"] - df["tail"]) / (df["range"] + 1e-6 + 1e-12)

        # Divergence metrics
        df["price_slope"] = df["close"].diff().fillna(0.0)
        df["div_rsi"] = df["price_slope"] - df["rsi_slope"]
        df["div_macd"] = df["price_slope"] - df["macd_slope"]

        # Ensure we have at least seq_len bars
        if len(df) < self.seq_len:
            # Prepend duplicates of first row to reach seq_len (rare)
            needed = self.seq_len - len(df)
            pad = pd.DataFrame([df.iloc[0].to_dict()] * needed)
            df = pd.concat([pad, df], ignore_index=True)

        df = df.iloc[-self.seq_len:].fillna(0.0).reset_index(drop=True)

        # Flatten sequence-level features (t-0..t-(seq_len-1), where t-0 is oldest in sequence)
        # Note: in generator you used t-0..t-19 with 0 being first in sequence. Replicate same ordering.
        flat = {}
        for i in range(self.seq_len):
            # keep same ordering as training generator
            for col in ["body", "wick", "tail", "body_ratio", "wick_ratio", "tail_ratio", "dir"]:
                flat[f"{col}_t-{i}"] = df.iloc[i][col]

        # Aggregate (last-bar) features used in training
        last = df.iloc[-1]
        for col in ["roc_smooth", "rsi", "rsi_slope", "macd_hist", "macd_slope",
                    "velocity", "body_shrink", "wick_tail_asym", "div_rsi", "div_macd"]:
            flat[col] = last.get(col, 0.0)

        # Return in DataFrame form
        feat_df = pd.DataFrame([flat])

        # Reindex to expected feature order and fill missing with 0
        feat_df = feat_df.reindex(columns=self.feature_cols, fill_value=0.0)

        return feat_df

    def predict(self, symbol="EURUSDm", timeframe=mt5.TIMEFRAME_M15, n_bars=200):
        """
        Fetch n_bars of historical data from MT5, compute features and return predictions.
        Returns a dict with preds and confidences for each head.
        """
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
        if rates is None or len(rates) < max(30, self.seq_len):
            return {"error": "Insufficient data"}

        df = pd.DataFrame(rates)
        feat_df = self._compute_features(df)

        # Scale and convert
        scaled = self.scaler.transform(feat_df.values)
        x = torch.tensor(scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(x)

        preds = [int(torch.argmax(out, dim=1).cpu().item()) for out in outputs]
        probs = [torch.softmax(out, dim=1).cpu().numpy()[0] for out in outputs]
        confidences = [float(np.max(p)) for p in probs]

        return {
            "momentum_direction": preds[0],       # 0 neutral, 1 up, 2 down
            "sustainability": preds[1],           # 0 weak, 1 medium, 2 strong
            "exhaustion": preds[2],               # 0 no, 1 yes
            "ignition_quality": preds[3],         # 0 none, 1 clean, 2 fake
            "probabilities": {
                "momentum_direction": probs[0].tolist(),
                "sustainability": probs[1].tolist(),
                "exhaustion": probs[2].tolist(),
                "ignition_quality": probs[3].tolist(),
            },
            "confidences": {
                "momentum_direction": confidences[0],
                "sustainability": confidences[1],
                "exhaustion": confidences[2],
                "ignition_quality": confidences[3],
            }
        }

    def shutdown(self):
        mt5.shutdown()


# --- Example ---
if __name__ == "__main__":
    predictor = MomentumPredictor()
    result = predictor.predict("EURUSDm", mt5.TIMEFRAME_M15)
    print("📊 Momentum–Shift Prediction:", result)
    predictor.shutdown()
