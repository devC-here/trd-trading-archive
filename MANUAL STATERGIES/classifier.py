import os
import sys
import MetaTrader5 as mt5
import pandas as pd
import torch
import torch.nn as nn
import joblib

# =====================================================
# -------- Model (Residual Net for Candle Ratios) -----
# =====================================================
class ResidualBlock(nn.Module):
    def __init__(self, dim, drop=0.35):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.seq(x) + x)


class CandleNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.35),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(512, 0.35),
            ResidualBlock(512, 0.35),
            ResidualBlock(512, 0.35),
        )
        self.mid = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.30),
        )
        self.head = nn.Linear(256, 3)  # 0 = Below EMA, 1 = Neutral, 2 = Above EMA

    def forward(self, x):
        h = self.stem(x)
        h = self.blocks(h)
        h = self.mid(h)
        return self.head(h)


# =====================================================
# ---------------- Candle Predictor -------------------
# =====================================================
class CandlePredictor:
    def __init__(self,
                 model_path="candle_ratios_model_best.pth",
                 scaler_path="candle_ratios_scaler.pkl"):
        
        # Determine the base path for bundled files
        if getattr(sys, 'frozen', False):
            # When running as a PyInstaller executable
            base_path = sys._MEIPASS
        else:
            # When running as a regular Python script
            base_path = os.path.dirname(os.path.abspath(__file__))

        # Join the base path with the file names
        full_model_path = os.path.join(base_path, model_path)
        full_scaler_path = os.path.join(base_path, scaler_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load scaler bundle using the corrected path
        bundle = joblib.load(full_scaler_path)
        self.scaler = bundle["scaler"]
        self.feature_cols = bundle["feature_names"]

        # Load trained model using the corrected path
        self.model = CandleNet(input_dim=len(self.feature_cols)).to(self.device)
        self.model.load_state_dict(torch.load(full_model_path, map_location=self.device))
        self.model.eval()

        # MT5 init
        if not mt5.initialize():
            raise RuntimeError("❌ MT5 initialization failed")

    # ----------------- Feature Builder -----------------
    def _build_features(self, candle):
        """Generate ratios (upper_ratio, lower_ratio) from last candle vs EMA."""

        o, h, l, c, ema_val = candle["open"], candle["high"], candle["low"], candle["close"], candle["ema"]

        # ---- Body High/Low ----
        high_body = max(o, c)
        low_body = min(o, c)
        body_size = abs(c - o)

        if body_size == 0:  # Doji case
            upper_ratio, lower_ratio = 0.5, 0.5
        else:
            # Clamp EMA within the body
            ema_clamped = min(max(ema_val, low_body), high_body)
            upper_part = high_body - ema_clamped
            lower_part = ema_clamped - low_body

            upper_ratio = upper_part / body_size
            lower_ratio = lower_part / body_size

        # ---- Debug Print ----
        print("\n📊 Last Candle OHLCV+EMA:")
        print(f"O={o}, H={h}, L={l}, C={c}, EMA={ema_val}")
        print(f"Body High={high_body}, Body Low={low_body}, Body Size={body_size}")
        print(f"➡️ Features -> upper_ratio={upper_ratio:.4f}, lower_ratio={lower_ratio:.4f}")

        # ---- Feature Vector ----
        features = {
            "upper_ratio": upper_ratio,
            "lower_ratio": lower_ratio,
        }

        feat_df = pd.DataFrame([features])
        feat_df = feat_df.reindex(columns=self.feature_cols, fill_value=0.0)
        return feat_df

    # ---------------- Prediction ----------------
    def predict(self, symbol="EURUSDm", timeframe=mt5.TIMEFRAME_M15, n_bars=50):
        """Fetch OHLCV (last candle only) and predict with 10-period EMA."""
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, n_bars)
        if rates is None or len(rates) < 11:  # need at least 10 candles for EMA
            return {"error": "Insufficient data"}

        df = pd.DataFrame(rates)[["open", "high", "low", "close"]]

        # ---- Calculate 10-period EMA ----
        df["ema"] = df["close"].ewm(span=10, adjust=False).mean()

        last_candle = df.iloc[-1]

        # Build features
        feat_df = self._build_features(last_candle)
        X = torch.tensor(self.scaler.transform(feat_df.values), dtype=torch.float32).to(self.device)

        # Model inference
        with torch.no_grad():
            outs = self.model(X)

        pred = int(outs.argmax(1).item())  # 0 = Below EMA, 1 = Neutral, 2 = Above EMA
        probs = torch.softmax(outs, dim=1).cpu().numpy()[0].tolist()
        confidence = float(max(probs))

        return {
            "prediction": pred,
            "probabilities": probs,
            "confidence": confidence
        }

    def shutdown(self):
        mt5.shutdown()


# =====================================================
# ----------------- Example Run -----------------------
# =====================================================
if __name__ == "__main__":
    p = CandlePredictor()
    result = p.predict("EURUSD", mt5.TIMEFRAME_M15, 50)
    print("\n📈 Candlestick Prediction:", result)
    p.shutdown()