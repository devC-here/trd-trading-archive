# momentum_deploy.py
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import math

# =====================================================================
# Model Architecture (must match training exactly)
# =====================================================================

class ResidualBlock(nn.Module):
    def __init__(self, dim, drop=0.25):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.seq(x) + x)

class MomentumMultiHeadNet(nn.Module):
    def __init__(self, input_dim, trunk_dim=192, drop=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, trunk_dim),
            nn.BatchNorm1d(trunk_dim),
            nn.GELU(),
            nn.Dropout(drop)
        )
        self.res = nn.Sequential(
            ResidualBlock(trunk_dim, drop),
            ResidualBlock(trunk_dim, drop)
        )
        self.mid = nn.Sequential(
            nn.Linear(trunk_dim, trunk_dim // 2),
            nn.BatchNorm1d(trunk_dim // 2),
            nn.GELU(),
            nn.Dropout(drop)
        )
        self.head_momentum = nn.Linear(trunk_dim // 2, 3)
        self.head_slope = nn.Linear(trunk_dim // 2, 3)
        self.head_bias = nn.Linear(trunk_dim // 2, 3)

    def forward(self, x):
        h = self.stem(x)
        h = self.res(h)
        h = self.mid(h)
        return self.head_momentum(h), self.head_slope(h), self.head_bias(h)

# =====================================================================
# Deployment Class
# =====================================================================

class MomentumPredictor:
    def __init__(self,
                 model_path="momentum_multihead_best.pth",
                 scaler_path="momentum_scaler.pkl",
                 seq_len=20):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = seq_len

        # Load scaler bundle
        bundle = joblib.load(scaler_path)
        self.scaler = bundle["scaler"]
        self.feature_cols = bundle["feature_names"]

        # Init model
        self.model = MomentumMultiHeadNet(len(self.feature_cols)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Init MT5
        if not mt5.initialize():
            raise RuntimeError("MT5 initialization failed")

    # -------------------------------------------------------------
    # Core: compute features EXACTLY like generator
    # -------------------------------------------------------------
    @staticmethod
    def _rolling_z(series, window):
        return (series - series.rolling(window).mean()) / (series.rolling(window).std() + 1e-12)

    def _compute_features(self, df):
        eps = 1e-12
        df = df.copy().reset_index(drop=True)

        df["close_prev1"] = df["close"].shift(1)
        df["logret1"] = np.log(df["close"] / df["close_prev1"]).fillna(0.0)
        df["logret3"] = np.log(df["close"] / df["close"].shift(3)).fillna(0.0)
        df["roc3"] = df["close"].pct_change(3).fillna(0.0)
        df["roc5"] = df["close"].pct_change(5).fillna(0.0)

        # RSI (same method as generator)
        delta = df["close"].diff().fillna(0.0)
        up = delta.clip(lower=0)
        down = (-delta).clip(lower=0)
        roll_up = up.rolling(14, min_periods=1).mean()
        roll_dn = down.rolling(14, min_periods=1).mean() + eps
        rs = roll_up / roll_dn
        df["rsi14"] = 100 - (100 / (1 + rs))
        df["rsi_ema5"] = df["rsi14"].ewm(span=5, adjust=False).mean()
        df["rsi_z"] = self._rolling_z(df["rsi14"], 40).fillna(0.0)

        # MACD
        ema_fast = df["close"].ewm(span=12, adjust=False).mean()
        ema_slow = df["close"].ewm(span=26, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_sig = macd_line.ewm(span=9, adjust=False).mean()
        df["macd_hist"] = (macd_line - macd_sig).fillna(0.0)
        df["macd_hist_slope"] = df["macd_hist"].diff().fillna(0.0)
        df["macd_ema5"] = df["macd_hist"].ewm(span=5, adjust=False).mean()

        # return Z-score
        df["return_zscore"] = self._rolling_z(df["logret3"], 40).fillna(0.0)

        # slope regression (same as generator logic)
        def slope_calc(window):
            arr = window.values
            if len(arr) < 3:
                return 0.0
            x = np.arange(len(arr))
            return float(np.polyfit(x, arr, 1)[0])

        df["close_slope_10"] = df["close"].rolling(10).apply(slope_calc, raw=False).fillna(0.0)

        # ATR14 approx
        df["range"] = df["high"] - df["low"]
        df["atr14"] = df["range"].rolling(14).mean().fillna(0.0)
        df["trend_strength_score"] = (df["close_slope_10"].abs() * (df["atr14"] / (df["close"] + eps))).fillna(0.0)

        # interactions
        df["rsi_x_macd"] = (df["rsi_z"] * df["macd_hist"]).clip(-3, 3)
        df["roc3_x_macd"] = (df["roc3"] * df["macd_hist"]).clip(-3, 3)

        # seq-level stats
        df["seq_ret_sum"] = df["logret1"].rolling(self.seq_len).sum().fillna(0.0)
        df["seq_volatility"] = df["logret1"].rolling(self.seq_len).std().fillna(0.0)

        # pad if needed
        if len(df) < self.seq_len:
            pad = pd.DataFrame([df.iloc[0].to_dict()] * (self.seq_len - len(df)))
            df = pd.concat([pad, df], ignore_index=True)

        df = df.iloc[-self.seq_len:].reset_index(drop=True)
        last = df.iloc[-1]

        feat = {
            "logret1": last["logret1"],
            "logret3": last["logret3"],
            "roc3": last["roc3"],
            "roc5": last["roc5"],
            "rsi14": last["rsi14"],
            "rsi_ema5": last["rsi_ema5"],
            "rsi_z": last["rsi_z"],
            "macd_hist": last["macd_hist"],
            "macd_hist_slope": last["macd_hist_slope"],
            "macd_ema5": last["macd_ema5"],
            "return_zscore": last["return_zscore"],
            "close_slope_10": last["close_slope_10"],
            "trend_strength_score": last["trend_strength_score"],
            "rsi_x_macd": last["rsi_x_macd"],
            "roc3_x_macd": last["roc3_x_macd"],
            "seq_ret_sum": last["seq_ret_sum"],
            "seq_volatility": last["seq_volatility"],
            # extra seq stats (generator)
            "seq_dir_ratio": float((df["close"].diff().fillna(0.0) > 0).mean()),
            "seq_mean_range": float(df["range"].mean()),
            "seq_std_range": float(df["range"].std()),
        }

        # align order with training
        feat_df = pd.DataFrame([feat])
        feat_df = feat_df.reindex(columns=self.feature_cols, fill_value=0.0)

        return feat_df

    # -------------------------------------------------------------
    # MAIN INFERENCE
    # -------------------------------------------------------------
    def predict(self, symbol="EURUSD", timeframe=mt5.TIMEFRAME_M15, n_bars=200):

        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)

        if rates is None or len(rates) < max(30, self.seq_len):
            return {"error": "Insufficient data"}

        df = pd.DataFrame(rates)

        features = self._compute_features(df)
        scaled = self.scaler.transform(features.values)
        x = torch.tensor(scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            o1, o2, o3 = self.model(x)

        # Convert to outputs
        out = {}
        # momentum_state: 0=down,1=neutral,2=up
        momentum = torch.softmax(o1, dim=1).cpu().numpy()[0]
        slope = torch.softmax(o2, dim=1).cpu().numpy()[0]
        bias = torch.softmax(o3, dim=1).cpu().numpy()[0]

        out["momentum_state"] = int(np.argmax(momentum))
        out["momentum_prob"] = momentum.tolist()
        out["momentum_conf"] = float(momentum.max())

        out["slope_zone"] = int(np.argmax(slope))
        out["slope_prob"] = slope.tolist()
        out["slope_conf"] = float(slope.max())

        out["entry_bias"] = int(np.argmax(bias))
        out["entry_bias_prob"] = bias.tolist()
        out["entry_bias_conf"] = float(bias.max())

        # readable labels
        out["labels"] = {
            "momentum_state": ["down", "neutral", "up"][out["momentum_state"]],
            "slope_zone": ["falling", "flat", "rising"][out["slope_zone"]],
            "entry_bias": ["avoid", "neutral", "bias"][out["entry_bias"]],
        }

        # show input feature values (nice for debugging)
        feat_map = dict(zip(self.feature_cols, features.values.flatten().tolist()))
        out["feature_values"] = feat_map

        return out

    def shutdown(self):
        mt5.shutdown()


# =====================================================================
# Example run
# =====================================================================
if __name__ == "__main__":
    predictor = MomentumPredictor()

    result = predictor.predict("EURUSD", mt5.TIMEFRAME_M15)
    print("📊 Momentum Prediction:")
    print(result)

    predictor.shutdown()
