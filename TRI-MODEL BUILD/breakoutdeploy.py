# breakout_deploy.py
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib


# ============================================================
#                     MODEL DEFINITION
# ============================================================

class ResidualBlock(nn.Module):
    def __init__(self, dim, drop=0.25):
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


class BreakoutMultiHeadNet(nn.Module):
    def __init__(self, input_dim, trunk_dim=256, drop=0.25):
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

        # Output heads
        self.head_vol = nn.Linear(trunk_dim // 2, 3)      # vol_regime
        self.head_bias = nn.Linear(trunk_dim // 2, 3)     # breakout_bias
        self.head_strength = nn.Linear(trunk_dim // 2, 2) # breakout_strength

    def forward(self, x):
        h = self.stem(x)
        h = self.res(h)
        h = self.mid(h)
        return self.head_vol(h), self.head_bias(h), self.head_strength(h)


# ============================================================
#                 BREAKOUT PREDICTOR CLASS
# ============================================================

class BreakoutPredictor:
    def __init__(self,
                 model_path="breakout_multihead_best.pth",
                 scaler_path="breakout_scaler.pkl",
                 seq_len=20,
                 symbol="EURUSD",
                 timeframe=mt5.TIMEFRAME_M15):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = seq_len
        self.symbol = symbol
        self.timeframe = timeframe

        # Load scaler bundle
        bundle = joblib.load(scaler_path)
        self.scaler = bundle["scaler"]
        self.feature_cols = bundle["feature_names"]

        # Load model
        self.model = BreakoutMultiHeadNet(input_dim=len(self.feature_cols)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        if not mt5.initialize():
            raise RuntimeError("MT5 initialization failed")

    # ---------------- Utility ----------------
    @staticmethod
    def _zscore(x, w):
        return (x - x.rolling(w).mean()) / (x.rolling(w).std() + 1e-12)

    # ============================================================
    #                 FEATURE ENGINEERING (DEPLOY)
    # ============================================================

    def _compute_features(self, df):
        df = df.copy().reset_index(drop=True)
        eps = 1e-12

        # ---------------- Basic candle features ----------------
        df["range"] = df["high"] - df["low"]
        df["body"] = abs(df["close"] - df["open"])
        df["dir"] = np.sign(df["close"] - df["open"])

        # ---------------- ATR & volatility ----------------
        df["atr14"] = df["range"].rolling(14, min_periods=1).mean()
        df["atr_z"] = self._zscore(df["atr14"], 40).fillna(0.0)

        df["range_3"] = df["range"].rolling(3, min_periods=1).mean()
        df["range_12"] = df["range"].rolling(12, min_periods=1).mean()

        df["compression_ratio"] = df["range_3"] / (df["range_12"] + eps)
        df["compression_zscore"] = self._zscore(df["compression_ratio"], 40).fillna(0.0)

        # ---------------- Bollinger Bands ----------------
        mid = df["close"].rolling(20, min_periods=1).mean()
        sd = df["close"].rolling(20, min_periods=1).std().fillna(0.0)

        df["bb_upper"] = mid + 2 * sd
        df["bb_lower"] = mid - 2 * sd
        bw = df["bb_upper"] - df["bb_lower"]

        df["bb_position"] = (df["close"] - df["bb_lower"]) / (bw + eps)
        df["bb_position"] = df["bb_position"].clip(0.0, 1.0)

        df["bb_bandwidth"] = bw / (df["atr14"] + eps)
        df["bb_bandwidth_z"] = self._zscore(df["bb_bandwidth"], 40).fillna(0.0)

        # ---------------- MACD ----------------
        fast = df["close"].ewm(span=12, adjust=False).mean()
        slow = df["close"].ewm(span=26, adjust=False).mean()
        macd_line = fast - slow
        signal = macd_line.ewm(span=9, adjust=False).mean()

        df["macd_hist"] = macd_line - signal
        df["macd_hist_slope"] = df["macd_hist"].diff().fillna(0.0)

        df["macd_cross_flag"] = 1
        df.loc[(macd_line > signal) & (macd_line.shift(1) <= signal.shift(1)), "macd_cross_flag"] = 2
        df.loc[(macd_line < signal) & (macd_line.shift(1) >= signal.shift(1)), "macd_cross_flag"] = 0

        # ---------------- RSI ----------------
        delta = df["close"].diff().fillna(0.0)
        up = delta.clip(lower=0)
        down = (-delta).clip(lower=0)

        roll_up = up.rolling(14, min_periods=1).mean()
        roll_dn = down.rolling(14, min_periods=1).mean() + eps

        rs = roll_up / roll_dn
        df["rsi14"] = 100 - (100 / (1 + rs))
        df["rsi_z"] = self._zscore(df["rsi14"], 40).fillna(0.0)

        # ---------------- Interactions ----------------
        df["rsi_x_macd"] = (df["rsi_z"] * df["macd_hist"]).clip(-3, 3)
        df["bbpos_x_macd"] = (df["bb_position"] * df["macd_hist"]).clip(-3, 3)

        # ---------------- Guarantee Sufficient Data ----------------
        if len(df) < self.seq_len:
            pad = pd.DataFrame([df.iloc[0]] * (self.seq_len - len(df)))
            df = pd.concat([pad, df], ignore_index=True)

        df = df.iloc[-self.seq_len:]

        # ==========================================================
        #      ADD MISSING SEQUENCE-LEVEL FEATURES (VERY IMPORTANT)
        # ==========================================================

        seq_dir_ratio = float((df["dir"] > 0).mean())
        seq_mean_range = float(df["range"].mean())
        seq_mean_volume = float(
            df["tick_volume"].mean()
            if "tick_volume" in df.columns else
            df.get("volume", pd.Series([0]*len(df))).mean()
        )
        seq_std_range = float(df["range"].std())

        # ---------------- Collect Last-Bar Features ----------------
        last = df.iloc[-1]

        features = {
            "atr14": float(last["atr14"]),
            "atr_z": float(last["atr_z"]),
            "bb_bandwidth": float(last["bb_bandwidth"]),
            "bb_bandwidth_z": float(last["bb_bandwidth_z"]),
            "bb_position": float(last["bb_position"]),
            "bbpos_x_macd": float(last["bbpos_x_macd"]),
            "compression_ratio": float(last["compression_ratio"]),
            "compression_zscore": float(last["compression_zscore"]),
            "macd_cross_flag": float(last["macd_cross_flag"]),
            "macd_hist": float(last["macd_hist"]),
            "macd_hist_slope": float(last["macd_hist_slope"]),
            "range_12": float(last["range_12"]),
            "range_3": float(last["range_3"]),
            "rsi14": float(last["rsi14"]),
            "rsi_x_macd": float(last["rsi_x_macd"]),
            "rsi_z": float(last["rsi_z"]),

            # -------- NEW: match synthetic training dataset --------
            "seq_dir_ratio": seq_dir_ratio,
            "seq_mean_range": seq_mean_range,
            "seq_mean_volume": seq_mean_volume,
            "seq_std_range": seq_std_range,
        }

        feat_df = pd.DataFrame([features])
        feat_df = feat_df.reindex(columns=self.feature_cols, fill_value=0.0)
        return feat_df

    # ============================================================
    #                         PREDICT
    # ============================================================

    def predict(self, symbol=None, timeframe=None, n_bars=200):
        symbol = symbol or self.symbol
        timeframe = timeframe or self.timeframe

        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
        if rates is None or len(rates) < self.seq_len:
            return {"error": "Insufficient MT5 data"}

        df = pd.DataFrame(rates)
        feat_df = self._compute_features(df)

        scaled = self.scaler.transform(feat_df[self.feature_cols])
        x = torch.tensor(scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            out_vol, out_bias, out_strength = self.model(x)

        pv = torch.softmax(out_vol, dim=1).cpu().numpy()[0].tolist()
        pb = torch.softmax(out_bias, dim=1).cpu().numpy()[0].tolist()
        ps = torch.softmax(out_strength, dim=1).cpu().numpy()[0].tolist()

        vol = int(np.argmax(pv))
        bias = int(np.argmax(pb))
        strength = int(np.argmax(ps))

        return {
            "vol_regime": vol,
            "vol_regime_label": ["stable", "compressing", "expanding"][vol],
            "vol_regime_prob": pv,

            "breakout_bias": bias,
            "breakout_bias_label": ["down", "neutral", "up"][bias],
            "breakout_bias_prob": pb,

            "breakout_strength": strength,
            "breakout_strength_label": ["weak", "strong"][strength],
            "breakout_strength_prob": ps,

            "feature_values": {c: float(feat_df[c].iloc[0]) for c in self.feature_cols},
        }

    def shutdown(self):
        mt5.shutdown()


# ============================================================
#                        TEST RUN
# ============================================================

if __name__ == "__main__":
    p = BreakoutPredictor()
    out = p.predict()
    print("📊 Breakout Prediction:", out)
    p.shutdown()
