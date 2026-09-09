# vwap_deploy.py
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

# ---------------- Model (match training) ----------------
class ResidualBlock(nn.Module):
    def __init__(self, dim, drop=0.35):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()
    def forward(self, x): return self.act(self.net(x)+x)

class VWAPNet(nn.Module):
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
            nn.Dropout(0.3),
        )
        # 7 heads: 3,2,4,3,3,3,3
        self.heads = nn.ModuleList([
            nn.Linear(256, 3),  # trend_state
            nn.Linear(256, 2),  # reversion_vs_trend
            nn.Linear(256, 4),  # anchor_bias
            nn.Linear(256, 3),  # entry_window
            nn.Linear(256, 3),  # band_touch
            nn.Linear(256, 3),  # cross_strength
            nn.Linear(256, 3),  # reversion_target_band
        ])
    def forward(self, x):
        h = self.stem(x); h = self.blocks(h); h = self.mid(h)
        return [head(h) for head in self.heads]

# ---------------- Predictor ----------------
class VWAPPredictor:
    def __init__(self,
                 model_path="vwap_model_best.pth",
                 scaler_path="vwap_scaler.pkl",
                 seq_len=20):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = seq_len

        bundle = joblib.load(scaler_path)
        self.scaler = bundle["scaler"]
        self.feature_cols = bundle["feature_names"]

        self.model = VWAPNet(input_dim=len(self.feature_cols)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        if not mt5.initialize():
            raise RuntimeError("MT5 initialization failed")

    # ---------- feature helpers ----------
    @staticmethod
    def _tp(df):
        return (df["high"] + df["low"] + df["close"]) / 3.0

    def _vwap_series(self, df):
        tp = self._tp(df)
        vol = df["volume"].values
        pv = (tp * vol).cumsum()
        vv = (vol).cumsum() + 1e-12
        return pv / vv

    @staticmethod
    def _rolling_z(s, w):
        return (s - s.rolling(w).mean()) / (s.rolling(w).std() + 1e-12)

    @staticmethod
    def _local_swings(df, lookback=4):
        highs = df["high"].values; lows = df["low"].values
        sh, sl = None, None
        for i in range(len(df)-lookback-1, lookback-1, -1):
            if sh is None and highs[i] == highs[i-lookback:i+lookback+1].max():
                sh = i
            if sl is None and lows[i] == lows[i-lookback:i+lookback+1].min():
                sl = i
            if sh is not None and sl is not None: break
        if sh is None: sh = max(len(df)//3, 1)
        if sl is None: sl = max(2*len(df)//3, 2)
        return int(sh), int(sl)

    def _anchored_vwap(self, df, start_idx):
        tp = self._tp(df).values.copy()
        vol = df["volume"].values.copy()
        pv = tp * vol
        pv[:start_idx] = 0.0; vol[:start_idx] = 0.0
        cpv = pv.cumsum(); cv = vol.cumsum() + 1e-12
        return cpv / cv

    # ---------- core feature build ----------
    def _build_features(self, raw_df):
        df = raw_df.copy()
        if "tick_volume" in df.columns and "volume" not in df.columns:
            df["volume"] = df["tick_volume"]
        elif "volume" not in df.columns:
            df["volume"] = 0

        # keep last N+extra for rolling
        df = df.tail(max(self.seq_len*3, 100)).reset_index(drop=True)

        # Base
        df["range"] = df["high"] - df["low"]
        df["body"] = (df["close"] - df["open"]).abs()
        df["dir"] = np.sign(df["close"] - df["open"]).fillna(0).astype(int)

        # VWAP & bands
        df["vwap"] = self._vwap_series(df)
        dev = df["close"] - df["vwap"]
        df["dev"] = dev
        df["dev_abs"] = dev.abs()
        df["dev_z20"] = self._rolling_z(dev, 20).fillna(0.0)
        df["vwap_slope"] = df["vwap"].diff().fillna(0.0)
        dev_std20 = dev.rolling(20).std().fillna(dev.rolling(10).std()).fillna(1e-6)
        df["band_upper_1"] = df["vwap"] + dev_std20 * 1.0
        df["band_lower_1"] = df["vwap"] - dev_std20 * 1.0
        df["band_upper_2"] = df["vwap"] + dev_std20 * 2.0
        df["band_lower_2"] = df["vwap"] - dev_std20 * 2.0

        # Anchors
        sh, sl = self._local_swings(df, lookback=4)
        df["awap_high"] = self._anchored_vwap(df, sh)
        df["awap_low"]  = self._anchored_vwap(df, sl)

        # Distances / z
        df["dev_awap_high"] = df["close"] - df["awap_high"]
        df["dev_awap_low"]  = df["close"] - df["awap_low"]
        df["dev_awap_high_z20"] = self._rolling_z(df["dev_awap_high"], 20).fillna(0.0)
        df["dev_awap_low_z20"]  = self._rolling_z(df["dev_awap_low"], 20).fillna(0.0)

        # Cross / confluence
        df["above_vwap"] = (df["close"] > df["vwap"]).astype(int)
        df["above_high_awap"] = (df["close"] > df["awap_high"]).astype(int)
        df["above_low_awap"]  = (df["close"] > df["awap_low"]).astype(int)
        df["cross_vwap"] = (df["above_vwap"].diff().abs() > 0).astype(int)

        # time_since_cross (robust)
        tmp = df["cross_vwap"].copy()
        last_seen = -1; tsc = []
        for i,flag in enumerate(tmp):
            if flag == 1: last_seen = i
            tsc.append(0 if last_seen < 0 else i - last_seen)
        df["time_since_cross"] = tsc

        # more distances
        df["dist_vwap"] = df["close"] - df["vwap"]
        df["dist_awap_high"] = df["close"] - df["awap_high"]
        df["dist_awap_low"]  = df["close"] - df["awap_low"]

        df["confluence_support_score"] = (
            (df["close"] > df["awap_low"]).astype(int) +
            (df["close"] > df["vwap"]).astype(int) +
            (df["close"] > df["band_lower_1"]).astype(int)
        )
        df["confluence_resist_score"] = (
            (df["close"] < df["awap_high"]).astype(int) +
            (df["close"] < df["vwap"]).astype(int) +
            (df["close"] < df["band_upper_1"]).astype(int)
        )

        df["pct_above_vwap_10"] = df["above_vwap"].rolling(10).mean().fillna(0.0)
        df["vwap_curvature"] = df["vwap_slope"].diff().fillna(0.0)

        df = df.fillna(0.0).tail(self.seq_len).reset_index(drop=True)

        # Flatten per-bar features (must match generator)
        flat = {}
        for i in range(self.seq_len):
            for col in ["close","vwap","dev","dev_z20","vwap_slope","above_vwap","volume"]:
                flat[f"{col}_t-{i}"] = float(df.iloc[i][col])

        # Last-bar aggregates (same as generator)
        last = df.iloc[-1]
        for c in [
            "dev_abs","band_upper_1","band_lower_1","band_upper_2","band_lower_2",
            "awap_high","awap_low","dev_awap_high","dev_awap_low",
            "dev_awap_high_z20","dev_awap_low_z20","time_since_cross",
            "confluence_support_score","confluence_resist_score",
            "pct_above_vwap_10","vwap_curvature",
            "dist_vwap","dist_awap_high","dist_awap_low"
        ]:
            flat[c] = float(last[c])

        feat_df = pd.DataFrame([flat])
        feat_df = feat_df.reindex(columns=self.feature_cols, fill_value=0.0)
        return feat_df

    # ---------- predict ----------
    def predict(self, symbol="EURUSDm", timeframe=mt5.TIMEFRAME_M15, n_bars=300):
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
        if rates is None or len(rates) < max(80, self.seq_len+5):
            return {"error": "Insufficient data"}
        df = pd.DataFrame(rates)
        # MT5 columns: time, open, high, low, close, tick_volume, spread, real_volume
        if "volume" not in df.columns:
            df["volume"] = df.get("real_volume", df.get("tick_volume", 0))

        feat_df = self._build_features(df)

        x = torch.tensor(self.scaler.transform(feat_df.values), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outs = self.model(x)

        preds = [int(o.argmax(1).item()) for o in outs]
        probs = [torch.softmax(o, dim=1).cpu().numpy()[0].tolist() for o in outs]
        confs = [float(max(p)) for p in probs]

        return {
            "trend_state": preds[0],               # 0 range, 1 up, 2 down
            "reversion_vs_trend": preds[1],        # 0 revert, 1 continue
            "anchor_bias": preds[2],               # 0 neutral,1 above_all,2 below_all,3 mixed
            "entry_window": preds[3],              # 0 no_trade,1 immediate,2 wait_pullback
            "band_touch": preds[4],                # 0 none,1 upper,2 lower
            "cross_strength": preds[5],            # 0 weak,1 med,2 strong
            "reversion_target_band": preds[6],     # 0 vwap,1 mid,2 far
            "probabilities": {
                "trend_state": probs[0],
                "reversion_vs_trend": probs[1],
                "anchor_bias": probs[2],
                "entry_window": probs[3],
                "band_touch": probs[4],
                "cross_strength": probs[5],
                "reversion_target_band": probs[6],
            },
            "confidences": {
                "trend_state": confs[0],
                "reversion_vs_trend": confs[1],
                "anchor_bias": confs[2],
                "entry_window": confs[3],
                "band_touch": confs[4],
                "cross_strength": confs[5],
                "reversion_target_band": confs[6],
            }
        }

    def shutdown(self): mt5.shutdown()

# --------- Example ---------
if __name__ == "__main__":
    p = VWAPPredictor()
    out = p.predict("EURUSDm", mt5.TIMEFRAME_M15)
    print("📊 VWAP–Anchored Prediction:", out)
    p.shutdown()
