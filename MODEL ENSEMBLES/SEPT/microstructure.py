# microstructure_deploy.py
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

# ---------------- Model (must match training) ----------------
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.35):
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

class MicrostructureNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.35)
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(512, dropout=0.35),
            ResidualBlock(512, dropout=0.35),
            ResidualBlock(512, dropout=0.35)
        )
        self.mid_layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        # Heads (sizes): 2, 4, 4, 4, 3, 3
        self.heads = nn.ModuleList([
            nn.Linear(256, 2),  # continuation
            nn.Linear(256, 4),  # cluster_strength
            nn.Linear(256, 4),  # pullback_quality
            nn.Linear(256, 4),  # rejection_type
            nn.Linear(256, 3),  # absorption_score_cat
            nn.Linear(256, 3),  # expected_move_mult
        ])

    def forward(self, x):
        h = self.input_layer(x)
        h = self.res_blocks(h)
        h = self.mid_layer(h)
        return [head(h) for head in self.heads]

# ---------------- Predictor ----------------
class MicrostructurePredictor:
    def __init__(self,
                 model_path="microstructure_model_best.pth",
                 scaler_path="microstructure_scaler.pkl",
                 seq_len=20):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = seq_len

        bundle = joblib.load(scaler_path)
        self.scaler = bundle["scaler"]
        self.feature_cols = bundle["feature_names"]

        self.model = MicrostructureNet(input_dim=len(self.feature_cols)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        if not mt5.initialize():
            raise RuntimeError("MT5 initialization failed")

    # ---------- feature engineering to match generator ----------
    @staticmethod
    def _rolling_zscore(s, window):
        return (s - s.rolling(window).mean()) / (s.rolling(window).std() + 1e-9)

    def _compute_features(self, df):
        df = df.copy().reset_index(drop=True)

        # Basic candle features
        df["range"] = df["high"] - df["low"]
        df["body"] = (df["close"] - df["open"]).abs()
        df["dir"] = np.sign(df["close"] - df["open"]).fillna(0).astype(int)
        df["wick"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["tail"] = df[["open", "close"]].min(axis=1) - df["low"]

        df["body_ratio"] = df["body"] / (df["range"] + 1e-12)
        df["wick_ratio"] = df["wick"] / (df["range"] + 1e-12)
        df["tail_ratio"] = df["tail"] / (df["range"] + 1e-12)

        # Stats
        df["range_mean3"] = df["range"].rolling(3).mean()
        df["range_mean5"] = df["range"].rolling(5).mean()
        df["range_std5"] = df["range"].rolling(5).std().fillna(0.0)
        df["body_mean5"] = df["body"].rolling(5).mean()
        df["body_mean3"] = df["body"].rolling(3).mean()

        # Wick-tail asym
        df["wick_tail_asym"] = (df["wick"] - df["tail"]) / (df["range"] + 1e-12)

        # Volume features & imbalance proxies
        if "tick_volume" in df.columns and "volume" not in df.columns:
            df["volume"] = df["tick_volume"]
        elif "volume" not in df.columns:
            df["volume"] = 0

        df["signed_volume"] = df["volume"] * df["dir"]
        df["vol_mean5"] = df["volume"].rolling(5).mean()
        df["vol_z5"] = self._rolling_zscore(df["volume"], 5).fillna(0.0)
        df["vol_imbalance_5"] = df["signed_volume"].rolling(5).sum() / (df["volume"].rolling(5).sum() + 1e-12)

        # Consecutive body cluster length
        cons = []
        cur = 0
        prev = 0
        for d in df["dir"].fillna(0).tolist():
            if d == prev and d != 0:
                cur += 1
            elif d != 0:
                cur = 1
            else:
                cur = 0
            cons.append(cur)
            prev = d
        df["consec_body_cluster_len"] = cons

        # Orderflow proxies
        df["signed_body_vol"] = df["dir"] * df["body"] * df["volume"]
        df["orderflow_5"] = df["signed_body_vol"].rolling(5).sum()
        df["orderflow_10"] = df["signed_body_vol"].rolling(10).sum()

        # Absorption proxy
        df["is_small_body"] = (df["body"] < (df["body"].rolling(5).mean() * 0.6)).astype(int)
        df["opposite_wick"] = 0.0
        df.loc[(df["dir"] > 0), "opposite_wick"] = df["tail"]
        df.loc[(df["dir"] < 0), "opposite_wick"] = df["wick"]
        df["absorption_raw"] = (df["is_small_body"] * df["opposite_wick"]).rolling(5).sum()

        # Pullback depth
        df["pullback_depth"] = df["range"] / (df["range_mean5"] + 1e-12)
        df["pullback_depth_z"] = self._rolling_zscore(df["pullback_depth"], 5).fillna(0.0)

        # Exhaustion proxy
        df["exhaustion_proxy"] = ((df["wick"] + df["tail"]) / (df["body"] + 1e-12)).fillna(0.0)

        # ATR-ish and expected move
        df["atr5"] = df["range"].rolling(5).mean().fillna(0.0)
        df["expected_move_raw"] = df["atr5"] * (1 + np.abs(df["orderflow_5"]) / (df["vol_mean5"] + 1e-12))

        # Interactions
        df["vol_body_inter"] = df["volume"] * df["body"]
        df["wick_vol_inter"] = df["wick"] * df["volume"]
        df["body_ratio_volimb"] = df["body_ratio"] * df["vol_imbalance_5"]

        df = df.fillna(0.0)

        # Ensure enough bars
        if len(df) < self.seq_len:
            need = self.seq_len - len(df)
            pad = pd.DataFrame([df.iloc[0].to_dict()] * need)
            df = pd.concat([pad, df], ignore_index=True)

        df = df.iloc[-self.seq_len:].reset_index(drop=True)

        # Flatten per-bar features (order must match generator)
        flat = {}
        for i in range(self.seq_len):
            for col in ["body", "range", "wick", "tail", "body_ratio", "wick_ratio", "tail_ratio", "dir", "volume"]:
                flat[f"{col}_t-{i}"] = df.iloc[i][col]

        # Add aggregated/last-bar features (same list as generator)
        agg_cols = [
            "range_mean3","range_mean5","range_std5","body_mean5","body_mean3",
            "wick_tail_asym","vol_mean5","vol_z5","vol_imbalance_5",
            "consec_body_cluster_len","orderflow_5","orderflow_10","absorption_raw",
            "pullback_depth","pullback_depth_z","exhaustion_proxy","atr5","expected_move_raw",
            "vol_body_inter","wick_vol_inter","body_ratio_volimb"
        ]
        last = df.iloc[-1]
        for col in agg_cols:
            flat[col] = last.get(col, 0.0)

        # Extra aggregated stats used in generator
        flat["mean_body_last5"] = df["body"].tail(5).mean()
        flat["std_range_last5"] = df["range"].tail(5).std()
        flat["sum_signed_vol_last5"] = df["signed_volume"].tail(5).sum()

        feat_df = pd.DataFrame([flat])
        feat_df = feat_df.reindex(columns=self.feature_cols, fill_value=0.0)
        return feat_df

    def predict(self, symbol="EURUSDm", timeframe=mt5.TIMEFRAME_M15, n_bars=200):
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
        if rates is None or len(rates) < max(30, self.seq_len):
            return {"error": "Insufficient data"}

        df = pd.DataFrame(rates)
        feat_df = self._compute_features(df)

        scaled = self.scaler.transform(feat_df.values)
        x = torch.tensor(scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(x)

        preds = [int(torch.argmax(out, dim=1).cpu().item()) for out in outputs]
        probs = [torch.softmax(out, dim=1).cpu().numpy()[0] for out in outputs]
        confidences = [float(np.max(p)) for p in probs]

        return {
            "continuation": preds[0],
            "cluster_strength": preds[1],
            "pullback_quality": preds[2],
            "rejection_type": preds[3],
            "absorption_score_cat": preds[4],
            "expected_move_mult": preds[5],
            "probabilities": {
                "continuation": probs[0].tolist(),
                "cluster_strength": probs[1].tolist(),
                "pullback_quality": probs[2].tolist(),
                "rejection_type": probs[3].tolist(),
                "absorption_score_cat": probs[4].tolist(),
                "expected_move_mult": probs[5].tolist(),
            },
            "confidences": {
                "continuation": confidences[0],
                "cluster_strength": confidences[1],
                "pullback_quality": confidences[2],
                "rejection_type": confidences[3],
                "absorption_score_cat": confidences[4],
                "expected_move_mult": confidences[5],
            }
        }

    def shutdown(self):
        mt5.shutdown()

# -------------- Example --------------
if __name__ == "__main__":
    predictor = MicrostructurePredictor()
    result = predictor.predict("EURUSDm", mt5.TIMEFRAME_M15)
    print("📊 Microstructure Flow Prediction:", result)
    predictor.shutdown()
