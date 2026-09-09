# macro_deploy_noevents.py
"""
Deploy wrapper for Macro (no-events) model.
Loads macro_noevents_scaler.pkl and macro_noevents_model_best.pth.
Reconstructs features from MT5 M15 bars (H1 proxied from M15).
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

SEQ_LEN = 20

class ResidualBlock(nn.Module):
    def __init__(self, dim, drop=0.35):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(drop),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim)
        )
        self.act = nn.GELU()
    def forward(self, x): return self.act(self.seq(x)+x)

class MacroNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(input_dim,512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.35))
        self.blocks = nn.Sequential(ResidualBlock(512,0.35), ResidualBlock(512,0.35), ResidualBlock(512,0.35))
        self.mid = nn.Sequential(nn.Linear(512,256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.30))
        self.heads = nn.ModuleList([
            nn.Linear(256,3),  # session_bias
            nn.Linear(256,3),  # tf_alignment_head
            nn.Linear(256,3),  # volatility_shift
            nn.Linear(256,3),  # session_phase
            nn.Linear(256,3),  # expected_move_cat
            nn.Linear(256,2),  # continuity
            nn.Linear(256,3),  # vol_regime
        ])
    def forward(self, x):
        h = self.stem(x); h = self.blocks(h); h = self.mid(h)
        return [head(h) for head in self.heads]

class MacroPredictor:
    def __init__(self, model_path="macro_noevents_model_best.pth", scaler_path="macro_noevents_scaler.pkl", seq_len=SEQ_LEN):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = seq_len
        bundle = joblib.load(scaler_path)
        self.scaler = bundle["scaler"]
        self.feature_cols = bundle["feature_names"]

        self.model = MacroNet(input_dim=len(self.feature_cols)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        if not mt5.initialize():
            raise RuntimeError("MT5 initialization failed")

    @staticmethod
    def _rolling_z(s, w):
        return (s - s.rolling(w).mean()) / (s.rolling(w).std() + 1e-12)

    def _anchored_vwap(self, df, start_idx):
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        pv = (tp * df["volume"]).copy().values
        vv = df["volume"].copy().values
        pv[:start_idx] = 0.0; vv[:start_idx] = 0.0
        cpv = np.cumsum(pv); cv = np.cumsum(vv)+1e-12
        return pd.Series(cpv / cv, index=df.index)

    def _vwap(self, df):
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        pv = (tp * df["volume"]).cumsum()
        vv = df["volume"].cumsum() + 1e-12
        return pv / vv

    def _build_features(self, raw_df):
        df = raw_df.copy().reset_index(drop=True)
        if "volume" not in df.columns:
            df["volume"] = df.get("tick_volume", 0)

        if len(df) < max(self.seq_len*3, 80):
            raise ValueError("Insufficient bars for feature building")
        df = df.tail(max(self.seq_len*3, 120)).reset_index(drop=True)

        # basics
        df["range"] = df["high"] - df["low"]
        df["body"] = (df["close"] - df["open"]).abs()
        df["dir"] = np.sign(df["close"] - df["open"]).fillna(0).astype(int)
        df["mid"] = (df["high"] + df["low"]) / 2
        df["typ"] = (df["high"] + df["low"] + df["close"]) / 3.0

        df["atr5"] = df["range"].rolling(5).mean()
        df["atr20"] = df["range"].rolling(20).mean()
        df["atr_ratio"] = df["atr5"] / (df["atr20"] + 1e-12)
        df["vol_z20"] = self._rolling_z(df["range"], 20).fillna(0.0)
        df["vwap"] = self._vwap(df)

        df["awap_session_start"] = ( ( (df["high"] + df["low"] + df["close"]) / 3.0 ) * df["volume"] ).cumsum() / (df["volume"].cumsum()+1e-12)
        idx_max = df["high"].idxmax(); idx_min = df["low"].idxmin()
        df["awap_from_high"] = self._anchored_vwap(df, idx_max)
        df["awap_from_low"] = self._anchored_vwap(df, idx_min)

        df["h1_close_proxy"] = df["close"].rolling(4, min_periods=1).mean()
        df["h1_slope"] = df["h1_close_proxy"].diff().fillna(0.0)
        df["m15_slope"] = df["close"].diff().fillna(0.0)
        df["tf_alignment"] = np.sign(df["h1_slope"].rolling(5, min_periods=1).mean()) * np.sign(df["m15_slope"])

        df["bar_idx"] = np.arange(len(df))
        df["session_pos"] = df["bar_idx"] / max(1, (len(df)-1))
        df["phase_open"] = (df["session_pos"] < 0.2).astype(int)
        df["phase_mid"] = ((df["session_pos"] >= 0.2) & (df["session_pos"] < 0.8)).astype(int)
        df["phase_close"] = (df["session_pos"] >= 0.8).astype(int)

        df["vol_mean5"] = df["volume"].rolling(5).mean()
        df["vol_z5"] = self._rolling_z(df["volume"], 5).fillna(0.0)
        df["liquidity_score"] = df["vol_mean5"] / (df["atr5"] + 1e-12)

        df["dir_entropy_10"] = - ( (df["dir"].rolling(10).apply(lambda x: (x==1).mean(), raw=True) *
                                   np.log2(df["dir"].rolling(10).apply(lambda x: (x==1).mean(), raw=True)+1e-9))
                                 + (df["dir"].rolling(10).apply(lambda x: (x==-1).mean(), raw=True) *
                                   np.log2(df["dir"].rolling(10).apply(lambda x: (x==-1).mean(), raw=True)+1e-9)) ).fillna(0.0)

        df["vwap_dev"] = df["close"] - df["vwap"]
        df["vwap_dev_z"] = self._rolling_z(df["vwap_dev"], 20).fillna(0.0)
        df["awap_gap_high"] = df["close"] - df["awap_from_high"]
        df["awap_gap_low"] = df["close"] - df["awap_from_low"]

        df["liquidity_x_vwapdev"] = df["liquidity_score"] * df["vwap_dev_z"]

        df = df.fillna(0.0).tail(self.seq_len).reset_index(drop=True)

        flat = {}
        per_bar_cols = ["close","vwap","mid","typ","range","body","dir","volume","session_pos","phase_open","phase_mid","phase_close"]
        for i in range(self.seq_len):
            for c in per_bar_cols:
                flat[f"{c}_t-{i}"] = float(df.iloc[i].get(c, 0.0))

        last_cols = [
            "atr5","atr20","atr_ratio","vol_z20","vwap_dev","vwap_dev_z",
            "h1_slope","m15_slope","tf_alignment","liquidity_score",
            "vol_mean5","vol_z5","dir_entropy_10","awap_gap_high","awap_gap_low",
            "liquidity_x_vwapdev"
        ]
        last = df.iloc[-1]
        for c in last_cols:
            flat[c] = float(last.get(c, 0.0))

        flat["mean_close"] = float(df["close"].mean())
        flat["std_range"] = float(df["range"].std())
        flat["pct_positive"] = float((df["dir"]>0).mean())

        feat_df = pd.DataFrame([flat])
        feat_df = feat_df.reindex(columns=self.feature_cols, fill_value=0.0)
        return feat_df

    def predict(self, symbol="EURUSDm", timeframe=mt5.TIMEFRAME_M15, n_bars=300):
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
        if rates is None or len(rates) < max(120, self.seq_len+20):
            return {"error":"Insufficient data"}
        df = pd.DataFrame(rates)
        if "volume" not in df.columns:
            df["volume"] = df.get("real_volume", df.get("tick_volume", 0))

        feat_df = self._build_features(df)
        scaled = self.scaler.transform(feat_df.values)
        x = torch.tensor(scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outs = self.model(x)

        preds = [int(o.argmax(1).item()) for o in outs]
        probs = [torch.softmax(o, dim=1).cpu().numpy()[0].tolist() for o in outs]
        confs = [float(max(p)) for p in probs]

        keys = ["session_bias","tf_alignment_head","volatility_shift","session_phase","expected_move_cat","continuity","vol_regime"]
        return {
            **{k: preds[i] for i,k in enumerate(keys)},
            "probabilities": {k: probs[i] for i,k in enumerate(keys)},
            "confidences": {k: confs[i] for i,k in enumerate(keys)}
        }

    def shutdown(self): mt5.shutdown()

if __name__ == "__main__":
    p = MacroPredictor()
    out = p.predict("EURUSDm", mt5.TIMEFRAME_M15)
    print("Macro (no-events) prediction:", out)
    p.shutdown()
