"""
backtest_predictors.py
----------------------
Backtest-compatible wrappers for all 5 predictors.
Instead of pulling data from MT5 internally, each predictor
accepts a pre-fetched DataFrame slice (up to the current bar).
No lookahead: only rows[:i] are ever passed in.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

from utils import resource_path

# ──────────────────────────────────────────────
#  MOMENTUM
# ──────────────────────────────────────────────
from ta.momentum import RSIIndicator
from ta.trend import MACD

class MomentumOscillatorNet(nn.Module):
    def __init__(self, input_dim=14):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(0.2),
        )
        self.heads = nn.ModuleList([nn.Linear(64, 3) for _ in range(3)])

    def forward(self, x):
        x = self.shared(x)
        return [head(x) for head in self.heads]

def _compute_momentum_features(df):
    df = df.copy()
    rsi  = RSIIndicator(df["close"], window=14)
    macd = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["rsi_14"]          = rsi.rsi()
    df["rsi_zscore_20"]   = (df["rsi_14"] - df["rsi_14"].rolling(10).mean()) / (df["rsi_14"].rolling(10).std() + 1e-9)
    df["roc_3"]           = df["close"].pct_change(3)
    df["roc_5"]           = df["close"].pct_change(5)
    df["roc_slope_3"]     = df["roc_3"].diff(2)
    df["macd_line"]       = macd.macd()
    df["macd_signal"]     = macd.macd_signal()
    df["macd_hist"]       = macd.macd_diff()
    df["macd_hist_slope"] = df["macd_hist"].diff(1)
    df["return_3"]        = df["close"].pct_change(3)
    df["return_zscore_20"]= (df["return_3"] - df["return_3"].rolling(10).mean()) / (df["return_3"].rolling(10).std() + 1e-9)
    df["rsi_x_roc"]        = np.clip(df["rsi_14"] * df["roc_3"],       -2, 2)
    df["rsi_x_macd_hist"]  = np.clip(df["rsi_14"] * df["macd_hist"],   -5, 5)
    df["macd_hist_x_roc"]  = np.clip(df["macd_hist"] * df["roc_3"],    -5, 5)
    return df

class MomentumPredictorBT:
    FEATURE_COLS = [
        "rsi_14","rsi_zscore_20","roc_3","roc_5","roc_slope_3",
        "macd_line","macd_signal","macd_hist","macd_hist_slope",
        "return_3","return_zscore_20","rsi_x_roc","rsi_x_macd_hist","macd_hist_x_roc"
    ]

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = MomentumOscillatorNet(input_dim=14).to(self.device)
        self.model.load_state_dict(torch.load(resource_path("momentum_model.pth"), map_location=self.device))
        self.model.eval()
        self.scaler = joblib.load(resource_path("momentum_scaler.pkl"))

    def predict(self, df_slice: pd.DataFrame):
        """df_slice: all bars up to and including the current bar."""
        if len(df_slice) < 50:
            return {"error": "insufficient data"}
        df = _compute_momentum_features(df_slice).dropna()
        if df.empty:
            return {"error": "no valid rows after feature computation"}
        latest = df.iloc[-1:][self.FEATURE_COLS]
        scaled = self.scaler.transform(latest)
        x = torch.tensor(scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(x)
        preds = [int(torch.argmax(o).cpu()) for o in outputs]
        probs = [torch.softmax(o, dim=1).cpu().numpy()[0] for o in outputs]
        confs = [float(np.max(p)) for p in probs]
        return {"momentum_state": preds[0], "momentum_slope_zone": preds[1],
                "entry_bias": preds[2], "confidence": confs[2]}


# ──────────────────────────────────────────────
#  BREAKOUT
# ──────────────────────────────────────────────
from ta.volatility import BollingerBands

class BreakoutNet(nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(0.2),
        )
        self.heads = nn.ModuleList([nn.Linear(64, 3) for _ in range(3)])

    def forward(self, x):
        x = self.shared(x)
        return [head(x) for head in self.heads]

def _compute_breakout_features(df):
    df = df.copy()
    bb   = BollingerBands(df["close"], window=14, window_dev=2)
    macd = MACD(df["close"], window_slow=18, window_fast=6, window_sign=5)
    from ta.momentum import RSIIndicator
    rsi  = RSIIndicator(df["close"], window=9)

    df["boll_upper"]  = bb.bollinger_hband()
    df["boll_lower"]  = bb.bollinger_lband()
    df["boll_middle"] = bb.bollinger_mavg()
    df["atr"]         = (df["high"] - df["low"]).rolling(10).mean()
    df["boll_bandwidth"] = (df["boll_upper"] - df["boll_lower"]) / (df["atr"] + 1e-9)
    rng = (df["boll_upper"] - df["boll_lower"]).replace(0, 1e-9)
    df["boll_position"] = (df["close"] - df["boll_lower"]) / (rng + 1e-9)

    macd_line   = macd.macd()
    macd_signal = macd.macd_signal()
    macd_hist   = macd.macd_diff()
    df["macd_hist"] = macd_hist * 1.5
    df["macd_cross_flag"] = 1
    up_cross   = (macd_line > macd_signal) & (macd_line.shift(1) <= macd_signal.shift(1))
    down_cross = (macd_line < macd_signal) & (macd_line.shift(1) >= macd_signal.shift(1))
    df.loc[up_cross,   "macd_cross_flag"] = 2
    df.loc[down_cross, "macd_cross_flag"] = 0

    df["rsi_14"]   = rsi.rsi()
    df["rsi_zscore"] = (df["rsi_14"] - df["rsi_14"].rolling(20).mean()) / (df["rsi_14"].rolling(20).std() + 1e-9)
    df["range"]    = df["high"] - df["low"]
    df["range_2"]  = df["range"].rolling(2).mean()
    df["range_8"]  = df["range"].rolling(8).mean()
    df["compression_ratio"] = df["range_2"] / (df["range_8"] + 1e-9)
    df["squeeze_zscore"]    = (df["boll_bandwidth"] - df["boll_bandwidth"].rolling(20).mean()) / (df["boll_bandwidth"].rolling(20).std() + 1e-9)
    df["boll_macd"] = np.clip(df["boll_position"] * df["macd_hist"], -3, 3)
    df["boll_rsi"]  = np.clip(df["boll_position"] * df["rsi_zscore"], -3, 3)
    return df

class BreakoutPredictorBT:
    FEATURE_COLS = [
        "boll_bandwidth","boll_position","compression_ratio","squeeze_zscore",
        "macd_hist","macd_cross_flag","rsi_14","rsi_zscore","boll_macd","boll_rsi"
    ]

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = BreakoutNet(input_dim=10).to(self.device)
        self.model.load_state_dict(torch.load(resource_path("breakout_model.pth"), map_location=self.device))
        self.model.eval()
        self.scaler = joblib.load(resource_path("breakout_scaler.pkl"))

    def predict(self, df_slice: pd.DataFrame):
        if len(df_slice) < 50:
            return {"error": "insufficient data"}
        df = _compute_breakout_features(df_slice).dropna()
        if df.empty:
            return {"error": "no valid rows after feature computation"}
        latest = df.iloc[-1:][self.FEATURE_COLS]
        scaled = self.scaler.transform(latest)
        x = torch.tensor(scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(x)
        preds = [int(torch.argmax(o).cpu()) for o in outputs]
        probs = [torch.softmax(o, dim=1).cpu().numpy()[0] for o in outputs]
        confs = [float(np.max(p)) for p in probs]
        return {"squeeze_state": preds[0], "breakout_bias": preds[1],
                "entry_viability": preds[2], "confidence": confs[2]}


# ──────────────────────────────────────────────
#  MICROSTRUCTURE
# ──────────────────────────────────────────────
class MicrostructureNet(nn.Module):
    def __init__(self, input_dim=15):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(0.2),
        )
        self.heads = nn.ModuleList([nn.Linear(64, 3) for _ in range(3)])

    def forward(self, x):
        x = self.shared(x)
        return [head(x) for head in self.heads]

def _compute_micro_features(df):
    df = df.copy()
    df["range"]     = df["high"] - df["low"] + 1e-9
    df["body"]      = (df["close"] - df["open"]).abs()
    df["wick"]      = df["high"] - df[["close","open"]].max(axis=1)
    df["tail"]      = df[["close","open"]].min(axis=1) - df["low"]
    df["body_ratio"] = (df["body"] / df["range"]) ** 0.9
    df["wick_ratio"] = (df["wick"] / df["range"]) ** 1.2
    df["tail_ratio"] = (df["tail"] / df["range"]) ** 1.2
    df["body_direction"] = np.sign(df["close"] - df["open"])
    df["body_dir"] = df["body"] * df["body_direction"]
    df["tail_dir"] = df["tail"] * df["body_direction"]
    df["wick_dir"] = df["wick"] * df["body_direction"]
    df["ret1"]            = df["close"].pct_change()
    df["ret1_zscore_10"]  = (df["ret1"] - df["ret1"].rolling(5).mean()) / (df["ret1"].rolling(5).std() + 1e-9)
    df["range_zscore_10"] = (df["range"] - df["range"].rolling(5).mean()) / (df["range"].rolling(5).std() + 1e-9)
    df["ret1_body_ratio"] = df["ret1"] * df["body_ratio"]
    return df

class MicrostructurePredictorBT:
    FEATURE_COLS = [
        "body","wick","tail","body_ratio","wick_ratio","tail_ratio",
        "body_direction","body_dir","tail_dir","wick_dir",
        "ret1","ret1_zscore_10","range","range_zscore_10","ret1_body_ratio"
    ]

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = MicrostructureNet(input_dim=15).to(self.device)
        self.model.load_state_dict(torch.load(resource_path("microstructure_model.pth"), map_location=self.device))
        self.model.eval()
        self.scaler = joblib.load(resource_path("microstructure_scaler.pkl"))

    def predict(self, df_slice: pd.DataFrame):
        if len(df_slice) < 20:
            return {"error": "insufficient data"}
        df = _compute_micro_features(df_slice).dropna()
        if df.empty:
            return {"error": "no valid rows after feature computation"}
        latest = df.iloc[-1:][self.FEATURE_COLS]
        scaled = self.scaler.transform(latest)
        x = torch.tensor(scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(x)
        preds = [int(torch.argmax(o).cpu()) for o in outputs]
        probs = [torch.softmax(o, dim=1).cpu().numpy()[0] for o in outputs]
        confs = [float(np.max(p)) for p in probs]
        return {"burst_type": preds[0], "tail_pressure_direction": preds[1],
                "micro_risk_zone": preds[2], "confidence": confs[2]}


# ──────────────────────────────────────────────
#  CANDLESTICK SEQUENCE
# ──────────────────────────────────────────────
class CandlestickSequenceNet(nn.Module):
    def __init__(self, input_dim=35):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(0.2),
        )
        self.heads = nn.ModuleList([nn.Linear(64, 5), nn.Linear(64, 3), nn.Linear(64, 3)])

    def forward(self, x):
        x = self.shared(x)
        return [head(x) for head in self.heads]

def _engineer_candle_features(df):
    df = df.copy()
    df["range"]    = df["high"] - df["low"] + 1e-9
    df["body"]     = (df["close"] - df["open"]).abs()
    df["wick"]     = df["high"] - df[["open","close"]].max(axis=1)
    df["tail"]     = df[["open","close"]].min(axis=1) - df["low"]
    df["body_dir"] = np.sign(df["close"] - df["open"])
    df["body_ratio"] = (df["body"] / df["range"]) ** 0.9
    df["tail_ratio"] = (df["tail"] / df["range"]) ** 1.2
    df["wick_ratio"] = (df["wick"] / df["range"]) ** 1.2
    return df

def _get_sequence_features(df, seq_len=5):
    out = pd.DataFrame()
    for i in range(seq_len):
        shifted = df.shift(i).add_suffix(f"_t-{i}")
        out = pd.concat([out, shifted], axis=1)
    return out.dropna()

class CandlestickSequencePredictorBT:
    FEATURE_COLS = [
        f"{feat}_t-{i}" for i in range(5)
        for feat in ["body","wick","tail","body_ratio","tail_ratio","wick_ratio","body_dir"]
    ]

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = CandlestickSequenceNet(input_dim=35).to(self.device)
        self.model.load_state_dict(torch.load(resource_path("candle_seq_model.pth"), map_location=self.device))
        self.model.eval()
        self.scaler = joblib.load(resource_path("candle_seq_scaler.pkl"))

    def predict(self, df_slice: pd.DataFrame):
        if len(df_slice) < 10:
            return {"error": "insufficient data"}
        df     = _engineer_candle_features(df_slice)
        seq_df = _get_sequence_features(df)
        if seq_df.empty:
            return {"error": "not enough valid sequence"}
        latest = seq_df.iloc[-1:][self.scaler.feature_names_in_]
        scaled = self.scaler.transform(latest)
        x = torch.tensor(scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(x)
        preds = [int(torch.argmax(o).cpu()) for o in outputs]
        probs = [torch.softmax(o, dim=1).cpu().numpy()[0] for o in outputs]
        confs = [float(np.max(p)) for p in probs]
        return {"sequence_type": preds[0], "candle_dominance": preds[1],
                "pattern_strength": preds[2], "confidence": confs[2]}


# ──────────────────────────────────────────────
#  VWAP / VOLUME
# ──────────────────────────────────────────────
class VwapVolumeNet(nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(0.2),
        )
        self.heads = nn.ModuleList([nn.Linear(64, 3) for _ in range(3)])

    def forward(self, x):
        x = self.shared(x)
        return [head(x) for head in self.heads]

def _compute_vwap_features(df):
    df = df.copy()

    # volume column normalisation (MT5 uses tick_volume)
    if "tick_volume" in df.columns and "volume" not in df.columns:
        df["volume"] = df["tick_volume"]

    # Session VWAP — daily reset
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    session_id    = pd.to_datetime(df["time"], unit="s").dt.date
    cum_tp_vol    = (typical_price * df["volume"]).groupby(session_id).cumsum()
    cum_vol       = df["volume"].groupby(session_id).cumsum()
    df["vwap"]    = cum_tp_vol / (cum_vol + 1e-9)

    # ATR for normalisation
    high_low    = df["high"] - df["low"]
    high_close  = np.abs(df["high"] - df["close"].shift())
    low_close   = np.abs(df["low"]  - df["close"].shift())
    true_range  = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr         = true_range.rolling(14).mean() + 1e-9

    df["close_minus_vwap"]    = (df["close"] - df["vwap"]) / atr
    df["is_above_vwap"]       = (df["close"] > df["vwap"]).astype(int)
    df["vwap_distance_ratio"] = np.abs(df["close"] - df["vwap"]) / atr

    vol_rolling = df["volume"].rolling(10)
    df["volume_z"]              = (df["volume"] - vol_rolling.mean()) / (vol_rolling.std() + 1e-9)
    df["volume_rolling_change"] = df["volume"].diff(1) / (vol_rolling.std() + 1e-9)

    df["volatility"]           = df["close"].pct_change().rolling(3).std()
    df["vwap_x_volume"]        = df["close_minus_vwap"] * df["volume_z"]
    df["vwap_vol_displacement"] = df["close_minus_vwap"] * df["volume_z"]
    df["volatility_x_volume"]  = df["volatility"] * df["volume_z"]

    return df

class VwapVolumePredictorBT:
    FEATURE_COLS = [
        "close_minus_vwap", "is_above_vwap", "vwap_distance_ratio",
        "volume", "volume_z", "volume_rolling_change", "volatility",
        "vwap_x_volume", "vwap_vol_displacement", "volatility_x_volume",
    ]

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = VwapVolumeNet(input_dim=10).to(self.device)
        self.model.load_state_dict(torch.load(resource_path("vwap_volume_model.pth"), map_location=self.device))
        self.model.eval()
        self.scaler = joblib.load(resource_path("vwap_volume_scaler.pkl"))

    def predict(self, df_slice: pd.DataFrame):
        if len(df_slice) < 50:
            return {"error": "insufficient data"}
        df = _compute_vwap_features(df_slice).dropna()
        if df.empty:
            return {"error": "no valid rows after feature computation"}
        latest = df.iloc[-1:][self.FEATURE_COLS]
        scaled = self.scaler.transform(latest)
        x = torch.tensor(scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(x)
        preds = [int(torch.argmax(o).cpu()) for o in outputs]
        probs = [torch.softmax(o, dim=1).cpu().numpy()[0] for o in outputs]
        confs = [float(np.max(p)) for p in probs]
        return {"vwap_zone": preds[0], "volume_pressure": preds[1],
                "liquidity_structure": preds[2], "confidence": confs[2]}
