# momentum_deploy.py  ·  v2
# ─────────────────────────────────────────────────────────────────────────────
# Live inference wrapper for MomentumMultiHeadNet (30 features, 6 heads).
# Feature computation is byte-identical to synthetic_momentum.py.
# Supports both MT5 live feed and offline DataFrame input.
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

EPS = 1e-12

# ─────────────────────────────────────────────
# MODEL — must be identical to train_momentum.py
# ─────────────────────────────────────────────
class ResidualBlock(nn.Module):
    def __init__(self, dim: int, drop: float = 0.25):
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


class MomentumMultiHeadNet(nn.Module):
    def __init__(self, input_dim: int, trunk_dim: int = 256, drop: float = 0.25):
        super().__init__()
        half = trunk_dim // 2
        self.stem = nn.Sequential(
            nn.Linear(input_dim, trunk_dim),
            nn.BatchNorm1d(trunk_dim),
            nn.GELU(),
            nn.Dropout(drop),
        )
        self.res = nn.Sequential(
            ResidualBlock(trunk_dim, drop),
            ResidualBlock(trunk_dim, drop),
        )
        self.mid = nn.Sequential(
            nn.Linear(trunk_dim, half),
            nn.BatchNorm1d(half),
            nn.GELU(),
            nn.Dropout(drop * 0.7),
        )
        self.head_momentum_state   = nn.Linear(half, 3)
        self.head_momentum_slope   = nn.Linear(half, 3)
        self.head_regime           = nn.Linear(half, 4)
        self.head_volatility_state = nn.Linear(half, 3)
        self.head_structure_bias   = nn.Linear(half, 3)
        self.head_entry_quality    = nn.Linear(half, 3)

    def forward(self, x):
        h = self.mid(self.res(self.stem(x)))
        return (
            self.head_momentum_state(h),
            self.head_momentum_slope(h),
            self.head_regime(h),
            self.head_volatility_state(h),
            self.head_structure_bias(h),
            self.head_entry_quality(h),
        )


# ─────────────────────────────────────────────
# FEATURE COMPUTATION
# !! MUST REMAIN BYTE-IDENTICAL TO synthetic_momentum.py !!
# Any change here must be mirrored there and vice versa.
# ─────────────────────────────────────────────
def _rolling_z(series: pd.Series, window: int) -> pd.Series:
    return (series - series.rolling(window).mean()) / (series.rolling(window).std() + EPS)


def _rolling_autocorr(series: pd.Series, window: int, lag: int = 1) -> pd.Series:
    def _ac(arr):
        if len(arr) < lag + 2:
            return 0.0
        s = pd.Series(arr)
        r = s.autocorr(lag=lag)
        return r if np.isfinite(r) else 0.0
    return series.rolling(window).apply(_ac, raw=False)


def compute_momentum_features(df: pd.DataFrame) -> dict:
    """
    30 momentum features from OHLCV DataFrame (min 120 bars recommended).
    Returns dict of float values at the last bar.
    """
    df = df.copy().reset_index(drop=True)

    close  = df["close"].astype(float)
    high   = df["high"].astype(float)
    low    = df["low"].astype(float)
    open_  = df["open"].astype(float)
    volume = df["volume"].astype(float)

    # ── BASE: ATR ─────────────────────────────────────────────────
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr      = tr.rolling(14).mean()
    atr_safe = atr + EPS

    # ── A: DISPLACEMENT ───────────────────────────────────────────
    ret_3          = close.pct_change(3)
    ret_3_atr_norm = (ret_3 / (atr_safe / (close + EPS))).fillna(0.0)
    ret_3_z        = _rolling_z(ret_3, 50).fillna(0.0)

    # ── B: TREND QUALITY ──────────────────────────────────────────
    direction        = (close - close.shift(10)).abs()
    noise            = close.diff().abs().rolling(10).sum()
    efficiency_ratio = (direction / (noise + EPS)).fillna(0.0)
    er_slope         = efficiency_ratio.diff(5).fillna(0.0)

    ret_1                = close.pct_change(1).fillna(0.0)
    momentum_persistence = _rolling_autocorr(ret_1, 30).fillna(0.0)

    # ── C: ACCELERATION ───────────────────────────────────────────
    ema_20         = close.ewm(span=20, adjust=False).mean()
    ema_slope      = ema_20.diff(5)
    ema_slope_norm = (ema_slope / atr_safe).fillna(0.0)
    ema_acc_norm   = (ema_slope.diff(3) / atr_safe).fillna(0.0)

    # ── D: VOLATILITY REGIME ──────────────────────────────────────
    atr_z           = _rolling_z(atr, 50).fillna(0.0)
    rng             = high - low
    range_expansion = (rng.rolling(3).mean() / (rng.rolling(20).mean() + EPS)).fillna(1.0)
    vol_regime_confirm = (atr_z * (range_expansion - 1.0)).clip(-4, 4).fillna(0.0)

    # ── E: STRUCTURE ──────────────────────────────────────────────
    roll_min_20    = close.rolling(20).min()
    roll_max_20    = close.rolling(20).max()
    trend_position = ((close - roll_min_20) / (roll_max_20 - roll_min_20 + EPS)).fillna(0.5)
    rolling_high   = high.rolling(20).max().shift(1)
    break_strength = ((close - rolling_high) / atr_safe).clip(-3, 3).fillna(0.0)
    mean_50                = close.rolling(50).mean()
    mean_reversion_stretch = ((close - mean_50) / atr_safe).clip(-4, 4).fillna(0.0)

    # ── F: TREND STRENGTH — ADX ───────────────────────────────────
    up_move   = high.diff()
    down_move = (-low.diff())
    plus_dm   = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm  = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    sm_plus   = plus_dm.ewm(span=14, adjust=False).mean()
    sm_minus  = minus_dm.ewm(span=14, adjust=False).mean()
    sm_atr    = atr.ewm(span=14, adjust=False).mean() + EPS
    plus_di   = 100 * (sm_plus  / sm_atr)
    minus_di  = 100 * (sm_minus / sm_atr)
    dx        = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + EPS)
    adx       = dx.ewm(span=14, adjust=False).mean()
    adx_z     = _rolling_z(adx, 50).fillna(0.0)
    adx_slope = _rolling_z(adx, 30).diff(3).fillna(0.0)
    di_balance = ((plus_di - minus_di) / (plus_di + minus_di + EPS)).fillna(0.0)

    # ── G: ICHIMOKU GEOMETRY ──────────────────────────────────────
    tenkan   = (high.rolling(9).max()  + low.rolling(9).min())  / 2
    kijun    = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)

    tk_spread       = ((tenkan - kijun) / atr_safe).clip(-3, 3).fillna(0.0)
    cloud_thickness = ((senkou_a - senkou_b).abs() / atr_safe).clip(0, 5).fillna(0.0)
    price_vs_cloud  = ((close - senkou_a) / atr_safe).clip(-4, 4).fillna(0.0)
    cloud_twist     = np.sign(senkou_a - senkou_b).fillna(0.0)

    # ── H: DONCHIAN CHANNELS ──────────────────────────────────────
    don_high  = high.rolling(20).max()
    don_low   = low.rolling(20).min()
    don_width = don_high - don_low

    don_position = ((close - don_low) / (don_width + EPS)).fillna(0.5)
    don_width_z  = _rolling_z(don_width, 50).fillna(0.0)
    don_breakout = ((close - don_high.shift(1)) / atr_safe).clip(-3, 3).fillna(0.0)

    # ── I: MOMENTUM PHYSICS ───────────────────────────────────────
    ret_5         = close.pct_change(5)
    ret_1_norm    = (ret_1 / (atr_safe / (close + EPS)))
    ret_5_norm    = (ret_5 / (atr_safe / (close + EPS)))
    momentum_decay = (ret_1_norm / ((ret_5_norm / 5) + EPS)).clip(-5, 5).fillna(0.0)

    tension = (ret_3_atr_norm * (0.5 - trend_position)).clip(-4, 4).fillna(0.0)

    ret_10          = close.pct_change(10)
    ret_10_atr_norm = (ret_10 / (atr_safe / (close + EPS))).fillna(0.0)
    ret_10_z        = _rolling_z(ret_10, 50).fillna(0.0)

    # ── J: MULTI-TF ALIGNMENT ─────────────────────────────────────
    ema_50         = close.ewm(span=50, adjust=False).mean()
    ema_50_slope   = ema_50.diff(10)
    slope_alignment = (np.sign(ema_slope) * np.sign(ema_50_slope)).fillna(0.0)

    # ── K: PARTICIPATION ──────────────────────────────────────────
    volume_z        = _rolling_z(volume, 30).fillna(0.0)
    vol_displacement = (volume_z * ret_3_z).clip(-5, 5).fillna(0.0)

    # ── COMPILE ───────────────────────────────────────────────────
    i = -1
    return {
        "ret_3_atr_norm":         float(ret_3_atr_norm.iloc[i]),
        "ret_3_z":                float(ret_3_z.iloc[i]),
        "efficiency_ratio":       float(efficiency_ratio.iloc[i]),
        "er_slope":               float(er_slope.iloc[i]),
        "momentum_persistence":   float(momentum_persistence.iloc[i]),
        "ema_slope_norm":         float(ema_slope_norm.iloc[i]),
        "ema_acc_norm":           float(ema_acc_norm.iloc[i]),
        "atr_z":                  float(atr_z.iloc[i]),
        "range_expansion":        float(range_expansion.iloc[i]),
        "vol_regime_confirm":     float(vol_regime_confirm.iloc[i]),
        "trend_position":         float(trend_position.iloc[i]),
        "break_strength":         float(break_strength.iloc[i]),
        "mean_reversion_stretch": float(mean_reversion_stretch.iloc[i]),
        "adx_z":                  float(adx_z.iloc[i]),
        "adx_slope":              float(adx_slope.iloc[i]),
        "di_balance":             float(di_balance.iloc[i]),
        "tk_spread":              float(tk_spread.iloc[i]),
        "cloud_thickness":        float(cloud_thickness.iloc[i]),
        "price_vs_cloud":         float(price_vs_cloud.iloc[i]),
        "cloud_twist":            float(cloud_twist.iloc[i]),
        "don_position":           float(don_position.iloc[i]),
        "don_width_z":            float(don_width_z.iloc[i]),
        "don_breakout":           float(don_breakout.iloc[i]),
        "momentum_decay":         float(momentum_decay.iloc[i]),
        "tension":                float(tension.iloc[i]),
        "ret_10_atr_norm":        float(ret_10_atr_norm.iloc[i]),
        "ret_10_z":               float(ret_10_z.iloc[i]),
        "slope_alignment":        float(slope_alignment.iloc[i]),
        "volume_z":               float(volume_z.iloc[i]),
        "vol_displacement":       float(vol_displacement.iloc[i]),
    }


# ─────────────────────────────────────────────
# OUTPUT MAPS
# ─────────────────────────────────────────────
HEAD_LABEL_MAPS = {
    "momentum_state":   {0: "down",        1: "neutral",    2: "up"},
    "momentum_slope":   {0: "falling",     1: "flat",       2: "rising"},
    "regime":           {0: "ranging",     1: "weak_trend", 2: "trending", 3: "breakout"},
    "volatility_state": {0: "contracting", 1: "normal",     2: "expanding"},
    "structure_bias":   {0: "bearish",     1: "neutral",    2: "bullish"},
    "entry_quality":    {0: "avoid",       1: "wait",       2: "go"},
}
HEAD_NAMES = list(HEAD_LABEL_MAPS.keys())


# ─────────────────────────────────────────────
# PREDICTOR
# ─────────────────────────────────────────────
class MomentumPredictor:
    def __init__(self,
                 model_path:  str = "momentum_best.pth",
                 scaler_path: str = "momentum_scaler.pkl",
                 n_bars:      int = 300):
        """
        n_bars: bars to pull from MT5 (need 120+ for all indicators to warm up).
                300 is safe for all rolling windows including Ichimoku 52+26.
        """
        self.n_bars = n_bars
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        bundle            = joblib.load(scaler_path)
        self.scaler       = bundle["scaler"]
        self.feature_cols = bundle["feature_names"]   # preserves training order

        self.model = MomentumMultiHeadNet(input_dim=len(self.feature_cols)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        if MT5_AVAILABLE:
            if not mt5.initialize():
                raise RuntimeError("MT5 initialization failed")

    def predict_from_df(self, df: pd.DataFrame) -> dict:
        """Offline inference from any OHLCV DataFrame."""
        if len(df) < 120:
            return {"error": "Need at least 120 bars for full indicator warmup"}
        return self._run_inference(compute_momentum_features(df))

    def predict(self, symbol: str = "EURUSD", timeframe=None) -> dict:
        """Live inference from MT5."""
        if not MT5_AVAILABLE:
            return {"error": "MetaTrader5 not installed"}
        if timeframe is None:
            timeframe = mt5.TIMEFRAME_M15

        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, self.n_bars)
        if rates is None or len(rates) < 120:
            return {"error": f"Insufficient data for {symbol}"}

        df = pd.DataFrame(rates)
        if "tick_volume" in df.columns and "volume" not in df.columns:
            df = df.rename(columns={"tick_volume": "volume"})

        return self._run_inference(compute_momentum_features(df))

    def _run_inference(self, raw_feats: dict) -> dict:
        feat_df = pd.DataFrame([raw_feats]).reindex(columns=self.feature_cols, fill_value=0.0)
        scaled  = self.scaler.transform(feat_df.values)
        x       = torch.tensor(scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(x)

        result = {}
        for i, name in enumerate(HEAD_NAMES):
            probs    = torch.softmax(logits[i], dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            lmap     = HEAD_LABEL_MAPS[name]
            result[name] = {
                "pred":  pred_idx,
                "label": lmap[pred_idx],
                "conf":  round(float(probs.max()), 4),
                "probs": {lmap[k]: round(float(probs[k]), 4) for k in range(len(probs))},
            }

        # Flat summary for context engine
        result["_summary"] = {
            "momentum":    result["momentum_state"]["label"],
            "slope":       result["momentum_slope"]["label"],
            "regime":      result["regime"]["label"],
            "vol":         result["volatility_state"]["label"],
            "structure":   result["structure_bias"]["label"],
            "signal":      result["entry_quality"]["label"],
            "signal_conf": result["entry_quality"]["conf"],
        }

        result["_features"] = raw_feats
        return result

    def shutdown(self):
        if MT5_AVAILABLE:
            mt5.shutdown()


# ─────────────────────────────────────────────
# EXAMPLE RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    pred = MomentumPredictor()

    if MT5_AVAILABLE:
        result = pred.predict("EURUSD", mt5.TIMEFRAME_M15)
    else:
        # Offline test: synthetic trending price series
        np.random.seed(99)
        n = 250
        ret = np.random.normal(0.0004, 0.0006, n)   # slight upward drift
        close = np.cumprod(1 + ret) * 1.0800
        df_test = pd.DataFrame({
            "open":   close * (1 + np.random.normal(0, 0.00015, n)),
            "high":   close * (1 + np.abs(np.random.normal(0, 0.00035, n))),
            "low":    close * (1 - np.abs(np.random.normal(0, 0.00035, n))),
            "close":  close,
            "volume": np.random.randint(400, 2200, n).astype(float),
        })
        result = pred.predict_from_df(df_test)

    print("\n── Summary ─────────────────────────────────────")
    for k, v in result["_summary"].items():
        print(f"  {k:<14}: {v}")

    print("\n── Head Details ────────────────────────────────")
    for name in HEAD_NAMES:
        h = result[name]
        print(f"  {name:<22} → {h['label']:<13} conf={h['conf']:.4f}  {h['probs']}")

    print("\n── Feature Values ──────────────────────────────")
    blocks = {
        "A Displacement":   ["ret_3_atr_norm","ret_3_z"],
        "B Trend Quality":  ["efficiency_ratio","er_slope","momentum_persistence"],
        "C Acceleration":   ["ema_slope_norm","ema_acc_norm"],
        "D Vol Regime":     ["atr_z","range_expansion","vol_regime_confirm"],
        "E Structure":      ["trend_position","break_strength","mean_reversion_stretch"],
        "F Trend Strength": ["adx_z","adx_slope","di_balance"],
        "G Ichimoku":       ["tk_spread","cloud_thickness","price_vs_cloud","cloud_twist"],
        "H Donchian":       ["don_position","don_width_z","don_breakout"],
        "I Physics":        ["momentum_decay","tension","ret_10_atr_norm","ret_10_z"],
        "J MTF Align":      ["slope_alignment"],
        "K Participation":  ["volume_z","vol_displacement"],
    }
    feats = result["_features"]
    for block, keys in blocks.items():
        vals = "  ".join(f"{k}={feats[k]:+.3f}" for k in keys)
        print(f"  [{block}]  {vals}")

    pred.shutdown()
