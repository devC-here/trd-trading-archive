# momentum_deploy.py
# Live inference for the Momentum model
# Architecture and features match train_momentum.py + synthetic_momentum.py exactly
# Features : 30  |  Heads : 6  |  HEAD_SIZES : [3, 2, 4, 3, 3, 3]
# NOTE: momentum dataset uses NO label_ prefix and excludes "scenario" column
# ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

try:
    from utils import resource_path
except ImportError:
    def resource_path(p): return p

EPS     = 1e-12
N_BARS  = 300           # bars fetched from MT5 (need 120+ for Ichimoku warmup)
TRUNK   = 256
DROP    = 0.25


# ─────────────────────────────────────────────────────────────────────
# Architecture  ·  must be byte-identical to train_momentum.py
# ─────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, dim, drop=DROP):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(drop),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.seq(x) + x)


class MomentumNet(nn.Module):
    # Heads: next_direction(3), next_magnitude(2), next_regime_behaviour(4),
    #        next_volatility(3), next_structural_bias(3), next_momentum_quality(3)
    HEAD_SIZES = [3, 2, 4, 3, 3, 3]

    def __init__(self, input_dim, trunk=TRUNK, drop=DROP):
        super().__init__()
        half = trunk // 2
        self.stem = nn.Sequential(
            nn.Linear(input_dim, trunk),
            nn.BatchNorm1d(trunk), nn.GELU(), nn.Dropout(drop),
        )
        self.res = nn.Sequential(
            ResidualBlock(trunk, drop),
            ResidualBlock(trunk, drop),
        )
        self.mid = nn.Sequential(
            nn.Linear(trunk, half),
            nn.BatchNorm1d(half), nn.GELU(), nn.Dropout(drop * 0.7),
        )
        self.heads = nn.ModuleList([nn.Linear(half, s) for s in self.HEAD_SIZES])

    def forward(self, x):
        h = self.mid(self.res(self.stem(x)))
        return tuple(head(h) for head in self.heads)


# ─────────────────────────────────────────────────────────────────────
# Feature computation  ·  identical to synthetic_momentum.py
# ─────────────────────────────────────────────────────────────────────

def _rolling_z(series: pd.Series, window: int) -> pd.Series:
    return (series - series.rolling(window).mean()) / (series.rolling(window).std() + EPS)


def _rolling_autocorr(series: pd.Series, window: int, lag: int = 1) -> pd.Series:
    def _ac(arr):
        if len(arr) < lag + 2: return 0.0
        r = pd.Series(arr).autocorr(lag=lag)
        return r if np.isfinite(r) else 0.0
    return series.rolling(window).apply(_ac, raw=False)


def compute_momentum_features(df: pd.DataFrame) -> dict:
    """
    df : DataFrame with [open, high, low, close, volume]. Min 120 rows.
    Returns dict of 30 float features — keys identical to original.
    """
    df = df.copy().reset_index(drop=True)

    close  = df["close"].astype(float)
    high   = df["high"].astype(float)
    low    = df["low"].astype(float)
    volume = df["volume"].astype(float)

    tr       = pd.concat([high - low,
                           (high - close.shift(1)).abs(),
                           (low  - close.shift(1)).abs()], axis=1).max(axis=1)
    atr      = tr.rolling(14).mean()
    atr_safe = atr + EPS

    # A: Displacement
    ret_3          = close.pct_change(3)
    ret_3_atr_norm = (ret_3 / (atr_safe / (close + EPS))).fillna(0.0)
    ret_3_z        = _rolling_z(ret_3, 50).fillna(0.0)

    # B: Trend quality
    direction        = (close - close.shift(10)).abs()
    noise            = close.diff().abs().rolling(10).sum()
    efficiency_ratio = (direction / (noise + EPS)).fillna(0.0)

    # FIX: er_slope z-scored to match scale of other features
    # Original .diff(5) produced raw -1..+1 deltas, inconsistent with z-scored peers
    er_slope = _rolling_z(efficiency_ratio.diff(5).fillna(0.0), 50).fillna(0.0)

    ret_1 = close.pct_change(1).fillna(0.0)

    # FIX: vectorised lag-1 rolling correlation — replaces rolling.apply(autocorr)
    # which was ~74× slower and created a pd.Series per window call.
    # Formula: corr(x_t, x_{t-1}) over window W
    ret_1_lag    = ret_1.shift(1)
    roll_mean_x  = ret_1.rolling(30).mean()
    roll_mean_xl = ret_1_lag.rolling(30).mean()
    roll_cov     = ((ret_1 * ret_1_lag).rolling(30).mean()
                    - roll_mean_x * roll_mean_xl)
    roll_std_x   = ret_1.rolling(30).std()
    roll_std_xl  = ret_1_lag.rolling(30).std()
    momentum_persistence = (roll_cov / (roll_std_x * roll_std_xl + EPS)).fillna(0.0)

    # C: Acceleration
    ema_20         = close.ewm(span=20, adjust=False).mean()
    ema_slope      = ema_20.diff(5)
    ema_slope_norm = (ema_slope / atr_safe).fillna(0.0)
    ema_acc_norm   = (ema_slope.diff(3) / atr_safe).fillna(0.0)

    # D: Volatility regime
    atr_z           = _rolling_z(atr, 50).fillna(0.0)
    rng             = high - low
    range_expansion = (rng.rolling(3).mean() / (rng.rolling(20).mean() + EPS)).fillna(1.0)
    vol_regime_confirm = (atr_z * (range_expansion - 1.0)).clip(-4, 4).fillna(0.0)

    # E: Structure
    roll_min_20    = close.rolling(20).min()
    roll_max_20    = close.rolling(20).max()
    trend_position = ((close - roll_min_20) / (roll_max_20 - roll_min_20 + EPS)).fillna(0.5)
    rolling_high   = high.rolling(20).max().shift(1)
    break_strength = ((close - rolling_high) / atr_safe).clip(-3, 3).fillna(0.0)
    mean_50               = close.rolling(50).mean()
    mean_reversion_stretch = ((close - mean_50) / atr_safe).clip(-4, 4).fillna(0.0)

    # F: ADX
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

    # G: Ichimoku
    tenkan   = (high.rolling(9).max()  + low.rolling(9).min())  / 2
    kijun    = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    tk_spread       = ((tenkan - kijun) / atr_safe).clip(-3, 3).fillna(0.0)
    cloud_thickness = ((senkou_a - senkou_b).abs() / atr_safe).clip(0, 5).fillna(0.0)
    price_vs_cloud  = ((close - senkou_a) / atr_safe).clip(-4, 4).fillna(0.0)
    cloud_twist     = np.sign(senkou_a - senkou_b).fillna(0.0)

    # H: Donchian
    don_high     = high.rolling(20).max()
    don_low      = low.rolling(20).min()
    don_width    = don_high - don_low
    don_position = ((close - don_low) / (don_width + EPS)).fillna(0.5)
    don_width_z  = _rolling_z(don_width, 50).fillna(0.0)
    don_breakout = ((close - don_high.shift(1)) / atr_safe).clip(-3, 3).fillna(0.0)

    # I: Momentum physics
    ret_5           = close.pct_change(5)
    ret_1_norm      = (ret_1 / (atr_safe / (close + EPS)))
    ret_5_norm      = (ret_5 / (atr_safe / (close + EPS)))
    momentum_decay  = (ret_1_norm / ((ret_5_norm / 5) + EPS)).clip(-5, 5).fillna(0.0)
    tension         = (ret_3_atr_norm * (0.5 - trend_position)).clip(-4, 4).fillna(0.0)
    ret_10          = close.pct_change(10)
    ret_10_atr_norm = (ret_10 / (atr_safe / (close + EPS))).fillna(0.0)
    ret_10_z        = _rolling_z(ret_10, 50).fillna(0.0)

    # J: Multi-TF alignment
    ema_50          = close.ewm(span=50, adjust=False).mean()
    ema_50_slope    = ema_50.diff(10)
    slope_alignment = (np.sign(ema_slope) * np.sign(ema_50_slope)).fillna(0.0)

    # K: Participation
    volume_z         = _rolling_z(volume, 30).fillna(0.0)
    vol_displacement = (volume_z * ret_3_z).clip(-5, 5).fillna(0.0)

    i = -1
    return {
        "adx_slope":              float(adx_slope.iloc[i]),
        "adx_z":                  float(adx_z.iloc[i]),
        "atr_z":                  float(atr_z.iloc[i]),
        "break_strength":         float(break_strength.iloc[i]),
        "cloud_thickness":        float(cloud_thickness.iloc[i]),
        "cloud_twist":            float(cloud_twist.iloc[i]),
        "di_balance":             float(di_balance.iloc[i]),
        "don_breakout":           float(don_breakout.iloc[i]),
        "don_position":           float(don_position.iloc[i]),
        "don_width_z":            float(don_width_z.iloc[i]),
        "efficiency_ratio":       float(efficiency_ratio.iloc[i]),
        "ema_acc_norm":           float(ema_acc_norm.iloc[i]),
        "ema_slope_norm":         float(ema_slope_norm.iloc[i]),
        "er_slope":               float(er_slope.iloc[i]),
        "mean_reversion_stretch": float(mean_reversion_stretch.iloc[i]),
        "momentum_decay":         float(momentum_decay.iloc[i]),
        "momentum_persistence":   float(momentum_persistence.iloc[i]),
        "price_vs_cloud":         float(price_vs_cloud.iloc[i]),
        "range_expansion":        float(range_expansion.iloc[i]),
        "ret_10_atr_norm":        float(ret_10_atr_norm.iloc[i]),
        "ret_10_z":               float(ret_10_z.iloc[i]),
        "ret_3_atr_norm":         float(ret_3_atr_norm.iloc[i]),
        "ret_3_z":                float(ret_3_z.iloc[i]),
        "slope_alignment":        float(slope_alignment.iloc[i]),
        "tension":                float(tension.iloc[i]),
        "tk_spread":              float(tk_spread.iloc[i]),
        "trend_position":         float(trend_position.iloc[i]),
        "vol_displacement":       float(vol_displacement.iloc[i]),
        "vol_regime_confirm":     float(vol_regime_confirm.iloc[i]),
        "volume_z":               float(volume_z.iloc[i]),
    }


# ─────────────────────────────────────────────────────────────────────
# Label maps  ·  sourced from synthetic_momentum.py label_maps dict
# ─────────────────────────────────────────────────────────────────────

HEAD_LABELS = {
    "next_direction":        {0: "bearish",       1: "flat",          2: "bullish"},
    "next_magnitude":        {0: "small",         1: "large"},
    "next_regime_behaviour": {0: "chop",          1: "pullback",      2: "continuation", 3: "breakout"},
    "next_volatility":       {0: "contracting",   1: "normal",        2: "expanding"},
    "next_structural_bias":  {0: "bearish",       1: "neutral",       2: "bullish"},
    "next_momentum_quality": {0: "fading",        1: "stable",        2: "accelerating"},
}

HEAD_NAMES = list(HEAD_LABELS.keys())


# ─────────────────────────────────────────────────────────────────────
# Predictor
# ─────────────────────────────────────────────────────────────────────

class MomentumPredictor:
    def __init__(self,
                 model_path:  str = "momentum_best.pth",
                 scaler_path: str = "momentum_scaler.pkl",
                 n_bars:      int = N_BARS):

        self.n_bars = n_bars
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        bundle            = joblib.load(scaler_path)
        self.scaler       = bundle["scaler"]
        self.feature_cols = bundle["feature_names"]

        self.model = MomentumNet(input_dim=len(self.feature_cols)).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()

        if MT5_AVAILABLE:
            if not mt5.initialize():
                raise RuntimeError("MT5 initialization failed")

    def predict(self, symbol: str = "EURUSD", timeframe=None) -> dict:
        if not MT5_AVAILABLE:
            return {"error": "MetaTrader5 not installed"}
        if timeframe is None:
            timeframe = mt5.TIMEFRAME_H1
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, self.n_bars)
        if rates is None or len(rates) < 120:
            return {"error": f"Insufficient bars for {symbol}"}
        df = pd.DataFrame(rates)
        if "tick_volume" in df.columns and "volume" not in df.columns:
            df = df.rename(columns={"tick_volume": "volume"})
        return self._run(compute_momentum_features(df))

    def predict_from_df(self, df: pd.DataFrame) -> dict:
        if len(df) < 120:
            return {"error": "Need at least 120 bars for indicator warmup"}
        return self._run(compute_momentum_features(df))

    def _run(self, raw_feats: dict) -> dict:
        feat_df = pd.DataFrame([raw_feats]).reindex(columns=self.feature_cols, fill_value=0.0)
        scaled  = self.scaler.transform(feat_df.values)
        x       = torch.tensor(scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outs = self.model(x)

        result: dict = {}
        for name, out in zip(HEAD_NAMES, outs):
            probs    = torch.softmax(out, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            lmap     = HEAD_LABELS[name]
            result[name] = {
                "class":      pred_idx,
                "label":      lmap[pred_idx],
                "confidence": float(probs.max()),
                "probs":      {lmap[k]: round(float(probs[k]), 4) for k in range(len(probs))},
            }

        result["_summary"] = {
            "direction":    result["next_direction"]["label"],
            "magnitude":    result["next_magnitude"]["label"],
            "regime":       result["next_regime_behaviour"]["label"],
            "volatility":   result["next_volatility"]["label"],
            "structure":    result["next_structural_bias"]["label"],
            "momentum":     result["next_momentum_quality"]["label"],
            "signal":       result["next_direction"]["label"],
            "signal_conf":  result["next_direction"]["confidence"],
            "trade_active": bool(
                result["next_direction"]["class"] != 1           # not flat
                and result["next_regime_behaviour"]["class"] >= 1  # not chop
                and result["next_momentum_quality"]["class"] >= 1  # not fading
            ),
        }
        result["_features"] = raw_feats
        return result

    def shutdown(self):
        if MT5_AVAILABLE:
            mt5.shutdown()


# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pred   = MomentumPredictor()
    result = pred.predict("EURUSD", mt5.TIMEFRAME_M15)
    print("\n── Summary ──────────────────────────────────────────")
    for k, v in result["_summary"].items():
        print(f"  {k:<14}: {v}")
    print("\n── Head Details ─────────────────────────────────────")
    for name in HEAD_NAMES:
        h = result[name]
        print(f"  {name:<26} → {h['label']:<14} conf={h['confidence']:.4f}")
    pred.shutdown()
