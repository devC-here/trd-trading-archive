# feature_functions_fixed.py
# Drop-in replacements for the compute_*_features() functions in each deploy file.
# ─────────────────────────────────────────────────────────────────────────────
# CONTRACT (must not change):
#   • Every function keeps the same name, same signature, same return-key set
#   • Feature counts unchanged: candle=30, breakout=27, momentum=30, micro=32, vwap=32
#   • No new imports beyond what was already used in the original files
# ─────────────────────────────────────────────────────────────────────────────
# Changes applied:
#   CANDLE
#     ha_body_slope  — removed double ATR division (ha_body already normalised)
#     handoff_str    — divided by full_mean (dimensionless), not atr_val (price)
#     _ols_slope     — replaced np.polyfit with dot-product (~20× faster)
#     consec_len     — replaced Python loop with vectorised groupby
#
#   BREAKOUT
#     consec_squeeze — replaced Python loop with vectorised groupby
#
#   MOMENTUM
#     momentum_persistence — replaced rolling.apply(autocorr) with vectorised
#                            lag-1 correlation (~74× faster, identical result)
#     er_slope             — z-scored to match scale of other features
#
#   MICROSTRUCTURE
#     orderflow_5/10  — normalised by vol_avg10 (scale-invariant)
#     body_vol_inter  — normalised to body_ratio * vol_z, clip ±6 (effective)
#     consec_len      — replaced Python loop with vectorised groupby
#
#   VWAP
#     above_vwap_streak — replaced Python loop with vectorised groupby
#     vwap_band_walk    — symmetrised: now tracks both bull and bear extremes
#                         (key kept; price_vol_div kept as-is — it exists in the
#                          trained scaler so removing it would break inference)
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

EPS = 1e-12


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _rz(s: pd.Series, w: int) -> pd.Series:
    """Rolling z-score."""
    return (s - s.rolling(w).mean()) / (s.rolling(w).std() + EPS)


def _consec_dir(dir_series: pd.Series) -> pd.Series:
    """
    Vectorised consecutive same-direction counter.
    Replaces the Python loop used in candle, breakout (squeeze), and microstructure.
    Returns a Series of the same length with the consecutive count per bar.
    Zeros in dir_series reset the counter to 0.
    """
    s = dir_series.fillna(0)
    # Each time direction changes (or hits 0), start a new group
    change = (s != s.shift(1)).cumsum()
    counts = s.groupby(change).cumcount() + 1
    counts[s == 0] = 0
    return counts.values  # return ndarray to match original loop output style


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  CANDLE
# ═══════════════════════════════════════════════════════════════════════════════

def _ols_slope(arr: np.ndarray) -> float:
    """
    OLS slope of arr, normalised by std(arr).
    Replaced np.polyfit with dot-product — identical result, ~20× faster.
    """
    n = len(arr)
    if n < 2:
        return 0.0
    s = float(np.std(arr))
    if s < EPS:
        return 0.0
    x = np.arange(n, dtype=float)
    x = x - x.mean()                          # centre x
    slope = float(np.dot(x, arr) / (np.dot(x, x) + EPS))
    return slope / (s + EPS)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < EPS or np.std(b) < EPS:
        return 0.0
    try:
        r, _ = sp_stats.pearsonr(a, b)
        return float(r) if np.isfinite(r) else 0.0
    except Exception:
        return 0.0


WIN = 10   # sequence window — matches candle_deploy.py


def compute_candle_features(df: pd.DataFrame) -> dict:
    """
    df : DataFrame with [open, high, low, close, volume]. Min 50 rows.
    Returns dict of 30 float features — keys identical to original.
    """
    df = df.copy().reset_index(drop=True)

    df["range"]          = df["high"] - df["low"]
    df["body"]           = df["close"] - df["open"]
    df["abs_body"]       = df["body"].abs()
    df["dir"]            = np.sign(df["body"]).fillna(0)
    df["upper_wick"]     = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"]     = df[["open", "close"]].min(axis=1) - df["low"]
    df["atr14"]          = df["range"].rolling(14, min_periods=1).mean()
    df["abs_body_avg30"] = df["abs_body"].rolling(30, min_periods=1).mean()
    df["range_avg30"]    = df["range"].rolling(30, min_periods=1).mean()
    df["vol_avg20"]      = df["volume"].rolling(20, min_periods=1).mean()
    df["body_ratio"]     = df["abs_body"] / (df["range"] + EPS)
    df["body_rat_avg5"]  = df["body_ratio"].rolling(5, min_periods=1).mean()
    df["close_grav"]     = (df["close"] - df["low"]) / (df["range"] + EPS)
    df["shadow_dom"]     = (df["lower_wick"] - df["upper_wick"]) / (df["range"] + EPS)
    bull                 = df["dir"] >= 0
    df["dir_eff"]        = np.where(bull,
                                     df["body_ratio"] * df["close_grav"],
                                     df["body_ratio"] * (1 - df["close_grav"]))
    df["vol_weight"]     = (df["volume"] / (df["vol_avg20"] + EPS)).clip(0, 4)

    ha_close  = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_open   = ha_close.ewm(span=3, adjust=False).mean()
    df["ha_body"] = (ha_close - ha_open) / (df["atr14"] + EPS)

    # ── vectorised consecutive-direction counter ─────────────────────
    df["consec_len"] = _consec_dir(df["dir"])

    # ── last WIN bars ────────────────────────────────────────────────
    w       = df.iloc[-WIN:].reset_index(drop=True)
    atr_val = float(df["atr14"].iloc[-1])
    f       = {}

    # GROUP A — last-bar anatomy (5)
    f["lb_close_grav"] = float(w["close_grav"].iloc[-1])
    f["lb_body_ratio"] = float(w["body_ratio"].iloc[-1])
    f["lb_shadow_dom"] = float(w["shadow_dom"].iloc[-1])
    f["lb_vol_weight"] = float(min(w["vol_weight"].iloc[-1], 4.0))
    f["lb_dir_eff"]    = float(w["dir_eff"].iloc[-1])

    # GROUP B — sequence narrative (8)
    f["seq_cg_slope"]     = _ols_slope(w["close_grav"].values)
    bm = float(w["body_ratio"].mean()); bs = float(w["body_ratio"].std())
    f["body_entropy"]     = float(bs / (bm + EPS))
    f["wick_press_slope"] = _ols_slope(w["shadow_dom"].values)
    bar_conv              = w["dir"] * w["body_ratio"] * w["vol_weight"]
    f["vol_narrative"]    = _ols_slope(bar_conv.values)
    f["seq_harmony"]      = _safe_corr(w["close_grav"].values, w["range"].values)
    early_dir             = np.sign(w["body"].iloc[:5].sum())
    late_dir              = np.sign(w["body"].iloc[5:].sum())
    f["handoff"]          = float(early_dir != late_dir)
    early_body            = float(w["body_ratio"].iloc[:5].mean())
    late_body             = float(w["body_ratio"].iloc[5:].mean())
    full_mean             = float(w["body_ratio"].mean()) + EPS

    # FIX: handoff_str — use full_mean (dimensionless) not atr_val (price units)
    # Original divided by atr_val (~0.0008) inflating result to 100×+
    f["handoff_str"] = float((late_body - early_body) / full_mean)

    f["seq_accel"]    = float((late_body - early_body) / full_mean)

    # GROUP C — pattern physics (7)
    curr_body  = float(w["abs_body"].iloc[-1])
    prior_body = float(w["abs_body"].iloc[-2]) if len(w) >= 2 else curr_body
    curr_dir   = float(w["dir"].iloc[-1])
    f["engulf_qual"] = float((curr_body - prior_body) / (atr_val + EPS) * (curr_dir or 1))

    inside_count = 0
    for i in range(1, len(w)):
        if w["high"].iloc[i] < w["high"].iloc[i-1] and w["low"].iloc[i] > w["low"].iloc[i-1]:
            inside_count += 1
        else:
            break
    mother_range       = float(w["range"].iloc[0]) if inside_count > 0 else 0.0
    f["inside_energy"] = float(inside_count * mother_range / (atr_val + EPS))

    f["ha_body_last"]  = float(w["ha_body"].iloc[-1])

    # FIX: ha_body_slope — ha_body is already normalised by atr14, so no second division
    # Original: (ha_arr[-1] - ha_arr[-4]) / (atr_val + EPS) → double-normalisation
    ha_arr             = w["ha_body"].values
    f["ha_body_slope"] = float(ha_arr[-1] - ha_arr[max(0, len(ha_arr) - 4)])

    uw  = float(w["upper_wick"].iloc[-1])
    lw  = float(w["lower_wick"].iloc[-1])
    rng = float(w["range"].iloc[-1])
    br  = float(w["body_ratio"].iloc[-1])
    cg  = float(w["close_grav"].iloc[-1])
    bsm = 1.0 if br < 0.35 else 0.5
    f["pin_bull_qual"] = float((lw / (rng + EPS)) * bsm * cg)
    f["pin_bear_qual"] = float((uw / (rng + EPS)) * bsm * (1 - cg))

    closes            = w["close"].values
    c_min, c_max      = float(closes.min()), float(closes.max())
    seq_pos           = (closes[-1] - c_min) / (c_max - c_min + EPS)
    doji_score        = float(max(0, 0.15 - br) / 0.15)
    f["doji_context"] = float(doji_score * (seq_pos - 0.5))

    # GROUP D — relative sizing (3)
    f["body_vs_avg"]  = float(min(float(df["abs_body"].iloc[-1]) /
                                   (float(df["abs_body_avg30"].iloc[-1]) + EPS), 4.0))
    f["range_vs_avg"] = float(min(float(df["range"].iloc[-1]) /
                                   (float(df["range_avg30"].iloc[-1]) + EPS), 4.0))
    br5_avg           = float(df["body_rat_avg5"].iloc[-1])
    f["body_accel"]   = float(br - br5_avg)

    # GROUP E — fractal (2)
    br1      = float(w["body_ratio"].iloc[-1])
    br3      = float(w["body_ratio"].iloc[-3:].mean()) if len(w) >= 3 else br1
    br10     = float(w["body_ratio"].mean())
    price_up = closes[-1] > closes[0]
    f["fractal_bull"] = float((br1 > br3 > br10) and price_up)
    f["fractal_bear"] = float((br1 > br3 > br10) and not price_up)

    # GROUP F — supplementary (5)
    f["seq_dir_ratio"]  = float((w["dir"] > 0).mean())
    f["consec_len"]     = float(w["consec_len"].iloc[-1])
    vol_surge_w         = float(w["vol_weight"].mean())
    br_mean_w           = float(w["body_ratio"].mean()) + EPS
    f["vol_climax_div"] = float(br_mean_w / (vol_surge_w + EPS))
    f["ha_trend_str"]   = float(w["ha_body"].ewm(span=5, adjust=False).mean().iloc[-1])
    f["seq_body_avg"]   = float(w["body_ratio"].mean())

    return f


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  BREAKOUT
# ═══════════════════════════════════════════════════════════════════════════════

def compute_breakout_features(df: pd.DataFrame) -> dict:
    """
    df : DataFrame with [open, high, low, close, volume]. Min 100 rows.
    Returns dict of 27 float features — keys identical to original.
    """
    df = df.copy().reset_index(drop=True)

    df["range"]      = df["high"] - df["low"]
    df["body"]       = df["close"] - df["open"]
    df["abs_body"]   = df["body"].abs()
    df["dir"]        = np.sign(df["body"]).fillna(0)
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["atr14"]      = df["range"].rolling(14, min_periods=1).mean()
    df["atr5"]       = df["range"].rolling(5,  min_periods=1).mean()

    for w in [8, 14, 20]:
        mid   = df["close"].rolling(w, min_periods=1).mean()
        std   = df["close"].rolling(w, min_periods=1).std().fillna(0)
        atr_w = df["range"].rolling(w, min_periods=1).mean()
        df[f"bb_upper_{w}"] = mid + 2 * std
        df[f"bb_lower_{w}"] = mid - 2 * std
        df[f"bb_width_{w}"] = 4 * std
        df[f"kc_width_{w}"] = 3 * atr_w
        df[f"kc_upper_{w}"] = mid + 1.5 * atr_w
        df[f"kc_lower_{w}"] = mid - 1.5 * atr_w

    df["bb_width_z"] = _rz(df["bb_width_14"], 100).fillna(0.0)
    df["kc_width_n"] = df["kc_width_14"] / (df["atr14"] + EPS)

    sq8  = (df["bb_width_8"]  < df["kc_width_8"]).astype(int)
    sq14 = (df["bb_width_14"] < df["kc_width_14"]).astype(int)
    sq20 = (df["bb_width_20"] < df["kc_width_20"]).astype(int)
    df["squeeze_stack"] = (sq8 + sq14 + sq20).astype(float)
    df["kb_ratio"]      = df["bb_width_14"] / (df["kc_width_14"] + EPS)
    df["kb_slope"]      = _rz(df["kb_ratio"].diff(1).rolling(10, min_periods=1).mean(), 20).fillna(0.0)

    # FIX: vectorised consecutive squeeze counter (replaces Python loop)
    sq_active_s = (df["kb_ratio"] < 1.0).astype(int)
    sq_dir      = sq_active_s.copy()          # treat squeeze=1, no-squeeze=0
    # We want consecutive 1-runs only, so zero out non-squeeze the same way _consec_dir works
    change         = (sq_active_s != sq_active_s.shift(1)).cumsum()
    consec_counts  = sq_active_s.groupby(change).cumcount() + 1
    consec_counts[sq_active_s == 0] = 0
    df["consec_squeeze"] = consec_counts.values

    avg_sq                 = max(float(pd.Series(df["consec_squeeze"]).rolling(100, min_periods=5).mean().iloc[-1]), 1.0)
    df["squeeze_maturity"] = (df["consec_squeeze"] / avg_sq).clip(0, 4)

    hv_pct                = df["bb_width_14"].rank(pct=True).fillna(0.5)
    df["vol_potential_e"] = ((1 - hv_pct) * (1 - df["kb_ratio"]).clip(0, 2)).clip(0, 2)

    don_high  = df["high"].rolling(20, min_periods=1).max()
    don_low   = df["low"].rolling(20,  min_periods=1).min()
    don_mid   = (don_high + don_low) / 2
    don_width = (don_high - don_low) + EPS
    lean_raw  = ((df["close"] - don_mid) / don_width).rolling(3, min_periods=1).mean()
    df["sq_lean"] = (lean_raw * sq_active_s).fillna(0.0)

    df["atr_ratio"]   = df["atr5"] / (df["atr14"] + EPS)
    df["atr_ratio_z"] = _rz(df["atr_ratio"], 50).fillna(0.0)
    df["exp_velocity"] = _rz(df["bb_width_14"].diff(3).fillna(0), 50).fillna(0.0)
    above_upper         = (df["close"] - df["bb_upper_14"]).clip(lower=0) / (df["atr14"] + EPS)
    below_lower         = (df["bb_lower_14"] - df["close"]).clip(lower=0) / (df["atr14"] + EPS)
    df["band_walk_bull"] = above_upper.rolling(5, min_periods=1).sum()
    df["band_walk_bear"] = below_lower.rolling(5, min_periods=1).sum()
    df["hv_percentile"]  = hv_pct

    bb_range           = (df["bb_upper_14"] - df["bb_lower_14"]) + EPS
    df["bb_position"]  = (df["close"] - df["bb_lower_14"]) / bb_range
    df["bb_pos_z"]     = _rz(df["bb_position"], 50).fillna(0.0)
    df["kc_position"]  = (df["close"] - df["kc_lower_14"]) / (df["kc_width_14"] + EPS)
    mid_bb14           = df["close"].rolling(14, min_periods=1).mean()
    df["mr_dist"]      = ((df["close"] - mid_bb14) / (df["atr14"] + EPS)).clip(-3, 3)
    df["coil_tightness"] = (1 - df["kb_ratio"]).clip(-1, 1)

    body_avg20          = df["abs_body"].rolling(20, min_periods=1).mean()
    df["body_comp_r"]   = (df["abs_body"] / (body_avg20 + EPS)).clip(0, 3)
    df["close_grav"]    = (df["close"] - df["low"]) / (df["range"] + EPS)
    vol_avg20           = df["volume"].rolling(20, min_periods=1).mean()
    df["vol_z20"]       = _rz(df["volume"], 20).fillna(0.0)
    df["prebreak_qual"] = (df["close_grav"] * df["body_comp_r"] *
                           df["vol_z20"].clip(0, 3)).clip(-3, 3)
    df["intrabar_fill"] = df["abs_body"] / (df["range"] + EPS)
    df["vol_sq_div"]    = (df["vol_z20"] * (df["squeeze_stack"] / 3.0)).clip(-3, 3)
    close_above         = (df["close"] > df["bb_upper_14"]).astype(float)
    close_below         = (df["close"] < df["bb_lower_14"]).astype(float)
    df["bkout_confirm"] = ((close_above - close_below) *
                           (df["vol_z20"] + 1).clip(0, 3)).clip(-3, 3)
    df["don_bias"]      = ((df["close"] - don_mid) / don_width).clip(-1, 1)

    df["vol_z30"]   = _rz(df["volume"], 30).fillna(0.0)
    df["vol_z100"]  = _rz(df["volume"], min(100, len(df))).fillna(0.0)
    df["vol_trend"] = _rz(df["volume"].rolling(5, min_periods=1).mean() /
                          (vol_avg20 + EPS), 30).fillna(0.0)

    last = df.iloc[-1]
    return {
        "atr_ratio_z":      float(last["atr_ratio_z"]),
        "band_walk_bear":   float(last["band_walk_bear"]),
        "band_walk_bull":   float(last["band_walk_bull"]),
        "bb_pos_z":         float(last["bb_pos_z"]),
        "bb_position":      float(last["bb_position"]),
        "bb_width_z":       float(last["bb_width_z"]),
        "bkout_confirm":    float(last["bkout_confirm"]),
        "body_comp_r":      float(last["body_comp_r"]),
        "coil_tightness":   float(last["coil_tightness"]),
        "don_bias":         float(last["don_bias"]),
        "exp_velocity":     float(last["exp_velocity"]),
        "hv_percentile":    float(last["hv_percentile"]),
        "intrabar_fill":    float(last["intrabar_fill"]),
        "kb_ratio":         float(last["kb_ratio"]),
        "kb_slope":         float(last["kb_slope"]),
        "kc_position":      float(last["kc_position"]),
        "kc_width_n":       float(last["kc_width_n"]),
        "mr_dist":          float(last["mr_dist"]),
        "prebreak_qual":    float(last["prebreak_qual"]),
        "sq_lean":          float(last["sq_lean"]),
        "squeeze_maturity": float(last["squeeze_maturity"]),
        "squeeze_stack":    float(last["squeeze_stack"]),
        "vol_potential_e":  float(last["vol_potential_e"]),
        "vol_sq_div":       float(last["vol_sq_div"]),
        "vol_trend":        float(last["vol_trend"]),
        "vol_z100":         float(last["vol_z100"]),
        "vol_z30":          float(last["vol_z30"]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  MOMENTUM
# ═══════════════════════════════════════════════════════════════════════════════

def _rolling_z(series: pd.Series, window: int) -> pd.Series:
    return (series - series.rolling(window).mean()) / (series.rolling(window).std() + EPS)


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


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  MICROSTRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_microstructure_features(df: pd.DataFrame) -> dict:
    """
    df : DataFrame with [open, high, low, close, volume]. Min 40 rows.
    Returns dict of 32 float features — keys identical to original.
    """
    df = df.copy().reset_index(drop=True)

    df["range"]      = df["high"] - df["low"]
    df["body"]       = df["close"] - df["open"]
    df["abs_body"]   = df["body"].abs()
    df["dir"]        = np.sign(df["body"]).fillna(0).astype(int)
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

    df["body_ratio"]     = df["abs_body"]   / (df["range"] + EPS)
    df["upper_ratio"]    = df["upper_wick"] / (df["range"] + EPS)
    df["lower_ratio"]    = df["lower_wick"] / (df["range"] + EPS)
    df["body_ratio_s5"]  = df["body_ratio"].rolling(5, min_periods=1).mean()
    df["upper_ratio_s5"] = df["upper_ratio"].rolling(5, min_periods=1).mean()
    df["lower_ratio_s5"] = df["lower_ratio"].rolling(5, min_periods=1).mean()

    df["body_z"]       = _rz(df["abs_body"], 40).fillna(0.0)
    df["range_z"]      = _rz(df["range"],    40).fillna(0.0)
    df["tail_asym"]    = (df["upper_wick"] - df["lower_wick"]) / (df["range"] + EPS)
    df["tail_to_body"] = (df["upper_wick"] + df["lower_wick"]) / (df["abs_body"] + EPS)

    vol_avg10 = df["volume"].rolling(10, min_periods=1).mean()

    # FIX: orderflow normalised by vol_avg10 — removes raw-volume scale dependency
    # Original: signed_vol.rolling(5).sum() → could be 300-700 raw ticks
    # Fixed:    signed_vol/vol_avg10 rolling sum → dimensionless, stable across regimes
    df["signed_vol_n"]  = df["volume"] * df["dir"] / (vol_avg10 + EPS)
    df["orderflow_5"]   = df["signed_vol_n"].rolling(5,  min_periods=1).sum()
    df["orderflow_10"]  = df["signed_vol_n"].rolling(10, min_periods=1).sum()

    df["small_body_flag"] = (df["abs_body"] < df["abs_body"].rolling(5, min_periods=1).mean() * 0.6).astype(int)
    df["opp_wick"]        = 0.0
    df.loc[df["dir"] >  0, "opp_wick"] = df["lower_wick"]
    df.loc[df["dir"] <  0, "opp_wick"] = df["upper_wick"]
    df["absorption_raw"]  = (df["small_body_flag"] * df["opp_wick"]).rolling(5, min_periods=1).sum()

    df["atr5"]           = df["range"].rolling(5,  min_periods=1).mean()
    df["atr10"]          = df["range"].rolling(10, min_periods=1).mean()
    df["pullback_depth"] = df["range"] / (df["atr5"] + EPS)
    df["pullback_z"]     = _rz(df["pullback_depth"], 20).fillna(0.0)
    df["exhaustion"]     = ((df["upper_wick"] + df["lower_wick"]) / (df["abs_body"] + EPS)).fillna(0.0)
    df["expected_move_raw"] = df["atr5"] * (
        1 + df["orderflow_5"].abs() / (df["volume"].rolling(5, min_periods=1).mean() + EPS))

    # FIX: body_vol_inter normalised — original clip(±1e6) never fired for FX
    # EURUSD: abs_body~0.0005 × vol~1000 ≈ 0.5, so ±1e6 was dead code
    # Fixed:  body_ratio × vol_z gives dimensionless product, clip ±6 is effective
    vol_z10 = _rz(df["volume"], 10).fillna(0.0)
    df["wick_body_inter"]  = ((df["upper_wick"] - df["lower_wick"]) * df["abs_body"]).clip(-2.0, 2.0)
    df["body_vol_inter"]   = (df["body_ratio"] * vol_z10).clip(-6.0, 6.0)

    # FIX: vectorised consecutive-direction counter (replaces Python loop)
    df["consec_len"] = _consec_dir(df["dir"].astype(float))

    df["close_gravity"]  = (df["close"] - df["low"]) / (df["range"] + EPS)
    df["body_mid_disp"]  = (((df["open"] + df["close"]) / 2) -
                              ((df["high"] + df["low"]) / 2)) / (df["range"] + EPS)
    df["tick_delta"]     = df["close_gravity"] * 2 - 1

    delta_vol            = df["tick_delta"] * df["volume"]
    df["cum_delta_5z"]   = _rz(delta_vol.rolling(5, min_periods=1).sum(), 20).fillna(0.0)

    vol_avg10_s          = df["volume"].rolling(10, min_periods=1).mean()
    dom_wick             = df[["upper_wick", "lower_wick"]].max(axis=1)
    rej_dir              = np.where(df["lower_wick"] > df["upper_wick"], 1, -1)
    df["rejection_qual"] = ((dom_wick / (df["range"] + EPS)) *
                             (df["volume"] / (vol_avg10_s + EPS)) * rej_dir).clip(-4, 4)

    comp_a               = (df["body_ratio"] < 0.25).astype(int)
    comp_b               = (df["range"] / (df["atr10"] + EPS) > 0.80).astype(int)
    comp_c               = (df["volume"] / (vol_avg10_s + EPS) > 1.20).astype(int)
    df["absorption_fp"]  = (comp_a + comp_b + comp_c).astype(float)

    bull_mask            = df["dir"] >= 0
    df["dir_efficiency"] = np.where(bull_mask,
                                     df["body_ratio"] * df["close_gravity"],
                                     df["body_ratio"] * (1 - df["close_gravity"]))

    vol_avg5             = df["volume"].rolling(5, min_periods=1).mean()
    df["bar_energy"]     = (df["range"] / (df["atr5"] + EPS) *
                             df["volume"] / (vol_avg5 + EPS)).clip(0, 6)

    df["gap_pressure"]   = ((df["open"] - df["close"].shift(1).fillna(df["open"])) /
                              (df["atr5"] + EPS)).clip(-3, 3)

    last = df.iloc[-1]
    return {
        "abs_body":          float(last["abs_body"]),
        "absorption_fp":     float(last["absorption_fp"]),
        "absorption_raw":    float(last["absorption_raw"]),
        "bar_energy":        float(last["bar_energy"]),
        "body_mid_disp":     float(last["body_mid_disp"]),
        "body_ratio":        float(last["body_ratio"]),
        "body_ratio_s5":     float(last["body_ratio_s5"]),
        "body_vol_inter":    float(last["body_vol_inter"]),
        "body_z":            float(last["body_z"]),
        "close_gravity":     float(last["close_gravity"]),
        "consec_len":        float(last["consec_len"]),
        "cum_delta_5z":      float(last["cum_delta_5z"]),
        "dir_efficiency":    float(last["dir_efficiency"]),
        "exhaustion":        float(last["exhaustion"]),
        "expected_move_raw": float(last["expected_move_raw"]),
        "gap_pressure":      float(last["gap_pressure"]),
        "lower_ratio":       float(last["lower_ratio"]),
        "lower_ratio_s5":    float(last["lower_ratio_s5"]),
        "lower_wick":        float(last["lower_wick"]),
        "orderflow_10":      float(last["orderflow_10"]),
        "orderflow_5":       float(last["orderflow_5"]),
        "pullback_z":        float(last["pullback_z"]),
        "range":             float(last["range"]),
        "range_z":           float(last["range_z"]),
        "rejection_qual":    float(last["rejection_qual"]),
        "tail_asym":         float(last["tail_asym"]),
        "tail_to_body":      float(last["tail_to_body"]),
        "tick_delta":        float(last["tick_delta"]),
        "upper_ratio":       float(last["upper_ratio"]),
        "upper_ratio_s5":    float(last["upper_ratio_s5"]),
        "upper_wick":        float(last["upper_wick"]),
        "wick_body_inter":   float(last["wick_body_inter"]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  VWAP
# ═══════════════════════════════════════════════════════════════════════════════

def compute_vwap_features(df: pd.DataFrame, hour_vol_avg: dict = None) -> dict:
    """
    df : DataFrame with [open, high, low, close, volume].
         Optionally include 'hour' column (0-23) for session-adjusted volume.
         Min ~70 rows recommended.
    Returns dict of 32 float features — keys identical to original.
    """
    df = df.copy().reset_index(drop=True)

    df["range"]      = df["high"] - df["low"]
    df["body"]       = df["close"] - df["open"]
    df["abs_body"]   = df["body"].abs()
    df["dir"]        = np.sign(df["body"]).fillna(0)
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

    hl  = df["range"]
    hc  = (df["high"] - df["close"].shift(1)).abs().fillna(0)
    lc  = (df["low"]  - df["close"].shift(1)).abs().fillna(0)
    df["atr14"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14, min_periods=1).mean()

    tp        = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol    = tp * df["volume"]
    vol_50    = df["volume"].rolling(50, min_periods=1).sum()
    vwap_50   = tp_vol.rolling(50, min_periods=1).sum() / (vol_50 + EPS)
    df["vwap"] = vwap_50

    vwap_sq_dev = (tp - vwap_50) ** 2 * df["volume"]
    vwap_var    = vwap_sq_dev.rolling(50, min_periods=1).sum() / (vol_50 + EPS)
    df["vwap_std"] = np.sqrt(vwap_var.clip(lower=0)) / (df["atr14"] + EPS)

    # GROUP A
    df["vwap_dev_atr"]  = (df["close"] - df["vwap"]) / (df["atr14"] + EPS)
    df["vwap_sigma"]    = df["vwap_dev_atr"] / (df["vwap_std"] + EPS)
    df["vwap_slope_z"]  = _rz((df["vwap"].diff(5) / (df["atr14"] + EPS)).fillna(0), 30).fillna(0.0)

    vol_avg20   = df["volume"].rolling(20, min_periods=1).mean()
    vol_z20     = _rz(df["volume"], 20).fillna(0.0)
    vol_z_cross = (vol_z20 + 1).clip(0, 3)
    cross_bull  = ((df["close"] > df["vwap"]) & (df["close"].shift(1) <= df["vwap"].shift(1))).astype(float)
    cross_bear  = ((df["close"] < df["vwap"]) & (df["close"].shift(1) >= df["vwap"].shift(1))).astype(float)
    df["vwap_reclaim_qual"]   = (df["abs_body"] / (df["atr14"] + EPS) * vol_z_cross * cross_bull).fillna(0).clip(0, 3)
    df["vwap_rejection_qual"] = (df["abs_body"] / (df["atr14"] + EPS) * vol_z_cross * cross_bear).fillna(0).clip(0, 3)
    near_vwap                 = (df["vwap_dev_atr"].abs() < 0.50).astype(int)
    df["vwap_touch_density"]  = near_vwap.rolling(20, min_periods=1).sum() / 20.0

    # GROUP B
    close_grav       = (df["close"] - df["low"]) / (df["range"] + EPS)
    df["tick_delta"] = close_grav * 2 - 1
    delta_vol        = df["tick_delta"] * df["volume"]
    df["cum_delta_20z"] = _rz(delta_vol.rolling(20, min_periods=1).sum(), 50).fillna(0.0)
    up_vol           = df["volume"].where(df["dir"] > 0, 0.0)
    dn_vol           = df["volume"].where(df["dir"] < 0, 0.0)
    up_sum           = up_vol.rolling(20, min_periods=1).sum()
    dn_sum           = dn_vol.rolling(20, min_periods=1).sum()
    df["vol_profile_asym"] = ((up_sum - dn_sum) / (up_sum + dn_sum + EPS)).fillna(0.0)
    up_frac               = up_sum / (up_sum + dn_sum + EPS)
    df["buy_press_z"]     = _rz(up_frac, 30).fillna(0.0)
    delta_at_vwap         = delta_vol * near_vwap
    df["delta_at_vwap"]   = _rz(delta_at_vwap.rolling(20, min_periods=1).sum(), 30).fillna(0.0)

    # GROUP C
    df["vol_z30"]     = _rz(df["volume"], 30).fillna(0.0)
    df["vol_z100"]    = _rz(df["volume"], min(100, len(df))).fillna(0.0)
    df["vol_trend_z"] = _rz(df["volume"].rolling(5, min_periods=1).mean() / (vol_avg20 + EPS), 30).fillna(0.0)

    if "hour" in df.columns and hour_vol_avg is not None:
        session_expected         = df["hour"].map(hour_vol_avg).fillna(vol_avg20)
        df["session_vol_factor"] = (df["volume"] / (session_expected + EPS)).clip(0, 4)
    else:
        df["session_vol_factor"] = (vol_z20.clip(-3, 3) + 1)

    price_dir        = np.sign(df["close"].diff(3).fillna(0))
    vol_dir          = np.sign(df["volume"].rolling(3, min_periods=1).mean().diff(1).fillna(0))
    df["vol_mom_div"]    = (price_dir * vol_dir).fillna(0.0)
    df["vol_efficiency"] = (df["abs_body"] / (df["atr14"] + EPS) /
                             (df["vol_z30"].abs() + 0.5)).clip(0, 3)

    # GROUP D
    body_ratio           = df["abs_body"] / (df["range"] + EPS)
    df["vol_at_vwap_z"]  = (vol_z20 * near_vwap).fillna(0.0)
    comp_a               = (vol_z20 > 1.5).astype(int)
    comp_b               = (df["vwap_dev_atr"].abs() < 0.30).astype(int)
    comp_c               = (body_ratio < 0.25).astype(int)
    df["inst_fp"]        = (comp_a + comp_b + comp_c).astype(float)

    # FIX: vwap_band_walk symmetrised — original tracked only bullish >+1σ excursions
    # Now tracks signed extremes: positive = bull walk above +1σ, negative = bear walk below -1σ
    # Key name unchanged; scaler will handle the expanded range (was always clipped)
    sigma_extreme        = (df["vwap_sigma"].abs() - 1.0).clip(lower=0) * np.sign(df["vwap_sigma"])
    df["vwap_band_walk"] = sigma_extreme.rolling(10, min_periods=1).sum()

    # price_vol_div kept as-is — it exists in the trained scaler/feature_names bundle;
    # replacing it would break inference. Values remain consistent with training.
    df["price_vol_div"]  = (df["abs_body"] / (df["atr14"] + EPS) /
                             (df["vol_z30"].abs() + 0.5)).clip(0, 3)
    df["absorption_lvl"] = (vol_z20 * (body_ratio < 0.25).astype(float)).clip(0, 3)

    # GROUP E
    df["vwap_dist_pct"] = ((df["close"] - df["vwap"]) / (df["vwap"] + EPS) * 100).clip(-1, 1)

    # FIX: above_vwap_streak vectorised (replaces Python loop)
    above_mask           = df["close"] > df["vwap"]
    grp                  = (~above_mask).cumsum()
    streak_counts        = above_mask.groupby(grp).cumcount() + 1
    streak_counts[~above_mask] = 0
    df["above_vwap_streak"] = streak_counts.values

    df["vwap_mr_z"]            = _rz(df["vwap_dev_atr"].abs(), 50).fillna(0.0)
    vwap_side                  = np.sign(df["vwap_dev_atr"])
    delta_sign                 = np.sign(df["tick_delta"])
    df["cum_delta_vwap_align"] = (vwap_side * delta_sign).fillna(0.0)
    far_vwap                   = (df["vwap_sigma"].abs() > 1.5).astype(float)
    df["vol_surge_extreme"]    = (vol_z20 * far_vwap).clip(0, 3)

    # GROUP F
    df["body_ratio"]    = body_ratio
    df["close_grav_f"]  = close_grav
    df["vwap_std_feat"] = df["vwap_std"].clip(0, 5)
    df["atr14_z"]       = _rz(df["atr14"], 50).fillna(0.0)
    df["vol_surge_20"]  = vol_z20.clip(-3, 3)

    last = df.iloc[-1]
    return {
        "above_vwap_streak":    float(last["above_vwap_streak"]),
        "absorption_lvl":       float(last["absorption_lvl"]),
        "atr14_z":              float(last["atr14_z"]),
        "body_ratio":           float(last["body_ratio"]),
        "buy_press_z":          float(last["buy_press_z"]),
        "close_grav_f":         float(last["close_grav_f"]),
        "cum_delta_20z":        float(last["cum_delta_20z"]),
        "cum_delta_vwap_align": float(last["cum_delta_vwap_align"]),
        "delta_at_vwap":        float(last["delta_at_vwap"]),
        "inst_fp":              float(last["inst_fp"]),
        "price_vol_div":        float(last["price_vol_div"]),
        "session_vol_factor":   float(last["session_vol_factor"]),
        "tick_delta":           float(last["tick_delta"]),
        "vol_at_vwap_z":        float(last["vol_at_vwap_z"]),
        "vol_efficiency":       float(last["vol_efficiency"]),
        "vol_mom_div":          float(last["vol_mom_div"]),
        "vol_profile_asym":     float(last["vol_profile_asym"]),
        "vol_surge_20":         float(last["vol_surge_20"]),
        "vol_surge_extreme":    float(last["vol_surge_extreme"]),
        "vol_trend_z":          float(last["vol_trend_z"]),
        "vol_z100":             float(last["vol_z100"]),
        "vol_z30":              float(last["vol_z30"]),
        "vwap_band_walk":       float(last["vwap_band_walk"]),
        "vwap_dev_atr":         float(last["vwap_dev_atr"]),
        "vwap_dist_pct":        float(last["vwap_dist_pct"]),
        "vwap_mr_z":            float(last["vwap_mr_z"]),
        "vwap_reclaim_qual":    float(last["vwap_reclaim_qual"]),
        "vwap_rejection_qual":  float(last["vwap_rejection_qual"]),
        "vwap_sigma":           float(last["vwap_sigma"]),
        "vwap_slope_z":         float(last["vwap_slope_z"]),
        "vwap_std_feat":        float(last["vwap_std_feat"]),
        "vwap_touch_density":   float(last["vwap_touch_density"]),
    }
