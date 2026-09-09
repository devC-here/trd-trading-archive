# synthetic_momentum.py  ·  v3  (next-candle edition)
# ─────────────────────────────────────────────────────────────────────
# Generates synthetic FX OHLCV sequences and computes 30 momentum features.
# ALL 6 labels answer: "what will the NEXT 1-2 candles look like?"
#
# Run:  python synthetic_momentum.py
# Out:  synthetic_momentum_dataset.csv
# ─────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import random

try:
    from tqdm import trange
except ImportError:
    def trange(n, **kw): return range(n)

N_SAMPLES    = 25000
SEQ_LEN      = 120    # context bars for features
FUTURE_BARS  = 2      # bars used for labelling
OUTPUT_PATH  = "synthetic_momentum_dataset.csv"
BASE_PRICE   = 1.0800
SEED         = 42

random.seed(SEED)
np.random.seed(SEED)

EPS = 1e-12


# ─────────────────────────────────────────────────────────────────────
# CANDLE GENERATOR
# ─────────────────────────────────────────────────────────────────────

def generate_candle(base_price, body_scale=0.00045, wick_mult=(0.15, 0.9),
                    vol_range=(300, 2000), bias=0.5):
    direction = 1 if random.random() < bias else -1
    body = float(np.random.uniform(0.2, 1.1) * body_scale)
    wick = float(body * np.random.uniform(*wick_mult))
    o    = float(base_price)
    c    = float(o + direction * body)
    h    = max(o, c) + wick + abs(np.random.normal(0, body * 0.08))
    l    = min(o, c) - wick - abs(np.random.normal(0, body * 0.08))
    v    = int(np.random.uniform(*vol_range))
    return {"open": o, "high": float(h), "low": float(l), "close": float(c), "volume": v}, c


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

def rolling_z(series, window):
    return (series - series.rolling(window).mean()) / (series.rolling(window).std() + EPS)


def rolling_autocorr(series, window, lag=1):
    def _ac(arr):
        if len(arr) < lag + 2:
            return 0.0
        s = pd.Series(arr)
        r = s.autocorr(lag=lag)
        return r if np.isfinite(r) else 0.0
    return series.rolling(window).apply(_ac, raw=False)


# ─────────────────────────────────────────────────────────────────────
# CORE FEATURE COMPUTATION — 30 FEATURES (identical to v2)
# ─────────────────────────────────────────────────────────────────────

def compute_momentum_features(df: pd.DataFrame) -> dict:
    df = df.copy().reset_index(drop=True)

    close  = df["close"].astype(float)
    high   = df["high"].astype(float)
    low    = df["low"].astype(float)
    volume = df["volume"].astype(float)

    tr  = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    atr_safe = atr + EPS

    ret_3          = close.pct_change(3)
    ret_3_atr_norm = (ret_3 / (atr_safe / (close + EPS))).fillna(0.0)
    ret_3_z        = rolling_z(ret_3, 50).fillna(0.0)

    direction        = (close - close.shift(10)).abs()
    noise            = close.diff().abs().rolling(10).sum()
    efficiency_ratio = (direction / (noise + EPS)).fillna(0.0)
    er_slope         = efficiency_ratio.diff(5).fillna(0.0)

    ret_1              = close.pct_change(1).fillna(0.0)
    momentum_persistence = rolling_autocorr(ret_1, 30).fillna(0.0)

    ema_20         = close.ewm(span=20, adjust=False).mean()
    ema_slope      = ema_20.diff(5)
    ema_slope_norm = (ema_slope / atr_safe).fillna(0.0)
    ema_acc_norm   = (ema_slope.diff(3) / atr_safe).fillna(0.0)

    atr_z           = rolling_z(atr, 50).fillna(0.0)
    rng             = high - low
    range_expansion = (rng.rolling(3).mean() / (rng.rolling(20).mean() + EPS)).fillna(1.0)
    vol_regime_confirm = (atr_z * (range_expansion - 1.0)).clip(-4, 4).fillna(0.0)

    roll_min_20    = close.rolling(20).min()
    roll_max_20    = close.rolling(20).max()
    trend_position = ((close - roll_min_20) / (roll_max_20 - roll_min_20 + EPS)).fillna(0.5)
    rolling_high   = high.rolling(20).max().shift(1)
    break_strength = ((close - rolling_high) / atr_safe).clip(-3, 3).fillna(0.0)
    mean_50                 = close.rolling(50).mean()
    mean_reversion_stretch  = ((close - mean_50) / atr_safe).clip(-4, 4).fillna(0.0)

    up_move    = high.diff()
    down_move  = (-low.diff())
    plus_dm    = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm   = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    sm_plus    = plus_dm.ewm(span=14, adjust=False).mean()
    sm_minus   = minus_dm.ewm(span=14, adjust=False).mean()
    sm_atr     = atr.ewm(span=14, adjust=False).mean() + EPS
    plus_di    = 100 * (sm_plus  / sm_atr)
    minus_di   = 100 * (sm_minus / sm_atr)
    dx         = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + EPS)
    adx        = dx.ewm(span=14, adjust=False).mean()
    adx_z      = rolling_z(adx, 50).fillna(0.0)
    adx_slope  = rolling_z(adx, 30).diff(3).fillna(0.0)
    di_balance = ((plus_di - minus_di) / (plus_di + minus_di + EPS)).fillna(0.0)

    tenkan   = (high.rolling(9).max()  + low.rolling(9).min())  / 2
    kijun    = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    tk_spread       = ((tenkan - kijun) / atr_safe).clip(-3, 3).fillna(0.0)
    cloud_thickness = ((senkou_a - senkou_b).abs() / atr_safe).clip(0, 5).fillna(0.0)
    price_vs_cloud  = ((close - senkou_a) / atr_safe).clip(-4, 4).fillna(0.0)
    cloud_twist     = np.sign(senkou_a - senkou_b).fillna(0.0)

    don_high  = high.rolling(20).max()
    don_low   = low.rolling(20).min()
    don_width = don_high - don_low
    don_position = ((close - don_low) / (don_width + EPS)).fillna(0.5)
    don_width_z  = rolling_z(don_width, 50).fillna(0.0)
    don_breakout = ((close - don_high.shift(1)) / atr_safe).clip(-3, 3).fillna(0.0)

    ret_5         = close.pct_change(5)
    ret_1_norm    = (ret_1 / (atr_safe / (close + EPS)))
    ret_5_norm    = (ret_5 / (atr_safe / (close + EPS)))
    momentum_decay = (ret_1_norm / ((ret_5_norm / 5) + EPS)).clip(-5, 5).fillna(0.0)
    tension = (ret_3_atr_norm * (0.5 - trend_position)).clip(-4, 4).fillna(0.0)
    ret_10          = close.pct_change(10)
    ret_10_atr_norm = (ret_10 / (atr_safe / (close + EPS))).fillna(0.0)
    ret_10_z        = rolling_z(ret_10, 50).fillna(0.0)

    ema_50      = close.ewm(span=50, adjust=False).mean()
    ema_50_slope = ema_50.diff(10)
    slope_alignment = (np.sign(ema_slope) * np.sign(ema_50_slope)).fillna(0.0)

    volume_z        = rolling_z(volume, 30).fillna(0.0)
    vol_displacement = (volume_z * ret_3_z).clip(-5, 5).fillna(0.0)

    i = -1
    features = {
        "ret_3_atr_norm":        float(ret_3_atr_norm.iloc[i]),
        "ret_3_z":               float(ret_3_z.iloc[i]),
        "efficiency_ratio":      float(efficiency_ratio.iloc[i]),
        "er_slope":              float(er_slope.iloc[i]),
        "momentum_persistence":  float(momentum_persistence.iloc[i]),
        "ema_slope_norm":        float(ema_slope_norm.iloc[i]),
        "ema_acc_norm":          float(ema_acc_norm.iloc[i]),
        "atr_z":                 float(atr_z.iloc[i]),
        "range_expansion":       float(range_expansion.iloc[i]),
        "vol_regime_confirm":    float(vol_regime_confirm.iloc[i]),
        "trend_position":        float(trend_position.iloc[i]),
        "break_strength":        float(break_strength.iloc[i]),
        "mean_reversion_stretch": float(mean_reversion_stretch.iloc[i]),
        "adx_z":                 float(adx_z.iloc[i]),
        "adx_slope":             float(adx_slope.iloc[i]),
        "di_balance":            float(di_balance.iloc[i]),
        "tk_spread":             float(tk_spread.iloc[i]),
        "cloud_thickness":       float(cloud_thickness.iloc[i]),
        "price_vs_cloud":        float(price_vs_cloud.iloc[i]),
        "cloud_twist":           float(cloud_twist.iloc[i]),
        "don_position":          float(don_position.iloc[i]),
        "don_width_z":           float(don_width_z.iloc[i]),
        "don_breakout":          float(don_breakout.iloc[i]),
        "momentum_decay":        float(momentum_decay.iloc[i]),
        "tension":               float(tension.iloc[i]),
        "ret_10_atr_norm":       float(ret_10_atr_norm.iloc[i]),
        "ret_10_z":              float(ret_10_z.iloc[i]),
        "slope_alignment":       float(slope_alignment.iloc[i]),
        "volume_z":              float(volume_z.iloc[i]),
        "vol_displacement":      float(vol_displacement.iloc[i]),
        # context for labelling (stripped before training)
        "_atr":                  float(atr.iloc[i]),
        "_context_close":        float(close.iloc[i]),
        "_di_balance":           float(di_balance.iloc[i]),
        "_trend_position":       float(trend_position.iloc[i]),
        "_ema_slope_norm":       float(ema_slope_norm.iloc[i]),
        "_mean_rev_stretch":     float(mean_reversion_stretch.iloc[i]),
        "_slope_alignment":      float(slope_alignment.iloc[i]),
        "_momentum_persistence": float(momentum_persistence.iloc[i]),
    }
    return features


# ─────────────────────────────────────────────────────────────────────
# FORWARD LABEL GENERATION — 6 heads, all about NEXT 1-2 candles
# ─────────────────────────────────────────────────────────────────────

def label_momentum_sample(features: dict, future_candles: list) -> tuple:
    """
    Returns 6 integer labels predicting the NEXT 1-2 candles:
      0  next_direction         bearish(0) / flat(1) / bullish(2)
      1  next_magnitude         small(0) / medium(1) / large(2)
      2  next_regime_behaviour  chop(0) / pullback(1) / continuation(2) / breakout(3)
      3  next_volatility        contracting(0) / normal(1) / expanding(2)
      4  next_structural_bias   bearish(0) / neutral(1) / bullish(2)
      5  next_momentum_quality  fading(0) / stable(1) / accelerating(2)
    """
    atr = max(features["_atr"], EPS)
    ref = features["_context_close"]

    c1 = future_candles[0]
    c2 = future_candles[1] if len(future_candles) >= 2 else c1

    c1_body_atr = abs(c1["close"] - c1["open"]) / atr
    c2_body_atr = abs(c2["close"] - c2["open"]) / atr
    c1_bull     = c1["close"] > c1["open"]
    c2_bull     = c2["close"] > c2["open"]
    net_atr     = (c2["close"] - ref) / atr
    seq_high    = max(c1["high"], c2["high"])
    seq_low     = min(c1["low"],  c2["low"])
    seq_range   = (seq_high - seq_low) / atr

    # Context features
    dib  = features["_di_balance"]
    tp   = features["_trend_position"]
    ems  = features["_ema_slope_norm"]
    mrs  = features["_mean_rev_stretch"]
    sla  = features["_slope_alignment"]
    mp   = features["_momentum_persistence"]

    # ── HEAD 1: next_direction ────────────────────────────────────────
    if   net_atr > 0.40:    next_dir = 2
    elif net_atr < -0.40:   next_dir = 0
    else:                    next_dir = 1

    # ── HEAD 2: next_magnitude ────────────────────────────────────────
    avg_body = (c1_body_atr + c2_body_atr) / 2
    if   avg_body > 1.5:    next_mag = 2
    elif avg_body > 0.5:    next_mag = 1
    else:                    next_mag = 0

    # ── HEAD 3: next_regime_behaviour ────────────────────────────────
    # Based on actual future movement pattern
    same_dir = (c1_bull == c2_bull)
    if avg_body < 0.30 and seq_range < 1.2:
        nxt_reg = 0   # chop
    elif not same_dir and c2_body_atr > c1_body_atr * 0.7:
        nxt_reg = 1   # pullback
    elif same_dir and c2_body_atr >= c1_body_atr * 0.75:
        if seq_range > 2.5:  nxt_reg = 3  # breakout
        else:                 nxt_reg = 2  # continuation
    else:
        nxt_reg = 0   # default chop

    # ── HEAD 4: next_volatility ───────────────────────────────────────
    if   seq_range > 2.5:   next_vol = 2
    elif seq_range < 1.0:   next_vol = 0
    else:                    next_vol = 1

    # ── HEAD 5: next_structural_bias ─────────────────────────────────
    # Will price be in a bullish or bearish structural position after c2?
    close_grav = (c2["close"] - seq_low) / (seq_range * atr + EPS)
    if   close_grav > 0.65 and net_atr > 0:    nxt_struct = 2
    elif close_grav < 0.35 and net_atr < 0:    nxt_struct = 0
    else:                                        nxt_struct = 1

    # ── HEAD 6: next_momentum_quality ────────────────────────────────
    # Is the momentum accelerating, stable, or fading?
    if c2_body_atr > c1_body_atr * 1.2 and same_dir:
        nxt_mom = 2   # accelerating
    elif c2_body_atr < c1_body_atr * 0.6 or not same_dir:
        nxt_mom = 0   # fading
    else:
        nxt_mom = 1   # stable

    return (int(next_dir), int(next_mag), int(nxt_reg),
            int(next_vol), int(nxt_struct), int(nxt_mom))


# ─────────────────────────────────────────────────────────────────────
# SCENARIOS
# ─────────────────────────────────────────────────────────────────────

SCENARIOS = {
    "strong_trend":    {"weight": 0.15, "bias": 0.73, "body_scale": 0.00085, "vol_range": (700, 2500)},
    "weak_trend":      {"weight": 0.18, "bias": 0.61, "body_scale": 0.00050, "vol_range": (300, 1500)},
    "range_bound":     {"weight": 0.32, "bias": 0.50, "body_scale": 0.00028, "vol_range": (120, 700)},
    "breakout":        {"weight": 0.12, "bias": 0.70, "body_scale": 0.00120, "vol_range": (1000, 5000)},
    "exhaustion_fade": {"weight": 0.23, "bias": 0.47, "body_scale": 0.00038, "vol_range": (180, 900)},
}


def generate_sequence(base_price: float = BASE_PRICE) -> dict:
    names    = list(SCENARIOS.keys())
    weights  = [SCENARIOS[n]["weight"] for n in names]
    scenario = random.choices(names, weights=weights, k=1)[0]
    cfg      = SCENARIOS[scenario]

    bearish_flip = random.random() < 0.50
    prices = [base_price + np.random.normal(0, 0.0010)]
    seq    = []

    total = SEQ_LEN + FUTURE_BARS

    for i in range(total):
        if scenario == "strong_trend":
            local_bias   = min(0.88, cfg["bias"] + i * 0.0018)
            local_vol    = (cfg["vol_range"][0], min(cfg["vol_range"][1] + i * 20, 5000))
        elif scenario == "exhaustion_fade":
            local_bias   = max(0.15, cfg["bias"] - i * 0.0028)
            local_vol    = cfg["vol_range"]
        elif scenario == "breakout":
            if i < SEQ_LEN * 0.55:
                local_bias   = 0.50
                local_vol    = (150, 600)
                body_s       = 0.00022
            else:
                local_bias   = cfg["bias"] + 0.10
                local_vol    = cfg["vol_range"]
                body_s       = cfg["body_scale"]
        elif scenario == "range_bound":
            local_bias = 0.50 + 0.15 * np.sin(i * 0.4)
            local_vol  = cfg["vol_range"]
        else:
            local_bias = cfg["bias"]
            local_vol  = cfg["vol_range"]

        if bearish_flip:
            local_bias = 1.0 - local_bias
        if random.random() < 0.07:
            local_bias = 1.0 - local_bias

        local_bias = float(np.clip(local_bias, 0.10, 0.90))
        body_scale = body_s if (scenario == "breakout" and i < SEQ_LEN * 0.55) else cfg["body_scale"]

        candle, new_close = generate_candle(
            prices[-1],
            body_scale=body_scale,
            vol_range=local_vol,
            bias=local_bias,
        )
        seq.append(candle)
        prices.append(new_close)

    df_context = pd.DataFrame(seq[:SEQ_LEN])
    future     = seq[SEQ_LEN:]

    features = compute_momentum_features(df_context)
    labels   = label_momentum_sample(features, future)

    row = {k: v for k, v in features.items() if not k.startswith("_")}
    row["scenario"]              = scenario
    row["next_direction"]        = labels[0]
    row["next_magnitude"]        = labels[1]
    row["next_regime_behaviour"] = labels[2]
    row["next_volatility"]       = labels[3]
    row["next_structural_bias"]  = labels[4]
    row["next_momentum_quality"] = labels[5]
    return row


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Generating {N_SAMPLES} sequences  |  SEQ_LEN={SEQ_LEN}  |  FUTURE={FUTURE_BARS}  |  features=30")

    rows = []
    for _ in trange(N_SAMPLES):
        rows.append(generate_sequence())

    df_out = pd.DataFrame(rows)

    label_cols   = ["next_direction", "next_magnitude", "next_regime_behaviour",
                    "next_volatility", "next_structural_bias", "next_momentum_quality"]
    feature_cols = [c for c in df_out.columns if c not in label_cols + ["scenario"]]

    df_out = df_out[["scenario"] + feature_cols + label_cols]
    df_out.to_csv(OUTPUT_PATH, index=False)

    label_maps = {
        "next_direction":        {0:"bearish", 1:"flat", 2:"bullish"},
        "next_magnitude":        {0:"small", 1:"medium", 2:"large"},
        "next_regime_behaviour": {0:"chop", 1:"pullback", 2:"continuation", 3:"breakout"},
        "next_volatility":       {0:"contracting", 1:"normal", 2:"expanding"},
        "next_structural_bias":  {0:"bearish", 1:"neutral", 2:"bullish"},
        "next_momentum_quality": {0:"fading", 1:"stable", 2:"accelerating"},
    }
    print(f"\n✅ Saved → {OUTPUT_PATH}   shape={df_out.shape}")
    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")
    print("\nLabel distributions:")
    for lc in label_cols:
        counts = df_out[lc].value_counts().sort_index()
        named  = {label_maps[lc][k]: f"{100*v/len(df_out):.1f}%" for k, v in counts.items()}
        print(f"  {lc:<25}: {named}")
