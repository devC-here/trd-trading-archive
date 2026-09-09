# synthetic_momentum.py  ·  v2
# ─────────────────────────────────────────────────────────────────────────────
# Generates synthetic FX OHLCV sequences and computes 30 momentum features.
# Features span: displacement, trend quality, acceleration, volatility regime,
# Ichimoku geometry, Donchian structure, ADX decomposition, momentum physics,
# multi-timeframe alignment, and volume participation.
#
# Run:  python synthetic_momentum.py
# Out:  synthetic_momentum_dataset.csv
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import random
from tqdm import trange

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
N_SAMPLES   = 25000
SEQ_LEN     = 120   # must exceed all rolling windows (ichimoku needs 52+26 shift = 78 min)
OUTPUT_PATH = "synthetic_momentum_dataset.csv"
BASE_PRICE  = 1.0800
SEED        = 42

random.seed(SEED)
np.random.seed(SEED)

EPS = 1e-12


# ─────────────────────────────────────────────
# CANDLE GENERATOR
# Produces realistic OHLCV with directional bias per scenario
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def rolling_z(series, window):
    return (series - series.rolling(window).mean()) / (series.rolling(window).std() + EPS)


def rolling_autocorr(series, window, lag=1):
    """Rolling lag-1 autocorrelation over window bars."""
    def _ac(arr):
        if len(arr) < lag + 2:
            return 0.0
        s = pd.Series(arr)
        r = s.autocorr(lag=lag)
        return r if np.isfinite(r) else 0.0
    return series.rolling(window).apply(_ac, raw=False)


# ─────────────────────────────────────────────
# CORE FEATURE COMPUTATION — 30 FEATURES
# Identical code used in synthetic generator AND deploy script.
# Any change here must be mirrored in momentum_deploy.py
# ─────────────────────────────────────────────
def compute_momentum_features(df: pd.DataFrame) -> dict:
    """
    Input : DataFrame with [open, high, low, close, volume], minimum SEQ_LEN rows.
    Output: dict of 30 float feature values at the last bar.

    Feature blocks:
      A  Displacement         (2)   ret_3_atr_norm, ret_3_z
      B  Trend Quality        (3)   efficiency_ratio, er_slope, momentum_persistence
      C  Acceleration         (2)   ema_slope_norm, ema_acc_norm
      D  Volatility Regime    (3)   atr_z, range_expansion, vol_regime_confirm
      E  Structure            (3)   trend_position, break_strength, mean_reversion_stretch
      F  Trend Strength       (3)   adx_z, adx_slope, di_balance
      G  Ichimoku Geometry    (4)   tk_spread, cloud_thickness, price_vs_cloud, cloud_twist
      H  Donchian             (3)   don_position, don_width_z, don_breakout
      I  Momentum Physics     (4)   momentum_decay, tension, ret_10_atr_norm, ret_10_z
      J  Multi-TF Alignment   (1)   slope_alignment
      K  Participation        (2)   volume_z, vol_displacement
    """
    df = df.copy().reset_index(drop=True)

    close  = df["close"].astype(float)
    high   = df["high"].astype(float)
    low    = df["low"].astype(float)
    open_  = df["open"].astype(float)
    volume = df["volume"].astype(float)

    # ── BASE: ATR ─────────────────────────────────────────────────
    tr  = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    atr_safe = atr + EPS

    # ── A: DISPLACEMENT ───────────────────────────────────────────
    ret_3          = close.pct_change(3)
    ret_3_atr_norm = (ret_3 / (atr_safe / (close + EPS))).fillna(0.0)
    ret_3_z        = rolling_z(ret_3, 50).fillna(0.0)

    # ── B: TREND QUALITY ──────────────────────────────────────────
    # Efficiency Ratio: 1 = straight-line trend, 0 = pure noise
    direction        = (close - close.shift(10)).abs()
    noise            = close.diff().abs().rolling(10).sum()
    efficiency_ratio = (direction / (noise + EPS)).fillna(0.0)

    # ER Slope: dropping ER after high reading = trend ending
    er_slope = efficiency_ratio.diff(5).fillna(0.0)

    # Momentum Persistence: rolling autocorrelation of 1-bar returns
    # H > 0 = trending (returns follow themselves), H < 0 = mean-reverting
    ret_1              = close.pct_change(1).fillna(0.0)
    momentum_persistence = rolling_autocorr(ret_1, 30).fillna(0.0)

    # ── C: ACCELERATION ───────────────────────────────────────────
    ema_20         = close.ewm(span=20, adjust=False).mean()
    ema_slope      = ema_20.diff(5)
    ema_slope_norm = (ema_slope / atr_safe).fillna(0.0)
    ema_acc_norm   = (ema_slope.diff(3) / atr_safe).fillna(0.0)

    # ── D: VOLATILITY REGIME ──────────────────────────────────────
    atr_z           = rolling_z(atr, 50).fillna(0.0)
    rng             = high - low
    range_expansion = (rng.rolling(3).mean() / (rng.rolling(20).mean() + EPS)).fillna(1.0)

    # Double-confirmed vol expansion: both ATR z-score AND range expansion agree
    vol_regime_confirm = (atr_z * (range_expansion - 1.0)).clip(-4, 4).fillna(0.0)

    # ── E: STRUCTURE ──────────────────────────────────────────────
    # Where are we in the 20-bar range? 0 = bottom, 1 = top
    roll_min_20    = close.rolling(20).min()
    roll_max_20    = close.rolling(20).max()
    trend_position = ((close - roll_min_20) / (roll_max_20 - roll_min_20 + EPS)).fillna(0.5)

    # Breakout strength: distance above prior 20-bar high, ATR-normalized
    rolling_high   = high.rolling(20).max().shift(1)
    break_strength = ((close - rolling_high) / atr_safe).clip(-3, 3).fillna(0.0)

    # Mean reversion stretch: distance from 50-bar mean, ATR-normalized
    # Large positive = overbought overextension. Large negative = oversold overextension.
    mean_50                 = close.rolling(50).mean()
    mean_reversion_stretch  = ((close - mean_50) / atr_safe).clip(-4, 4).fillna(0.0)

    # ── F: TREND STRENGTH — ADX DECOMPOSED ───────────────────────
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
    adx_slope  = rolling_z(adx, 30).diff(3).fillna(0.0)    # z-score the ADX first, then diff — scale-invariant
    di_balance = ((plus_di - minus_di) / (plus_di + minus_di + EPS)).fillna(0.0)
    # di_balance: +1 = pure bull flow, -1 = pure bear flow, 0 = balanced

    # ── G: ICHIMOKU GEOMETRY ──────────────────────────────────────
    # We engineer from Ichimoku's geometry, NOT its crossover signals.
    # These capture multi-period structure that no other block encodes.
    tenkan   = (high.rolling(9).max()  + low.rolling(9).min())  / 2
    kijun    = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)

    # Tenkan-Kijun spread: directional momentum strength, ATR-normalized
    tk_spread       = ((tenkan - kijun) / atr_safe).clip(-3, 3).fillna(0.0)

    # Cloud thickness: wide cloud = strong trend structure
    cloud_thickness = ((senkou_a - senkou_b).abs() / atr_safe).clip(0, 5).fillna(0.0)

    # Price vs Cloud: how far above/below Senkou A, ATR-normalized
    price_vs_cloud  = ((close - senkou_a) / atr_safe).clip(-4, 4).fillna(0.0)

    # Cloud twist direction: +1 = bullish cloud, -1 = bearish cloud
    cloud_twist     = np.sign(senkou_a - senkou_b).fillna(0.0)

    # ── H: DONCHIAN CHANNELS ──────────────────────────────────────
    don_high  = high.rolling(20).max()
    don_low   = low.rolling(20).min()
    don_width = don_high - don_low

    # Position within Donchian channel [0=bottom, 1=top]
    don_position = ((close - don_low) / (don_width + EPS)).fillna(0.5)

    # Channel width z-score: expanding = trending, contracting = coiling
    don_width_z  = rolling_z(don_width, 50).fillna(0.0)

    # Donchian breakout: distance above prior upper band, ATR-normalized
    don_breakout = ((close - don_high.shift(1)) / atr_safe).clip(-3, 3).fillna(0.0)

    # ── I: MOMENTUM PHYSICS ───────────────────────────────────────
    # Momentum Decay: fresh momentum vs fading momentum
    # ret_1/atr relative to (ret_5/atr/5) — ratio > 1 = momentum fresh, < 1 = fading
    ret_5         = close.pct_change(5)
    ret_1_norm    = (ret_1 / (atr_safe / (close + EPS)))
    ret_5_norm    = (ret_5 / (atr_safe / (close + EPS)))
    momentum_decay = (ret_1_norm / ((ret_5_norm / 5) + EPS)).clip(-5, 5).fillna(0.0)

    # Tension: displacement × structural positioning
    # Positive = move has room to run (aligned with structure)
    # Negative = move is exhausting into structure wall
    tension = (ret_3_atr_norm * (0.5 - trend_position)).clip(-4, 4).fillna(0.0)

    # 10-bar displacement: medium-term momentum context
    ret_10          = close.pct_change(10)
    ret_10_atr_norm = (ret_10 / (atr_safe / (close + EPS))).fillna(0.0)
    ret_10_z        = rolling_z(ret_10, 50).fillna(0.0)

    # ── J: MULTI-TIMEFRAME ALIGNMENT ──────────────────────────────
    # Are short and medium term trends aligned?
    # +1 = both pointing same direction, 0 = one flat, -1 = opposing
    ema_50      = close.ewm(span=50, adjust=False).mean()
    ema_50_slope = ema_50.diff(10)
    slope_alignment = (
        np.sign(ema_slope) * np.sign(ema_50_slope)
    ).fillna(0.0)

    # ── K: PARTICIPATION ──────────────────────────────────────────
    volume_z        = rolling_z(volume, 30).fillna(0.0)
    vol_displacement = (volume_z * ret_3_z).clip(-5, 5).fillna(0.0)

    # ── COMPILE LAST BAR ──────────────────────────────────────────
    i = -1
    return {
        # A — Displacement (2)
        "ret_3_atr_norm":        float(ret_3_atr_norm.iloc[i]),
        "ret_3_z":               float(ret_3_z.iloc[i]),
        # B — Trend Quality (3)
        "efficiency_ratio":      float(efficiency_ratio.iloc[i]),
        "er_slope":              float(er_slope.iloc[i]),
        "momentum_persistence":  float(momentum_persistence.iloc[i]),
        # C — Acceleration (2)
        "ema_slope_norm":        float(ema_slope_norm.iloc[i]),
        "ema_acc_norm":          float(ema_acc_norm.iloc[i]),
        # D — Volatility Regime (3)
        "atr_z":                 float(atr_z.iloc[i]),
        "range_expansion":       float(range_expansion.iloc[i]),
        "vol_regime_confirm":    float(vol_regime_confirm.iloc[i]),
        # E — Structure (3)
        "trend_position":        float(trend_position.iloc[i]),
        "break_strength":        float(break_strength.iloc[i]),
        "mean_reversion_stretch": float(mean_reversion_stretch.iloc[i]),
        # F — Trend Strength (3)
        "adx_z":                 float(adx_z.iloc[i]),
        "adx_slope":             float(adx_slope.iloc[i]),
        "di_balance":            float(di_balance.iloc[i]),
        # G — Ichimoku Geometry (4)
        "tk_spread":             float(tk_spread.iloc[i]),
        "cloud_thickness":       float(cloud_thickness.iloc[i]),
        "price_vs_cloud":        float(price_vs_cloud.iloc[i]),
        "cloud_twist":           float(cloud_twist.iloc[i]),
        # H — Donchian (3)
        "don_position":          float(don_position.iloc[i]),
        "don_width_z":           float(don_width_z.iloc[i]),
        "don_breakout":          float(don_breakout.iloc[i]),
        # I — Momentum Physics (4)
        "momentum_decay":        float(momentum_decay.iloc[i]),
        "tension":               float(tension.iloc[i]),
        "ret_10_atr_norm":       float(ret_10_atr_norm.iloc[i]),
        "ret_10_z":              float(ret_10_z.iloc[i]),
        # J — Multi-TF Alignment (1)
        "slope_alignment":       float(slope_alignment.iloc[i]),
        # K — Participation (2)
        "volume_z":              float(volume_z.iloc[i]),
        "vol_displacement":      float(vol_displacement.iloc[i]),
    }


# ─────────────────────────────────────────────
# LABEL GENERATION
# Rule-based heuristics encoding what each head should learn.
# Thresholds calibrated to real feature distributions.
# ─────────────────────────────────────────────
def label_momentum_sample(features: dict) -> tuple:
    """
    Returns 6 integer labels:
      0  momentum_state      down(0) / neutral(1) / up(2)
      1  momentum_slope      falling(0) / flat(1) / rising(2)
      2  regime              ranging(0) / weak_trend(1) / trending(2) / breakout(3)
      3  volatility_state    contracting(0) / normal(1) / expanding(2)
      4  structure_bias      bearish(0) / neutral(1) / bullish(2)
      5  entry_quality       avoid(0) / wait(1) / go(2)
    """
    r   = features["ret_3_atr_norm"]
    rz  = features["ret_3_z"]
    er  = features["efficiency_ratio"]
    ers = features["er_slope"]
    mp  = features["momentum_persistence"]
    es  = features["ema_slope_norm"]
    ea  = features["ema_acc_norm"]
    az  = features["atr_z"]
    re  = features["range_expansion"]
    vrc = features["vol_regime_confirm"]
    tp  = features["trend_position"]
    bs  = features["break_strength"]
    mrs = features["mean_reversion_stretch"]
    adz = features["adx_z"]
    ads = features["adx_slope"]
    dib = features["di_balance"]
    tks = features["tk_spread"]
    clt = features["cloud_thickness"]
    pvc = features["price_vs_cloud"]
    cwt = features["cloud_twist"]
    dnp = features["don_position"]
    dnw = features["don_width_z"]
    dnb = features["don_breakout"]
    mdc = features["momentum_decay"]
    ten = features["tension"]
    r10 = features["ret_10_atr_norm"]
    rz10= features["ret_10_z"]
    sla = features["slope_alignment"]
    vz  = features["volume_z"]
    vd  = features["vol_displacement"]

    # ── HEAD 1: momentum_state ────────────────────────────────────
    # Requires multi-signal confirmation: short + medium displacement, DI flow, Ichimoku direction
    bullish_score = (
        (r  > 0.7) +
        (rz > 0.5) +
        (r10 > 0.5) +
        (dib > 0.15) +
        (tks > 0.1) +
        (pvc > 0.0 and cwt > 0)
    )
    bearish_score = (
        (r   < -0.7) +
        (rz  < -0.5) +
        (r10 < -0.5) +
        (dib < -0.15) +
        (tks < -0.1) +
        (pvc < 0.0 and cwt < 0)
    )
    if bullish_score >= 4:
        momentum_state = 2
    elif bearish_score >= 4:
        momentum_state = 0
    else:
        momentum_state = 1

    # ── HEAD 2: momentum_slope ────────────────────────────────────
    # ema acceleration + ADX slope + decay ratio all must agree
    rising_score  = (es > 0.4) + (ea > 0.04) + (ads > 0.02) + (mdc > 0.5)
    falling_score = (es < -0.4) + (ea < -0.04) + (ads < -0.02) + (mdc < -0.5)
    if rising_score >= 3:
        momentum_slope = 2
    elif falling_score >= 3:
        momentum_slope = 0
    else:
        momentum_slope = 1

    # ── HEAD 3: regime ────────────────────────────────────────────
    # Breakout: range exploding + don breakout + vol surge + break_strength active
    breakout_score = (
        (re > 1.3) +
        (abs(dnb) > 0.5) +
        (vz > 0.5) +
        (abs(bs) > 0.4) +
        (vrc > 0.5)
    )
    # Trending: efficient movement + ADX elevated + Ichimoku aligned
    trending_score = (
        (er > 0.50) +
        (adz > 0.4) +
        (abs(tks) > 0.15) +
        (clt > 0.3) +
        (mp > 0.1) +
        (sla != 0)
    )
    if breakout_score >= 3:
        regime = 3
    elif trending_score >= 4:
        regime = 2
    elif trending_score >= 2 or er > 0.2:
        regime = 1
    else:
        regime = 0

    # ── HEAD 4: volatility_state ──────────────────────────────────
    expand_score  = (az > 0.6) + (re > 1.25) + (vrc > 0.4) + (dnw > 0.5)
    contract_score= (az < -0.6) + (re < 0.8) + (dnw < -0.5)
    if expand_score >= 2:
        volatility_state = 2
    elif contract_score >= 2:
        volatility_state = 0
    else:
        volatility_state = 1

    # ── HEAD 5: structure_bias ────────────────────────────────────
    # Uses Donchian position, trend_position, cloud positioning, DI balance, tension
    bull_struct = (
        (tp > 0.58) +
        (dnp > 0.58) +
        (pvc > 0.2) +
        (cwt > 0) +
        (dib > 0.2) +
        (ten > 0.1)   # positive tension = room to run
    )
    bear_struct = (
        (tp < 0.42) +
        (dnp < 0.42) +
        (pvc < -0.2) +
        (cwt < 0) +
        (dib < -0.2) +
        (ten < -0.1)
    )
    if bull_struct >= 4:
        structure_bias = 2
    elif bear_struct >= 4:
        structure_bias = 0
    else:
        structure_bias = 1

    # ── HEAD 6: entry_quality ─────────────────────────────────────
    # The synthesis gate — ALL pillars must agree
    regime_ready    = regime >= 2
    vol_confirmed   = (vd > 0.4 or vz > 0.5)
    trend_efficient = er > 0.30
    mtf_aligned     = sla != 0 and mp > 0.0    # multi-TF + persistence
    slope_matches   = (
        (momentum_state == 2 and momentum_slope == 2) or
        (momentum_state == 0 and momentum_slope == 0)
    )
    no_overextension = abs(mrs) < 2.5           # not too stretched

    go_score = (
        regime_ready +
        vol_confirmed +
        trend_efficient +
        mtf_aligned +
        slope_matches +
        no_overextension
    )

    chop = (regime == 0 or (er < 0.12 and abs(r) < 0.25))

    if go_score >= 5 and not chop:
        entry_quality = 2
    elif chop or (go_score <= 1):
        entry_quality = 0
    else:
        entry_quality = 1

    return (
        int(momentum_state),
        int(momentum_slope),
        int(regime),
        int(volatility_state),
        int(structure_bias),
        int(entry_quality),
    )


# ─────────────────────────────────────────────
# SCENARIOS — 5 market regimes with realistic physics
# ─────────────────────────────────────────────
SCENARIOS = {
    "strong_trend":    {"weight": 0.15, "bias": 0.73, "body_scale": 0.00085, "vol_range": (700, 2500)},
    "weak_trend":      {"weight": 0.18, "bias": 0.61, "body_scale": 0.00050, "vol_range": (300, 1500)},
    "range_bound":     {"weight": 0.32, "bias": 0.50, "body_scale": 0.00028, "vol_range": (120, 700)},
    "breakout":        {"weight": 0.12, "bias": 0.70, "body_scale": 0.00120, "vol_range": (1000, 5000)},
    "exhaustion_fade": {"weight": 0.23, "bias": 0.47, "body_scale": 0.00038, "vol_range": (180, 900)},
}


def generate_sequence(base_price: float = BASE_PRICE) -> dict:
    """Generate one full scenario sequence and return feature + label dict."""
    names    = list(SCENARIOS.keys())
    weights  = [SCENARIOS[n]["weight"] for n in names]
    scenario = random.choices(names, weights=weights, k=1)[0]
    cfg      = SCENARIOS[scenario]

    # 50% chance of bearish flip — ensures symmetric up/down label distribution
    bearish_flip = random.random() < 0.50

    prices = [base_price + np.random.normal(0, 0.0010)]
    seq    = []

    for i in range(SEQ_LEN):
        # Scenario-specific drift dynamics
        if scenario == "strong_trend":
            # accelerates through sequence, both price and volume
            local_bias   = min(0.88, cfg["bias"] + i * 0.0018)
            local_vol    = (cfg["vol_range"][0], min(cfg["vol_range"][1] + i * 20, 5000))
        elif scenario == "exhaustion_fade":
            # starts directional, steadily loses conviction
            local_bias   = max(0.15, cfg["bias"] - i * 0.0028)
            local_vol    = cfg["vol_range"]
        elif scenario == "breakout":
            if i < SEQ_LEN * 0.55:
                # coiling phase: tight range, low vol
                local_bias   = 0.50
                local_vol    = (150, 600)
                body_s       = 0.00022
            else:
                # explosion phase: directional + vol surge
                local_bias   = cfg["bias"] + 0.10
                local_vol    = cfg["vol_range"]
                body_s       = cfg["body_scale"]
        elif scenario == "range_bound":
            # mean-reverting: bias oscillates around 0.50
            local_bias = 0.50 + 0.15 * np.sin(i * 0.4)
            local_vol  = cfg["vol_range"]
        else:
            local_bias = cfg["bias"]
            local_vol  = cfg["vol_range"]

        # Apply bearish flip: invert directional bias around 0.50
        if bearish_flip:
            local_bias = 1.0 - local_bias

        # 7% random noise flip across all scenarios
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

    df       = pd.DataFrame(seq)
    features = compute_momentum_features(df)
    labels   = label_momentum_sample(features)

    row = {**features, "scenario": scenario}
    row["momentum_state"]   = labels[0]
    row["momentum_slope"]   = labels[1]
    row["regime"]           = labels[2]
    row["volatility_state"] = labels[3]
    row["structure_bias"]   = labels[4]
    row["entry_quality"]    = labels[5]
    return row


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Generating {N_SAMPLES} sequences  |  SEQ_LEN={SEQ_LEN}  |  features=30")

    rows = []
    for _ in trange(N_SAMPLES):
        rows.append(generate_sequence())

    df_out = pd.DataFrame(rows)

    label_cols   = ["momentum_state", "momentum_slope", "regime",
                    "volatility_state", "structure_bias", "entry_quality"]
    feature_cols = [c for c in df_out.columns if c not in label_cols + ["scenario"]]

    df_out = df_out[["scenario"] + feature_cols + label_cols]
    df_out.to_csv(OUTPUT_PATH, index=False)

    print(f"\n✅ Saved → {OUTPUT_PATH}   shape={df_out.shape}")
    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")

    label_maps = {
        "momentum_state":   {0:"down", 1:"neutral", 2:"up"},
        "momentum_slope":   {0:"falling", 1:"flat", 2:"rising"},
        "regime":           {0:"ranging", 1:"weak_trend", 2:"trending", 3:"breakout"},
        "volatility_state": {0:"contracting", 1:"normal", 2:"expanding"},
        "structure_bias":   {0:"bearish", 1:"neutral", 2:"bullish"},
        "entry_quality":    {0:"avoid", 1:"wait", 2:"go"},
    }
    print("\nLabel distributions:")
    for lc in label_cols:
        counts = df_out[lc].value_counts().sort_index()
        named  = {label_maps[lc][k]: f"{100*v/len(df_out):.1f}%" for k, v in counts.items()}
        print(f"  {lc:<20}: {named}")
