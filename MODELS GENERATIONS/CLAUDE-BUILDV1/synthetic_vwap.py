# synthetic_vwap.py
# Generates training data for the VWAP / Volume model
# 32 features | 6 heads | Rolling 50-bar VWAP | Institutional flow physics | FX M15
import pandas as pd
import numpy as np
import random
from collections import defaultdict

try:
    from tqdm import trange
except ImportError:
    def trange(n, **kw): return range(n)

N_SAMPLES   = 20000
LOOKBACK    = 100          # bars; 50-bar VWAP needs context
OUTPUT_PATH = "synthetic_vwap_dataset.csv"
BASE_PRICE  = 1.2000
SEED        = 42
random.seed(SEED)
np.random.seed(SEED)
EPS = 1e-12


# ─────────────────────────────────────────────────────────────────────
# Candle generator
# ─────────────────────────────────────────────────────────────────────

def _gen_candle(price, body_scale, wl, wh, vl, vh, bias):
    d    = 1 if random.random() < bias else -1
    body = float(np.random.uniform(0.1, 1.0) * body_scale)
    wk   = float(body * np.random.uniform(wl, wh))
    tl   = float(body * np.random.uniform(wl, wh))
    o    = float(price)
    c    = float(o + d * body)
    # hour label (simulated 0-23, cycling every 4 bars = 1h for M15)
    return {"open": o, "high": max(o,c)+wk, "low": min(o,c)-tl, "close": float(c),
            "volume": int(np.random.uniform(vl, vh))}, c


def _rz(s, w):
    return (s - s.rolling(w).mean()) / (s.rolling(w).std() + EPS)


# ─────────────────────────────────────────────────────────────────────
# Feature computation (matches deploy exactly)
# ─────────────────────────────────────────────────────────────────────

def compute_vwap_features(df: pd.DataFrame, hour_vol_avg: dict = None) -> dict:
    """
    Input : DataFrame with [open, high, low, close, volume] + optional 'hour' column.
            Min ~60 bars for stable rolling VWAP.
    Output: dict of 32 features.
    """
    df = df.copy().reset_index(drop=True)

    # ── base ─────────────────────────────────────────────────────────
    df["range"]      = df["high"] - df["low"]
    df["body"]       = df["close"] - df["open"]
    df["abs_body"]   = df["body"].abs()
    df["dir"]        = np.sign(df["body"]).fillna(0)
    df["upper_wick"] = df["high"] - df[["open","close"]].max(axis=1)
    df["lower_wick"] = df[["open","close"]].min(axis=1) - df["low"]

    # ATR
    hl  = df["range"]
    hc  = (df["high"] - df["close"].shift(1)).abs().fillna(0)
    lc  = (df["low"]  - df["close"].shift(1)).abs().fillna(0)
    df["atr14"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14, min_periods=1).mean()

    # ── Rolling 50-bar VWAP (session-independent — FX safe) ──────────
    tp           = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol       = tp * df["volume"]
    vwap_50      = tp_vol.rolling(50, min_periods=1).sum() / (df["volume"].rolling(50, min_periods=1).sum() + EPS)
    df["vwap"]   = vwap_50

    # VWAP std (volume-weighted)
    vwap_sq_dev  = (tp - vwap_50) ** 2 * df["volume"]
    vwap_var     = vwap_sq_dev.rolling(50, min_periods=1).sum() / (df["volume"].rolling(50, min_periods=1).sum() + EPS)
    df["vwap_std"] = np.sqrt(vwap_var.clip(lower=0)) / (df["atr14"] + EPS)

    # ── GROUP A: VWAP construction (6) ───────────────────────────────

    # A1: vwap_dev_atr — core fair value distance
    df["vwap_dev_atr"] = (df["close"] - df["vwap"]) / (df["atr14"] + EPS)

    # A2: vwap_sigma — volume-weighted Bollinger band position
    df["vwap_sigma"]   = df["vwap_dev_atr"] / (df["vwap_std"] + EPS)

    # A3: vwap_slope_z — rising VWAP = net buying
    df["vwap_slope_z"] = _rz((df["vwap"].diff(5) / (df["atr14"] + EPS)).fillna(0), 30).fillna(0.0)

    # A4+A5: VWAP reclaim / rejection quality (volume-confirmed crosses)
    vol_avg20     = df["volume"].rolling(20, min_periods=1).mean()
    vol_z_cross   = (_rz(df["volume"], 20) + 1).clip(0, 3)
    cross_bull    = ((df["close"] > df["vwap"]) & (df["close"].shift(1) <= df["vwap"].shift(1))).astype(float)
    cross_bear    = ((df["close"] < df["vwap"]) & (df["close"].shift(1) >= df["vwap"].shift(1))).astype(float)
    df["vwap_reclaim_qual"]   = (df["abs_body"] / (df["atr14"]+EPS) * vol_z_cross * cross_bull).fillna(0).clip(0,3)
    df["vwap_rejection_qual"] = (df["abs_body"] / (df["atr14"]+EPS) * vol_z_cross * cross_bear).fillna(0).clip(0,3)

    # A6: vwap_touch_density — how often price near VWAP
    near_vwap     = (df["vwap_dev_atr"].abs() < 0.50).astype(int)
    df["vwap_touch_density"] = near_vwap.rolling(20, min_periods=1).sum() / 20.0

    # ── GROUP B: Tick delta directional volume reconstruction (5) ────

    # B1: tick_delta_bar — single-bar directional pressure
    close_grav    = (df["close"] - df["low"]) / (df["range"] + EPS)
    df["tick_delta"] = close_grav * 2 - 1

    # B2: cumulative 20-bar delta z-scored
    delta_vol     = df["tick_delta"] * df["volume"]
    df["cum_delta_20z"] = _rz(delta_vol.rolling(20, min_periods=1).sum(), 50).fillna(0.0)

    # B3: vol_profile_asymmetry — bull vol vs bear vol ratio
    up_vol        = df["volume"].where(df["dir"] > 0, 0.0)
    dn_vol        = df["volume"].where(df["dir"] < 0, 0.0)
    up_sum        = up_vol.rolling(20, min_periods=1).sum()
    dn_sum        = dn_vol.rolling(20, min_periods=1).sum()
    df["vol_profile_asym"] = ((up_sum - dn_sum) / (up_sum + dn_sum + EPS)).fillna(0.0)

    # B4: buying_pressure_ratio_z — fraction of volume on up-bars, z-scored
    up_frac       = up_sum / (up_sum + dn_sum + EPS)
    df["buy_press_z"] = _rz(up_frac, 30).fillna(0.0)

    # B5: delta_at_vwap — delta measured only when near VWAP (most informative)
    delta_at_vwap = delta_vol * near_vwap
    df["delta_at_vwap"] = _rz(delta_at_vwap.rolling(20, min_periods=1).sum(), 30).fillna(0.0)

    # ── GROUP C: Volume character (6) ────────────────────────────────

    df["vol_z30"]  = _rz(df["volume"], 30).fillna(0.0)
    df["vol_z100"] = _rz(df["volume"], min(100, len(df))).fillna(0.0)
    df["vol_trend_z"] = _rz(df["volume"].rolling(5, min_periods=1).mean() / (vol_avg20 + EPS), 30).fillna(0.0)

    # C4: session_vol_factor — removes time-of-day bias (FX critical)
    if "hour" in df.columns and hour_vol_avg is not None:
        session_expected = df["hour"].map(hour_vol_avg).fillna(vol_avg20)
        df["session_vol_factor"] = (df["volume"] / (session_expected + EPS)).clip(0, 4)
    else:
        df["session_vol_factor"] = df["vol_z30"].clip(-3, 3) + 1   # fallback

    # C5: vol_momentum_divergence — price rising but volume falling = distribution
    price_dir     = np.sign(df["close"].diff(3).fillna(0))
    vol_dir       = np.sign(df["volume"].rolling(3, min_periods=1).mean().diff(1).fillna(0))
    df["vol_mom_div"] = (price_dir * vol_dir).fillna(0.0)

    # C6: vol_efficiency — price movement per unit volume (friction)
    df["vol_efficiency"] = (df["abs_body"] / (df["atr14"] + EPS) /
                             (df["vol_z30"].abs() + 0.5)).clip(0, 3)

    # ── GROUP D: Institutional detection (5) ─────────────────────────

    vol_z20       = _rz(df["volume"], 20).fillna(0.0)

    # D1: vol_at_vwap_z — volume surge specifically near VWAP
    df["vol_at_vwap_z"] = (vol_z20 * near_vwap).fillna(0.0)

    # D2: institutional_footprint — 3-component iceberg at VWAP test
    body_ratio    = df["abs_body"] / (df["range"] + EPS)
    comp_a        = (vol_z20 > 1.5).astype(int)
    comp_b        = (df["vwap_dev_atr"].abs() < 0.30).astype(int)
    comp_c        = (body_ratio < 0.25).astype(int)
    df["inst_fp"]  = (comp_a + comp_b + comp_c).astype(float)

    # D3: vwap_band_walk — sustained extension above VWAP+1σ
    above_1sd     = (df["vwap_sigma"] - 1.0).clip(lower=0)
    df["vwap_band_walk"] = above_1sd.rolling(10, min_periods=1).sum()

    # D4: price_vol_divergence — large move on thin volume = suspicious
    df["price_vol_div"] = (df["abs_body"] / (df["atr14"] + EPS) *
                            (1 / (df["vol_z30"].abs() + 0.5))).clip(0, 3)

    # D5: absorption_at_level — vol surge with tiny body (any level)
    df["absorption_lvl"] = (vol_z20 * (body_ratio < 0.25).astype(float)).clip(0, 3)

    # ── GROUP E: VWAP structural (5) ─────────────────────────────────

    # E1: vwap_distance_pct
    df["vwap_dist_pct"]  = ((df["close"] - df["vwap"]) / (df["vwap"] + EPS) * 100).clip(-1, 1)

    # E2: above_vwap_streak
    above_streak = []
    cur = 0
    for v in (df["close"] > df["vwap"]).tolist():
        cur = cur + 1 if v else 0
        above_streak.append(cur)
    df["above_vwap_streak"] = above_streak

    # E3: vwap_mean_reversion_z — extreme deviation → mean reversion pressure
    df["vwap_mr_z"]      = _rz(df["vwap_dev_atr"].abs(), 50).fillna(0.0)

    # E4: cum_delta_vwap_align — is delta direction aligned with VWAP side?
    vwap_side = np.sign(df["vwap_dev_atr"])
    delta_sign = np.sign(df["tick_delta"])
    df["cum_delta_vwap_align"] = (vwap_side * delta_sign).fillna(0.0)

    # E5: vol_surge_at_extreme — volume surging when price far from VWAP
    far_vwap      = (df["vwap_sigma"].abs() > 1.5).astype(float)
    df["vol_surge_extreme"] = (vol_z20 * far_vwap).clip(0, 3)

    # ── GROUP F: auxiliary price-level features (5) ──────────────────

    # F1: body_ratio (last bar)
    df["body_ratio"]      = df["abs_body"] / (df["range"] + EPS)

    # F2: close_grav (last bar — already computed as close_grav above; reuse)
    df["close_grav_f"]    = close_grav

    # F3: vwap_std_last — normalised volume-weighted std dev (magnitude)
    # already stored in df["vwap_std"]; expose as feature
    df["vwap_std_feat"]   = df["vwap_std"].clip(0, 5)

    # F4: atr14_z — volatility regime
    df["atr14_z"]         = _rz(df["atr14"], 50).fillna(0.0)

    # F5: vol_surge_20 — single-bar volume z-score (shorter window for responsiveness)
    df["vol_surge_20"]    = vol_z20.clip(-3, 3)

    # ── Collect last-row features ─────────────────────────────────────
    last = df.iloc[-1]
    f = {
        # A (6)
        "vwap_dev_atr":       float(last["vwap_dev_atr"]),
        "vwap_sigma":         float(last["vwap_sigma"]),
        "vwap_slope_z":       float(last["vwap_slope_z"]),
        "vwap_reclaim_qual":  float(last["vwap_reclaim_qual"]),
        "vwap_rejection_qual":float(last["vwap_rejection_qual"]),
        "vwap_touch_density": float(last["vwap_touch_density"]),
        # B (5)
        "tick_delta":         float(last["tick_delta"]),
        "cum_delta_20z":      float(last["cum_delta_20z"]),
        "vol_profile_asym":   float(last["vol_profile_asym"]),
        "buy_press_z":        float(last["buy_press_z"]),
        "delta_at_vwap":      float(last["delta_at_vwap"]),
        # C (6)
        "vol_z30":            float(last["vol_z30"]),
        "vol_z100":           float(last["vol_z100"]),
        "vol_trend_z":        float(last["vol_trend_z"]),
        "session_vol_factor": float(last["session_vol_factor"]),
        "vol_mom_div":        float(last["vol_mom_div"]),
        "vol_efficiency":     float(last["vol_efficiency"]),
        # D (5)
        "vol_at_vwap_z":      float(last["vol_at_vwap_z"]),
        "inst_fp":            float(last["inst_fp"]),
        "vwap_band_walk":     float(last["vwap_band_walk"]),
        "price_vol_div":      float(last["price_vol_div"]),
        "absorption_lvl":     float(last["absorption_lvl"]),
        # E (5)
        "vwap_dist_pct":      float(last["vwap_dist_pct"]),
        "above_vwap_streak":  float(last["above_vwap_streak"]),
        "vwap_mr_z":          float(last["vwap_mr_z"]),
        "cum_delta_vwap_align": float(last["cum_delta_vwap_align"]),
        "vol_surge_extreme":  float(last["vol_surge_extreme"]),
        # F (5)
        "body_ratio":         float(last["body_ratio"]),
        "close_grav_f":       float(last["close_grav_f"]),
        "vwap_std_feat":      float(last["vwap_std_feat"]),
        "atr14_z":            float(last["atr14_z"]),
        "vol_surge_20":       float(last["vol_surge_20"]),
        # context for labelling only (stripped before training)
        "_vwap_sigma":        float(last["vwap_sigma"]),
        "_inst_fp":           float(last["inst_fp"]),
        "_above_streak":      float(last["above_vwap_streak"]),
        "_vol_z30":           float(last["vol_z30"]),
        "_cum_delta":         float(last["cum_delta_20z"]),
        "_vwap_band_walk":    float(last["vwap_band_walk"]),
    }
    return f, df


# ─────────────────────────────────────────────────────────────────────
# Label generation — 6 heads
# ─────────────────────────────────────────────────────────────────────

def compute_labels(f: dict) -> tuple:
    sigma      = f["_vwap_sigma"]
    inst_fp    = f["_inst_fp"]
    streak     = f["_above_streak"]
    vol_z      = f["_vol_z30"]
    cum_d      = f["_cum_delta"]
    bwalk      = f["_vwap_band_walk"]
    dev_atr    = f["vwap_dev_atr"]
    slope_z    = f["vwap_slope_z"]
    rec_q      = f["vwap_reclaim_qual"]
    rej_q      = f["vwap_rejection_qual"]
    touch_d    = f["vwap_touch_density"]
    tick_d     = f["tick_delta"]
    pa         = f["vol_profile_asym"]
    bpz        = f["buy_press_z"]
    mom_div    = f["vol_mom_div"]

    # ── HEAD 1: vwap_zone [0=below_2sd 1=below_1sd 2=near_vwap 3=above_1sd 4=above_2sd]
    if   sigma > 1.8:    zone = 4
    elif sigma > 0.75:   zone = 3
    elif sigma < -1.8:   zone = 0
    elif sigma < -0.75:  zone = 1
    else:                zone = 2

    # ── HEAD 2: volume_character [0=absence 1=normal 2=surge 3=climax_surge]
    # climax = high vol but body shrinking (distribution / exhaustion)
    if   vol_z > 2.0 and f["vol_efficiency"] < 0.6:  vol_char = 3  # climax
    elif vol_z > 1.2:                                  vol_char = 2  # surge
    elif vol_z < -0.8:                                 vol_char = 0  # absence
    else:                                              vol_char = 1  # normal

    # ── HEAD 3: institutional_pressure [0=selling 1=neutral 2=buying]
    # absorption → forces neutral (iceberg holds, not pushes)
    if   inst_fp >= 2:                                inst_p = 1   # absorption = neutral
    elif cum_d > 0.6 and bpz > 0.4 and pa > 0.15:    inst_p = 2
    elif cum_d < -0.6 and bpz < -0.4 and pa < -0.15:  inst_p = 0
    else:                                              inst_p = 1

    # ── HEAD 4: vwap_interaction [0=rejection 1=testing 2=reclaim 3=walk]
    if   bwalk > 1.0 and streak > 6:                  vwap_int = 3  # band walk
    elif rec_q > 0.2:                                  vwap_int = 2  # reclaim
    elif rej_q > 0.2:                                  vwap_int = 0  # rejection
    elif abs(dev_atr) > 0.8 and abs(dev_atr) < 0.4:   vwap_int = 1  # testing (unreachable guard)
    elif abs(dev_atr) < 0.5:                           vwap_int = 1  # testing
    else:                                              vwap_int = 0  # rejection by default far from VWAP

    # ── HEAD 5: volume_trend [0=declining 1=stable 2=building]
    vt = f["vol_trend_z"]
    if   vt > 0.5:   vol_trend = 2
    elif vt < -0.5:  vol_trend = 0
    else:            vol_trend = 1

    # ── HEAD 6: vwap_signal [0=avoid 1=wait 2=act]
    bullish_act = (inst_p == 2 and vol_char in (1,2) and
                   vwap_int in (2,3) and zone >= 3 and slope_z > 0.3)
    bearish_act = (inst_p == 0 and vol_char in (1,2) and
                   vwap_int in (0,3) and zone <= 1 and slope_z < -0.3)
    act_cond   = bullish_act or bearish_act
    avoid_cond = (vol_char == 0 or vol_char == 3 or
                  (inst_fp >= 2 and vol_char == 3))

    if   act_cond:    sig = 2
    elif avoid_cond:  sig = 0
    else:             sig = 1

    return (int(zone), int(vol_char), int(inst_p),
            int(vwap_int), int(vol_trend), int(sig))


# ─────────────────────────────────────────────────────────────────────
# Scenario definitions
# ─────────────────────────────────────────────────────────────────────

SCENARIOS = {
    "institutional_buy":   dict(bias=0.70, body=0.0008, wl=0.10, wh=0.40, vl=800,  vh=3000),
    "institutional_sell":  dict(bias=0.30, body=0.0008, wl=0.10, wh=0.40, vl=800,  vh=3000),
    "vwap_reclaim_bull":   dict(bias=0.65, body=0.0007, wl=0.15, wh=0.55, vl=600,  vh=2200),
    "vwap_reclaim_bear":   dict(bias=0.35, body=0.0007, wl=0.15, wh=0.55, vl=600,  vh=2200),
    "vwap_rejection":      dict(bias=0.48, body=0.0005, wl=0.35, wh=1.20, vl=500,  vh=1800),
    "vol_climax_bull":     dict(bias=0.65, body=0.0010, wl=0.15, wh=0.40, vl=2000, vh=7000),
    "vol_climax_bear":     dict(bias=0.35, body=0.0010, wl=0.15, wh=0.40, vl=2000, vh=7000),
    "vol_absence":         dict(bias=0.50, body=0.0003, wl=0.05, wh=0.30, vl=50,   vh=350),
    "absorption":          dict(bias=0.55, body=0.0002, wl=0.20, wh=0.70, vl=1200, vh=4000),
    "band_walk_bull":      dict(bias=0.72, body=0.0009, wl=0.08, wh=0.35, vl=700,  vh=2500),
    "band_walk_bear":      dict(bias=0.28, body=0.0009, wl=0.08, wh=0.35, vl=700,  vh=2500),
    "chop":                dict(bias=0.50, body=0.0004, wl=0.10, wh=0.50, vl=200,  vh=900),
}
WEIGHTS = [0.08, 0.08, 0.08, 0.08, 0.08, 0.07, 0.07, 0.07, 0.08, 0.08, 0.08, 0.07]


def generate_sequence(base_price=BASE_PRICE):
    sc_name = random.choices(list(SCENARIOS.keys()), weights=WEIGHTS, k=1)[0]
    sc      = SCENARIOS[sc_name]
    price   = base_price + np.random.normal(0, 0.0010)
    candles = []

    for i in range(LOOKBACK):
        local_bias = sc["bias"] if random.random() > 0.08 else 1 - sc["bias"]

        # vol_climax: ramp volume then collapse body late
        if "vol_climax" in sc_name and i > LOOKBACK * 0.7:
            fade   = 1 - (i - LOOKBACK*0.7) / (LOOKBACK*0.3 + 1) * 0.65
            bs     = sc["body"] * max(fade, 0.12)
        # band_walk: sustained directional pressure
        elif "band_walk" in sc_name:
            bs     = sc["body"] * np.random.uniform(0.8, 1.3)
        else:
            bs     = sc["body"] * np.random.uniform(0.6, 1.4)

        c, price = _gen_candle(price, bs, sc["wl"], sc["wh"], sc["vl"], sc["vh"], local_bias)
        # add simulated hour for session_vol_factor
        c["hour"] = (i * 3) % 24   # rough M15 cycle
        candles.append(c)

    df = pd.DataFrame(candles)

    # build per-hour volume baseline from this sequence (proxy for rolling 20-day)
    hour_vol_avg = df.groupby("hour")["volume"].mean().to_dict()

    f, _   = compute_vwap_features(df, hour_vol_avg=hour_vol_avg)
    labels = compute_labels(f)

    feat = {k: v for k, v in f.items() if not k.startswith("_")}
    row  = dict(feat)
    (row["label_vwap_zone"],    row["label_vol_char"],
     row["label_inst_pressure"],row["label_vwap_interaction"],
     row["label_vol_trend"],    row["label_vwap_signal"]) = labels
    return row


# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Generating {N_SAMPLES} VWAP/Volume samples (FX M15) ...")
    rows = []
    for _ in trange(N_SAMPLES):
        rows.append(generate_sequence())

    df_out     = pd.DataFrame(rows)
    label_cols = [c for c in df_out.columns if c.startswith("label_")]
    feat_cols  = [c for c in df_out.columns if not c.startswith("label_")]
    df_out     = df_out[sorted(feat_cols) + label_cols]
    df_out.to_csv(OUTPUT_PATH, index=False)

    print(f"\n✅  Saved → {OUTPUT_PATH}  shape={df_out.shape}")
    print(f"    Features : {len(feat_cols)}")
    for lc in label_cols:
        print(f"    {lc}: {dict(df_out[lc].value_counts().sort_index())}")
