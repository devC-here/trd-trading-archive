# synthetic_vwap.py  ·  v2  (next-candle edition)
# ─────────────────────────────────────────────────────────────────────
# Generates training data for the VWAP / Volume model
# 32 features | 6 heads — ALL labels predict the NEXT 1-2 candles
# Rolling 50-bar VWAP | Institutional flow physics | FX M15
# ─────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import random

try:
    from tqdm import trange
except ImportError:
    def trange(n, **kw): return range(n)

N_SAMPLES    = 20000
LOOKBACK     = 100          # bars; 50-bar VWAP needs context
FUTURE_BARS  = 2            # bars used for labelling
OUTPUT_PATH  = "synthetic_vwap_dataset.csv"
BASE_PRICE   = 1.2000
SEED         = 42
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
    return {"open": o, "high": max(o,c)+wk, "low": min(o,c)-tl, "close": float(c),
            "volume": int(np.random.uniform(vl, vh))}, c


def _rz(s, w):
    return (s - s.rolling(w).mean()) / (s.rolling(w).std() + EPS)


# ─────────────────────────────────────────────────────────────────────
# Feature computation — unchanged from v1, on LOOKBACK bars only
# ─────────────────────────────────────────────────────────────────────

def compute_vwap_features(df: pd.DataFrame, hour_vol_avg: dict = None) -> dict:
    df = df.copy().reset_index(drop=True)

    df["range"]      = df["high"] - df["low"]
    df["body"]       = df["close"] - df["open"]
    df["abs_body"]   = df["body"].abs()
    df["dir"]        = np.sign(df["body"]).fillna(0)
    df["upper_wick"] = df["high"] - df[["open","close"]].max(axis=1)
    df["lower_wick"] = df[["open","close"]].min(axis=1) - df["low"]

    hl  = df["range"]
    hc  = (df["high"] - df["close"].shift(1)).abs().fillna(0)
    lc  = (df["low"]  - df["close"].shift(1)).abs().fillna(0)
    df["atr14"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14, min_periods=1).mean()

    tp           = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol       = tp * df["volume"]
    vwap_50      = tp_vol.rolling(50, min_periods=1).sum() / (df["volume"].rolling(50, min_periods=1).sum() + EPS)
    df["vwap"]   = vwap_50

    vwap_sq_dev  = (tp - vwap_50) ** 2 * df["volume"]
    vwap_var     = vwap_sq_dev.rolling(50, min_periods=1).sum() / (df["volume"].rolling(50, min_periods=1).sum() + EPS)
    df["vwap_std"] = np.sqrt(vwap_var.clip(lower=0)) / (df["atr14"] + EPS)

    df["vwap_dev_atr"] = (df["close"] - df["vwap"]) / (df["atr14"] + EPS)
    df["vwap_sigma"]   = df["vwap_dev_atr"] / (df["vwap_std"] + EPS)
    df["vwap_slope_z"] = _rz((df["vwap"].diff(5) / (df["atr14"] + EPS)).fillna(0), 30).fillna(0.0)

    vol_avg20     = df["volume"].rolling(20, min_periods=1).mean()
    vol_z_cross   = (_rz(df["volume"], 20) + 1).clip(0, 3)
    cross_bull    = ((df["close"] > df["vwap"]) & (df["close"].shift(1) <= df["vwap"].shift(1))).astype(float)
    cross_bear    = ((df["close"] < df["vwap"]) & (df["close"].shift(1) >= df["vwap"].shift(1))).astype(float)
    df["vwap_reclaim_qual"]   = (df["abs_body"] / (df["atr14"]+EPS) * vol_z_cross * cross_bull).fillna(0).clip(0,3)
    df["vwap_rejection_qual"] = (df["abs_body"] / (df["atr14"]+EPS) * vol_z_cross * cross_bear).fillna(0).clip(0,3)

    near_vwap     = (df["vwap_dev_atr"].abs() < 0.50).astype(int)
    df["vwap_touch_density"] = near_vwap.rolling(20, min_periods=1).sum() / 20.0

    close_grav    = (df["close"] - df["low"]) / (df["range"] + EPS)
    df["tick_delta"] = close_grav * 2 - 1
    delta_vol     = df["tick_delta"] * df["volume"]
    df["cum_delta_20z"] = _rz(delta_vol.rolling(20, min_periods=1).sum(), 50).fillna(0.0)

    up_vol        = df["volume"].where(df["dir"] > 0, 0.0)
    dn_vol        = df["volume"].where(df["dir"] < 0, 0.0)
    up_sum        = up_vol.rolling(20, min_periods=1).sum()
    dn_sum        = dn_vol.rolling(20, min_periods=1).sum()
    df["vol_profile_asym"] = ((up_sum - dn_sum) / (up_sum + dn_sum + EPS)).fillna(0.0)
    up_frac       = up_sum / (up_sum + dn_sum + EPS)
    df["buy_press_z"] = _rz(up_frac, 30).fillna(0.0)
    delta_at_vwap = delta_vol * near_vwap
    df["delta_at_vwap"] = _rz(delta_at_vwap.rolling(20, min_periods=1).sum(), 30).fillna(0.0)

    df["vol_z30"]  = _rz(df["volume"], 30).fillna(0.0)
    df["vol_z100"] = _rz(df["volume"], min(100, len(df))).fillna(0.0)
    df["vol_trend_z"] = _rz(df["volume"].rolling(5, min_periods=1).mean() / (vol_avg20 + EPS), 30).fillna(0.0)

    if "hour" in df.columns and hour_vol_avg is not None:
        session_expected = df["hour"].map(hour_vol_avg).fillna(vol_avg20)
        df["session_vol_factor"] = (df["volume"] / (session_expected + EPS)).clip(0, 4)
    else:
        df["session_vol_factor"] = df["vol_z30"].clip(-3, 3) + 1

    price_dir     = np.sign(df["close"].diff(3).fillna(0))
    vol_dir       = np.sign(df["volume"].rolling(3, min_periods=1).mean().diff(1).fillna(0))
    df["vol_mom_div"] = (price_dir * vol_dir).fillna(0.0)

    body_ratio    = df["abs_body"] / (df["range"] + EPS)
    df["vol_efficiency"] = (df["abs_body"] / (df["atr14"] + EPS) /
                             (df["vol_z30"].abs() + 0.5)).clip(0, 3)

    vol_z20       = _rz(df["volume"], 20).fillna(0.0)
    df["vol_at_vwap_z"] = (vol_z20 * near_vwap).fillna(0.0)
    comp_a        = (vol_z20 > 1.5).astype(int)
    comp_b        = (df["vwap_dev_atr"].abs() < 0.30).astype(int)
    comp_c        = (body_ratio < 0.25).astype(int)
    df["inst_fp"]  = (comp_a + comp_b + comp_c).astype(float)

    above_1sd     = (df["vwap_sigma"] - 1.0).clip(lower=0)
    df["vwap_band_walk"] = above_1sd.rolling(10, min_periods=1).sum()

    df["price_vol_div"] = (df["abs_body"] / (df["atr14"] + EPS) *
                            (1 / (df["vol_z30"].abs() + 0.5))).clip(0, 3)
    df["absorption_lvl"] = (vol_z20 * (body_ratio < 0.25).astype(float)).clip(0, 3)

    df["vwap_dist_pct"]  = ((df["close"] - df["vwap"]) / (df["vwap"] + EPS) * 100).clip(-1, 1)
    above_streak = []
    cur = 0
    for v in (df["close"] > df["vwap"]).tolist():
        cur = cur + 1 if v else 0
        above_streak.append(cur)
    df["above_vwap_streak"] = above_streak
    df["vwap_mr_z"]      = _rz(df["vwap_dev_atr"].abs(), 50).fillna(0.0)
    vwap_side = np.sign(df["vwap_dev_atr"])
    delta_sign = np.sign(df["tick_delta"])
    df["cum_delta_vwap_align"] = (vwap_side * delta_sign).fillna(0.0)
    far_vwap      = (df["vwap_sigma"].abs() > 1.5).astype(float)
    df["vol_surge_extreme"] = (vol_z20 * far_vwap).clip(0, 3)

    df["body_ratio"]      = body_ratio
    df["close_grav_f"]    = close_grav
    df["vwap_std_feat"]   = df["vwap_std"].clip(0, 5)
    df["atr14_z"]         = _rz(df["atr14"], 50).fillna(0.0)
    df["vol_surge_20"]    = vol_z20.clip(-3, 3)

    last = df.iloc[-1]
    f = {
        "vwap_dev_atr":       float(last["vwap_dev_atr"]),
        "vwap_sigma":         float(last["vwap_sigma"]),
        "vwap_slope_z":       float(last["vwap_slope_z"]),
        "vwap_reclaim_qual":  float(last["vwap_reclaim_qual"]),
        "vwap_rejection_qual":float(last["vwap_rejection_qual"]),
        "vwap_touch_density": float(last["vwap_touch_density"]),
        "tick_delta":         float(last["tick_delta"]),
        "cum_delta_20z":      float(last["cum_delta_20z"]),
        "vol_profile_asym":   float(last["vol_profile_asym"]),
        "buy_press_z":        float(last["buy_press_z"]),
        "delta_at_vwap":      float(last["delta_at_vwap"]),
        "vol_z30":            float(last["vol_z30"]),
        "vol_z100":           float(last["vol_z100"]),
        "vol_trend_z":        float(last["vol_trend_z"]),
        "session_vol_factor": float(last["session_vol_factor"]),
        "vol_mom_div":        float(last["vol_mom_div"]),
        "vol_efficiency":     float(last["vol_efficiency"]),
        "vol_at_vwap_z":      float(last["vol_at_vwap_z"]),
        "inst_fp":            float(last["inst_fp"]),
        "vwap_band_walk":     float(last["vwap_band_walk"]),
        "price_vol_div":      float(last["price_vol_div"]),
        "absorption_lvl":     float(last["absorption_lvl"]),
        "vwap_dist_pct":      float(last["vwap_dist_pct"]),
        "above_vwap_streak":  float(last["above_vwap_streak"]),
        "vwap_mr_z":          float(last["vwap_mr_z"]),
        "cum_delta_vwap_align": float(last["cum_delta_vwap_align"]),
        "vol_surge_extreme":  float(last["vol_surge_extreme"]),
        "body_ratio":         float(last["body_ratio"]),
        "close_grav_f":       float(last["close_grav_f"]),
        "vwap_std_feat":      float(last["vwap_std_feat"]),
        "atr14_z":            float(last["atr14_z"]),
        "vol_surge_20":       float(last["vol_surge_20"]),
        # context for labelling (stripped before training)
        "_atr14":             float(last["atr14"]),
        "_vwap":              float(last["vwap"]),
        "_vwap_sigma":        float(last["vwap_sigma"]),
        "_context_close":     float(last["close"]),
        "_cum_delta":         float(last["cum_delta_20z"]),
        "_vol_z30":           float(last["vol_z30"]),
        "_inst_fp":           float(last["inst_fp"]),
        "_slope_z":           float(last["vwap_slope_z"]),
        "_tick_delta":        float(last["tick_delta"]),
        "_vol_efficiency":    float(last["vol_efficiency"]),
    }
    return f, df


# ─────────────────────────────────────────────────────────────────────
# FORWARD LABEL GENERATION — 6 heads, all about NEXT 1-2 candles
# ─────────────────────────────────────────────────────────────────────

def compute_labels(f: dict, future_candles: list) -> tuple:
    """
    f               : feature dict (with _ context keys)
    future_candles  : list of 2 raw dicts {open, high, low, close, volume}
    """
    atr  = max(f["_atr14"], EPS)
    vwap = f["_vwap"]
    ref  = f["_context_close"]

    c1   = future_candles[0]
    c2   = future_candles[1] if len(future_candles) >= 2 else c1

    c1_body_atr = abs(c1["close"] - c1["open"]) / atr
    c2_body_atr = abs(c2["close"] - c2["open"]) / atr
    c1_bull     = c1["close"] > c1["open"]
    c2_bull     = c2["close"] > c2["open"]
    net_atr     = (c2["close"] - ref) / atr
    seq_high    = max(c1["high"], c2["high"])
    seq_low     = min(c1["low"],  c2["low"])
    seq_range   = (seq_high - seq_low) / atr
    close_grav  = (c2["close"] - seq_low) / (seq_range * atr + EPS)

    # VWAP relationship of future candles
    c1_above_vwap = c1["close"] > vwap
    c2_above_vwap = c2["close"] > vwap
    c1_crossed_vwap = (ref > vwap) != c1_above_vwap  # crossed in c1

    # Context signals
    cum_d  = f["_cum_delta"]
    vol_z  = f["_vol_z30"]
    sigma  = f["_vwap_sigma"]
    slope  = f["_slope_z"]
    tick   = f["_tick_delta"]
    eff    = f["_vol_efficiency"]
    inst   = f["_inst_fp"]

    # ── HEAD 1: next_price_direction_vs_vwap [0=move_away_below 1=oscillate_near 2=move_away_above]
    if   net_atr > 0.35 and c2_above_vwap:   vwap_dir = 2
    elif net_atr < -0.35 and not c2_above_vwap: vwap_dir = 0
    else:                                        vwap_dir = 1

    # ── HEAD 2: next_volume_character [0=declining 1=stable 2=surge]
    avg_body = (c1_body_atr + c2_body_atr) / 2
    if   avg_body > 1.2:    next_vol_char = 2
    elif avg_body < 0.3:    next_vol_char = 0
    else:                    next_vol_char = 1

    # ── HEAD 3: vwap_interaction_next [0=rejection 1=test_return 2=reclaim_hold 3=walk_extension]
    if c1_crossed_vwap and c2_above_vwap:
        if c2_body_atr > 0.5 and avg_body > 0.7:  vwap_int = 2  # reclaim hold
        else:                                        vwap_int = 1  # test return
    elif c2_above_vwap and net_atr > 0.5:
        vwap_int = 3   # walk extension
    elif not c1_above_vwap and not c2_above_vwap and net_atr < -0.3:
        vwap_int = 0   # rejection
    else:
        vwap_int = 1   # test return / oscillate

    # ── HEAD 4: next_institutional_signal [0=distribution 1=neutral 2=accumulation]
    # Based on volume efficiency of future bars: high vol + small body = absorption/distribution
    if avg_body > 0.8 and seq_range > 1.5:
        if net_atr > 0.3:   next_inst = 2
        else:                next_inst = 0
    elif avg_body < 0.25:
        next_inst = 1   # thin bars = absorption or neutral
    else:
        next_inst = 1

    # ── HEAD 5: next_close_location [0=lower_third 1=mid 2=upper_third]
    if   close_grav > 0.67:  next_close = 2
    elif close_grav < 0.33:  next_close = 0
    else:                     next_close = 1

    # ── HEAD 6: vwap_follow_through [0=no 1=partial 2=confirmed]
    # Did price follow through in the direction implied by VWAP deviation + flow?
    bullish_context = (sigma > 0.3 and cum_d > 0.3 and slope > 0.2)
    bearish_context = (sigma < -0.3 and cum_d < -0.3 and slope < -0.2)

    if bullish_context:
        if net_atr > 0.60:   follow = 2
        elif net_atr > 0.15: follow = 1
        else:                 follow = 0
    elif bearish_context:
        if net_atr < -0.60:  follow = 2
        elif net_atr < -0.15:follow = 1
        else:                 follow = 0
    else:
        if avg_body > 0.8:   follow = 2
        elif avg_body > 0.3: follow = 1
        else:                 follow = 0

    return (int(vwap_dir), int(next_vol_char), int(vwap_int),
            int(next_inst), int(next_close), int(follow))


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
    total   = LOOKBACK + FUTURE_BARS

    for i in range(total):
        local_bias = sc["bias"] if random.random() > 0.08 else 1 - sc["bias"]

        if "vol_climax" in sc_name and i > LOOKBACK * 0.7:
            fade   = 1 - (i - LOOKBACK*0.7) / (LOOKBACK*0.3 + 1) * 0.65
            bs     = sc["body"] * max(fade, 0.12)
        elif "band_walk" in sc_name:
            bs     = sc["body"] * np.random.uniform(0.8, 1.3)
        else:
            bs     = sc["body"] * np.random.uniform(0.6, 1.4)

        c, price = _gen_candle(price, bs, sc["wl"], sc["wh"], sc["vl"], sc["vh"], local_bias)
        c["hour"] = (i * 3) % 24
        candles.append(c)

    df_context = pd.DataFrame(candles[:LOOKBACK])
    future     = candles[LOOKBACK:]

    hour_vol_avg = df_context.groupby("hour")["volume"].mean().to_dict()

    f, _ = compute_vwap_features(df_context, hour_vol_avg=hour_vol_avg)
    labels = compute_labels(f, future)

    feat = {k: v for k, v in f.items() if not k.startswith("_")}
    row  = dict(feat)
    (row["label_next_vwap_dir"],       row["label_next_vol_char"],
     row["label_vwap_interaction"],    row["label_next_inst_signal"],
     row["label_next_close_location"], row["label_vwap_follow_through"]) = labels
    return row


# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Generating {N_SAMPLES} VWAP/Volume samples — NEXT-CANDLE labels (FX M15) ...")
    rows = []
    for _ in trange(N_SAMPLES):
        rows.append(generate_sequence())

    df_out     = pd.DataFrame(rows)
    label_cols = [c for c in df_out.columns if c.startswith("label_")]
    feat_cols  = [c for c in df_out.columns if not c.startswith("label_")]
    df_out     = df_out[sorted(feat_cols) + label_cols]
    df_out.to_csv(OUTPUT_PATH, index=False)

    label_maps = {
        "label_next_vwap_dir":        {0:"below_away", 1:"oscillate_near", 2:"above_away"},
        "label_next_vol_char":        {0:"declining", 1:"stable", 2:"surge"},
        "label_vwap_interaction":     {0:"rejection", 1:"test_return", 2:"reclaim_hold", 3:"walk_extension"},
        "label_next_inst_signal":     {0:"distribution", 1:"neutral", 2:"accumulation"},
        "label_next_close_location":  {0:"lower_third", 1:"mid", 2:"upper_third"},
        "label_vwap_follow_through":  {0:"no", 1:"partial", 2:"confirmed"},
    }
    print(f"\n✅  Saved → {OUTPUT_PATH}  shape={df_out.shape}")
    print(f"    Features : {len(feat_cols)}")
    for lc in label_cols:
        counts = df_out[lc].value_counts().sort_index()
        named  = {label_maps[lc].get(k,k): f"{100*v/len(df_out):.1f}%" for k, v in counts.items()}
        print(f"    {lc}: {named}")
