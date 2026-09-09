# synthetic_breakout.py  ·  v2  (next-candle edition)
# ─────────────────────────────────────────────────────────────────────
# Generates training data for the Breakout / Squeeze model
# 32 features | 6 heads — ALL labels predict the NEXT 1-2 candles
# FX M15 | 14-100 bar compression → expansion cycle
# ─────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import random

try:
    from tqdm import trange
except ImportError:
    def trange(n, **kw): return range(n)

N_SAMPLES   = 20000
LOOKBACK    = 125          # bars; 120 context + 2 future candles for labelling
OUTPUT_PATH = "synthetic_breakout_dataset.csv"
BASE_PRICE  = 1.2000
SEED        = 42
random.seed(SEED)
np.random.seed(SEED)
EPS = 1e-12

CONTEXT_BARS = 120   # features computed on first 120 bars
FUTURE_BARS  = 2     # labels derived from candles 121-122


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
    h    = max(o, c) + wk
    l    = min(o, c) - tl
    v    = int(np.random.uniform(vl, vh))
    return {"open": o, "high": float(h), "low": float(l), "close": float(c), "volume": v}, c


def _rz(s, w):
    return (s - s.rolling(w).mean()) / (s.rolling(w).std() + EPS)


# ─────────────────────────────────────────────────────────────────────
# Feature computation (same as before — on context bars only)
# ─────────────────────────────────────────────────────────────────────

def compute_breakout_features(df: pd.DataFrame):
    """Returns (feature_dict, enriched_df). Run on CONTEXT_BARS only."""
    df = df.copy().reset_index(drop=True)

    df["range"]      = df["high"] - df["low"]
    df["body"]       = df["close"] - df["open"]
    df["abs_body"]   = df["body"].abs()
    df["dir"]        = np.sign(df["body"]).fillna(0)
    df["upper_wick"] = df["high"] - df[["open","close"]].max(axis=1)
    df["lower_wick"] = df[["open","close"]].min(axis=1) - df["low"]
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
        df[f"kc_mid_{w}"]   = mid
        df[f"kc_upper_{w}"] = mid + 1.5 * atr_w
        df[f"kc_lower_{w}"] = mid - 1.5 * atr_w

    df["bb_width_z"]    = _rz(df["bb_width_14"], 100).fillna(0.0)
    df["kc_width_n"]    = df["kc_width_14"] / (df["atr14"] + EPS)

    sq8  = (df["bb_width_8"]  < df["kc_width_8"]).astype(int)
    sq14 = (df["bb_width_14"] < df["kc_width_14"]).astype(int)
    sq20 = (df["bb_width_20"] < df["kc_width_20"]).astype(int)
    df["squeeze_stack"] = (sq8 + sq14 + sq20).astype(float)
    df["kb_ratio"]      = df["bb_width_14"] / (df["kc_width_14"] + EPS)
    df["kb_slope"]      = _rz(df["kb_ratio"].diff(1).rolling(10, min_periods=1).mean(), 20).fillna(0.0)

    sq_active_s = (df["kb_ratio"] < 1.0).astype(int)
    cons_sq     = []
    cur = 0
    for s in sq_active_s.tolist():
        cur = cur + 1 if s == 1 else 0
        cons_sq.append(cur)
    df["consec_squeeze"]    = cons_sq
    df["sq_active"]         = sq_active_s
    avg_sq = max(float(pd.Series(cons_sq).rolling(100, min_periods=5).mean().iloc[-1]), 1.0)
    df["squeeze_maturity"]  = (df["consec_squeeze"] / avg_sq).clip(0, 4)

    hv_pct                  = df["bb_width_14"].rank(pct=True).fillna(0.5)
    df["vol_potential_e"]   = ((1 - hv_pct) * (1 - df["kb_ratio"]).clip(0, 2)).clip(0, 2)

    don_high    = df["high"].rolling(20, min_periods=1).max()
    don_low     = df["low"].rolling(20,  min_periods=1).min()
    don_mid     = (don_high + don_low) / 2
    don_width   = (don_high - don_low) + EPS
    lean_raw    = ((df["close"] - don_mid) / don_width).rolling(3, min_periods=1).mean()
    df["sq_lean"] = (lean_raw * sq_active_s).fillna(0.0)

    df["atr_ratio"]         = df["atr5"] / (df["atr14"] + EPS)
    df["atr_ratio_z"]       = _rz(df["atr_ratio"], 50).fillna(0.0)
    df["exp_velocity"]      = _rz(df["bb_width_14"].diff(3).fillna(0), 50).fillna(0.0)
    above_upper             = (df["close"] - df["bb_upper_14"]).clip(lower=0) / (df["atr14"] + EPS)
    below_lower             = (df["bb_lower_14"] - df["close"]).clip(lower=0) / (df["atr14"] + EPS)
    df["band_walk_bull"]    = above_upper.rolling(5, min_periods=1).sum()
    df["band_walk_bear"]    = below_lower.rolling(5, min_periods=1).sum()
    df["hv_percentile"]     = hv_pct

    bb_range                = (df["bb_upper_14"] - df["bb_lower_14"]) + EPS
    df["bb_position"]       = (df["close"] - df["bb_lower_14"]) / bb_range
    df["bb_pos_z"]          = _rz(df["bb_position"], 50).fillna(0.0)
    df["kc_position"]       = (df["close"] - df["kc_lower_14"]) / (df["kc_width_14"] + EPS)
    mid_bb14                = df["close"].rolling(14, min_periods=1).mean()
    df["mr_dist"]           = ((df["close"] - mid_bb14) / (df["atr14"] + EPS)).clip(-3, 3)
    df["coil_tightness"]    = (1 - df["kb_ratio"]).clip(-1, 1)

    body_avg20              = df["abs_body"].rolling(20, min_periods=1).mean()
    df["body_comp_r"]       = (df["abs_body"] / (body_avg20 + EPS)).clip(0, 3)
    df["close_grav"]        = (df["close"] - df["low"]) / (df["range"] + EPS)
    vol_avg20               = df["volume"].rolling(20, min_periods=1).mean()
    df["vol_z20"]           = _rz(df["volume"], 20).fillna(0.0)
    df["prebreak_qual"]     = (df["close_grav"] * df["body_comp_r"] *
                                df["vol_z20"].clip(0, 3)).clip(-3, 3)
    df["intrabar_fill"]     = df["abs_body"] / (df["range"] + EPS)
    df["vol_sq_div"]        = (df["vol_z20"] * (df["squeeze_stack"] / 3.0)).clip(-3, 3)
    close_above             = (df["close"] > df["bb_upper_14"]).astype(float)
    close_below             = (df["close"] < df["bb_lower_14"]).astype(float)
    df["bkout_confirm"]     = ((close_above - close_below) *
                                (df["vol_z20"] + 1).clip(0, 3)).clip(-3, 3)
    df["don_bias"]          = ((df["close"] - don_mid) / don_width).clip(-1, 1)

    df["vol_z30"]           = _rz(df["volume"], 30).fillna(0.0)
    df["vol_z100"]          = _rz(df["volume"], min(100, len(df))).fillna(0.0)
    df["vol_trend"]         = _rz(
        df["volume"].rolling(5, min_periods=1).mean() /
        (vol_avg20 + EPS), 30).fillna(0.0)

    last = df.iloc[-1]

    feat = {
        "bb_width_z":         float(last["bb_width_z"]),
        "kc_width_n":         float(last["kc_width_n"]),
        "squeeze_stack":      float(last["squeeze_stack"]),
        "kb_ratio":           float(last["kb_ratio"]),
        "kb_slope":           float(last["kb_slope"]),
        "squeeze_maturity":   float(last["squeeze_maturity"]),
        "vol_potential_e":    float(last["vol_potential_e"]),
        "sq_lean":            float(last["sq_lean"]),
        "atr_ratio_z":        float(last["atr_ratio_z"]),
        "exp_velocity":       float(last["exp_velocity"]),
        "band_walk_bull":     float(last["band_walk_bull"]),
        "band_walk_bear":     float(last["band_walk_bear"]),
        "hv_percentile":      float(last["hv_percentile"]),
        "bb_position":        float(last["bb_position"]),
        "bb_pos_z":           float(last["bb_pos_z"]),
        "kc_position":        float(last["kc_position"]),
        "mr_dist":            float(last["mr_dist"]),
        "coil_tightness":     float(last["coil_tightness"]),
        "body_comp_r":        float(last["body_comp_r"]),
        "prebreak_qual":      float(last["prebreak_qual"]),
        "intrabar_fill":      float(last["intrabar_fill"]),
        "vol_sq_div":         float(last["vol_sq_div"]),
        "bkout_confirm":      float(last["bkout_confirm"]),
        "don_bias":           float(last["don_bias"]),
        "vol_z30":            float(last["vol_z30"]),
        "vol_z100":           float(last["vol_z100"]),
        "vol_trend":          float(last["vol_trend"]),
        # context for labelling (stripped before training)
        "_atr14":             float(last["atr14"]),
        "_bb_upper_14":       float(last["bb_upper_14"]),
        "_bb_lower_14":       float(last["bb_lower_14"]),
        "_kb_ratio":          float(last["kb_ratio"]),
        "_kb_slope":          float(last["kb_slope"]),
        "_sq_active":         float(last["sq_active"]),
        "_vol_z30":           float(last["vol_z30"]),
    }
    return feat, df


# ─────────────────────────────────────────────────────────────────────
# FORWARD LABEL GENERATION
# All labels answer: "what will happen in the NEXT 1-2 candles?"
# ─────────────────────────────────────────────────────────────────────

def compute_labels(f: dict, future_candles: list, context_close: float) -> tuple:
    """
    f               : feature dict from context bars
    future_candles  : list of 1-2 dicts {open, high, low, close, volume}
    context_close   : close price of the last context bar
    """
    atr       = max(f["_atr14"], EPS)
    c1        = future_candles[0]
    c2        = future_candles[1] if len(future_candles) >= 2 else c1

    # ── Derived future measurements ───────────────────────────────────

    # Net move over 2 candles (ATR-normalised)
    net_move_atr   = (c2["close"] - context_close) / atr

    # Direction of candle 1 and candle 2
    c1_bull        = c1["close"] > c1["open"]
    c2_bull        = c2["close"] > c2["open"]

    # Body size relative to ATR
    c1_body_atr    = abs(c1["close"] - c1["open"]) / atr
    c2_body_atr    = abs(c2["close"] - c2["open"]) / atr
    avg_body_atr   = (c1_body_atr + c2_body_atr) / 2

    # High/low of the 2-bar sequence
    seq_high       = max(c1["high"], c2["high"])
    seq_low        = min(c1["low"],  c2["low"])
    seq_range_atr  = (seq_high - seq_low) / atr

    # Close gravity of combined sequence
    seq_close_grav = (c2["close"] - seq_low) / (seq_range_atr * atr + EPS)

    # ── HEAD 1: next_direction [0=bearish 1=doji/flat 2=bullish]
    # Based on net 2-bar close-to-close move
    if   net_move_atr > 0.30:    next_dir = 2
    elif net_move_atr < -0.30:   next_dir = 0
    else:                         next_dir = 1

    # ── HEAD 2: next_magnitude [0=small <0.4ATR  1=medium 0.4-1.2ATR  2=large >1.2ATR]
    # Captures explosive vs quiet next candle(s)
    if   avg_body_atr > 1.20:    next_mag = 2
    elif avg_body_atr > 0.40:    next_mag = 1
    else:                         next_mag = 0

    # ── HEAD 3: next_pattern [0=continuation 1=reversal 2=indecision]
    # Does candle 2 continue or reverse candle 1's direction?
    if c1_bull == c2_bull:
        # both same direction = continuation
        if avg_body_atr > 0.35:  next_pat = 0
        else:                     next_pat = 2  # both small = indecision
    else:
        # direction flip = potential reversal
        if c2_body_atr > c1_body_atr * 0.80:  next_pat = 1   # strong flip
        else:                                   next_pat = 2   # weak flip = indecision

    # ── HEAD 4: next_volatility [0=contracting 1=normal 2=expanding]
    # Is the next 2-bar range expanding vs recent ATR?
    if   seq_range_atr > 2.5:    next_vol = 2
    elif seq_range_atr < 1.0:    next_vol = 0
    else:                         next_vol = 1

    # ── HEAD 5: next_close_location [0=lower_third 1=mid 2=upper_third]
    # Where does candle 2 close relative to the 2-bar high/low range?
    if   seq_close_grav > 0.67:  next_loc = 2
    elif seq_close_grav < 0.33:  next_loc = 0
    else:                         next_loc = 1

    # ── HEAD 6: breakout_follow_through [0=no 1=partial 2=yes]
    # Does price follow through in the direction implied by current context?
    sq_lean    = f["sq_lean"]
    bk_confirm = f["bkout_confirm"]
    expected_bull = (sq_lean > 0.05 or bk_confirm > 0.3)
    expected_bear = (sq_lean < -0.05 or bk_confirm < -0.3)

    if expected_bull:
        if   net_move_atr > 0.60:  follow = 2
        elif net_move_atr > 0.10:  follow = 1
        else:                       follow = 0
    elif expected_bear:
        if   net_move_atr < -0.60: follow = 2
        elif net_move_atr < -0.10: follow = 1
        else:                       follow = 0
    else:
        # No directional expectation — judge by magnitude
        if   avg_body_atr > 0.80:  follow = 2
        elif avg_body_atr > 0.30:  follow = 1
        else:                       follow = 0

    return (int(next_dir), int(next_mag), int(next_pat),
            int(next_vol), int(next_loc), int(follow))


# ─────────────────────────────────────────────────────────────────────
# Scenarios
# ─────────────────────────────────────────────────────────────────────

SCENARIOS = {
    "deep_squeeze":       dict(bias=0.50, body=0.0002, wl=0.05, wh=0.30, vl=100, vh=500,  sqb=70),
    "squeeze_forming":    dict(bias=0.50, body=0.0004, wl=0.10, wh=0.50, vl=200, vh=900,  sqb=25),
    "bull_breakout":      dict(bias=0.74, body=0.0014, wl=0.05, wh=0.28, vl=900, vh=3200, sqb=20),
    "bear_breakout":      dict(bias=0.26, body=0.0014, wl=0.05, wh=0.28, vl=900, vh=3200, sqb=20),
    "false_bull_break":   dict(bias=0.60, body=0.0008, wl=0.50, wh=1.50, vl=350, vh=1400, sqb=15),
    "false_bear_break":   dict(bias=0.40, body=0.0008, wl=0.50, wh=1.50, vl=350, vh=1400, sqb=15),
    "trending_expansion": dict(bias=0.68, body=0.0009, wl=0.10, wh=0.40, vl=600, vh=2000, sqb=0),
    "overdue_squeeze":    dict(bias=0.50, body=0.0002, wl=0.04, wh=0.22, vl=80,  vh=450,  sqb=90),
    "chop":               dict(bias=0.50, body=0.0005, wl=0.15, wh=0.60, vl=200, vh=900,  sqb=0),
    "vol_climax":         dict(bias=0.55, body=0.0016, wl=0.08, wh=0.35, vl=2000,vh=6000, sqb=10),
}
WEIGHTS = [0.10, 0.10, 0.12, 0.12, 0.08, 0.08, 0.10, 0.10, 0.12, 0.08]


def generate_sequence(base_price=BASE_PRICE):
    sc_name = random.choices(list(SCENARIOS.keys()), weights=WEIGHTS, k=1)[0]
    sc      = SCENARIOS[sc_name]
    price   = base_price + np.random.normal(0, 0.0010)
    sqb     = sc["sqb"]
    candles = []

    # Generate CONTEXT_BARS + FUTURE_BARS candles
    total_bars = CONTEXT_BARS + FUTURE_BARS

    for i in range(total_bars):
        local_bias = sc["bias"] if random.random() > 0.08 else 1 - sc["bias"]

        if i < sqb:
            shrink = max(0.12, 1 - i / (sqb + 1) * 0.65)
            bs     = sc["body"] * shrink * np.random.uniform(0.6, 1.1)
            vl, vh = sc["vl"] * 0.45, sc["vh"] * 0.45
            wl, wh = sc["wl"] * 0.4, sc["wh"] * 0.6
        else:
            bs     = sc["body"] * np.random.uniform(0.7, 1.4)
            vl, vh = sc["vl"], sc["vh"]
            wl, wh = sc["wl"], sc["wh"]
            if sc_name in ("false_bull_break","false_bear_break") and i > CONTEXT_BARS - 12:
                wl *= 1.5; wh *= 2.2

        c, price = _gen_candle(price, bs, wl, wh, vl, vh, local_bias)
        candles.append(c)

    df_context = pd.DataFrame(candles[:CONTEXT_BARS])
    future     = candles[CONTEXT_BARS:]  # list of 2 dicts

    f, _       = compute_breakout_features(df_context)
    context_close = df_context["close"].iloc[-1]
    labels     = compute_labels(f, future, context_close)

    feat = {k: v for k, v in f.items() if not k.startswith("_")}
    row  = dict(feat)
    (row["label_next_direction"], row["label_next_magnitude"],  row["label_next_pattern"],
     row["label_next_volatility"],row["label_next_close_loc"],  row["label_bk_follow_through"]) = labels
    return row


# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Generating {N_SAMPLES} breakout samples — NEXT-CANDLE labels (FX M15) ...")
    rows = []
    for _ in trange(N_SAMPLES):
        rows.append(generate_sequence())

    df_out     = pd.DataFrame(rows)
    label_cols = [c for c in df_out.columns if c.startswith("label_")]
    feat_cols  = [c for c in df_out.columns if not c.startswith("label_")]
    df_out     = df_out[sorted(feat_cols) + label_cols]
    df_out.to_csv(OUTPUT_PATH, index=False)

    label_maps = {
        "label_next_direction":     {0:"bearish", 1:"flat", 2:"bullish"},
        "label_next_magnitude":     {0:"small", 1:"medium", 2:"large"},
        "label_next_pattern":       {0:"continuation", 1:"reversal", 2:"indecision"},
        "label_next_volatility":    {0:"contracting", 1:"normal", 2:"expanding"},
        "label_next_close_loc":     {0:"lower_third", 1:"mid", 2:"upper_third"},
        "label_bk_follow_through":  {0:"no", 1:"partial", 2:"yes"},
    }
    print(f"\n✅  Saved → {OUTPUT_PATH}  shape={df_out.shape}")
    print(f"    Features : {len(feat_cols)}")
    for lc in label_cols:
        counts = df_out[lc].value_counts().sort_index()
        named  = {label_maps[lc][k]: f"{100*v/len(df_out):.1f}%" for k, v in counts.items()}
        print(f"    {lc}: {named}")
