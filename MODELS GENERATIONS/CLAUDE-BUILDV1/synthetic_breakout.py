# synthetic_breakout.py
# Generates training data for the Breakout / Squeeze model
# 32 features | 6 heads | 14-100 bar compression → expansion cycle | FX M15
import pandas as pd
import numpy as np
import random

try:
    from tqdm import trange
except ImportError:
    def trange(n, **kw): return range(n)

N_SAMPLES   = 20000
LOOKBACK    = 120          # bars; need 100+ for z-score stability
OUTPUT_PATH = "synthetic_breakout_dataset.csv"
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
    h    = max(o, c) + wk
    l    = min(o, c) - tl
    v    = int(np.random.uniform(vl, vh))
    return {"open": o, "high": float(h), "low": float(l), "close": float(c), "volume": v}, c


def _rz(s, w):
    return (s - s.rolling(w).mean()) / (s.rolling(w).std() + EPS)


# ─────────────────────────────────────────────────────────────────────
# Feature computation (matches deploy exactly)
# ─────────────────────────────────────────────────────────────────────

def compute_breakout_features(df: pd.DataFrame):
    """Returns (feature_dict, enriched_df)."""
    df = df.copy().reset_index(drop=True)

    df["range"]      = df["high"] - df["low"]
    df["body"]       = df["close"] - df["open"]
    df["abs_body"]   = df["body"].abs()
    df["dir"]        = np.sign(df["body"]).fillna(0)
    df["upper_wick"] = df["high"] - df[["open","close"]].max(axis=1)
    df["lower_wick"] = df[["open","close"]].min(axis=1) - df["low"]
    df["atr14"]      = df["range"].rolling(14, min_periods=1).mean()
    df["atr5"]       = df["range"].rolling(5,  min_periods=1).mean()

    # Bollinger + Keltner for squeeze_stack (3 window sizes)
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

    # ── GROUP A (8) ───────────────────────────────────────────────────

    # A1: bb_width_z
    df["bb_width_z"]    = _rz(df["bb_width_14"], 100).fillna(0.0)

    # A2: kc_width (ATR-normalised)
    df["kc_width_n"]    = df["kc_width_14"] / (df["atr14"] + EPS)

    # A3: squeeze_stack
    sq8  = (df["bb_width_8"]  < df["kc_width_8"]).astype(int)
    sq14 = (df["bb_width_14"] < df["kc_width_14"]).astype(int)
    sq20 = (df["bb_width_20"] < df["kc_width_20"]).astype(int)
    df["squeeze_stack"] = (sq8 + sq14 + sq20).astype(float)

    # A4: kb_ratio
    df["kb_ratio"]      = df["bb_width_14"] / (df["kc_width_14"] + EPS)

    # A5: kb_slope — inflection point when squeeze releases
    df["kb_slope"]      = _rz(df["kb_ratio"].diff(1).rolling(10, min_periods=1).mean(), 20).fillna(0.0)

    # A6: squeeze_maturity (time dimension)
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

    # A7: vol_potential_energy
    hv_pct                  = df["bb_width_14"].rank(pct=True).fillna(0.5)
    df["vol_potential_e"]   = ((1 - hv_pct) * (1 - df["kb_ratio"]).clip(0, 2)).clip(0, 2)

    # A8: squeeze_directional_lean (price lean inside donchian × squeeze flag)
    don_high    = df["high"].rolling(20, min_periods=1).max()
    don_low     = df["low"].rolling(20,  min_periods=1).min()
    don_mid     = (don_high + don_low) / 2
    don_width   = (don_high - don_low) + EPS
    lean_raw    = ((df["close"] - don_mid) / don_width).rolling(3, min_periods=1).mean()
    df["sq_lean"] = (lean_raw * sq_active_s).fillna(0.0)

    # ── GROUP B (5) ───────────────────────────────────────────────────

    df["atr_ratio"]         = df["atr5"] / (df["atr14"] + EPS)
    df["atr_ratio_z"]       = _rz(df["atr_ratio"], 50).fillna(0.0)
    df["exp_velocity"]      = _rz(df["bb_width_14"].diff(3).fillna(0), 50).fillna(0.0)
    above_upper             = (df["close"] - df["bb_upper_14"]).clip(lower=0) / (df["atr14"] + EPS)
    below_lower             = (df["bb_lower_14"] - df["close"]).clip(lower=0) / (df["atr14"] + EPS)
    df["band_walk_bull"]    = above_upper.rolling(5, min_periods=1).sum()
    df["band_walk_bear"]    = below_lower.rolling(5, min_periods=1).sum()
    df["hv_percentile"]     = hv_pct

    # ── GROUP C (5) ───────────────────────────────────────────────────

    bb_range                = (df["bb_upper_14"] - df["bb_lower_14"]) + EPS
    df["bb_position"]       = (df["close"] - df["bb_lower_14"]) / bb_range
    df["bb_pos_z"]          = _rz(df["bb_position"], 50).fillna(0.0)
    df["kc_position"]       = (df["close"] - df["kc_lower_14"]) / (df["kc_width_14"] + EPS)
    mid_bb14                = df["close"].rolling(14, min_periods=1).mean()
    df["mr_dist"]           = ((df["close"] - mid_bb14) / (df["atr14"] + EPS)).clip(-3, 3)
    df["coil_tightness"]    = (1 - df["kb_ratio"]).clip(-1, 1)

    # ── GROUP D (6) ───────────────────────────────────────────────────

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

    # ── GROUP E (3) ───────────────────────────────────────────────────

    df["vol_z30"]           = _rz(df["volume"], 30).fillna(0.0)
    df["vol_z100"]          = _rz(df["volume"], min(100, len(df))).fillna(0.0)
    df["vol_trend"]         = _rz(
        df["volume"].rolling(5, min_periods=1).mean() /
        (vol_avg20 + EPS), 30).fillna(0.0)

    last = df.iloc[-1]

    feat = {
        # A
        "bb_width_z":         float(last["bb_width_z"]),
        "kc_width_n":         float(last["kc_width_n"]),
        "squeeze_stack":      float(last["squeeze_stack"]),
        "kb_ratio":           float(last["kb_ratio"]),
        "kb_slope":           float(last["kb_slope"]),
        "squeeze_maturity":   float(last["squeeze_maturity"]),
        "vol_potential_e":    float(last["vol_potential_e"]),
        "sq_lean":            float(last["sq_lean"]),
        # B
        "atr_ratio_z":        float(last["atr_ratio_z"]),
        "exp_velocity":       float(last["exp_velocity"]),
        "band_walk_bull":     float(last["band_walk_bull"]),
        "band_walk_bear":     float(last["band_walk_bear"]),
        "hv_percentile":      float(last["hv_percentile"]),
        # C
        "bb_position":        float(last["bb_position"]),
        "bb_pos_z":           float(last["bb_pos_z"]),
        "kc_position":        float(last["kc_position"]),
        "mr_dist":            float(last["mr_dist"]),
        "coil_tightness":     float(last["coil_tightness"]),
        # D
        "body_comp_r":        float(last["body_comp_r"]),
        "prebreak_qual":      float(last["prebreak_qual"]),
        "intrabar_fill":      float(last["intrabar_fill"]),
        "vol_sq_div":         float(last["vol_sq_div"]),
        "bkout_confirm":      float(last["bkout_confirm"]),
        "don_bias":           float(last["don_bias"]),
        # E
        "vol_z30":            float(last["vol_z30"]),
        "vol_z100":           float(last["vol_z100"]),
        "vol_trend":          float(last["vol_trend"]),
        # context for labelling only (prefixed _ — stripped before training)
        "_sq_active":         float(last["sq_active"]),
        "_consec_sq":         float(last["consec_squeeze"]),
        "_bkout_bull":        float(close_above.iloc[-1]),
        "_bkout_bear":        float(close_below.iloc[-1]),
        "_bwb":               float(last["band_walk_bull"]),
    }
    return feat, df


# ─────────────────────────────────────────────────────────────────────
# Label generation
# ─────────────────────────────────────────────────────────────────────

def compute_labels(f: dict) -> tuple:
    sq_stack  = f["squeeze_stack"]
    kb        = f["kb_ratio"]
    kbs       = f["kb_slope"]
    sq_mat    = f["squeeze_maturity"]
    sq_lean   = f["sq_lean"]
    sq_active = f["_sq_active"]
    aratio_z  = f["atr_ratio_z"]
    exp_vel   = f["exp_velocity"]
    bb_pos    = f["bb_position"]
    bwb       = f["band_walk_bull"]
    bwbe      = f["band_walk_bear"]
    hv_pct    = f["hv_percentile"]
    bkout_b   = f["_bkout_bull"]
    bkout_be  = f["_bkout_bear"]
    bk_conf   = f["bkout_confirm"]
    don_bias  = f["don_bias"]
    vol_z30   = f["vol_z30"]
    body_comp = f["body_comp_r"]

    # ── HEAD 1: squeeze_state [0=no_squeeze 1=forming 2=active 3=overdue]
    if   sq_mat > 1.8 and sq_active:           sq_state = 3
    elif sq_stack >= 2 and sq_active:           sq_state = 2
    elif sq_stack >= 1 or kb < 1.15:           sq_state = 1
    else:                                       sq_state = 0

    # ── HEAD 2: breakout_direction [0=bearish 1=neutral 2=bullish]
    # Relax thresholds — use bb_pos + don_bias as primary signal
    if   (sq_lean > 0.05 or don_bias > 0.12) and bb_pos > 0.55:   bk_dir = 2
    elif (sq_lean < -0.05 or don_bias < -0.12) and bb_pos < 0.45:  bk_dir = 0
    else:                                                            bk_dir = 1

    # ── HEAD 3: breakout_quality [0=no_bk 1=false 2=developing 3=confirmed]
    if bkout_b > 0 or bkout_be > 0:
        real = kbs > 0.2 and vol_z30 > 0.2 and body_comp > 0.8
        if   real and abs(bk_conf) > 0.6:      bk_qual = 3
        elif real:                              bk_qual = 2
        else:                                   bk_qual = 1
    else:
        # developing = close near band + good vol + kb_slope turning
        if kbs > 0.5 and vol_z30 > 0.4 and body_comp > 1.1 and (bb_pos > 0.90 or bb_pos < 0.10):
            bk_qual = 2
        else:
            bk_qual = 0

    # ── HEAD 4: volatility_phase [0=deeply_compressing 1=compressing 2=transitioning 3=expanding]
    if   aratio_z > 1.0 or exp_vel > 1.2:      vol_phase = 3
    elif kbs > 0.3 and sq_active:              vol_phase = 2
    elif hv_pct < 0.30 and sq_active:          vol_phase = 0
    elif sq_active:                             vol_phase = 1
    else:                                       vol_phase = 3

    # ── HEAD 5: band_structure [0=lower_walk 1=mid_range 2=upper_walk]
    if   bwb > 0.3 and bb_pos > 0.65:          band_str = 2
    elif bwbe > 0.3 and bb_pos < 0.35:         band_str = 0
    else:                                       band_str = 1

    # ── HEAD 6: entry_timing [0=avoid 1=wait 2=now]
    act   = (sq_state in (2,3) and vol_phase in (2,3) and
             bk_qual >= 2 and bk_dir != 1 and vol_z30 > 0.2)
    avoid = (bk_qual == 1 or sq_state == 0 or vol_phase == 0)
    if   act:   timing = 2
    elif avoid: timing = 0
    else:       timing = 1

    return (int(sq_state), int(bk_dir), int(bk_qual),
            int(vol_phase), int(band_str), int(timing))


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

    for i in range(LOOKBACK):
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
            if sc_name in ("false_bull_break","false_bear_break") and i > LOOKBACK - 12:
                wl *= 1.5; wh *= 2.2

        c, price = _gen_candle(price, bs, wl, wh, vl, vh, local_bias)
        candles.append(c)

    df     = pd.DataFrame(candles)
    f, _   = compute_breakout_features(df)
    labels = compute_labels(f)

    feat = {k: v for k, v in f.items() if not k.startswith("_")}
    row  = dict(feat)
    (row["label_sq_state"], row["label_bk_dir"],   row["label_bk_qual"],
     row["label_vol_phase"],row["label_band_str"],  row["label_entry_timing"]) = labels
    return row


# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Generating {N_SAMPLES} breakout samples (FX M15) ...")
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
