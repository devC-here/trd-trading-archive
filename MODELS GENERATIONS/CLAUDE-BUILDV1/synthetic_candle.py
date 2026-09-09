# synthetic_candle.py
# Generates training data for the Candle Sequence model
# 30 features | 6 heads | 10-bar sequential auction narrative | FX M15
import pandas as pd
import numpy as np
import random
from scipy import stats as sp_stats

try:
    from tqdm import trange
except ImportError:
    def trange(n, **kw): return range(n)

N_SAMPLES   = 20000
WIN         = 10           # bar window for sequence features
LOOKBACK    = 50           # total bars fetched; last WIN used for features
OUTPUT_PATH = "synthetic_candle_dataset.csv"
BASE_PRICE  = 1.2000
SEED        = 42
random.seed(SEED)
np.random.seed(SEED)
EPS = 1e-12


# ─────────────────────────────────────────────────────────────────────
# Candle generator
# ─────────────────────────────────────────────────────────────────────

def generate_candle(price, body_scale, wick_lo, wick_hi, vol_lo, vol_hi, bias):
    direction = 1 if random.random() < bias else -1
    body      = float(np.random.uniform(0.1, 1.0) * body_scale)
    wk        = float(body * np.random.uniform(wick_lo, wick_hi))
    tl        = float(body * np.random.uniform(wick_lo, wick_hi))
    o  = float(price)
    c  = float(o + direction * body)
    h  = max(o, c) + wk
    l  = min(o, c) - tl
    v  = int(np.random.uniform(vol_lo, vol_hi))
    return {"open": o, "high": float(h), "low": float(l), "close": float(c), "volume": v}, c


def _ols_slope(arr):
    """Normalised OLS slope of a 1-D array."""
    n = len(arr)
    if n < 2: return 0.0
    s = np.std(arr)
    if s < EPS: return 0.0
    x = np.arange(n, dtype=float)
    slope, *_ = np.polyfit(x, arr, 1)
    return float(slope / (s + EPS))


def _safe_corr(a, b):
    if np.std(a) < EPS or np.std(b) < EPS: return 0.0
    r, _ = sp_stats.pearsonr(a, b)
    return float(r) if np.isfinite(r) else 0.0


# ─────────────────────────────────────────────────────────────────────
# Feature computation over a LOOKBACK-bar DataFrame
# Returns dict of 30 features from the final WIN bars
# ─────────────────────────────────────────────────────────────────────

def compute_candle_features(df: pd.DataFrame) -> dict:
    df    = df.copy().reset_index(drop=True)
    n_all = len(df)

    # ── full-df base columns (need rolling context) ───────────────────
    df["range"]      = df["high"] - df["low"]
    df["body"]       = df["close"] - df["open"]
    df["abs_body"]   = df["body"].abs()
    df["dir"]        = np.sign(df["body"]).fillna(0)
    df["upper_wick"] = df["high"] - df[["open","close"]].max(axis=1)
    df["lower_wick"] = df[["open","close"]].min(axis=1) - df["low"]

    # atr for normalisation (full lookback)
    df["atr14"]      = df["range"].rolling(14, min_periods=1).mean()

    # relative sizing baselines
    df["abs_body_avg30"] = df["abs_body"].rolling(30, min_periods=1).mean()
    df["range_avg30"]    = df["range"].rolling(30, min_periods=1).mean()
    df["vol_avg20"]      = df["volume"].rolling(20, min_periods=1).mean()

    # body_ratio
    df["body_ratio"]  = df["abs_body"] / (df["range"] + EPS)
    df["body_rat_avg5"]= df["body_ratio"].rolling(5, min_periods=1).mean()

    # close gravity (intrabar vote)
    df["close_grav"]  = (df["close"] - df["low"]) / (df["range"] + EPS)

    # shadow dominance (signed)
    df["shadow_dom"]  = (df["lower_wick"] - df["upper_wick"]) / (df["range"] + EPS)

    # directional efficiency
    bull = df["dir"] >= 0
    df["dir_eff"]     = np.where(bull,
                                  df["body_ratio"] * df["close_grav"],
                                  df["body_ratio"] * (1 - df["close_grav"]))

    # volume weight (relative)
    df["vol_weight"]  = (df["volume"] / (df["vol_avg20"] + EPS)).clip(0, 4)

    # Heikin-Ashi body (noise-filtered direction)
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_open  = ha_close.ewm(span=3, adjust=False).mean()
    df["ha_body"]     = (ha_close - ha_open) / (df["atr14"] + EPS)

    # consecutive run length
    cons, cur, prev = [], 0, 0
    for d in df["dir"].fillna(0).tolist():
        if d == prev and d != 0:  cur += 1
        elif d != 0:               cur  = 1
        else:                      cur  = 0
        cons.append(cur)
        prev = d
    df["consec_len"]  = cons

    # ── extract the last WIN bars ─────────────────────────────────────
    w  = df.iloc[-WIN:].reset_index(drop=True)
    atr_val = float(df["atr14"].iloc[-1])

    # ────────────────────────────────────────────────────────────────
    # GROUP A — last-bar anatomy  (5 features)
    # ────────────────────────────────────────────────────────────────
    f = {}
    f["lb_close_grav"]   = float(w["close_grav"].iloc[-1])
    f["lb_body_ratio"]   = float(w["body_ratio"].iloc[-1])
    f["lb_shadow_dom"]   = float(w["shadow_dom"].iloc[-1])
    f["lb_vol_weight"]   = float(min(w["vol_weight"].iloc[-1], 4.0))
    f["lb_dir_eff"]      = float(w["dir_eff"].iloc[-1])

    # ────────────────────────────────────────────────────────────────
    # GROUP B — sequence narrative physics  (8 features)
    # ────────────────────────────────────────────────────────────────

    # B1: progressive close gravity slope
    f["seq_cg_slope"]    = _ols_slope(w["close_grav"].values)

    # B2: body entropy (std/mean of body_ratio over window)
    bm = float(w["body_ratio"].mean())
    bs = float(w["body_ratio"].std())
    f["body_entropy"]    = float(bs / (bm + EPS))

    # B3: wick pressure slope (shadow_dom OLS slope)
    f["wick_press_slope"]= _ols_slope(w["shadow_dom"].values)

    # B4: volume narrative (OLS slope of signed directional conviction)
    bar_conv = w["dir"] * w["body_ratio"] * w["vol_weight"]
    f["vol_narrative"]   = _ols_slope(bar_conv.values)

    # B5: sequence close harmony (corr of close_grav vs range over WIN)
    f["seq_harmony"]     = _safe_corr(w["close_grav"].values,
                                       w["range"].values)

    # B6: momentum handoff (direction change mid-sequence)
    early_dir = np.sign(w["body"].iloc[:5].sum())
    late_dir  = np.sign(w["body"].iloc[5:].sum())
    f["handoff"]         = float(early_dir != late_dir)
    early_body = float(w["body_ratio"].iloc[:5].mean())
    late_body  = float(w["body_ratio"].iloc[5:].mean())
    f["handoff_str"]     = float((late_body - early_body) / (atr_val + EPS))

    # B7: sequence acceleration (late 5 vs early 5)
    full_mean = float(w["body_ratio"].mean()) + EPS
    f["seq_accel"]       = float((late_body - early_body) / full_mean)

    # ────────────────────────────────────────────────────────────────
    # GROUP C — pattern physics as continuous scores  (7 features)
    # ────────────────────────────────────────────────────────────────

    # C1: engulf quality (continuous)
    curr_body  = float(w["abs_body"].iloc[-1])
    prior_body = float(w["abs_body"].iloc[-2]) if len(w) >= 2 else curr_body
    curr_dir   = float(w["dir"].iloc[-1])
    f["engulf_qual"]     = float((curr_body - prior_body) / (atr_val + EPS) * (curr_dir or 1))

    # C2: inside bar compression energy
    inside_count = 0
    for i in range(1, len(w)):
        if w["high"].iloc[i] < w["high"].iloc[i-1] and w["low"].iloc[i] > w["low"].iloc[i-1]:
            inside_count += 1
        else:
            break   # only consecutive from the end
    mother_range = float(w["range"].iloc[0]) if inside_count > 0 else 0.0
    f["inside_energy"]   = float(inside_count * mother_range / (atr_val + EPS))

    # C3+C4: Heikin-Ashi body & slope (last bar)
    f["ha_body_last"]    = float(w["ha_body"].iloc[-1])
    ha_arr = w["ha_body"].values
    f["ha_body_slope"]   = float((ha_arr[-1] - ha_arr[max(0,len(ha_arr)-4)]) / (atr_val + EPS))

    # C5: pin bull quality (continuous)
    uw  = float(w["upper_wick"].iloc[-1])
    lw  = float(w["lower_wick"].iloc[-1])
    rng = float(w["range"].iloc[-1])
    br  = float(w["body_ratio"].iloc[-1])
    cg  = float(w["close_grav"].iloc[-1])
    body_small = 1.0 if br < 0.35 else 0.5
    f["pin_bull_qual"]   = float((lw / (rng + EPS)) * body_small * cg)
    f["pin_bear_qual"]   = float((uw / (rng + EPS)) * body_small * (1 - cg))

    # C7: doji context (doji meaning relative to 10-bar position)
    closes      = w["close"].values
    c_min, c_max= float(closes.min()), float(closes.max())
    seq_pos     = (closes[-1] - c_min) / (c_max - c_min + EPS)
    doji_score  = float(max(0, 0.15 - br) / 0.15)
    f["doji_context"]    = float(doji_score * (seq_pos - 0.5))

    # ────────────────────────────────────────────────────────────────
    # GROUP D — relative sizing  (3 features)
    # ────────────────────────────────────────────────────────────────
    f["body_vs_avg"]     = float(min(float(df["abs_body"].iloc[-1]) /
                                      (float(df["abs_body_avg30"].iloc[-1]) + EPS), 4.0))
    f["range_vs_avg"]    = float(min(float(df["range"].iloc[-1]) /
                                      (float(df["range_avg30"].iloc[-1]) + EPS), 4.0))
    br5_avg = float(df["body_rat_avg5"].iloc[-1])
    f["body_accel"]      = float(br - br5_avg)

    # ────────────────────────────────────────────────────────────────
    # GROUP E — fractal body acceleration  (2 features)
    # ────────────────────────────────────────────────────────────────
    br1  = float(w["body_ratio"].iloc[-1])
    br3  = float(w["body_ratio"].iloc[-3:].mean()) if len(w) >= 3 else br1
    br10 = float(w["body_ratio"].mean())
    price_up = closes[-1] > closes[0]
    f["fractal_bull"]    = float((br1 > br3 > br10) and price_up)
    f["fractal_bear"]    = float((br1 > br3 > br10) and not price_up)

    # ────────────────────────────────────────────────────────────────
    # GROUP F — supplementary sequence stats  (5 features)
    # ────────────────────────────────────────────────────────────────

    # F1: sequential direction ratio (fraction of bullish bars in window)
    f["seq_dir_ratio"]   = float((w["dir"] > 0).mean())

    # F2: consecutive run length (last bar)
    f["consec_len"]      = float(w["consec_len"].iloc[-1])

    # F3: vol climax divergence (body per unit of volume relative surge)
    vol_surge_w          = float(w["vol_weight"].mean())   # avg relative vol in window
    body_ratio_mean_w    = float(w["body_ratio"].mean()) + EPS
    f["vol_climax_div"]  = float(body_ratio_mean_w / (vol_surge_w + EPS))

    # F4: HA trend strength (EWM of HA body over window)
    f["ha_trend_str"]    = float(w["ha_body"].ewm(span=5, adjust=False).mean().iloc[-1])

    # F5: sequence body ratio average (smoothed conviction baseline)
    f["seq_body_avg"]    = float(w["body_ratio"].mean())

    return f


# ─────────────────────────────────────────────────────────────────────
# Label generation — 6 heads
# ─────────────────────────────────────────────────────────────────────

def compute_labels(f: dict, w: pd.DataFrame) -> tuple:

    br    = f["lb_body_ratio"]
    cg    = f["lb_close_grav"]
    de    = f["lb_dir_eff"]
    bva   = f["body_vs_avg"]
    cgs   = f["seq_cg_slope"]
    vn    = f["vol_narrative"]
    ha    = f["ha_body_last"]
    acc   = f["seq_accel"]
    entr  = f["body_entropy"]
    harm  = f["seq_harmony"]
    ic    = f["inside_energy"]
    fb    = f["fractal_bull"]
    fbr   = f["fractal_bear"]
    haslp = f["ha_body_slope"]
    pin_b = f["pin_bull_qual"]
    pin_e = f["pin_bear_qual"]
    doji  = abs(f["doji_context"])
    hoff  = f["handoff"]
    hstr  = f["handoff_str"]
    cons  = int(f["consec_len"])
    sdr   = f["seq_dir_ratio"]     # fraction of bullish bars
    vcd   = f["vol_climax_div"]
    hts   = f["ha_trend_str"]

    # ── HEAD 1: pattern_type [0=impulse_bull 1=impulse_bear 2=pin_bull 3=pin_bear 4=inside_coil 5=indecision]
    if   pin_b > 0.28:                                     pat = 2
    elif pin_e > 0.28:                                     pat = 3
    elif ic > 0.5:                                         pat = 4
    elif doji > 0.04 and br < 0.12:                        pat = 5
    elif de > 0.35 and cg > 0.58 and bva > 1.0:           pat = 0
    elif de > 0.35 and cg < 0.42 and bva > 1.0:           pat = 1
    else:                                                  pat = 0 if cg >= 0.5 else 1

    # ── HEAD 2: sequence_dominance [0=bear 1=contested 2=bull]
    # Use seq_dir_ratio + cg_slope + HA as the three pillars
    if   sdr > 0.65 and cgs > 0.10 and hts > 0.05:       seq_dom = 2
    elif sdr < 0.35 and cgs < -0.10 and hts < -0.05:     seq_dom = 0
    else:                                                  seq_dom = 1

    # ── HEAD 3: conviction_level [0=low 1=medium 2=high]
    if   entr < 0.45 and harm > 0.25 and acc > 0.02:     conv = 2
    elif entr > 0.75 or harm < -0.15 or vcd < 0.20:      conv = 0
    else:                                                  conv = 1

    # ── HEAD 4: wick_pressure [0=upper 1=balanced 2=lower]
    wps = f["wick_press_slope"]
    sdm = float(w["shadow_dom"].mean())
    if   wps > 0.15 or sdm > 0.12:    wick_p = 2
    elif wps < -0.15 or sdm < -0.12:  wick_p = 0
    else:                              wick_p = 1

    # ── HEAD 5: sequence_phase [0=coiling 1=ignition 2=trending 3=exhausting]
    if   acc < -0.10 and cons >= 3:                        phase = 3
    elif ic > 0.8 or (entr > 0.65 and f["range_vs_avg"] < 0.88): phase = 0
    elif fb or fbr:                                        phase = 1
    elif haslp > 0 and acc >= 0 and entr < 0.60:          phase = 2
    else:                                                  phase = 2

    # ── HEAD 6: candle_signal [0=avoid 1=wait 2=act]
    # act: clear pattern + direction aligned + conviction + good phase
    dir_ok = ((pat == 0 and seq_dom == 2) or
               (pat == 1 and seq_dom == 0) or
               (pat in (2,3)) or
               (pat in (0,1) and seq_dom == 1 and conv == 2))
    act_cond  = (pat in (0,1,2,3) and dir_ok and
                 conv >= 1 and phase in (1,2) and bva > 1.0)
    avoid_cond= (phase == 3 or
                 (conv == 0) or
                 (pat == 5 and phase == 0))
    if   act_cond:    sig = 2
    elif avoid_cond:  sig = 0
    else:             sig = 1

    return (int(pat), int(seq_dom), int(conv), int(wick_p), int(phase), int(sig))


# ─────────────────────────────────────────────────────────────────────
# Scenario definitions
# ─────────────────────────────────────────────────────────────────────

SCENARIOS = {
    "bull_impulse":   dict(bias=0.75, body=0.0009, wl=0.05, wh=0.35, vl=600,  vh=2200),
    "bear_impulse":   dict(bias=0.25, body=0.0009, wl=0.05, wh=0.35, vl=600,  vh=2200),
    "pin_bull":       dict(bias=0.52, body=0.0003, wl=1.0,  wh=3.0,  vl=500,  vh=1800),
    "pin_bear":       dict(bias=0.48, body=0.0003, wl=1.0,  wh=3.0,  vl=500,  vh=1800),
    "inside_coil":    dict(bias=0.50, body=0.0004, wl=0.10, wh=0.60, vl=200,  vh=900),
    "bull_trend":     dict(bias=0.68, body=0.0007, wl=0.10, wh=0.50, vl=500,  vh=1800),
    "bear_trend":     dict(bias=0.32, body=0.0007, wl=0.10, wh=0.50, vl=500,  vh=1800),
    "exhaustion":     dict(bias=0.70, body=0.0006, wl=0.10, wh=0.40, vl=400,  vh=2000),
    "chop":           dict(bias=0.50, body=0.0003, wl=0.10, wh=0.50, vl=150,  vh=700),
    "reversal_bull":  dict(bias=0.62, body=0.0007, wl=0.15, wh=0.55, vl=500,  vh=1900),
}
WEIGHTS = [0.10, 0.10, 0.08, 0.08, 0.10, 0.12, 0.12, 0.10, 0.12, 0.08]


def generate_sequence(base_price=BASE_PRICE):
    sc_name = random.choices(list(SCENARIOS.keys()), weights=WEIGHTS, k=1)[0]
    sc      = SCENARIOS[sc_name]
    price   = base_price + np.random.normal(0, 0.0008)
    candles = []

    for i in range(LOOKBACK):
        local_bias = sc["bias"] if random.random() > 0.10 else 1 - sc["bias"]

        # Inside-coil: progressive range shrinkage in latter half
        if sc_name == "inside_coil" and i > LOOKBACK // 2:
            shrink = 1 - (i - LOOKBACK//2) / (LOOKBACK//2 + 1) * 0.60
            bs     = sc["body"] * shrink
        # Exhaustion: body shrinks while bias stays directional
        elif sc_name == "exhaustion" and i > LOOKBACK * 0.6:
            fade   = 1 - (i - LOOKBACK*0.6) / (LOOKBACK*0.4 + 1) * 0.70
            bs     = sc["body"] * max(fade, 0.15)
        else:
            bs = sc["body"]

        # Pin-bar: force a long wick on a random bar
        wl, wh = sc["wl"], sc["wh"]
        if sc_name in ("pin_bull","pin_bear") and i == LOOKBACK - 1:
            wl, wh = 2.5, 5.0   # last bar is the pin

        c, new_price = generate_candle(price, bs, wl, wh, sc["vl"], sc["vh"], local_bias)
        candles.append(c)
        price = new_price

    df = pd.DataFrame(candles)
    df["range"] = df["high"] - df["low"]
    df["body"]  = df["close"] - df["open"]
    df["abs_body"]   = df["body"].abs()
    df["dir"]        = np.sign(df["body"]).fillna(0)
    df["upper_wick"] = df["high"] - df[["open","close"]].max(axis=1)
    df["lower_wick"] = df[["open","close"]].min(axis=1) - df["low"]
    df["atr14"]      = df["range"].rolling(14,min_periods=1).mean()
    df["abs_body_avg30"] = df["abs_body"].rolling(30,min_periods=1).mean()
    df["range_avg30"]= df["range"].rolling(30,min_periods=1).mean()
    df["vol_avg20"]  = df["volume"].rolling(20,min_periods=1).mean()
    df["body_ratio"] = df["abs_body"] / (df["range"] + EPS)
    df["body_rat_avg5"]= df["body_ratio"].rolling(5,min_periods=1).mean()
    df["close_grav"] = (df["close"]-df["low"])/(df["range"]+EPS)
    df["shadow_dom"] = (df["lower_wick"]-df["upper_wick"])/(df["range"]+EPS)
    bull = df["dir"]>=0
    df["dir_eff"] = np.where(bull, df["body_ratio"]*df["close_grav"], df["body_ratio"]*(1-df["close_grav"]))
    df["vol_weight"] = (df["volume"]/(df["vol_avg20"]+EPS)).clip(0,4)
    ha_close = (df["open"]+df["high"]+df["low"]+df["close"])/4
    ha_open  = ha_close.ewm(span=3,adjust=False).mean()
    df["ha_body"] = (ha_close-ha_open)/(df["atr14"]+EPS)
    cons,cur,prev=[],0,0
    for d in df["dir"].fillna(0).tolist():
        if d==prev and d!=0: cur+=1
        elif d!=0: cur=1
        else: cur=0
        cons.append(cur); prev=d
    df["consec_len"] = cons

    feat   = compute_candle_features(df)
    w_last = df.iloc[-WIN:].reset_index(drop=True)
    labels = compute_labels(feat, w_last)

    row = dict(feat)
    row["label_pattern_type"],  row["label_seq_dominance"], \
    row["label_conviction"],    row["label_wick_pressure"], \
    row["label_seq_phase"],     row["label_candle_signal"]  = labels
    return row


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Generating {N_SAMPLES} candle-sequence samples (FX M15) ...")
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
