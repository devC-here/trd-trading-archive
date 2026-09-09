# synthetic_candle.py  ·  v2  (next-candle edition)
# ─────────────────────────────────────────────────────────────────────
# Generates training data for the Candle Sequence model
# 30 features | 6 heads — ALL labels predict the NEXT 1-2 candles
# 10-bar sequential auction narrative | FX M15
# ─────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import random
from scipy import stats as sp_stats

try:
    from tqdm import trange
except ImportError:
    def trange(n, **kw): return range(n)

N_SAMPLES    = 20000
WIN          = 10           # bar window for sequence features
LOOKBACK     = 50           # context bars for features
FUTURE_BARS  = 2            # candles used to build labels
OUTPUT_PATH  = "synthetic_candle_dataset.csv"
BASE_PRICE   = 1.2000
SEED         = 42
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
# Feature computation — unchanged from v1 (on LOOKBACK bars only)
# ─────────────────────────────────────────────────────────────────────

def compute_candle_features(df: pd.DataFrame) -> dict:
    df    = df.copy().reset_index(drop=True)

    df["range"]      = df["high"] - df["low"]
    df["body"]       = df["close"] - df["open"]
    df["abs_body"]   = df["body"].abs()
    df["dir"]        = np.sign(df["body"]).fillna(0)
    df["upper_wick"] = df["high"] - df[["open","close"]].max(axis=1)
    df["lower_wick"] = df[["open","close"]].min(axis=1) - df["low"]
    df["atr14"]      = df["range"].rolling(14, min_periods=1).mean()
    df["abs_body_avg30"] = df["abs_body"].rolling(30, min_periods=1).mean()
    df["range_avg30"]    = df["range"].rolling(30, min_periods=1).mean()
    df["vol_avg20"]      = df["volume"].rolling(20, min_periods=1).mean()
    df["body_ratio"]  = df["abs_body"] / (df["range"] + EPS)
    df["body_rat_avg5"]= df["body_ratio"].rolling(5, min_periods=1).mean()
    df["close_grav"]  = (df["close"] - df["low"]) / (df["range"] + EPS)
    df["shadow_dom"]  = (df["lower_wick"] - df["upper_wick"]) / (df["range"] + EPS)
    bull = df["dir"] >= 0
    df["dir_eff"]     = np.where(bull,
                                  df["body_ratio"] * df["close_grav"],
                                  df["body_ratio"] * (1 - df["close_grav"]))
    df["vol_weight"]  = (df["volume"] / (df["vol_avg20"] + EPS)).clip(0, 4)
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_open  = ha_close.ewm(span=3, adjust=False).mean()
    df["ha_body"]     = (ha_close - ha_open) / (df["atr14"] + EPS)
    cons, cur, prev = [], 0, 0
    for d in df["dir"].fillna(0).tolist():
        if d == prev and d != 0:  cur += 1
        elif d != 0:               cur  = 1
        else:                      cur  = 0
        cons.append(cur)
        prev = d
    df["consec_len"]  = cons

    w  = df.iloc[-WIN:].reset_index(drop=True)
    atr_val = float(df["atr14"].iloc[-1])

    f = {}
    f["lb_close_grav"]   = float(w["close_grav"].iloc[-1])
    f["lb_body_ratio"]   = float(w["body_ratio"].iloc[-1])
    f["lb_shadow_dom"]   = float(w["shadow_dom"].iloc[-1])
    f["lb_vol_weight"]   = float(min(w["vol_weight"].iloc[-1], 4.0))
    f["lb_dir_eff"]      = float(w["dir_eff"].iloc[-1])

    f["seq_cg_slope"]    = _ols_slope(w["close_grav"].values)
    bm = float(w["body_ratio"].mean())
    bs = float(w["body_ratio"].std())
    f["body_entropy"]    = float(bs / (bm + EPS))
    f["wick_press_slope"]= _ols_slope(w["shadow_dom"].values)
    bar_conv = w["dir"] * w["body_ratio"] * w["vol_weight"]
    f["vol_narrative"]   = _ols_slope(bar_conv.values)
    f["seq_harmony"]     = _safe_corr(w["close_grav"].values, w["range"].values)
    early_dir = np.sign(w["body"].iloc[:5].sum())
    late_dir  = np.sign(w["body"].iloc[5:].sum())
    f["handoff"]         = float(early_dir != late_dir)
    early_body = float(w["body_ratio"].iloc[:5].mean())
    late_body  = float(w["body_ratio"].iloc[5:].mean())
    f["handoff_str"]     = float((late_body - early_body) / (atr_val + EPS))
    full_mean = float(w["body_ratio"].mean()) + EPS
    f["seq_accel"]       = float((late_body - early_body) / full_mean)

    curr_body  = float(w["abs_body"].iloc[-1])
    prior_body = float(w["abs_body"].iloc[-2]) if len(w) >= 2 else curr_body
    curr_dir   = float(w["dir"].iloc[-1])
    f["engulf_qual"]     = float((curr_body - prior_body) / (atr_val + EPS) * (curr_dir or 1))
    inside_count = 0
    for i in range(1, len(w)):
        if w["high"].iloc[i] < w["high"].iloc[i-1] and w["low"].iloc[i] > w["low"].iloc[i-1]:
            inside_count += 1
        else:
            break
    mother_range = float(w["range"].iloc[0]) if inside_count > 0 else 0.0
    f["inside_energy"]   = float(inside_count * mother_range / (atr_val + EPS))
    f["ha_body_last"]    = float(w["ha_body"].iloc[-1])
    ha_arr = w["ha_body"].values
    f["ha_body_slope"]   = float((ha_arr[-1] - ha_arr[max(0,len(ha_arr)-4)]) / (atr_val + EPS))
    uw  = float(w["upper_wick"].iloc[-1])
    lw  = float(w["lower_wick"].iloc[-1])
    rng = float(w["range"].iloc[-1])
    br  = float(w["body_ratio"].iloc[-1])
    cg  = float(w["close_grav"].iloc[-1])
    body_small = 1.0 if br < 0.35 else 0.5
    f["pin_bull_qual"]   = float((lw / (rng + EPS)) * body_small * cg)
    f["pin_bear_qual"]   = float((uw / (rng + EPS)) * body_small * (1 - cg))
    closes      = w["close"].values
    c_min, c_max= float(closes.min()), float(closes.max())
    seq_pos     = (closes[-1] - c_min) / (c_max - c_min + EPS)
    doji_score  = float(max(0, 0.15 - br) / 0.15)
    f["doji_context"]    = float(doji_score * (seq_pos - 0.5))

    f["body_vs_avg"]     = float(min(float(df["abs_body"].iloc[-1]) /
                                      (float(df["abs_body_avg30"].iloc[-1]) + EPS), 4.0))
    f["range_vs_avg"]    = float(min(float(df["range"].iloc[-1]) /
                                      (float(df["range_avg30"].iloc[-1]) + EPS), 4.0))
    br5_avg = float(df["body_rat_avg5"].iloc[-1])
    f["body_accel"]      = float(br - br5_avg)

    br1  = float(w["body_ratio"].iloc[-1])
    br3  = float(w["body_ratio"].iloc[-3:].mean()) if len(w) >= 3 else br1
    br10 = float(w["body_ratio"].mean())
    price_up = closes[-1] > closes[0]
    f["fractal_bull"]    = float((br1 > br3 > br10) and price_up)
    f["fractal_bear"]    = float((br1 > br3 > br10) and not price_up)

    f["seq_dir_ratio"]   = float((w["dir"] > 0).mean())
    f["consec_len"]      = float(w["consec_len"].iloc[-1])
    vol_surge_w          = float(w["vol_weight"].mean())
    body_ratio_mean_w    = float(w["body_ratio"].mean()) + EPS
    f["vol_climax_div"]  = float(body_ratio_mean_w / (vol_surge_w + EPS))
    f["ha_trend_str"]    = float(w["ha_body"].ewm(span=5, adjust=False).mean().iloc[-1])
    f["seq_body_avg"]    = float(w["body_ratio"].mean())

    # Store context values for label computation (prefixed _)
    f["_atr14"]          = float(df["atr14"].iloc[-1])
    f["_context_close"]  = float(df["close"].iloc[-1])
    f["_ha_trend_str"]   = f["ha_trend_str"]
    f["_seq_accel"]      = f["seq_accel"]
    f["_seq_dir_ratio"]  = f["seq_dir_ratio"]

    return f


# ─────────────────────────────────────────────────────────────────────
# FORWARD LABEL GENERATION — 6 heads, all about NEXT 1-2 candles
# ─────────────────────────────────────────────────────────────────────

def compute_labels(f: dict, future_candles: list) -> tuple:
    """
    f               : feature dict (with _ context keys)
    future_candles  : list of 2 dicts {open, high, low, close, volume}
    """
    atr   = max(f["_atr14"], EPS)
    ref   = f["_context_close"]
    c1    = future_candles[0]
    c2    = future_candles[1] if len(future_candles) >= 2 else c1

    c1_body_atr = abs(c1["close"] - c1["open"]) / atr
    c2_body_atr = abs(c2["close"] - c2["open"]) / atr
    c1_bull     = c1["close"] > c1["open"]
    c2_bull     = c2["close"] > c2["open"]
    net_atr     = (c2["close"] - ref) / atr
    seq_high    = max(c1["high"], c2["high"])
    seq_low     = min(c1["low"],  c2["low"])
    seq_range   = (seq_high - seq_low) / atr
    close_grav  = (c2["close"] - seq_low) / (seq_range * atr + EPS)

    c1_cg = (c1["close"] - c1["low"]) / (c1["high"] - c1["low"] + EPS)
    c1_uw = (c1["high"] - max(c1["open"], c1["close"])) / (c1["high"] - c1["low"] + EPS)
    c1_lw = (min(c1["open"], c1["close"]) - c1["low"]) / (c1["high"] - c1["low"] + EPS)

    # ── HEAD 1: next_candle_direction [0=bearish 1=doji 2=bullish]
    if   net_atr > 0.30:     nxt_dir = 2
    elif net_atr < -0.30:    nxt_dir = 0
    else:                     nxt_dir = 1

    # ── HEAD 2: next_candle_body_size [0=doji/small 1=normal 2=marubozu/large]
    avg_body = (c1_body_atr + c2_body_atr) / 2
    if   avg_body > 1.2:     body_sz = 2
    elif avg_body > 0.4:     body_sz = 1
    else:                     body_sz = 0

    # ── HEAD 3: next_candle_pattern [0=impulse 1=pin 2=inside/doji 3=engulf]
    # Detect the most dominant single-bar pattern in candle 1
    if c1_body_atr < 0.12:                                      nxt_pat = 2   # doji
    elif (c1_lw > 0.40 and c1_uw < 0.25):                      nxt_pat = 1   # pin bull
    elif (c1_uw > 0.40 and c1_lw < 0.25):                      nxt_pat = 1   # pin bear
    elif c1_body_atr > c2_body_atr * 1.3 and c1_body_atr > 0.8: nxt_pat = 3 # engulf
    else:                                                        nxt_pat = 0   # impulse

    # ── HEAD 4: sequence_continuation [0=reversal 1=indecision 2=continuation]
    # Does the 2-bar future continue the momentum direction?
    ha  = f["_ha_trend_str"]
    sdr = f["_seq_dir_ratio"]
    if ha > 0.05 and sdr > 0.55:       # bullish context
        if nxt_dir == 2:               seq_cont = 2
        elif nxt_dir == 0:             seq_cont = 0
        else:                          seq_cont = 1
    elif ha < -0.05 and sdr < 0.45:   # bearish context
        if nxt_dir == 0:               seq_cont = 2
        elif nxt_dir == 2:             seq_cont = 0
        else:                          seq_cont = 1
    else:                              seq_cont = 1   # no clear context

    # ── HEAD 5: next_close_gravity [0=lower_third 1=mid 2=upper_third]
    if   close_grav > 0.67:  nxt_cg = 2
    elif close_grav < 0.33:  nxt_cg = 0
    else:                     nxt_cg = 1

    # ── HEAD 6: next_volatility [0=compression 1=normal 2=expansion]
    if   seq_range > 2.5:    nxt_vol = 2
    elif seq_range < 0.9:    nxt_vol = 0
    else:                     nxt_vol = 1

    return (int(nxt_dir), int(body_sz), int(nxt_pat),
            int(seq_cont), int(nxt_cg), int(nxt_vol))


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
    total   = LOOKBACK + FUTURE_BARS

    for i in range(total):
        local_bias = sc["bias"] if random.random() > 0.10 else 1 - sc["bias"]

        if sc_name == "inside_coil" and i > LOOKBACK // 2:
            shrink = 1 - (i - LOOKBACK//2) / (LOOKBACK//2 + 1) * 0.60
            bs     = sc["body"] * shrink
        elif sc_name == "exhaustion" and i > LOOKBACK * 0.6:
            fade   = 1 - (i - LOOKBACK*0.6) / (LOOKBACK*0.4 + 1) * 0.70
            bs     = sc["body"] * max(fade, 0.15)
        else:
            bs = sc["body"]

        wl, wh = sc["wl"], sc["wh"]
        if sc_name in ("pin_bull","pin_bear") and i == LOOKBACK - 1:
            wl, wh = 2.5, 5.0

        c, new_price = generate_candle(price, bs, wl, wh, sc["vl"], sc["vh"], local_bias)
        candles.append(c)
        price = new_price

    df_context = pd.DataFrame(candles[:LOOKBACK])
    future     = candles[LOOKBACK:]

    # Build the full df with rolling columns (needed by compute_candle_features)
    df_context["range"]    = df_context["high"] - df_context["low"]
    df_context["body"]     = df_context["close"] - df_context["open"]
    df_context["abs_body"] = df_context["body"].abs()
    df_context["dir"]      = np.sign(df_context["body"]).fillna(0)
    df_context["upper_wick"] = df_context["high"] - df_context[["open","close"]].max(axis=1)
    df_context["lower_wick"] = df_context[["open","close"]].min(axis=1) - df_context["low"]
    df_context["atr14"]    = df_context["range"].rolling(14,min_periods=1).mean()
    df_context["abs_body_avg30"] = df_context["abs_body"].rolling(30,min_periods=1).mean()
    df_context["range_avg30"]= df_context["range"].rolling(30,min_periods=1).mean()
    df_context["vol_avg20"]  = df_context["volume"].rolling(20,min_periods=1).mean()
    df_context["body_ratio"] = df_context["abs_body"] / (df_context["range"] + EPS)
    df_context["body_rat_avg5"]= df_context["body_ratio"].rolling(5,min_periods=1).mean()
    df_context["close_grav"] = (df_context["close"]-df_context["low"])/(df_context["range"]+EPS)
    df_context["shadow_dom"] = (df_context["lower_wick"]-df_context["upper_wick"])/(df_context["range"]+EPS)
    bull = df_context["dir"]>=0
    df_context["dir_eff"] = np.where(bull,df_context["body_ratio"]*df_context["close_grav"],df_context["body_ratio"]*(1-df_context["close_grav"]))
    df_context["vol_weight"] = (df_context["volume"]/(df_context["vol_avg20"]+EPS)).clip(0,4)
    ha_close = (df_context["open"]+df_context["high"]+df_context["low"]+df_context["close"])/4
    ha_open  = ha_close.ewm(span=3,adjust=False).mean()
    df_context["ha_body"] = (ha_close-ha_open)/(df_context["atr14"]+EPS)
    cons,cur,prev=[],0,0
    for d in df_context["dir"].fillna(0).tolist():
        if d==prev and d!=0: cur+=1
        elif d!=0: cur=1
        else: cur=0
        cons.append(cur); prev=d
    df_context["consec_len"] = cons

    feat   = compute_candle_features(df_context)
    labels = compute_labels(feat, future)

    row = {k: v for k, v in feat.items() if not k.startswith("_")}
    row["label_next_direction"],  row["label_next_body_size"], \
    row["label_next_pattern"],    row["label_seq_continuation"], \
    row["label_next_close_grav"], row["label_next_volatility"]  = labels
    return row


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Generating {N_SAMPLES} candle-sequence samples — NEXT-CANDLE labels (FX M15) ...")
    rows = []
    for _ in trange(N_SAMPLES):
        rows.append(generate_sequence())

    df_out     = pd.DataFrame(rows)
    label_cols = [c for c in df_out.columns if c.startswith("label_")]
    feat_cols  = [c for c in df_out.columns if not c.startswith("label_")]
    df_out     = df_out[sorted(feat_cols) + label_cols]
    df_out.to_csv(OUTPUT_PATH, index=False)

    label_maps = {
        "label_next_direction":    {0:"bearish", 1:"doji", 2:"bullish"},
        "label_next_body_size":    {0:"small", 1:"normal", 2:"large"},
        "label_next_pattern":      {0:"impulse", 1:"pin", 2:"inside_doji", 3:"engulf"},
        "label_seq_continuation":  {0:"reversal", 1:"indecision", 2:"continuation"},
        "label_next_close_grav":   {0:"lower_third", 1:"mid", 2:"upper_third"},
        "label_next_volatility":   {0:"compression", 1:"normal", 2:"expansion"},
    }
    print(f"\n✅  Saved → {OUTPUT_PATH}  shape={df_out.shape}")
    print(f"    Features : {len(feat_cols)}")
    for lc in label_cols:
        counts = df_out[lc].value_counts().sort_index()
        named  = {label_maps[lc].get(k, k): f"{100*v/len(df_out):.1f}%" for k, v in counts.items()}
        print(f"    {lc}: {named}")
