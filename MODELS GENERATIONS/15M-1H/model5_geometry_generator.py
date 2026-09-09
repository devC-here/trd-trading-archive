# =============================================================================
# model5_geometry_generator.py
# Model 5 — THE PRICE ACTION CANDLESTICK GEOMETRY SPECIALIST
# Features: 97_final_upper_wick_ratio, 99_final_lower_wick_ratio,
#           71_intraday_variance_ratio, 104_body_size_to_atr_ratio,
#           94_relative_body_to_range_ratio, 98_lower_wick_absorption_ratio
# Normalization: None (scale-invariant geometry ratios [0,1])
# Target: R_p in [-1.0 (Heavy Demand Floor), +1.0 (Heavy Supply Ceiling)]
# Input: ohlcv_40_m15.csv
# =============================================================================

import csv, random
import numpy as np
import pandas as pd
from collections import defaultdict

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw): return x

# =============================================================================
# A — CONFIGURATION
# =============================================================================
INPUT_CSV  = "ohlcv_40_m15.csv"
OUTPUT_CSV = "model5_geometry_dataset.csv"

N_PER_COND = 1250
WINDOW     = 16
STEPS      = [4, 8, 12, 16]
PIP        = 0.00010
EPS        = 1e-12
SEED       = 42
random.seed(SEED)
np.random.seed(SEED)

FEATURES = [
    "97_final_upper_wick_ratio",
    "99_final_lower_wick_ratio",
    "71_intraday_variance_ratio",
    "104_body_size_to_atr_ratio",
    "94_relative_body_to_range_ratio",
    "98_lower_wick_absorption_ratio",
]

# Target: wick rejection geometry polarity
# BUY  → lower wick absorption (demand floor)  → -1
# SELL → upper wick rejection (supply ceiling)  → +1
# HOLD → balanced / no clear wick bias          →  0
LABEL_MAP = {"BUY": -1, "SELL": 1, "HOLD": 0}

# =============================================================================
# B — LOAD TEMPLATES
# =============================================================================
with open(INPUT_CSV, newline="") as fh:
    raw_rows = list(csv.DictReader(fh))

_raw_seqs = defaultdict(list)
for r in raw_rows:
    _raw_seqs[(r["condition_id"], r["sample"])].append({
        "m15_sequence":   int(r["m15_sequence"]),
        "open":           float(r["open"]),
        "high":           float(r["high"]),
        "low":            float(r["low"]),
        "close":          float(r["close"]),
        "volume":         int(r["volume"]),
        "label":          r["label"],
        "condition_id":   r["condition_id"],
        "condition_name": r["condition_name"],
    })
for k in _raw_seqs:
    _raw_seqs[k].sort(key=lambda x: x["m15_sequence"])

COND_MAP = defaultdict(list)
for (cid, _), seq in _raw_seqs.items():
    COND_MAP[cid].append(seq)

print(f"Loaded {len(COND_MAP)} conditions | "
      f"{sum(len(v) for v in COND_MAP.values())} sequences")

# =============================================================================
# C — SCENARIO DERIVATION
# =============================================================================
def derive_scenario(template_sequences):
    bodies, ranges, volumes, directions = [], [], [], []
    for seq in template_sequences:
        for c in seq:
            bodies.append(abs(c["close"] - c["open"]))
            ranges.append(c["high"] - c["low"])
            volumes.append(c["volume"])
            directions.append(1 if c["close"] > c["open"] else 0)
    avg_body = np.mean(bodies)    if bodies     else 0.001
    avg_rng  = np.mean(ranges)    if ranges     else 0.002
    avg_vol  = np.mean(volumes)   if volumes    else 500.0
    std_vol  = np.std(volumes)    if volumes    else 200.0
    bias     = np.mean(directions) if directions else 0.5
    avg_wick = max(0.0, (avg_rng - avg_body)) / 2.0
    wl = max(0.05, avg_wick / (avg_body + EPS) * 0.6)
    wh = max(wl + 0.10, avg_wick / (avg_body + EPS) * 1.4)
    return {
        "bias":       float(np.clip(bias,     0.10, 0.90)),
        "body_scale": float(np.clip(avg_body, PIP,  80*PIP)),
        "wl":         float(np.clip(wl,       0.02, 1.00)),
        "wh":         float(np.clip(wh,       0.10, 2.50)),
        "vl":         float(max(10.0,  avg_vol - std_vol)),
        "vh":         float(max(50.0,  avg_vol + std_vol)),
    }

SCENARIOS = {cid: derive_scenario(seqs) for cid, seqs in COND_MAP.items()}

# =============================================================================
# D — CANDLE & SEQUENCE GENERATORS
# =============================================================================
def gen_candle(open_price, sc, noise=1.0):
    d    = 1 if random.random() < sc["bias"] else -1
    body = float(np.random.uniform(0.5, 1.5) * sc["body_scale"] * noise)
    uw   = float(body * np.random.uniform(sc["wl"], sc["wh"]))
    lw   = float(body * np.random.uniform(sc["wl"], sc["wh"]))
    o = float(open_price)
    c = float(o + d * body)
    h = max(o, c) + abs(uw)
    l = min(o, c) - abs(lw)
    v = max(1, int(np.random.uniform(sc["vl"], sc["vh"]) * noise))
    return {"open": round(o,5), "high": round(h,5),
            "low":  round(l,5), "close": round(c,5), "volume": v}, c

def generate_sequence(sc, template_seqs):
    price = random.choice(template_seqs)[0]["open"] + np.random.normal(0, 15*PIP)
    candles = []
    for _ in range(WINDOW):
        candle, price = gen_candle(price, sc, noise=np.random.lognormal(0, 0.15))
        candles.append(candle)
    return candles

# =============================================================================
# E — NEUTRAL PADDING
# =============================================================================
def neutral_candle(ref, avg_vol):
    return {"open": ref, "high": ref+PIP, "low": ref-PIP,
            "close": ref, "volume": max(1, int(avg_vol))}

def build_padded_window(real_candles, step):
    avg_vol = max(1, int(np.mean([c["volume"] for c in real_candles])))
    ref     = real_candles[0]["open"]
    pads    = [neutral_candle(ref, avg_vol) for _ in range(WINDOW - step)]
    return pads + real_candles

# =============================================================================
# F — FEATURE ENGINE
# =============================================================================
def compute_features(window):
    C = np.array([c["close"] for c in window], dtype=float)
    O = np.array([c["open"]  for c in window], dtype=float)
    H = np.array([c["high"]  for c in window], dtype=float)
    L = np.array([c["low"]   for c in window], dtype=float)

    # True Range for ATR
    TR = np.array(
        [H[0] - L[0]] +
        [max(H[i]-L[i], abs(H[i]-C[i-1]), abs(L[i]-C[i-1]))
         for i in range(1, WINDOW)]
    )
    ATR     = TR.mean()
    bodies  = np.abs(C - O)
    ranges  = (H - L) + EPS
    total_h = (H[-1] - L[-1]) + EPS

    # 97_final_upper_wick_ratio
    # (High[t] - max(Open[t], Close[t])) / (High[t] - Low[t])
    # High → strong overhead rejection (sellers stepping in)
    # Low  → little upper wick (bullish close, buyers in control)
    # Neutral pads: High=ref+pip, Low=ref-pip, O=C=ref
    #   → (ref+pip - ref) / 2pip = 0.5 (symmetric neutral baseline)
    upper_wick_ratio = float((H[-1] - max(O[-1], C[-1])) / total_h)

    # 99_final_lower_wick_ratio
    # (min(Open[t], Close[t]) - Low[t]) / (High[t] - Low[t])
    # High → strong lower floor absorption (buyers stepping in)
    # Low  → little lower wick (bearish close, sellers in control)
    # Neutral pads: same symmetric 0.5 baseline
    lower_wick_ratio = float((min(O[-1], C[-1]) - L[-1]) / total_h)

    # 71_intraday_variance_ratio
    # Var(High-Low) / Var(|Close-Open|)
    # High → wick variance dominates body variance → erratic / reversal energy
    # Low  → body variance dominates → directional conviction bars
    # Neutral pads: all ranges=2pip → var(range)≈0, var(body)≈0 → ratio≈1
    var_ranges = np.var(H - L)
    var_bodies = np.var(bodies) + EPS
    intraday_var_ratio = float(var_ranges / var_bodies)

    # 104_body_size_to_atr_ratio
    # |Close[t] - Open[t]| / ATR
    # > 1.0 → final candle body larger than avg true range (conviction)
    # < 1.0 → small body relative to historical range (indecision)
    # Neutral pads: body≈0, ATR≈2pip → ratio≈0 (no body signal yet)
    body_to_atr = float(bodies[-1] / (ATR + EPS))

    # 94_relative_body_to_range_ratio
    # Mean of (|C-O| / (H-L)) across all 16 bars
    # High → most candles are full-bodied (strong directional sessions)
    # Low  → most candles are wick-heavy (indecision / reversal sessions)
    rel_body_to_range = float(np.mean(bodies / ranges))

    # 98_lower_wick_absorption_ratio
    # Mean of ((min(O,C)-L) / (H-L)) across all 16 bars
    # High → persistent lower wick presence → consistent demand absorption
    # Low  → little lower wick → either strong bearish closes or small wicks
    lower_wicks = np.minimum(O, C) - L
    lower_wick_abs = float(np.mean(lower_wicks / ranges))

    return {
        "97_final_upper_wick_ratio":      upper_wick_ratio,
        "99_final_lower_wick_ratio":      lower_wick_ratio,
        "71_intraday_variance_ratio":     intraday_var_ratio,
        "104_body_size_to_atr_ratio":     body_to_atr,
        "94_relative_body_to_range_ratio":rel_body_to_range,
        "98_lower_wick_absorption_ratio": lower_wick_abs,
    }

# =============================================================================
# G — MAIN LOOP
# =============================================================================
print(f"\nGenerating {N_PER_COND:,} x {len(COND_MAP)} x {len(STEPS)} = "
      f"{N_PER_COND*len(COND_MAP)*len(STEPS):,} rows...\n")

out_rows = []
seq_id   = 0

for cid in tqdm(sorted(COND_MAP.keys()), desc="Model-5"):
    sc, tmpl = SCENARIOS[cid], COND_MAP[cid]
    raw_lbl  = tmpl[0][0]["label"]
    cname    = tmpl[0][0]["condition_name"]
    for _ in range(N_PER_COND):
        candles = generate_sequence(sc, tmpl)
        for step in STEPS:
            seen  = candles[:step]
            feats = compute_features(build_padded_window(seen, step))
            fc    = seen[-1]
            out_rows.append({
                "sequence_id":                     seq_id,
                "condition_id":                    cid,
                "condition_name":                  cname,
                "label":                           raw_lbl,
                "label_geometry":                  LABEL_MAP[raw_lbl],
                "recognition_step":                step,
                "open":  fc["open"],  "high": fc["high"],
                "low":   fc["low"],   "close":fc["close"],
                "volume":fc["volume"],
                "97_final_upper_wick_ratio":       round(feats["97_final_upper_wick_ratio"],       8),
                "99_final_lower_wick_ratio":       round(feats["99_final_lower_wick_ratio"],       8),
                "71_intraday_variance_ratio":      round(feats["71_intraday_variance_ratio"],      8),
                "104_body_size_to_atr_ratio":      round(feats["104_body_size_to_atr_ratio"],      8),
                "94_relative_body_to_range_ratio": round(feats["94_relative_body_to_range_ratio"], 8),
                "98_lower_wick_absorption_ratio":  round(feats["98_lower_wick_absorption_ratio"],  8),
            })
        seq_id += 1

# =============================================================================
# H — VALIDATE & SAVE
# =============================================================================
df   = pd.DataFrame(out_rows)
df16 = df[df["recognition_step"]==16]
bad_h = (df["high"] < df[["open","close"]].max(axis=1)).sum()
bad_l = (df["low"]  > df[["open","close"]].min(axis=1)).sum()
print(f"Rows: {len(df):,} | Sequences: {seq_id:,} | "
      f"Physics violations: {bad_h+bad_l} {'✅' if bad_h+bad_l==0 else '❌'}")
print("Label dist (step=16):", df16["label"].value_counts().to_dict())
print("Feature ranges (step=16):")
for f in FEATURES:
    print(f"  {f:38s} min={df16[f].min():+.4f}  max={df16[f].max():+.4f}  mean={df16[f].mean():+.4f}")

COLS = (["sequence_id","condition_id","condition_name","label","label_geometry",
         "recognition_step","open","high","low","close","volume"] + FEATURES)
df[COLS].to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved → {OUTPUT_CSV}  shape={df[COLS].shape}")
