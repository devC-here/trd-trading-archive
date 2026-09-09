# =============================================================================
# model3_volatility_generator.py
# Model 3 — THE VOLATILITY SQUEEZE SPECIALIST
# Features: 62_bollinger_bandwidth_z_score, 63_keltner_channel_squeeze,
#           72_true_range_z_score, 65_fractal_dimension_index
# Normalization: None (standardized structural metrics)
# Target: A_v in [0.0 (Pure Chop), 1.0 (Pure Breakout Expansion)]
# Input: ohlcv_40_m15.csv
# =============================================================================

import csv, random, math
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
OUTPUT_CSV = "model3_volatility_dataset.csv"

N_PER_COND = 1250
WINDOW     = 16
STEPS      = [4, 8, 12, 16]
PIP        = 0.00010
EPS        = 1e-12
SEED       = 42
random.seed(SEED)
np.random.seed(SEED)

FEATURES = [
    "62_bollinger_bandwidth_z_score",
    "63_keltner_channel_squeeze",
    "72_true_range_z_score",
    "65_fractal_dimension_index",
]

# Target: 0.0 = pure chop/squeeze, 1.0 = pure breakout expansion
# BUY/SELL → directional breakout → high activation (1)
# HOLD     → compression/chop    → low activation  (0)
LABEL_MAP = {"BUY": 1, "SELL": 1, "HOLD": 0}

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
    H = np.array([c["high"]  for c in window], dtype=float)
    L = np.array([c["low"]   for c in window], dtype=float)

    H_max  = H.max(); L_min = L.min()
    mean_C = C.mean()
    std_C  = max(C.std(), EPS)

    # True Range array
    TR = np.array(
        [H[0] - L[0]] +
        [max(H[i]-L[i], abs(H[i]-C[i-1]), abs(L[i]-C[i-1]))
         for i in range(1, WINDOW)]
    )
    ATR = TR.mean()

    # 62_bollinger_bandwidth_z_score
    # Bollinger Bandwidth = (4 × std(closes)) / mean(closes)
    # Z-score relative to rolling sub-window bandwidths within this block
    # Low  Z → bandwidth historically tight → squeeze building
    # High Z → bandwidth historically wide  → expansion active
    # Neutral pads: all closes flat → std ≈ 0 → BB_BW ≈ 0 → low Z-score
    bb_bw    = (4.0 * std_C) / (mean_C + EPS)
    sub_bws  = np.array([
        (4.0 * max(C[:i].std(), EPS)) / (C[:i].mean() + EPS)
        for i in range(4, WINDOW + 1)
    ])
    sub_mean, sub_std = sub_bws.mean(), max(sub_bws.std(), EPS)
    bb_z = float((bb_bw - sub_mean) / sub_std)

    # 63_keltner_channel_squeeze
    # BB_BW / KC_BW  where KC_BW = (2 × ATR) / mean(closes)
    # < 1.0 → BB inside Keltner → classic squeeze (compression)
    # > 1.0 → BB outside Keltner → expansion breakout active
    # Neutral pads: ATR ≈ 2pip, BB_BW ≈ 0 → ratio ≈ 0 (deep squeeze signal)
    kc_bw   = (2.0 * ATR) / (mean_C + EPS)
    kc_sq   = float(bb_bw / (kc_bw + EPS))

    # 72_true_range_z_score
    # Z-score of the final candle's True Range vs the 16-bar TR distribution
    # High positive Z → final candle is unusually wide  → expansion
    # Negative Z       → final candle is unusually tight → compression
    # Neutral pads: all TR ≈ 2pip → std ≈ 0 → Z ≈ 0
    tr_mean, tr_std = TR.mean(), max(TR.std(), EPS)
    tr_z = float((TR[-1] - tr_mean) / tr_std)

    # 65_fractal_dimension_index
    # 1 + ln(path_length / total_range) / ln(16)
    # Near 1.0 → price moved in a straight line (trending / directional)
    # Near 2.0 → price filled the space randomly (pure chop / noise)
    # Neutral pads: flat closes → path_length ≈ 0 → FDI → 1.0 (linear)
    path_len  = float(np.abs(np.diff(C)).sum())
    total_rng = float(H_max - L_min) + EPS
    fdi = float(1.0 + math.log(path_len / total_rng + EPS) / math.log(WINDOW))

    return {
        "62_bollinger_bandwidth_z_score": bb_z,
        "63_keltner_channel_squeeze":     kc_sq,
        "72_true_range_z_score":          tr_z,
        "65_fractal_dimension_index":     fdi,
    }

# =============================================================================
# G — MAIN LOOP
# =============================================================================
print(f"\nGenerating {N_PER_COND:,} x {len(COND_MAP)} x {len(STEPS)} = "
      f"{N_PER_COND*len(COND_MAP)*len(STEPS):,} rows...\n")

out_rows = []
seq_id   = 0

for cid in tqdm(sorted(COND_MAP.keys()), desc="Model-3"):
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
                "sequence_id":                    seq_id,
                "condition_id":                   cid,
                "condition_name":                 cname,
                "label":                          raw_lbl,
                "label_volatility":               LABEL_MAP[raw_lbl],
                "recognition_step":               step,
                "open":  fc["open"],  "high": fc["high"],
                "low":   fc["low"],   "close":fc["close"],
                "volume":fc["volume"],
                "62_bollinger_bandwidth_z_score": round(feats["62_bollinger_bandwidth_z_score"], 8),
                "63_keltner_channel_squeeze":     round(feats["63_keltner_channel_squeeze"],     8),
                "72_true_range_z_score":          round(feats["72_true_range_z_score"],          8),
                "65_fractal_dimension_index":     round(feats["65_fractal_dimension_index"],     8),
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
    print(f"  {f:40s} min={df16[f].min():+.4f}  max={df16[f].max():+.4f}  mean={df16[f].mean():+.4f}")

COLS = (["sequence_id","condition_id","condition_name","label","label_volatility",
         "recognition_step","open","high","low","close","volume"] + FEATURES)
df[COLS].to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved → {OUTPUT_CSV}  shape={df[COLS].shape}")
