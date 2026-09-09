# =============================================================================
# model2_structure_generator.py
# Model 2 — THE STRUCTURAL PRICE LOCATION SPECIALIST
# Features: 43_fib_382_dist, 49_s1_dist, 48_r1_dist,
#           35_lin_reg_intercept, 69_high_low_ratio
# Normalization: StandardScaler (raw pip decimals — must be scaled)
# Target: E_s in [-1.0 (Support Floor), +1.0 (Resistance Ceiling)]
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
OUTPUT_CSV = "model2_structure_dataset.csv"

N_PER_COND = 1250
WINDOW     = 16
STEPS      = [4, 8, 12, 16]
PIP        = 0.00010
EPS        = 1e-12
SEED       = 42
random.seed(SEED)
np.random.seed(SEED)

FEATURES = [
    "43_fib_382_dist",
    "49_s1_dist",
    "48_r1_dist",
    "35_lin_reg_intercept",
    "69_high_low_ratio",
]

# Target scalar: +1 = near resistance ceiling, -1 = near support floor
# BUY  → price near support / below fair value → negative spatial score
# SELL → price near resistance / above fair value → positive spatial score
# HOLD → price at equilibrium → near zero
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
# C — SCENARIO DERIVATION  (shared logic)
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
def ols_slope_intercept(y):
    """OLS regression: returns (slope, intercept)."""
    n  = len(y); x = np.arange(n, dtype=float)
    mx, my = x.mean(), y.mean()
    slope     = ((x-mx)*(y-my)).sum() / (((x-mx)**2).sum()+EPS)
    intercept = my - slope * mx
    return float(slope), float(intercept)

def compute_features(window):
    C = np.array([c["close"] for c in window], dtype=float)
    H = np.array([c["high"]  for c in window], dtype=float)
    L = np.array([c["low"]   for c in window], dtype=float)

    H_max = H.max()
    L_min = L.min()
    pivot = (H_max + L_min + C[-1]) / 3.0

    # 43_fib_382_dist
    # Distance of close from 38.2% Fibonacci retracement level
    # Positive  → close ABOVE fib level (in upper half)
    # Negative  → close BELOW fib level (closer to support)
    # Neutral pads → H_max ≈ L_min ≈ ref → fib level ≈ ref → dist ≈ 0
    fib_382 = L_min + 0.382 * (H_max - L_min)
    fib_382_dist = float(C[-1] - fib_382)

    # 49_s1_dist
    # Distance from Support 1 pivot node
    # S1 = 2 × Pivot - High_max
    # Positive → close above S1 (price not yet at support)
    # Negative → close below S1 (price broken through support)
    s1 = 2.0 * pivot - H_max
    s1_dist = float(C[-1] - s1)

    # 48_r1_dist
    # Distance from Resistance 1 pivot node
    # R1 = 2 × Pivot - Low_min
    # Positive → close below R1 (space to resistance ceiling)
    # Negative → close above R1 (price broken through resistance)
    r1 = 2.0 * pivot - L_min
    r1_dist = float(r1 - C[-1])

    # 35_lin_reg_intercept
    # OLS trendline y-intercept — the statistical "anchor" of the price path
    # Represents where the fitted trend line starts at x=0
    # Neutral pads → flat closes → intercept ≈ ref_price
    _, intercept = ols_slope_intercept(C)
    lin_reg_intercept = intercept

    # 69_high_low_ratio
    # H_max / L_min — scale-invariant envelope elongation ratio
    # Near 1.0 → tight range (compression)
    # Higher   → wide range (expansion / volatility)
    # Neutral pads: H_max ≈ ref+pip, L_min ≈ ref-pip → ratio ≈ 1.0001
    high_low_ratio = float(H_max / (L_min + EPS))

    return {
        "43_fib_382_dist":      fib_382_dist,
        "49_s1_dist":           s1_dist,
        "48_r1_dist":           r1_dist,
        "35_lin_reg_intercept": lin_reg_intercept,
        "69_high_low_ratio":    high_low_ratio,
    }

# =============================================================================
# G — MAIN LOOP
# =============================================================================
print(f"\nGenerating {N_PER_COND:,} x {len(COND_MAP)} x {len(STEPS)} = "
      f"{N_PER_COND*len(COND_MAP)*len(STEPS):,} rows...\n")

out_rows = []
seq_id   = 0

for cid in tqdm(sorted(COND_MAP.keys()), desc="Model-2"):
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
                "sequence_id":          seq_id,
                "condition_id":         cid,
                "condition_name":       cname,
                "label":                raw_lbl,
                "label_structure":      LABEL_MAP[raw_lbl],
                "recognition_step":     step,
                "open":  fc["open"],  "high": fc["high"],
                "low":   fc["low"],   "close":fc["close"],
                "volume":fc["volume"],
                "43_fib_382_dist":      round(feats["43_fib_382_dist"],      8),
                "49_s1_dist":           round(feats["49_s1_dist"],           8),
                "48_r1_dist":           round(feats["48_r1_dist"],           8),
                "35_lin_reg_intercept": round(feats["35_lin_reg_intercept"], 8),
                "69_high_low_ratio":    round(feats["69_high_low_ratio"],    8),
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
print("NOTE: Apply StandardScaler to all 5 features before training.")
print("Feature ranges (step=16):")
for f in FEATURES:
    print(f"  {f:35s} min={df16[f].min():+.6f}  max={df16[f].max():+.6f}  mean={df16[f].mean():+.6f}")

COLS = (["sequence_id","condition_id","condition_name","label","label_structure",
         "recognition_step","open","high","low","close","volume"] + FEATURES)
df[COLS].to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved → {OUTPUT_CSV}  shape={df[COLS].shape}")
