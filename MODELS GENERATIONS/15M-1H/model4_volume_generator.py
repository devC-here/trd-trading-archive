# =============================================================================
# model4_volume_generator.py
# Model 4 — THE INSTITUTIONAL VOLUME & LIQUIDITY SPECIALIST
# Features (11): 90_money_flow_acceleration, 37_volume_trend_slope,
#   92_volume_to_range_ratio_final, 93_mean_volume_to_range_ratio,
#   86_volume_oscillator, 81_volume_rate_of_change, 84_negative_volume_index,
#   80_relative_volume_z_score, 89_volume_climax_filter,
#   82_force_index_final, 88_range_volume_correlation
# Normalization: MinMaxScaler (large integers + small indices → [0,1])
# Target: I_c in [0.0 (Liquidity Drain), +5.0 (Extreme Institutional Influx)]
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
OUTPUT_CSV = "model4_volume_dataset.csv"

N_PER_COND = 1250
WINDOW     = 16
STEPS      = [4, 8, 12, 16]
PIP        = 0.00010
EPS        = 1e-12
SEED       = 42
random.seed(SEED)
np.random.seed(SEED)

FEATURES = [
    "90_money_flow_acceleration",
    "37_volume_trend_slope",
    "92_volume_to_range_ratio_final",
    "93_mean_volume_to_range_ratio",
    "86_volume_oscillator",
    "81_volume_rate_of_change",
    "84_negative_volume_index",
    "80_relative_volume_z_score",
    "89_volume_climax_filter",
    "82_force_index_final",
    "88_range_volume_correlation",
]

# Target: institutional commitment intensity
# BUY/SELL → high institutional commitment → higher score
# HOLD     → low / drained liquidity       → lower score
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
def ols_slope(y):
    """OLS slope only."""
    n  = len(y); x = np.arange(n, dtype=float)
    mx, my = x.mean(), y.mean()
    return float(((x-mx)*(y-my)).sum() / (((x-mx)**2).sum()+EPS))

def ema_val(series, period):
    """EMA scalar — final value."""
    k = 2.0 / (period + 1)
    e = float(series[0])
    for v in series[1:]: e = float(v)*k + e*(1-k)
    return e

def pearson_r(x, y):
    mx, my = x.mean(), y.mean()
    num = ((x-mx)*(y-my)).sum()
    den = np.sqrt(((x-mx)**2).sum() * ((y-my)**2).sum())
    return float(num / (den + EPS))

def compute_features(window):
    C = np.array([c["close"]  for c in window], dtype=float)
    H = np.array([c["high"]   for c in window], dtype=float)
    L = np.array([c["low"]    for c in window], dtype=float)
    V = np.array([c["volume"] for c in window], dtype=float)

    TP = (H + L + C) / 3.0   # Typical Price

    # 90_money_flow_acceleration
    # (TP[t] × V[t]) − (TP[t-1] × V[t-1])
    # Positive → money flowing in faster (buying pressure increasing)
    # Negative → money flowing out (selling pressure or deceleration)
    mf_accel = float(TP[-1]*V[-1] - TP[-2]*V[-2])

    # 37_volume_trend_slope
    # OLS slope across 16 volume bars
    # Positive → volume trending up over window (building interest)
    # Negative → volume trending down (fading participation)
    vol_slope = ols_slope(V)

    # 92_volume_to_range_ratio_final
    # V[t] / (range in pips) — absorption efficiency on final candle
    # High → lots of volume per pip → institutional absorption / tight spread
    # Low  → little volume per pip  → low conviction move
    final_rng_pips = (H[-1] - L[-1]) * 10000
    v2r_final = float(V[-1] / (final_rng_pips + EPS))

    # 93_mean_volume_to_range_ratio
    # Row-by-row V/range averaged across 16 bars
    v2r_all  = V / ((H - L) * 10000 + EPS)
    v2r_mean = float(v2r_all.mean())

    # 86_volume_oscillator
    # ((EMA3(V) - EMA8(V)) / EMA8(V)) × 100
    # Positive → short-term volume > long-term → acceleration
    # Negative → short-term volume < long-term → deceleration
    e3 = ema_val(V, 3); e8 = ema_val(V, 8)
    vol_osc = float(((e3 - e8) / (e8 + EPS)) * 100)

    # 81_volume_rate_of_change
    # ((V[t] - V[t-3]) / V[t-3]) × 100
    # Percentage speed change of volume over last 3 bars
    vol_roc = float(((V[-1] - V[-4]) / (V[-4] + EPS)) * 100)

    # 84_negative_volume_index
    # Cumulative smart-money tracker. Initialises at 1000.
    # Updates ONLY on bars where volume < previous volume.
    # Rising NVI on down-volume → smart money buying quietly
    nvi = 1000.0
    for i in range(1, WINDOW):
        if V[i] < V[i-1]:
            nvi += ((C[i] - C[i-1]) / (C[i-1] + EPS)) * nvi

    # 80_relative_volume_z_score
    # Z-score of final candle volume vs window distribution
    vm, vs = V.mean(), max(V.std(), EPS)
    rv_z = float((V[-1] - vm) / vs)

    # 89_volume_climax_filter
    # Binary: 1 if V[t] > mean + 2σ (exhaustion / climax spike), else 0
    climax = int(V[-1] > vm + 2*vs)

    # 82_force_index_final
    # (Close[t] - Close[t-1]) × Volume[t]
    # Large positive → strong bull bar with high volume (conviction)
    # Large negative → strong bear bar with high volume (distribution)
    force = float((C[-1] - C[-2]) * V[-1])

    # 88_range_volume_correlation
    # Pearson r between candle ranges and volumes across 16 bars
    # High positive → bigger candles have more volume (healthy structure)
    # Near zero     → volume and range uncorrelated (noise / manipulation)
    ranges = H - L
    rv_corr = pearson_r(ranges, V)

    return {
        "90_money_flow_acceleration":     mf_accel,
        "37_volume_trend_slope":          vol_slope,
        "92_volume_to_range_ratio_final": v2r_final,
        "93_mean_volume_to_range_ratio":  v2r_mean,
        "86_volume_oscillator":           vol_osc,
        "81_volume_rate_of_change":       vol_roc,
        "84_negative_volume_index":       nvi,
        "80_relative_volume_z_score":     rv_z,
        "89_volume_climax_filter":        climax,
        "82_force_index_final":           force,
        "88_range_volume_correlation":    rv_corr,
    }

# =============================================================================
# G — MAIN LOOP
# =============================================================================
print(f"\nGenerating {N_PER_COND:,} x {len(COND_MAP)} x {len(STEPS)} = "
      f"{N_PER_COND*len(COND_MAP)*len(STEPS):,} rows...\n")

out_rows = []
seq_id   = 0

for cid in tqdm(sorted(COND_MAP.keys()), desc="Model-4"):
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
                "label_volume":                   LABEL_MAP[raw_lbl],
                "recognition_step":               step,
                "open":  fc["open"],  "high": fc["high"],
                "low":   fc["low"],   "close":fc["close"],
                "volume":fc["volume"],
                "90_money_flow_acceleration":     round(feats["90_money_flow_acceleration"],     8),
                "37_volume_trend_slope":          round(feats["37_volume_trend_slope"],          8),
                "92_volume_to_range_ratio_final": round(feats["92_volume_to_range_ratio_final"], 8),
                "93_mean_volume_to_range_ratio":  round(feats["93_mean_volume_to_range_ratio"],  8),
                "86_volume_oscillator":           round(feats["86_volume_oscillator"],           8),
                "81_volume_rate_of_change":       round(feats["81_volume_rate_of_change"],       8),
                "84_negative_volume_index":       round(feats["84_negative_volume_index"],       8),
                "80_relative_volume_z_score":     round(feats["80_relative_volume_z_score"],     8),
                "89_volume_climax_filter":        int(feats["89_volume_climax_filter"]),
                "82_force_index_final":           round(feats["82_force_index_final"],           8),
                "88_range_volume_correlation":    round(feats["88_range_volume_correlation"],    8),
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
print("NOTE: Apply MinMaxScaler to all 11 features before training.")
print("Feature ranges (step=16):")
for f in FEATURES:
    print(f"  {f:38s} min={df16[f].min():+.4f}  max={df16[f].max():+.4f}  mean={df16[f].mean():+.4f}")

COLS = (["sequence_id","condition_id","condition_name","label","label_volume",
         "recognition_step","open","high","low","close","volume"] + FEATURES)
df[COLS].to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved → {OUTPUT_CSV}  shape={df[COLS].shape}")
