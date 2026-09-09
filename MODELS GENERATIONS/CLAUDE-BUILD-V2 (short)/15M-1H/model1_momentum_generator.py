# =============================================================================
# model1_momentum_generator.py
# =============================================================================
# Model 1 — THE DIRECTIONAL MOMENTUM SPECIALIST
#
# Core Objective:
#   Isolate the velocity, acceleration, and linear stability of the current
#   lookback path. Answers: "How fast is price moving, and is the trend
#   numerically stable or decaying?"
#
# Target Output:
#   T_m = [P(Bullish Momentum), P(Bearish Momentum)]
#   A continuous directional probability vector.
#
# Features Generated (6 — no normalisation needed, naturally scale-invariant):
#   13_roc_4                         Rate of Change over 4 periods
#   58_skewness_returns              Fisher-Pearson skewness of log returns
#   101_consecutive_direction_counter Direction streak accumulator
#   103_final_candle_color           Final candle bull/bear/doji signal
#   15_roc_acceleration              First derivative of 1-period ROC
#   36_r_squared                     OLS R² linear trend stability
#
# Data Architecture:
#   • Source     : ohlcv_40_m15.csv  (40 conditions × 5 samples × 16 M15 candles)
#   • Window     : 16 M15 candles per sequence
#   • Steps      : 4 / 8 / 12 / 16  (partial-sequence recognition)
#   • Padding    : (16 - step) neutral candles prepended when step < 16
#   • Output     : one row per (sequence × step) combination
#
# OHLCV Physics Rules (enforced on every candle):
#   High  >= max(Open, Close)
#   Low   <= min(Open, Close)
#   Open[t] = Close[t-1]   (price continuity)
#   Volume  > 0
# =============================================================================

import math
import random
import numpy as np
import pandas as pd
from collections import defaultdict

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw): return x

# =============================================================================
# SECTION A — CONFIGURATION
# =============================================================================

# FIX: old code pointed to a non-existent absolute path ohlcv_40_unique.csv.
#      Use relative paths so the script runs from its own directory alongside
#      the CSV, and write output to the same directory.
INPUT_CSV   = "ohlcv_40_m15.csv"
OUTPUT_CSV  = "model1_momentum_dataset.csv"

N_PER_COND  = 5000          # synthetic sequences generated per condition
                             # 40 conditions × 5000 = 200,000 sequences
                             # × 4 steps = 800,000 rows total

WINDOW      = 16            # fixed M15 candle window size
STEPS       = [4, 8, 12, 16]  # recognition checkpoints

PIP         = 0.00010       # EUR/USD 1 pip
EPS         = 1e-12         # division-by-zero guard

SEED        = 42
random.seed(SEED)
np.random.seed(SEED)

# The 6 feature columns produced by this generator.
FEATURES = [
    "13_roc_4",                           # bounded %  (±~2%)
    "58_skewness_returns",                # statistical moment (±3 typical)
    "101_consecutive_direction_counter",  # signed integer [-15 … +15]
    "103_final_candle_color",             # signed binary {-1, 0, +1}
    "15_roc_acceleration",                # bounded % delta (±~1%)
    "36_r_squared",                       # regression fit [0.0 … 1.0]
]

# =============================================================================
# SECTION B — LOAD & PARSE TEMPLATE CONDITIONS
# =============================================================================
# FIX: old code used csv.DictReader and read column "candle" (doesn't exist in
#      ohlcv_40_m15.csv) and expected 4-candle groups. The new CSV has 16 M15
#      candles per (condition_id, sample), sorted by m15_sequence (1-16).
#      Load with pandas, validate required columns, and build 16-candle groups.

df_raw = pd.read_csv(INPUT_CSV)

# Validate expected columns are present before touching the data
_required = {"condition_id", "condition_name", "sample",
             "m15_sequence", "open", "high", "low", "close", "volume", "label"}
_missing = _required - set(df_raw.columns)
if _missing:
    raise ValueError(f"Input CSV is missing expected columns: {_missing}")

# condition_id → list of 16-candle template sequences
COND_MAP = defaultdict(list)

for (cid, _sid), grp in df_raw.groupby(["condition_id", "sample"], sort=False):
    grp_sorted = grp.sort_values("m15_sequence")
    seq = [
        {
            "open":           float(row["open"]),
            "high":           float(row["high"]),
            "low":            float(row["low"]),
            "close":          float(row["close"]),
            "volume":         int(row["volume"]),
            "label":          str(row["label"]),
            "condition_id":   str(row["condition_id"]),
            "condition_name": str(row["condition_name"]),
        }
        for _, row in grp_sorted.iterrows()
    ]
    # FIX: only add complete 16-candle sequences; partial groups would produce
    #      incorrect feature windows and corrupt scenario derivation.
    if len(seq) == WINDOW:
        COND_MAP[cid].append(seq)

# FIX: old code derived label/condition_name from tmpl_seqs[0][0] inline in the
#      loop — fragile if the map is empty or unsorted. Build a clean lookup once.
COND_META = (
    df_raw.groupby("condition_id")[["condition_name", "label"]]
    .first()
    .to_dict("index")
)

print(f"Loaded   : {len(COND_MAP)} conditions")
print(f"Templates: {sum(len(v) for v in COND_MAP.values())} sequences total")
print(f"Labels   : {sorted(set(m['label'] for m in COND_META.values()))}\n")

# =============================================================================
# SECTION C — SCENARIO DERIVATION
# =============================================================================
# Measure each condition's actual M15 template statistics and derive a scenario
# dict from them so generated sequences are structurally faithful.

def derive_scenario(template_sequences: list) -> dict:
    """
    Measure price behaviour statistics across all template sequences for
    one condition and return a synthetic generation parameter dict.

    Parameters
    ----------
    template_sequences : list of lists
        Each inner list is one 16-candle M15 template sequence.

    Returns
    -------
    dict
        bias       : P(bullish candle)       — drives candle direction
        body_scale : mean absolute body size  — drives move magnitude
        wl / wh    : lower/upper wick multipliers relative to body
        vl / vh    : volume lower / upper bounds
    """
    bodies, full_ranges, volumes, directions = [], [], [], []

    for seq in template_sequences:
        for candle in seq:
            body = abs(candle["close"] - candle["open"])
            rng  = candle["high"] - candle["low"]
            bodies.append(body)
            full_ranges.append(rng)
            volumes.append(candle["volume"])
            directions.append(1 if candle["close"] > candle["open"] else 0)

    avg_body = np.mean(bodies)      if bodies      else 0.001
    avg_rng  = np.mean(full_ranges) if full_ranges else 0.002
    avg_vol  = np.mean(volumes)     if volumes     else 500.0
    std_vol  = np.std(volumes)      if volumes     else 200.0
    bias     = np.mean(directions)  if directions  else 0.5

    avg_wick_each_side = max(0.0, (avg_rng - avg_body)) / 2.0
    wl = max(0.05, avg_wick_each_side / (avg_body + EPS) * 0.6)
    wh = max(wl + 0.10, avg_wick_each_side / (avg_body + EPS) * 1.4)

    return {
        "bias":       float(np.clip(bias,     0.10, 0.90)),
        "body_scale": float(np.clip(avg_body, PIP,  80 * PIP)),
        "wl":         float(np.clip(wl,       0.02, 1.00)),
        "wh":         float(np.clip(wh,       0.10, 2.50)),
        "vl":         float(max(10.0,  avg_vol - std_vol)),
        "vh":         float(max(50.0,  avg_vol + std_vol)),
    }

SCENARIOS = {cid: derive_scenario(seqs) for cid, seqs in COND_MAP.items()}

# =============================================================================
# SECTION D — CANDLE GENERATOR
# =============================================================================

def gen_candle(open_price: float, sc: dict, noise: float = 1.0) -> tuple:
    """
    Generate one M15 OHLCV candle from a scenario dict.

    Physics rules enforced unconditionally:
        High  >= max(Open, Close)
        Low   <= min(Open, Close)
        Volume > 0

    Price continuity (Open[t] = Close[t-1]) is maintained by the caller
    passing the previous candle's close as open_price.

    Returns
    -------
    (candle_dict, close_price)
    """
    direction  = 1 if random.random() < sc["bias"] else -1
    body       = float(np.random.uniform(0.5, 1.5) * sc["body_scale"] * noise)
    upper_wick = float(body * np.random.uniform(sc["wl"], sc["wh"]))
    lower_wick = float(body * np.random.uniform(sc["wl"], sc["wh"]))

    o = float(open_price)
    c = float(o + direction * body)
    h = max(o, c) + abs(upper_wick)   # High >= max(O,C)  always
    l = min(o, c) - abs(lower_wick)   # Low  <= min(O,C)  always
    v = max(1, int(np.random.uniform(sc["vl"], sc["vh"]) * noise))

    return {
        "open":   round(o, 5),
        "high":   round(h, 5),
        "low":    round(l, 5),
        "close":  round(c, 5),
        "volume": v,
    }, c

# =============================================================================
# SECTION E — SEQUENCE GENERATOR
# =============================================================================

def generate_sequence(sc: dict, template_seqs: list) -> list:
    """
    Generate one complete 16-candle M15 sequence for a condition.

    Starting price is jittered around a randomly chosen template's first open.
    FIX: old code used template_seqs[0]["open"] — templates are now 16-candle
         lists so the first candle is template_seqs[x][0]["open"].
    """
    ref_price = random.choice(template_seqs)[0]["open"]
    ref_price += np.random.normal(0, 15 * PIP)

    candles = []
    price   = ref_price

    for _ in range(WINDOW):
        noise = np.random.lognormal(mean=0.0, sigma=0.15)
        candle, price = gen_candle(price, sc, noise=noise)
        candles.append(candle)

    return candles

# =============================================================================
# SECTION F — NEUTRAL BASELINE PADDING  (Warm-Start Solution)
# =============================================================================
# Problem  : features like ROC_4 or R² need ≥ 5 bars of history.
#            At step=4 we only have 4 real candles.
# Solution : prepend (16 - step) perfectly flat "neutral" candles so the
#            feature engine always receives exactly 16 bars.
#
# Neutral candle design:
#   Open = Close = ref_price       → zero log-return → no directional signal
#   High = ref_price + 1pip        → prevents range = 0 → std/ATR never 0
#   Low  = ref_price - 1pip
#   Volume = mean volume of real candles → neutral volume baseline

def neutral_candle(ref_price: float, avg_vol: int) -> dict:
    """One perfectly flat neutral baseline candle."""
    return {
        "open":   ref_price,
        "high":   ref_price + PIP,
        "low":    ref_price - PIP,
        "close":  ref_price,
        "volume": max(1, int(avg_vol)),
    }

def build_padded_window(real_candles: list, step: int) -> list:
    """
    Prepend (WINDOW - step) neutral candles to real_candles.
    Always returns exactly WINDOW = 16 candles.
    """
    avg_vol = max(1, int(np.mean([c["volume"] for c in real_candles])))
    ref     = real_candles[0]["open"]
    n_pads  = WINDOW - step
    padding = [neutral_candle(ref, avg_vol) for _ in range(n_pads)]
    return padding + real_candles

# =============================================================================
# SECTION G — MODEL 1 FEATURE CALCULATION ENGINE
# =============================================================================

def ols_r_squared(y: np.ndarray) -> float:
    """
    Fit OLS linear regression through equally-spaced y-values (x = 0..n-1).
    Returns R² = 1 - SS_res / SS_tot.

    R² near 1.0 → clean straight-line trend (strong conviction)
    R² near 0.0 → choppy / random price (no linear structure)
    """
    n   = len(y)
    x   = np.arange(n, dtype=float)
    mx  = x.mean()
    my  = y.mean()
    ss_xx     = ((x - mx) ** 2).sum()
    ss_xy     = ((x - mx) * (y - my)).sum()
    slope     = ss_xy / (ss_xx + EPS)
    intercept = my - slope * mx
    ss_res    = ((y - (intercept + slope * x)) ** 2).sum()
    ss_tot    = ((y - my) ** 2).sum() + EPS
    return float(1.0 - ss_res / ss_tot)


def compute_model1_features(window: list) -> dict:
    """
    Compute all 6 Model 1 momentum features on a 16-candle window.

    Parameters
    ----------
    window : list of 16 dicts with keys: open / high / low / close / volume.
             First (16-step) entries may be neutral padding candles.

    Returns
    -------
    dict  feature_name → numeric value
    """
    C = np.array([c["close"] for c in window], dtype=float)
    O = np.array([c["open"]  for c in window], dtype=float)

    # ── 13_roc_4 ─────────────────────────────────────────────────────────────
    # Rate of Change over 4 periods: ((C[t] - C[t-4]) / C[t-4]) × 100
    roc_4 = float(((C[-1] - C[-5]) / (C[-5] + EPS)) * 100)

    # ── 58_skewness_returns ───────────────────────────────────────────────────
    # Adjusted Fisher-Pearson standardised skewness of 15 log returns.
    log_ret  = np.diff(np.log(C + EPS))
    n, m     = len(log_ret), log_ret.mean()
    s        = log_ret.std(ddof=1) + EPS
    skewness = float(
        (n / ((n - 1) * (n - 2) + EPS)) * np.sum(((log_ret - m) / s) ** 3)
    )

    # ── 101_consecutive_direction_counter ────────────────────────────────────
    # Signed integer streak: +1 per consecutive higher close, -1 per lower.
    # Resets to ±1 on direction swap; unchanged on equal close.
    counter = 0
    for i in range(1, WINDOW):
        if   C[i] > C[i - 1]: counter = (counter + 1) if counter > 0 else 1
        elif C[i] < C[i - 1]: counter = (counter - 1) if counter < 0 else -1

    # ── 103_final_candle_color ────────────────────────────────────────────────
    # +1 bull, -1 bear, 0 doji — only the last candle.
    final_color = int(1 if C[-1] > O[-1] else (-1 if C[-1] < O[-1] else 0))

    # ── 15_roc_acceleration ──────────────────────────────────────────────────
    # First derivative of 1-period ROC: ROC_1[t] - ROC_1[t-1]
    roc1_t   = ((C[-1] - C[-2]) / (C[-2] + EPS)) * 100
    roc1_t1  = ((C[-2] - C[-3]) / (C[-3] + EPS)) * 100
    roc_accel = float(roc1_t - roc1_t1)

    # ── 36_r_squared ─────────────────────────────────────────────────────────
    # OLS R² across all 16 closing prices.
    r_squared = ols_r_squared(C)

    return {
        "13_roc_4":                           roc_4,
        "58_skewness_returns":                skewness,
        "101_consecutive_direction_counter":  counter,
        "103_final_candle_color":             final_color,
        "15_roc_acceleration":                roc_accel,
        "36_r_squared":                       r_squared,
    }

# =============================================================================
# SECTION H — LABEL MAPPER
# =============================================================================
# Model 1 target: T_m = [P(Bullish Momentum), P(Bearish Momentum)]
#   BUY  → label_momentum = 1   (Bullish momentum)
#   SELL → label_momentum = 0   (Bearish momentum)
#   HOLD → label_momentum = -1  (No directional momentum — kept for ensemble)

def map_label(raw_label: str) -> int:
    mapping = {"BUY": 1, "SELL": 0, "HOLD": -1}
    # FIX: old code used a bare dict lookup which raises KeyError on unexpected
    #      label values. Use .get() with a clear error message instead.
    if raw_label not in mapping:
        raise ValueError(
            f"Unexpected label '{raw_label}'. Expected one of {list(mapping.keys())}."
        )
    return mapping[raw_label]

# =============================================================================
# SECTION I — MAIN GENERATION LOOP
# =============================================================================

total_rows = N_PER_COND * len(COND_MAP) * len(STEPS)
print(f"Generating  : {N_PER_COND:,} seqs × {len(COND_MAP)} conditions × {len(STEPS)} steps")
print(f"Total rows  : {total_rows:,}")
print(f"Output file : {OUTPUT_CSV}\n")

out_rows = []
seq_id   = 0

for cid in tqdm(sorted(COND_MAP.keys()), desc="Model-1 generation"):

    sc        = SCENARIOS[cid]
    tmpl_seqs = COND_MAP[cid]

    # FIX: old code pulled label/condition_name from tmpl_seqs[0][0] inline —
    #      fragile if sequences are unsorted or the map is empty. Use COND_META.
    meta      = COND_META[cid]
    raw_label = meta["label"]
    cname     = meta["condition_name"]
    mom_label = map_label(raw_label)

    for _ in range(N_PER_COND):

        candles = generate_sequence(sc, tmpl_seqs)

        for step in STEPS:

            seen   = candles[:step]
            window = build_padded_window(seen, step)
            feats  = compute_model1_features(window)
            fc     = seen[-1]

            out_rows.append({
                # Metadata
                "sequence_id":      seq_id,
                "condition_id":     cid,
                "condition_name":   cname,
                "label":            raw_label,
                "label_momentum":   mom_label,
                "recognition_step": step,
                # Final candle OHLCV at this step
                "open":    fc["open"],
                "high":    fc["high"],
                "low":     fc["low"],
                "close":   fc["close"],
                "volume":  fc["volume"],
                # Model 1 features
                "13_roc_4":                          round(feats["13_roc_4"], 8),
                "58_skewness_returns":               round(feats["58_skewness_returns"], 8),
                "101_consecutive_direction_counter": int(feats["101_consecutive_direction_counter"]),
                "103_final_candle_color":            int(feats["103_final_candle_color"]),
                "15_roc_acceleration":               round(feats["15_roc_acceleration"], 8),
                "36_r_squared":                      round(feats["36_r_squared"], 8),
            })

        seq_id += 1

# =============================================================================
# SECTION J — VALIDATION
# =============================================================================

df   = pd.DataFrame(out_rows)
df16 = df[df["recognition_step"] == 16]

print(f"\n{'═' * 65}")
print(f"  MODEL 1 — DIRECTIONAL MOMENTUM SPECIALIST")
print(f"{'═' * 65}")
print(f"  Total rows          : {len(df):,}")
print(f"  Total sequences     : {seq_id:,}")

print(f"\n  Label distribution (step=16):")
for lbl, cnt in df16["label"].value_counts().items():
    pct = 100 * cnt / len(df16)
    print(f"    {lbl:6s}  →  {cnt:,}  ({pct:.1f}%)")

print(f"\n  Momentum target distribution (step=16):")
lmap = {1: "Bullish (+1)", 0: "Bearish (0)", -1: "Hold (-1)"}
for val, cnt in sorted(df16["label_momentum"].value_counts().items(), reverse=True):
    pct = 100 * cnt / len(df16)
    print(f"    {lmap[val]:15s}  →  {cnt:,}  ({pct:.1f}%)")

print(f"\n  Rows per recognition step:")
for step in STEPS:
    cnt = (df["recognition_step"] == step).sum()
    print(f"    step={step:2d}  →  {cnt:,} rows")

# Feature range summary across all steps
print(f"\n  Feature ranges across recognition steps:")
col_w  = max(len(f) for f in FEATURES) + 2
header = f"    {'Feature':{col_w}}  {'Step':>5}  {'min':>9}  {'max':>9}  {'mean':>9}"
print(header)
print("    " + "─" * (len(header) - 4))
for feat in FEATURES:
    for step in STEPS:
        sub = df[df["recognition_step"] == step][feat]
        print(f"    {feat:{col_w}}  {step:>5}  "
              f"{sub.min():>+9.4f}  {sub.max():>+9.4f}  {sub.mean():>+9.4f}")
    print()

# OHLCV physics validation
bad_h = (df["high"]  < df[["open", "close"]].max(axis=1)).sum()
bad_l = (df["low"]   > df[["open", "close"]].min(axis=1)).sum()
print(f"  Physics check:")
print(f"    high < max(O,C)  violations : {bad_h}")
print(f"    low  > min(O,C)  violations : {bad_l}")
print(f"    Status : {'CLEAN' if bad_h + bad_l == 0 else 'VIOLATIONS FOUND'}")

# =============================================================================
# SECTION K — SAVE
# =============================================================================

ORDERED_COLS = (
    ["sequence_id", "condition_id", "condition_name",
     "label", "label_momentum", "recognition_step",
     "open", "high", "low", "close", "volume"]
    + FEATURES
)

df[ORDERED_COLS].to_csv(OUTPUT_CSV, index=False)

print(f"\n  Saved  →  {OUTPUT_CSV}")
print(f"  Shape  :  {df[ORDERED_COLS].shape}")
print(f"{'═' * 65}\n")