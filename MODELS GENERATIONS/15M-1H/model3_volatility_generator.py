# =============================================================================
# model3_volatility_generator.py  (FIXED v2)
# Model 3 — THE VOLATILITY SQUEEZE SPECIALIST
# Features: 62_bollinger_bandwidth_z_score, 63_keltner_channel_squeeze,
#           72_true_range_z_score, 65_fractal_dimension_index
# Normalization: None (standardized structural metrics)
# Target: A_v in [0.0 (Pure Chop), 1.0 (Pure Breakout Expansion)]
# Input: ohlcv_40_m15.csv
#
# ── WHAT WAS BROKEN IN v1 ───────────────────────────────────────────────────
# derive_scenario() only measured average body/wick/volume size and threw
# away the single most important signal: how WIDE the condition's actual
# 16-candle range was. HOLD_05/HOLD_07 templates are wider than half the
# BUY/SELL templates, so every condition ended up generating roughly the
# same "uniform noise" candle stream regardless of label. Result: BB-Z,
# KC-squeeze, TR-Z and FDI came out statistically identical between classes
# (means within 0.01-0.03 of each other) — the model could not learn anything.
#
# ── WHAT THIS VERSION FIXES ──────────────────────────────────────────────────
# 1. A `regime` flag (CHOP vs EXPANSION) is derived per condition straight
#    from its true label (HOLD → CHOP, BUY/SELL → EXPANSION) — not inferred
#    indirectly from body size.
# 2. CHOP sequences actively SUPPRESS body/wick scale for all 16 candles —
#    true sideways compression, tight Bollinger Bands the whole window.
# 3. EXPANSION sequences use a two-phase generator: candles 1..k stay
#    compressed (squeeze build-up), then candles k+1..16 burst into wide
#    trending bodies (the actual breakout) — exactly the texture these
#    4 features are designed to detect.
# 4. This produces real separation: low BB-Z / low TR-Z / FDI→2.0 for CHOP,
#    high BB-Z / high TR-Z / FDI→1.0 for EXPANSION.
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

N_PER_COND = 5000
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
# C — SCENARIO DERIVATION  (FIXED — now regime-aware)
# =============================================================================
def derive_scenario(template_sequences, raw_label):
    """
    Derive generation parameters AND an explicit volatility regime.

    KEY FIX: regime is set directly from the true label, not inferred from
    body statistics. This guarantees HOLD sequences are generated as
    genuine compression and BUY/SELL sequences are generated as genuine
    squeeze-then-breakout — giving the 4 volatility features real signal
    to learn from.
    """
    bodies, ranges, volumes = [], [], []
    for seq in template_sequences:
        for c in seq:
            bodies.append(abs(c["close"] - c["open"]))
            ranges.append(c["high"] - c["low"])
            volumes.append(c["volume"])

    avg_body = np.mean(bodies)  if bodies  else 0.001
    avg_vol  = np.mean(volumes) if volumes else 500.0
    std_vol  = np.std(volumes)  if volumes else 200.0

    # Explicit regime flag — this is the critical fix
    regime = "CHOP" if raw_label == "HOLD" else "EXPANSION"

    if regime == "CHOP":
        # Sustained tight range for the ENTIRE 16-candle window.
        # Body scale capped low and held constant — no breakout phase.
        chop_body_scale = float(np.clip(min(avg_body, 6 * PIP), PIP, 8 * PIP))
        return {
            "regime":        regime,
            "body_scale":    chop_body_scale,
            "breakout_mult": 1.0,     # no expansion multiplier — stays flat
            "squeeze_frac":  1.0,     # 100% of window stays compressed
            "wl": 0.15, "wh": 0.45,   # small, tight wicks
            "vl": float(max(10.0, avg_vol - std_vol * 0.5)),
            "vh": float(max(50.0, avg_vol)),
        }
    else:
        # EXPANSION regime: two-phase — squeeze build-up then breakout burst.
        # squeeze_frac controls what fraction of the 16 bars stay compressed
        # before the breakout phase begins (randomised per-condition flavor).
        base_body_scale = float(np.clip(avg_body, 10 * PIP, 80 * PIP))
        return {
            "regime":        regime,
            "body_scale":    float(np.clip(base_body_scale * 0.25, PIP, 10*PIP)),  # squeeze-phase body
            "breakout_mult": float(np.clip(base_body_scale / max(PIP, base_body_scale*0.25), 3.0, 12.0)),
            "squeeze_frac":  float(np.random.uniform(0.55, 0.80)),  # ~55-80% squeeze, then burst
            "wl": 0.10, "wh": 0.35,
            "vl": float(max(10.0, avg_vol - std_vol)),
            "vh": float(max(50.0, avg_vol + std_vol)),
        }

SCENARIOS = {
    cid: derive_scenario(seqs, seqs[0][0]["label"])
    for cid, seqs in COND_MAP.items()
}

# =============================================================================
# D — CANDLE & SEQUENCE GENERATORS  (FIXED — regime-driven volatility shaping)
# =============================================================================
def gen_candle(open_price, body_scale, wl, wh, vl, vh, noise=1.0):
    """Generate one candle with EXPLICIT body_scale (caller controls regime phase)."""
    d    = 1 if random.random() < 0.5 else -1   # direction is irrelevant to volatility texture
    body = float(np.random.uniform(0.6, 1.4) * body_scale * noise)
    uw   = float(body * np.random.uniform(wl, wh))
    lw   = float(body * np.random.uniform(wl, wh))
    o = float(open_price)
    c = float(o + d * body)
    h = max(o, c) + abs(uw)
    l = min(o, c) - abs(lw)
    v = max(1, int(np.random.uniform(vl, vh) * noise))
    return {"open": round(o,5), "high": round(h,5),
            "low":  round(l,5), "close": round(c,5), "volume": v}, c

def generate_sequence(sc, template_seqs):
    """
    Generate 16 candles whose volatility texture matches the condition's regime.

    CHOP regime      : all 16 candles use the same small body_scale → genuine
                        sustained compression (tight BB, low TR throughout).
    EXPANSION regime : candles 1..k use a SMALL body_scale (squeeze build-up,
                        matching real pre-breakout coiling), then candles
                        k+1..16 jump to body_scale × breakout_mult (the actual
                        breakout burst) → BB bandwidth and TR genuinely expand
                        exactly where the breakout starts.
    """
    price = random.choice(template_seqs)[0]["open"] + np.random.normal(0, 15*PIP)
    candles = []

    squeeze_bars = int(round(sc["squeeze_frac"] * WINDOW))
    squeeze_bars = max(3, min(WINDOW - 2, squeeze_bars))  # keep both phases non-trivial

    for i in range(WINDOW):
        noise = np.random.lognormal(0, 0.12)

        if sc["regime"] == "CHOP":
            body_scale = sc["body_scale"]
        else:
            if i < squeeze_bars:
                body_scale = sc["body_scale"]                       # squeeze phase
            else:
                # ramp the breakout in over 2 bars instead of an instant jump,
                # closer to how real expansions accelerate
                ramp = min(1.0, (i - squeeze_bars + 1) / 2.0)
                body_scale = sc["body_scale"] * (1.0 + ramp * (sc["breakout_mult"] - 1.0))

        candle, price = gen_candle(price, body_scale, sc["wl"], sc["wh"],
                                     sc["vl"], sc["vh"], noise=noise)
        candles.append(candle)

    return candles

# =============================================================================
# E — NEUTRAL PADDING  (unchanged — warm-start logic preserved)
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
# F — FEATURE ENGINE  (unchanged math — now receives real regime signal)
# =============================================================================
def compute_features(window):
    C = np.array([c["close"] for c in window], dtype=float)
    H = np.array([c["high"]  for c in window], dtype=float)
    L = np.array([c["low"]   for c in window], dtype=float)

    H_max  = H.max(); L_min = L.min()
    mean_C = C.mean()
    std_C  = max(C.std(), EPS)

    TR = np.array(
        [H[0] - L[0]] +
        [max(H[i]-L[i], abs(H[i]-C[i-1]), abs(L[i]-C[i-1]))
         for i in range(1, WINDOW)]
    )
    ATR = TR.mean()

    # 62_bollinger_bandwidth_z_score
    bb_bw    = (4.0 * std_C) / (mean_C + EPS)
    sub_bws  = np.array([
        (4.0 * max(C[:i].std(), EPS)) / (C[:i].mean() + EPS)
        for i in range(4, WINDOW + 1)
    ])
    sub_mean, sub_std = sub_bws.mean(), max(sub_bws.std(), EPS)
    bb_z = float((bb_bw - sub_mean) / sub_std)

    # 63_keltner_channel_squeeze
    kc_bw   = (2.0 * ATR) / (mean_C + EPS)
    kc_sq   = float(bb_bw / (kc_bw + EPS))

    # 72_true_range_z_score
    tr_mean, tr_std = TR.mean(), max(TR.std(), EPS)
    tr_z = float((TR[-1] - tr_mean) / tr_std)

    # 65_fractal_dimension_index
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

print("\nFeature separation check (step=16) — Chop vs Breakout:")
for f in FEATURES:
    g0 = df16[df16.label_volatility==0][f]
    g1 = df16[df16.label_volatility==1][f]
    sep = abs(g1.mean() - g0.mean())
    print(f"  {f:38s} CHOP mean={g0.mean():+.4f}  BREAKOUT mean={g1.mean():+.4f}  |sep|={sep:.4f}")

COLS = (["sequence_id","condition_id","condition_name","label","label_volatility",
         "recognition_step","open","high","low","close","volume"] + FEATURES)
df[COLS].to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved → {OUTPUT_CSV}  shape={df[COLS].shape}")
