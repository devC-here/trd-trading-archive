# ─────────────────────────────────────────────────────────────
# Synthetic Candle Dataset Generator (Robust + Minimal + Clean)
# Features: 5  |  Labels: 3 (bearish / neutral / bullish)
# Samples: 20,000+
# ─────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import random

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
N_SAMPLES   = 20000
OUTPUT_PATH = "synthetic_candle_clf_dataset.csv"
BASE_PRICE  = 1.2000
SEED        = 42

np.random.seed(SEED)
random.seed(SEED)

EPS = 1e-10

# ─────────────────────────────────────────────
# ARCHETYPES (realistic candle shapes)
# (p_bull, body_mu, body_sd, wick_mu, wick_sd, weight)
# ─────────────────────────────────────────────
ARCHETYPES = {
    "strong_bull": (0.95, 0.75, 0.08, 0.05, 0.03, 0.15),
    "strong_bear": (0.05, 0.75, 0.08, 0.05, 0.03, 0.15),

    "hammer":      (0.85, 0.30, 0.08, 0.05, 0.03, 0.10),
    "shooting":    (0.15, 0.30, 0.08, 0.50, 0.10, 0.10),

    "doji":        (0.50, 0.05, 0.02, 0.40, 0.15, 0.10),
    "spinning":    (0.50, 0.18, 0.05, 0.30, 0.10, 0.10),

    "balanced":    (0.50, 0.35, 0.10, 0.25, 0.10, 0.15),
    "volatile":    (0.50, 0.25, 0.08, 0.45, 0.15, 0.15),
}

ARC_KEYS    = list(ARCHETYPES.keys())
ARC_WEIGHTS = [ARCHETYPES[k][5] for k in ARC_KEYS]

# ─────────────────────────────────────────────
# CANDLE GENERATION
# ─────────────────────────────────────────────
def gen_single_candle(archetype):
    p_bull, bf_mu, bf_sd, wf_mu, wf_sd, _ = ARCHETYPES[archetype]

    is_bull = random.random() < p_bull

    body_frac = np.clip(np.random.normal(bf_mu, bf_sd), 0.02, 0.9)
    wick_frac = np.clip(np.random.normal(wf_mu, wf_sd), 0.0, 0.8)

    tail_frac = max(0.0, 1.0 - body_frac - wick_frac)

    total = body_frac + wick_frac + tail_frac + EPS
    body_frac /= total
    wick_frac /= total
    tail_frac /= total

    whole = np.random.uniform(0.0005, 0.002)

    upper_wick = wick_frac * whole
    lower_wick = tail_frac * whole
    body       = body_frac * whole

    open_ = BASE_PRICE + np.random.normal(0, 0.001)

    if is_bull:
        close_ = open_ + body
        high_  = close_ + upper_wick
        low_   = open_ - lower_wick
    else:
        close_ = open_ - body
        high_  = open_ + upper_wick
        low_   = close_ - lower_wick

    return {
        "open": open_,
        "high": high_,
        "low": low_,
        "close": close_,
    }

# ─────────────────────────────────────────────
# FEATURE ENGINEERING (ROBUST)
# ─────────────────────────────────────────────
def compute_features(candle):
    o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]

    whole = max(h - l, EPS)

    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    body       = abs(c - o)

    wick_ratio = upper_wick / whole
    tail_ratio = lower_wick / whole
    body_ratio = body / whole

    return {
        "wick_ratio": wick_ratio,
        "body_ratio": body_ratio,
        "tail_ratio": tail_ratio,
        "wick_vs_tail": (upper_wick - lower_wick) / whole,
        "body_vs_wick": body_ratio - (wick_ratio + tail_ratio),
    }

# ─────────────────────────────────────────────
# LABELING (STRUCTURE-AWARE)
# ─────────────────────────────────────────────
def compute_label(feat):
    br  = feat["body_ratio"]
    wvt = feat["wick_vs_tail"]

    # strong directional
    if br >= 0.30 and wvt < -0.05:
        return 2  # bullish

    if br >= 0.30 and wvt > 0.05:
        return 0  # bearish

    # moderate directional
    if br >= 0.20:
        if wvt < 0:
            return 2
        elif wvt > 0:
            return 0

    return 1  # neutral

# ─────────────────────────────────────────────
# SAMPLE GENERATION
# ─────────────────────────────────────────────
def generate_sample():
    arc = random.choices(ARC_KEYS, weights=ARC_WEIGHTS, k=1)[0]
    candle = gen_single_candle(arc)

    feat = compute_features(candle)
    label = compute_label(feat)

    return {
        **feat,
        "label_candle_type": label
    }

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Generating {N_SAMPLES} samples...")

    rows = [generate_sample() for _ in range(N_SAMPLES)]
    df = pd.DataFrame(rows)

    # OPTIONAL: balance dataset
    min_count = df["label_candle_type"].value_counts().min()
    df = df.groupby("label_candle_type").apply(
        lambda x: x.sample(min_count, random_state=SEED)
    ).reset_index(drop=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print("\nSaved:", OUTPUT_PATH)
    print("\nClass distribution:")
    print(df["label_candle_type"].value_counts(normalize=True))
    print("\nFeature stats:")
    print(df.describe())