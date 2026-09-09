# synthetic_momentum.py  ·  v5.1 (latent-state-aware / H1-prediction edition)
# ─────────────────────────────────────────────────────────────────────
# Generates synthetic FX training data for predicting the NEXT H1 candle.
#
# NEW v5 architecture:
#   Market Condition
#   → Latent Market State  (trend_age, volatility_regime, compression_duration,
#                           pullback_depth, liquidity_pressure, breakout_pressure,
#                           reversal_pressure, trend_strength)
#   → State-Biased Transition  (next condition chosen via latent-weighted probs)
#   → State-Modulated H1 Structure  (body_scale / vol_mult / wicks adjusted by state)
#   → M15 Expansion
#   → Momentum Features  (unchanged)
#   → Labels             (unchanged)
#
# SPECIALIST weights: overweights STRONG / ACCELERATING / PULLBACK_ENDING
#
# Run:  python synthetic_momentum.py
# Out:  synthetic_momentum_dataset.csv
# ─────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import random

try:
    from tqdm import trange
except ImportError:
    def trange(n, **kw): return range(n)

N_SAMPLES   = 10000
OUTPUT_PATH = "synthetic_momentum_dataset.csv"
BASE_PRICE  = 1.50000
SEED        = 42

random.seed(SEED)
np.random.seed(SEED)
EPS = 1e-12
EXPORT_METADATA = True  # set False to drop latent columns from CSV

# ─────────────────────────────────────────────────────────────────────
# MARKET CONDITION DEFINITIONS
# Specialist weights: momentum model overweights trending + pullback_ending
# ─────────────────────────────────────────────────────────────────────

CONDITIONS = {
    "STRONG_BULL":             {"weight": 0.08, "h1_seq": ["bull","bull","bull","bull"],               "vol_mult": 1.3,  "body_scale": 0.0010},
    "STRONG_BEAR":             {"weight": 0.08, "h1_seq": ["bear","bear","bear","bear"],               "vol_mult": 1.3,  "body_scale": 0.0010},
    "ACCELERATING_BULL":       {"weight": 0.08, "h1_seq": ["bull","bull_strong","bull_strong","bull_explosive"], "vol_mult": 1.5, "body_scale": 0.0012},
    "ACCELERATING_BEAR":       {"weight": 0.08, "h1_seq": ["bear","bear_strong","bear_strong","bear_explosive"], "vol_mult": 1.5, "body_scale": 0.0012},
    "MATURE_BULL":             {"weight": 0.03, "h1_seq": ["bull","bull","doji","bull_weak"],          "vol_mult": 1.0,  "body_scale": 0.0008},
    "MATURE_BEAR":             {"weight": 0.03, "h1_seq": ["bear","bear","doji","bear_weak"],          "vol_mult": 1.0,  "body_scale": 0.0008},
    "BULL_PULLBACK":           {"weight": 0.05, "h1_seq": ["bull","bull","bear","bear"],               "vol_mult": 0.9,  "body_scale": 0.0007},
    "BEAR_PULLBACK":           {"weight": 0.05, "h1_seq": ["bear","bear","bull","bull"],               "vol_mult": 0.9,  "body_scale": 0.0007},
    "BULL_PULLBACK_ENDING":    {"weight": 0.07, "h1_seq": ["bull","bear","bear_weak","doji"],          "vol_mult": 0.8,  "body_scale": 0.0006},
    "BEAR_PULLBACK_ENDING":    {"weight": 0.07, "h1_seq": ["bear","bull","bull_weak","doji"],          "vol_mult": 0.8,  "body_scale": 0.0006},
    "RANGE":                   {"weight": 0.03, "h1_seq": ["bull","bear","bull","bear"],               "vol_mult": 0.7,  "body_scale": 0.0005},
    "RANGE_HIGH":              {"weight": 0.02, "h1_seq": ["bull","bull_weak","doji","bear_weak"],     "vol_mult": 0.7,  "body_scale": 0.0005},
    "RANGE_LOW":               {"weight": 0.02, "h1_seq": ["bear","bear_weak","doji","bull_weak"],    "vol_mult": 0.7,  "body_scale": 0.0005},
    "RANGE_COMPRESSION":       {"weight": 0.02, "h1_seq": ["doji","doji","doji","doji"],              "vol_mult": 0.5,  "body_scale": 0.0003},
    "BREAKOUT_SETUP_BULL":     {"weight": 0.03, "h1_seq": ["doji","doji","bull_weak","bull"],          "vol_mult": 0.8,  "body_scale": 0.0006},
    "BREAKOUT_SETUP_BEAR":     {"weight": 0.03, "h1_seq": ["doji","doji","bear_weak","bear"],          "vol_mult": 0.8,  "body_scale": 0.0006},
    "BREAKOUT_CONFIRMED_BULL": {"weight": 0.04, "h1_seq": ["doji","bull","bull_strong","bull_explosive"], "vol_mult": 2.0, "body_scale": 0.0014},
    "BREAKOUT_CONFIRMED_BEAR": {"weight": 0.04, "h1_seq": ["doji","bear","bear_strong","bear_explosive"], "vol_mult": 2.0, "body_scale": 0.0014},
    "FALSE_BREAKOUT_BULL":     {"weight": 0.03, "h1_seq": ["bull_strong","spike_up","rejection","bear"], "vol_mult": 1.8, "body_scale": 0.0011},
    "FALSE_BREAKOUT_BEAR":     {"weight": 0.03, "h1_seq": ["bear_strong","spike_dn","rejection","bull"], "vol_mult": 1.8, "body_scale": 0.0011},
    "EARLY_BULL_REVERSAL":     {"weight": 0.04, "h1_seq": ["bear","bear","doji","bull_strong"],        "vol_mult": 1.2,  "body_scale": 0.0009},
    "EARLY_BEAR_REVERSAL":     {"weight": 0.04, "h1_seq": ["bull","bull","doji","bear_strong"],        "vol_mult": 1.2,  "body_scale": 0.0009},
    "LIQUIDITY_SWEEP_HIGH":    {"weight": 0.03, "h1_seq": ["bull","bull","spike_up","bear_strong"],   "vol_mult": 2.0,  "body_scale": 0.0013},
    "LIQUIDITY_SWEEP_LOW":     {"weight": 0.03, "h1_seq": ["bear","bear","spike_dn","bull_strong"],   "vol_mult": 2.0,  "body_scale": 0.0013},
}

TRANSITIONS = {
    "STRONG_BULL":             {"STRONG_BULL":0.30,"ACCELERATING_BULL":0.20,"MATURE_BULL":0.20,"EARLY_BEAR_REVERSAL":0.10,"BULL_PULLBACK":0.20},
    "STRONG_BEAR":             {"STRONG_BEAR":0.30,"ACCELERATING_BEAR":0.20,"MATURE_BEAR":0.20,"EARLY_BULL_REVERSAL":0.10,"BEAR_PULLBACK":0.20},
    "ACCELERATING_BULL":       {"STRONG_BULL":0.20,"ACCELERATING_BULL":0.20,"LIQUIDITY_SWEEP_HIGH":0.20,"EARLY_BEAR_REVERSAL":0.20,"FALSE_BREAKOUT_BULL":0.20},
    "ACCELERATING_BEAR":       {"STRONG_BEAR":0.20,"ACCELERATING_BEAR":0.20,"LIQUIDITY_SWEEP_LOW":0.20,"EARLY_BULL_REVERSAL":0.20,"FALSE_BREAKOUT_BEAR":0.20},
    "MATURE_BULL":             {"BULL_PULLBACK":0.30,"EARLY_BEAR_REVERSAL":0.20,"RANGE":0.20,"STRONG_BULL":0.15,"RANGE_HIGH":0.15},
    "MATURE_BEAR":             {"BEAR_PULLBACK":0.30,"EARLY_BULL_REVERSAL":0.20,"RANGE":0.20,"STRONG_BEAR":0.15,"RANGE_LOW":0.15},
    "BULL_PULLBACK":           {"BULL_PULLBACK_ENDING":0.35,"STRONG_BULL":0.25,"RANGE":0.20,"EARLY_BEAR_REVERSAL":0.20},
    "BEAR_PULLBACK":           {"BEAR_PULLBACK_ENDING":0.35,"STRONG_BEAR":0.25,"RANGE":0.20,"EARLY_BULL_REVERSAL":0.20},
    "BULL_PULLBACK_ENDING":    {"STRONG_BULL":0.35,"ACCELERATING_BULL":0.20,"RANGE":0.25,"BEAR_PULLBACK":0.20},
    "BEAR_PULLBACK_ENDING":    {"STRONG_BEAR":0.35,"ACCELERATING_BEAR":0.20,"RANGE":0.25,"BULL_PULLBACK":0.20},
    "RANGE":                   {"RANGE":0.30,"RANGE_HIGH":0.10,"RANGE_LOW":0.10,"RANGE_COMPRESSION":0.20,"BREAKOUT_SETUP_BULL":0.15,"BREAKOUT_SETUP_BEAR":0.15},
    "RANGE_HIGH":              {"RANGE":0.20,"FALSE_BREAKOUT_BULL":0.20,"BREAKOUT_CONFIRMED_BULL":0.20,"EARLY_BEAR_REVERSAL":0.20,"LIQUIDITY_SWEEP_HIGH":0.20},
    "RANGE_LOW":               {"RANGE":0.20,"FALSE_BREAKOUT_BEAR":0.20,"BREAKOUT_CONFIRMED_BEAR":0.20,"EARLY_BULL_REVERSAL":0.20,"LIQUIDITY_SWEEP_LOW":0.20},
    "RANGE_COMPRESSION":       {"BREAKOUT_SETUP_BULL":0.25,"BREAKOUT_SETUP_BEAR":0.25,"RANGE":0.25,"RANGE_COMPRESSION":0.25},
    "BREAKOUT_SETUP_BULL":     {"BREAKOUT_CONFIRMED_BULL":0.40,"FALSE_BREAKOUT_BULL":0.20,"RANGE":0.20,"STRONG_BULL":0.20},
    "BREAKOUT_SETUP_BEAR":     {"BREAKOUT_CONFIRMED_BEAR":0.40,"FALSE_BREAKOUT_BEAR":0.20,"RANGE":0.20,"STRONG_BEAR":0.20},
    "BREAKOUT_CONFIRMED_BULL": {"STRONG_BULL":0.40,"ACCELERATING_BULL":0.30,"BULL_PULLBACK":0.20,"MATURE_BULL":0.10},
    "BREAKOUT_CONFIRMED_BEAR": {"STRONG_BEAR":0.40,"ACCELERATING_BEAR":0.30,"BEAR_PULLBACK":0.20,"MATURE_BEAR":0.10},
    "FALSE_BREAKOUT_BULL":     {"RANGE":0.30,"STRONG_BEAR":0.30,"BEAR_PULLBACK":0.20,"EARLY_BEAR_REVERSAL":0.20},
    "FALSE_BREAKOUT_BEAR":     {"RANGE":0.30,"STRONG_BULL":0.30,"BULL_PULLBACK":0.20,"EARLY_BULL_REVERSAL":0.20},
    "EARLY_BULL_REVERSAL":     {"STRONG_BULL":0.30,"BREAKOUT_CONFIRMED_BULL":0.20,"RANGE":0.30,"BULL_PULLBACK_ENDING":0.20},
    "EARLY_BEAR_REVERSAL":     {"STRONG_BEAR":0.30,"BREAKOUT_CONFIRMED_BEAR":0.20,"RANGE":0.30,"BEAR_PULLBACK_ENDING":0.20},
    "LIQUIDITY_SWEEP_HIGH":    {"STRONG_BEAR":0.35,"EARLY_BEAR_REVERSAL":0.30,"RANGE":0.20,"FALSE_BREAKOUT_BULL":0.15},
    "LIQUIDITY_SWEEP_LOW":     {"STRONG_BULL":0.35,"EARLY_BULL_REVERSAL":0.30,"RANGE":0.20,"FALSE_BREAKOUT_BEAR":0.15},
}

# ─────────────────────────────────────────────────────────────────────
# LATENT MARKET STATE ENGINE
# Hidden state vector giving the market structure "memory".
# Drives candle generation; NOT exposed in training output.
# ─────────────────────────────────────────────────────────────────────

class _LS:
    """Latent state. All values are floats in [0,1] unless noted."""
    __slots__ = ("trend_strength","trend_age","vol_regime",
                 "compression_score","compression_duration",
                 "pullback_depth","liquidity_pressure",
                 "breakout_pressure","reversal_pressure","_vrn")
    def __init__(self):
        self.trend_strength       = float(np.random.uniform(0.3, 0.7))
        self.trend_age            = int(np.random.randint(1, 6))
        self.vol_regime           = "normal"
        self.compression_score    = float(np.random.uniform(0.0, 0.25))
        self.compression_duration = 0
        self.pullback_depth       = 0.0
        self.liquidity_pressure   = float(np.random.uniform(0.0, 0.25))
        self.breakout_pressure    = 0.0
        self.reversal_pressure    = 0.0
        self._vrn                 = 1  # 0=low 1=normal 2=high

def _init_ls(cond):
    ls = _LS()
    if any(k in cond for k in ("STRONG","ACCELERATING")):
        ls.trend_strength = np.random.uniform(0.70, 1.00)
        ls.trend_age      = np.random.randint(2, 8)
        ls._vrn = 2
    elif "MATURE" in cond:
        ls.trend_strength    = np.random.uniform(0.40, 0.70)
        ls.trend_age         = np.random.randint(10, 22)
        ls.reversal_pressure = np.random.uniform(0.25, 0.55)
        ls._vrn = 1
    elif "PULLBACK_ENDING" in cond:
        ls.trend_strength = np.random.uniform(0.30, 0.60)
        ls.pullback_depth = np.random.uniform(0.40, 0.75)
        ls.trend_age      = np.random.randint(1, 4)
        ls._vrn = 1
    elif "PULLBACK" in cond:
        ls.trend_strength = np.random.uniform(0.20, 0.50)
        ls.pullback_depth = np.random.uniform(0.15, 0.50)
        ls.trend_age      = np.random.randint(1, 5)
        ls._vrn = 1
    elif "COMPRESSION" in cond:
        ls.compression_score    = np.random.uniform(0.55, 0.90)
        ls.compression_duration = np.random.randint(5, 20)
        ls.breakout_pressure    = min(ls.compression_duration * 0.04, 0.80)
        ls.trend_strength       = np.random.uniform(0.00, 0.20)
        ls._vrn = 0
    elif "RANGE" in cond:
        ls.trend_strength    = np.random.uniform(0.00, 0.30)
        ls.compression_score = np.random.uniform(0.15, 0.45)
        ls._vrn = 0
    elif "BREAKOUT_CONFIRMED" in cond:
        ls.breakout_pressure    = np.random.uniform(0.65, 1.00)
        ls.compression_score    = np.random.uniform(0.00, 0.20)
        ls.compression_duration = 0
        ls.trend_strength       = np.random.uniform(0.65, 0.95)
        ls._vrn = 2
    elif "SWEEP" in cond:
        ls.liquidity_pressure = np.random.uniform(0.70, 1.00)
        ls.trend_strength     = np.random.uniform(0.50, 0.80)
        ls._vrn = 2
    elif "REVERSAL" in cond:
        ls.reversal_pressure = np.random.uniform(0.60, 0.90)
        ls.trend_age         = np.random.randint(1, 4)
        ls.trend_strength    = np.random.uniform(0.40, 0.70)
    ls.vol_regime = ["low","normal","high"][ls._vrn]
    return ls

def _step_ls(ls, cond):
    """Evolve latent state one H1 step given the new condition label."""
    # trend_age
    if any(k in cond for k in ("STRONG","ACCELERATING","MATURE")):
        ls.trend_age = min(ls.trend_age + 1, 40)
    elif any(k in cond for k in ("REVERSAL","BREAKOUT_CONFIRMED")):
        ls.trend_age = 1
    elif "PULLBACK_ENDING" in cond:
        ls.trend_age = max(1, ls.trend_age - 2)

    # trend_strength
    if "ACCELERATING" in cond:
        ls.trend_strength = min(1.0, ls.trend_strength + np.random.uniform(0.05, 0.14))
    elif "MATURE" in cond:
        ls.trend_strength = max(0.05, ls.trend_strength - np.random.uniform(0.02, 0.07))
    elif "PULLBACK" in cond and "ENDING" not in cond:
        ls.trend_strength = max(0.05, ls.trend_strength - np.random.uniform(0.05, 0.13))
    elif "PULLBACK_ENDING" in cond:
        ls.trend_strength = min(1.0,  ls.trend_strength + np.random.uniform(0.06, 0.13))
    elif any(k in cond for k in ("RANGE","COMPRESSION")):
        ls.trend_strength = max(0.0,  ls.trend_strength - np.random.uniform(0.04, 0.10))
    elif "BREAKOUT_CONFIRMED" in cond:
        ls.trend_strength = np.random.uniform(0.65, 0.95)
    ls.trend_strength = float(np.clip(ls.trend_strength, 0.0, 1.0))

    # compression
    if "COMPRESSION" in cond or "RANGE" in cond:
        ls.compression_score    = min(1.0, ls.compression_score + np.random.uniform(0.04, 0.09))
        ls.compression_duration += 1
        ls.breakout_pressure    = min(1.0, ls.breakout_pressure + 0.03 * ls.compression_duration)
    elif any(k in cond for k in ("BREAKOUT_CONFIRMED","STRONG","ACCELERATING")):
        ls.compression_score    = max(0.0, ls.compression_score - np.random.uniform(0.20, 0.50))
        ls.compression_duration = 0
        ls.breakout_pressure    = max(0.0, ls.breakout_pressure - np.random.uniform(0.40, 0.80))

    # pullback_depth
    if "PULLBACK" in cond and "ENDING" not in cond:
        ls.pullback_depth = min(1.0, ls.pullback_depth + np.random.uniform(0.08, 0.20))
    elif "PULLBACK_ENDING" in cond:
        ls.pullback_depth = max(0.0, ls.pullback_depth - np.random.uniform(0.10, 0.25))
    elif any(k in cond for k in ("STRONG","ACCELERATING","BREAKOUT_CONFIRMED")):
        ls.pullback_depth = max(0.0, ls.pullback_depth - np.random.uniform(0.15, 0.35))

    # liquidity_pressure
    if any(k in cond for k in ("MATURE","RANGE_HIGH","RANGE_LOW")):
        ls.liquidity_pressure = min(1.0, ls.liquidity_pressure + np.random.uniform(0.06, 0.14))
    elif any(k in cond for k in ("SWEEP","FALSE_BREAKOUT")):
        ls.liquidity_pressure = max(0.0, ls.liquidity_pressure - np.random.uniform(0.30, 0.60))

    # reversal_pressure — rises with trend_age and exhaustion signals
    age_factor = min(ls.trend_age / 20.0, 1.0)
    if any(k in cond for k in ("MATURE","SWEEP","FALSE_BREAKOUT")):
        ls.reversal_pressure = min(1.0, ls.reversal_pressure + np.random.uniform(0.05, 0.12) + age_factor * 0.04)
    elif any(k in cond for k in ("REVERSAL","BREAKOUT_CONFIRMED","PULLBACK_ENDING")):
        ls.reversal_pressure = max(0.0, ls.reversal_pressure - np.random.uniform(0.20, 0.50))
    elif any(k in cond for k in ("STRONG","ACCELERATING")):
        ls.reversal_pressure = max(0.0, ls.reversal_pressure - np.random.uniform(0.04, 0.09))

    # vol_regime
    if any(k in cond for k in ("ACCELERATING","BREAKOUT_CONFIRMED","SWEEP")):
        ls._vrn = min(2, ls._vrn + 1)
    elif "COMPRESSION" in cond:
        ls._vrn = max(0, ls._vrn - 1)
    elif any(k in cond for k in ("MATURE","PULLBACK","RANGE")) and np.random.random() < 0.35:
        ls._vrn = max(0, ls._vrn - 1)
    ls.vol_regime = ["low","normal","high"][ls._vrn]

    # clamp all
    ls.compression_score    = float(np.clip(ls.compression_score,    0.0, 1.0))
    ls.breakout_pressure    = float(np.clip(ls.breakout_pressure,     0.0, 1.0))
    ls.pullback_depth       = float(np.clip(ls.pullback_depth,        0.0, 1.0))
    ls.liquidity_pressure   = float(np.clip(ls.liquidity_pressure,    0.0, 1.0))
    ls.reversal_pressure    = float(np.clip(ls.reversal_pressure,     0.0, 1.0))
    return ls

def _next_cond(cond, ls):
    """Pick next condition using base probabilities biased by latent state."""
    base = TRANSITIONS.get(cond, {})
    if not base:
        keys = list(CONDITIONS.keys())
        return random.choices(keys, weights=[CONDITIONS[k]["weight"] for k in keys], k=1)[0]

    w = {}
    for c, p in base.items():
        b = 1.0
        # old trend → push toward maturity / reversal
        if ls.trend_age > 14:
            if any(k in c for k in ("MATURE","REVERSAL","PULLBACK")):
                b *= 1.0 + ls.trend_age * 0.025
            if any(k in c for k in ("STRONG","ACCELERATING")):
                b *= max(0.10, 1.0 - ls.trend_age * 0.035)
        # high reversal_pressure
        if ls.reversal_pressure > 0.55:
            if any(k in c for k in ("REVERSAL","SWEEP","FALSE_BREAKOUT")):
                b *= 1.0 + ls.reversal_pressure * 2.0
            if any(k in c for k in ("STRONG","ACCELERATING")):
                b *= max(0.10, 1.0 - ls.reversal_pressure * 0.90)
        # high breakout_pressure
        if ls.breakout_pressure > 0.55:
            if any(k in c for k in ("BREAKOUT_CONFIRMED","BREAKOUT_SETUP","FALSE_BREAKOUT")):
                b *= 1.0 + ls.breakout_pressure * 2.5
            if "COMPRESSION" in c:
                b *= max(0.05, 1.0 - ls.breakout_pressure * 0.80)
        # deep pullback → continuation more likely
        if ls.pullback_depth > 0.55:
            if "PULLBACK_ENDING" in c:
                b *= 1.0 + ls.pullback_depth * 2.0
            if "PULLBACK" in c and "ENDING" not in c:
                b *= 0.35
        # liquidity build-up → sweep / false breakout
        if ls.liquidity_pressure > 0.65:
            if any(k in c for k in ("SWEEP","FALSE_BREAKOUT")):
                b *= 1.0 + ls.liquidity_pressure * 2.0
        w[c] = max(1e-6, p * b)

    total = sum(w.values())
    keys  = list(w.keys())
    probs = [w[k] / total for k in keys]
    return random.choices(keys, weights=probs, k=1)[0]

def _modulate(cond, ls, base_bs, base_vm):
    """
    Apply latent state to produce (body_scale, vol_mult, wick_mult, body_taper).
    wick_mult  : multiplier on all wick std-devs   (>1 = longer wicks)
    body_taper : multiplier on all body deltas      (<1 = smaller bodies)
    """
    vol_scale = {"low": 0.60, "normal": 1.00, "high": 1.40}[ls.vol_regime]
    bs = base_bs * vol_scale
    vm = base_vm * vol_scale

    body_taper  = float(np.clip(0.45 + 0.55 * ls.trend_strength, 0.25, 1.50))
    body_taper *= float(np.clip(1.0 - ls.trend_age / 40.0,        0.40, 1.20))  # age fatigue

    wick_mult   = float(np.clip(1.0 + ls.trend_age / 30.0,        1.00, 1.60))  # age → longer wicks
    wick_mult  *= float(np.clip(1.0 + ls.liquidity_pressure * 0.80, 1.00, 2.20))
    wick_mult  *= float(np.clip(1.0 + ls.reversal_pressure  * 0.60, 1.00, 1.80))

    if "COMPRESSION" in cond or "RANGE" in cond:
        squeeze = float(np.clip(1.0 - ls.compression_score * 0.65, 0.25, 1.00))
        bs     *= squeeze
        vm     *= max(0.30, 1.0 - ls.compression_score * 0.50)

    if "BREAKOUT_CONFIRMED" in cond:
        bs *= 1.0 + ls.breakout_pressure * 0.80
        vm *= 1.0 + ls.breakout_pressure * 0.60

    if "PULLBACK" in cond and "ENDING" not in cond:
        body_taper *= max(0.35, 1.0 - ls.pullback_depth * 0.50)
        wick_mult  *= 1.0 + ls.pullback_depth * 0.45

    bs = float(np.clip(bs, 0.0001, 0.0035))
    vm = float(np.clip(vm, 0.10,   6.00))
    return bs, vm, float(np.clip(wick_mult, 0.50, 3.50)), float(np.clip(body_taper, 0.20, 1.60))


# ─────────────────────────────────────────────────────────────────────
# H1 ARCHETYPE → M15 EXPANSION  (wick_mult applied to all wick params)
# ─────────────────────────────────────────────────────────────────────

def _m15_from_h1_archetype(archetype, base_price, body_scale, vol_mult, wick_mult=1.0):
    bs = body_scale
    vm = vol_mult
    wx = wick_mult
    p  = base_price

    def candle(open_, close_, wick_up, wick_dn, vol_lo, vol_hi):
        h = max(open_, close_) + abs(np.random.normal(0, wick_up * wx))
        l = min(open_, close_) - abs(np.random.normal(0, wick_dn * wx))
        h = max(h, open_, close_)
        l = min(l, open_, close_)
        v = int(np.random.uniform(vol_lo, vol_hi))
        return {"open": float(open_), "high": float(h),
                "low": float(l), "close": float(close_), "volume": v}

    def rnd(lo, hi): return float(np.random.uniform(lo, hi))

    if archetype == "bull":
        c1o = p;     c1c = p    + rnd(0.10, 0.40) * bs
        c2o = c1c;   c2c = c2o  + rnd(0.20, 0.50) * bs
        c3o = c2c;   c3c = c3o  + rnd(0.20, 0.50) * bs
        c4o = c3c;   c4c = c4o  + rnd(0.30, 0.70) * bs
        candles = [candle(c1o,c1c,bs*0.10,bs*0.20,200*vm,600*vm),
                   candle(c2o,c2c,bs*0.10,bs*0.15,300*vm,800*vm),
                   candle(c3o,c3c,bs*0.10,bs*0.15,400*vm,1000*vm),
                   candle(c4o,c4c,bs*0.05,bs*0.10,500*vm,1200*vm)]

    elif archetype == "bull_strong":
        c1o = p;     c1c = p    + rnd(0.30, 0.60) * bs
        c2o = c1c;   c2c = c2o  + rnd(0.40, 0.70) * bs
        c3o = c2c;   c3c = c3o  + rnd(0.50, 0.90) * bs
        c4o = c3c;   c4c = c4o  + rnd(0.60, 1.00) * bs
        candles = [candle(c1o,c1c,bs*0.05,bs*0.15,500*vm,1200*vm),
                   candle(c2o,c2c,bs*0.05,bs*0.10,600*vm,1500*vm),
                   candle(c3o,c3c,bs*0.05,bs*0.10,700*vm,1800*vm),
                   candle(c4o,c4c,bs*0.03,bs*0.08,800*vm,2000*vm)]

    elif archetype == "bull_explosive":
        c1o = p;     c1c = p    + rnd(0.50, 0.90) * bs
        c2o = c1c;   c2c = c2o  + rnd(0.70, 1.20) * bs
        c3o = c2c;   c3c = c3o  + rnd(0.80, 1.40) * bs
        c4o = c3c;   c4c = c4o  + rnd(1.00, 1.80) * bs
        candles = [candle(c1o,c1c,bs*0.03,bs*0.10,800*vm,2000*vm),
                   candle(c2o,c2c,bs*0.03,bs*0.08,1000*vm,2500*vm),
                   candle(c3o,c3c,bs*0.03,bs*0.07,1200*vm,3000*vm),
                   candle(c4o,c4c,bs*0.02,bs*0.06,1500*vm,4000*vm)]

    elif archetype == "bull_weak":
        c1o = p;     c1c = p    + rnd(0.05, 0.20) * bs
        c2o = c1c;   c2c = c2o  + rnd(-0.10,0.20) * bs
        c3o = c2c;   c3c = c3o  + rnd(0.05, 0.20) * bs
        c4o = c3c;   c4c = c4o  + rnd(0.00, 0.15) * bs
        candles = [candle(c1o,c1c,bs*0.20,bs*0.30,150*vm,400*vm),
                   candle(c2o,c2c,bs*0.25,bs*0.35,120*vm,350*vm),
                   candle(c3o,c3c,bs*0.20,bs*0.30,130*vm,380*vm),
                   candle(c4o,c4c,bs*0.25,bs*0.35,110*vm,320*vm)]

    elif archetype == "bear":
        c1o = p;     c1c = p    - rnd(0.10, 0.40) * bs
        c2o = c1c;   c2c = c2o  - rnd(0.20, 0.50) * bs
        c3o = c2c;   c3c = c3o  - rnd(0.20, 0.50) * bs
        c4o = c3c;   c4c = c4o  - rnd(0.30, 0.70) * bs
        candles = [candle(c1o,c1c,bs*0.20,bs*0.10,200*vm,600*vm),
                   candle(c2o,c2c,bs*0.15,bs*0.10,300*vm,800*vm),
                   candle(c3o,c3c,bs*0.15,bs*0.10,400*vm,1000*vm),
                   candle(c4o,c4c,bs*0.10,bs*0.05,500*vm,1200*vm)]

    elif archetype == "bear_strong":
        c1o = p;     c1c = p    - rnd(0.30, 0.60) * bs
        c2o = c1c;   c2c = c2o  - rnd(0.40, 0.70) * bs
        c3o = c2c;   c3c = c3o  - rnd(0.50, 0.90) * bs
        c4o = c3c;   c4c = c4o  - rnd(0.60, 1.00) * bs
        candles = [candle(c1o,c1c,bs*0.15,bs*0.05,500*vm,1200*vm),
                   candle(c2o,c2c,bs*0.10,bs*0.05,600*vm,1500*vm),
                   candle(c3o,c3c,bs*0.10,bs*0.05,700*vm,1800*vm),
                   candle(c4o,c4c,bs*0.08,bs*0.03,800*vm,2000*vm)]

    elif archetype == "bear_explosive":
        c1o = p;     c1c = p    - rnd(0.50, 0.90) * bs
        c2o = c1c;   c2c = c2o  - rnd(0.70, 1.20) * bs
        c3o = c2c;   c3c = c3o  - rnd(0.80, 1.40) * bs
        c4o = c3c;   c4c = c4o  - rnd(1.00, 1.80) * bs
        candles = [candle(c1o,c1c,bs*0.10,bs*0.03,800*vm,2000*vm),
                   candle(c2o,c2c,bs*0.08,bs*0.03,1000*vm,2500*vm),
                   candle(c3o,c3c,bs*0.07,bs*0.03,1200*vm,3000*vm),
                   candle(c4o,c4c,bs*0.06,bs*0.02,1500*vm,4000*vm)]

    elif archetype == "bear_weak":
        c1o = p;     c1c = p    - rnd(0.05, 0.20) * bs
        c2o = c1c;   c2c = c2o  - rnd(-0.10,0.20) * bs
        c3o = c2c;   c3c = c3o  - rnd(0.05, 0.20) * bs
        c4o = c3c;   c4c = c4o  - rnd(0.00, 0.15) * bs
        candles = [candle(c1o,c1c,bs*0.30,bs*0.20,150*vm,400*vm),
                   candle(c2o,c2c,bs*0.35,bs*0.25,120*vm,350*vm),
                   candle(c3o,c3c,bs*0.30,bs*0.20,130*vm,380*vm),
                   candle(c4o,c4c,bs*0.35,bs*0.25,110*vm,320*vm)]

    elif archetype == "doji":
        drift = rnd(-0.05, 0.05) * bs
        c1o = p;     c1c = p    + rnd(-0.15, 0.15) * bs
        c2o = c1c;   c2c = c2o  + rnd(-0.15, 0.15) * bs
        c3o = c2c;   c3c = c3o  + rnd(-0.10, 0.10) * bs
        c4o = c3c;   c4c = c4o  + drift
        candles = [candle(c1o,c1c,bs*0.40,bs*0.40,150*vm,500*vm),
                   candle(c2o,c2c,bs*0.40,bs*0.40,120*vm,450*vm),
                   candle(c3o,c3c,bs*0.35,bs*0.35,110*vm,400*vm),
                   candle(c4o,c4c,bs*0.30,bs*0.30,100*vm,380*vm)]

    elif archetype == "spike_up":
        c1o = p;     c1c = p    + rnd(0.40, 0.80) * bs
        c2o = c1c;   c2c = c2o  + rnd(0.60, 1.20) * bs
        c3o = c2c;   c3c = c3o  + rnd(0.30, 0.70) * bs
        c4o = c3c;   c4c = c4o  + rnd(-0.20, 0.20) * bs
        candles = [candle(c1o,c1c,bs*0.10,bs*0.05,600*vm,1500*vm),
                   candle(c2o,c2c,bs*0.15,bs*0.05,1000*vm,3000*vm),
                   candle(c3o,c3c,bs*0.30,bs*0.10,800*vm,2000*vm),
                   candle(c4o,c4c,bs*0.50,bs*0.50,500*vm,1200*vm)]

    elif archetype == "spike_dn":
        c1o = p;     c1c = p    - rnd(0.40, 0.80) * bs
        c2o = c1c;   c2c = c2o  - rnd(0.60, 1.20) * bs
        c3o = c2c;   c3c = c3o  - rnd(0.30, 0.70) * bs
        c4o = c3c;   c4c = c4o  - rnd(-0.20, 0.20) * bs
        candles = [candle(c1o,c1c,bs*0.05,bs*0.10,600*vm,1500*vm),
                   candle(c2o,c2c,bs*0.05,bs*0.15,1000*vm,3000*vm),
                   candle(c3o,c3c,bs*0.10,bs*0.30,800*vm,2000*vm),
                   candle(c4o,c4c,bs*0.50,bs*0.50,500*vm,1200*vm)]

    elif archetype == "rejection":
        c1o = p;     c1c = p    + rnd(0.30, 0.70) * bs
        c2o = c1c;   c2c = c2o  + rnd(0.50, 1.00) * bs
        c3o = c2c;   c3c = c3o  - rnd(0.50, 1.20) * bs
        c4o = c3c;   c4c = c4o  - rnd(0.30, 0.70) * bs
        candles = [candle(c1o,c1c,bs*0.10,bs*0.10,400*vm,1000*vm),
                   candle(c2o,c2c,bs*0.50,bs*0.05,800*vm,2500*vm),
                   candle(c3o,c3c,bs*0.10,bs*0.20,900*vm,2800*vm),
                   candle(c4o,c4c,bs*0.10,bs*0.15,600*vm,1500*vm)]
    else:
        candles = []
        cp = p
        for _ in range(4):
            cc = cp + rnd(-0.10, 0.10) * bs
            candles.append(candle(cp, cc, bs*0.30, bs*0.30, 200*vm, 600*vm))
            cp = cc

    return candles, candles[-1]["close"]


# ─────────────────────────────────────────────────────────────────────
# MARKET HISTORY GENERATOR  (latent-state-aware)
# ─────────────────────────────────────────────────────────────────────

def _pick_condition():
    keys = list(CONDITIONS.keys())
    return random.choices(keys, weights=[CONDITIONS[k]["weight"] for k in keys], k=1)[0]

def generate_market_history(base_price=BASE_PRICE):
    price = base_price + np.random.normal(0, 0.0020)
    cond  = _pick_condition()
    ls    = _init_ls(cond)
    all_m15 = []

    for _ in range(random.randint(22, 34)):
        cfg              = CONDITIONS[cond]
        bs, vm, wx, bt   = _modulate(cond, ls, cfg["body_scale"], cfg["vol_mult"])
        for arch in cfg["h1_seq"]:
            m15s, price = _m15_from_h1_archetype(arch, price, bs * bt, vm, wx)
            all_m15.extend(m15s)
        ls   = _step_ls(ls, cond)
        cond = _next_cond(cond, ls)

    active_condition = cond
    cfg              = CONDITIONS[active_condition]
    bs, vm, wx, bt   = _modulate(active_condition, ls, cfg["body_scale"], cfg["vol_mult"])
    for arch in cfg["h1_seq"]:
        m15s, price = _m15_from_h1_archetype(arch, price, bs * bt, vm, wx)
        all_m15.extend(m15s)

    ls               = _step_ls(ls, active_condition)
    future_condition = _next_cond(active_condition, ls)
    fut_cfg          = CONDITIONS[future_condition]
    bs, vm, wx, bt   = _modulate(future_condition, ls, fut_cfg["body_scale"], fut_cfg["vol_mult"])

    future_m15 = []
    for arch in fut_cfg["h1_seq"][:1]:
        m15s, price = _m15_from_h1_archetype(arch, price, bs * bt, vm, wx)
        future_m15.extend(m15s)
        if len(future_m15) >= 4: break
    while len(future_m15) < 4:
        future_m15.append(future_m15[-1])
    future_m15 = future_m15[:4]

    return all_m15, active_condition, future_condition, future_m15, ls


# ─────────────────────────────────────────────────────────────────────
# FEATURE COMPUTATION — 30 MOMENTUM FEATURES
# Computed on full history; last-bar values extracted
# ─────────────────────────────────────────────────────────────────────

def rolling_z(series, window):
    return (series - series.rolling(window).mean()) / (series.rolling(window).std() + EPS)

def rolling_autocorr(series, window, lag=1):
    def _ac(arr):
        if len(arr) < lag + 2: return 0.0
        s = pd.Series(arr)
        r = s.autocorr(lag=lag)
        return r if np.isfinite(r) else 0.0
    return series.rolling(window).apply(_ac, raw=False)

def compute_momentum_features(df: pd.DataFrame) -> dict:
    df = df.copy().reset_index(drop=True)

    close  = df["close"].astype(float)
    high   = df["high"].astype(float)
    low    = df["low"].astype(float)
    volume = df["volume"].astype(float)

    tr  = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    atr_safe = atr + EPS

    ret_3          = close.pct_change(3)
    ret_3_atr_norm = (ret_3 / (atr_safe / (close + EPS))).fillna(0.0)
    ret_3_z        = rolling_z(ret_3, 50).fillna(0.0)

    direction        = (close - close.shift(10)).abs()
    noise            = close.diff().abs().rolling(10).sum()
    efficiency_ratio = (direction / (noise + EPS)).fillna(0.0)
    er_slope         = efficiency_ratio.diff(5).fillna(0.0)

    ret_1                = close.pct_change(1).fillna(0.0)
    momentum_persistence = rolling_autocorr(ret_1, 30).fillna(0.0)

    ema_20         = close.ewm(span=20, adjust=False).mean()
    ema_slope      = ema_20.diff(5)
    ema_slope_norm = (ema_slope / atr_safe).fillna(0.0)
    ema_acc_norm   = (ema_slope.diff(3) / atr_safe).fillna(0.0)

    atr_z           = rolling_z(atr, 50).fillna(0.0)
    rng             = high - low
    range_expansion = (rng.rolling(3).mean() / (rng.rolling(20).mean() + EPS)).fillna(1.0)
    vol_regime_confirm = (atr_z * (range_expansion - 1.0)).clip(-4, 4).fillna(0.0)

    roll_min_20    = close.rolling(20).min()
    roll_max_20    = close.rolling(20).max()
    trend_position = ((close - roll_min_20) / (roll_max_20 - roll_min_20 + EPS)).fillna(0.5)
    rolling_high   = high.rolling(20).max().shift(1)
    break_strength = ((close - rolling_high) / atr_safe).clip(-3, 3).fillna(0.0)
    mean_50                = close.rolling(50).mean()
    mean_reversion_stretch = ((close - mean_50) / atr_safe).clip(-4, 4).fillna(0.0)

    up_move    = high.diff()
    down_move  = (-low.diff())
    plus_dm    = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm   = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    sm_plus    = plus_dm.ewm(span=14, adjust=False).mean()
    sm_minus   = minus_dm.ewm(span=14, adjust=False).mean()
    sm_atr     = atr.ewm(span=14, adjust=False).mean() + EPS
    plus_di    = 100 * (sm_plus  / sm_atr)
    minus_di   = 100 * (sm_minus / sm_atr)
    dx         = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + EPS)
    adx        = dx.ewm(span=14, adjust=False).mean()
    adx_z      = rolling_z(adx, 50).fillna(0.0)
    adx_slope  = rolling_z(adx, 30).diff(3).fillna(0.0)
    di_balance = ((plus_di - minus_di) / (plus_di + minus_di + EPS)).fillna(0.0)

    tenkan   = (high.rolling(9).max()  + low.rolling(9).min())  / 2
    kijun    = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    tk_spread       = ((tenkan - kijun) / atr_safe).clip(-3, 3).fillna(0.0)
    cloud_thickness = ((senkou_a - senkou_b).abs() / atr_safe).clip(0, 5).fillna(0.0)
    price_vs_cloud  = ((close - senkou_a) / atr_safe).clip(-4, 4).fillna(0.0)
    cloud_twist     = np.sign(senkou_a - senkou_b).fillna(0.0)

    don_high  = high.rolling(20).max()
    don_low   = low.rolling(20).min()
    don_width = don_high - don_low
    don_position = ((close - don_low) / (don_width + EPS)).fillna(0.5)
    don_width_z  = rolling_z(don_width, 50).fillna(0.0)
    don_breakout = ((close - don_high.shift(1)) / atr_safe).clip(-3, 3).fillna(0.0)

    ret_5         = close.pct_change(5)
    ret_1_norm    = (ret_1 / (atr_safe / (close + EPS)))
    ret_5_norm    = (ret_5 / (atr_safe / (close + EPS)))
    momentum_decay = (ret_1_norm / ((ret_5_norm / 5) + EPS)).clip(-5, 5).fillna(0.0)
    tension = (ret_3_atr_norm * (0.5 - trend_position)).clip(-4, 4).fillna(0.0)
    ret_10          = close.pct_change(10)
    ret_10_atr_norm = (ret_10 / (atr_safe / (close + EPS))).fillna(0.0)
    ret_10_z        = rolling_z(ret_10, 50).fillna(0.0)

    ema_50       = close.ewm(span=50, adjust=False).mean()
    ema_50_slope = ema_50.diff(10)
    slope_alignment = (np.sign(ema_slope) * np.sign(ema_50_slope)).fillna(0.0)

    volume_z         = rolling_z(volume, 30).fillna(0.0)
    vol_displacement = (volume_z * ret_3_z).clip(-5, 5).fillna(0.0)

    i = -1
    features = {
        "ret_3_atr_norm":         float(ret_3_atr_norm.iloc[i]),
        "ret_3_z":                float(ret_3_z.iloc[i]),
        "efficiency_ratio":       float(efficiency_ratio.iloc[i]),
        "er_slope":               float(er_slope.iloc[i]),
        "momentum_persistence":   float(momentum_persistence.iloc[i]),
        "ema_slope_norm":         float(ema_slope_norm.iloc[i]),
        "ema_acc_norm":           float(ema_acc_norm.iloc[i]),
        "atr_z":                  float(atr_z.iloc[i]),
        "range_expansion":        float(range_expansion.iloc[i]),
        "vol_regime_confirm":     float(vol_regime_confirm.iloc[i]),
        "trend_position":         float(trend_position.iloc[i]),
        "break_strength":         float(break_strength.iloc[i]),
        "mean_reversion_stretch": float(mean_reversion_stretch.iloc[i]),
        "adx_z":                  float(adx_z.iloc[i]),
        "adx_slope":              float(adx_slope.iloc[i]),
        "di_balance":             float(di_balance.iloc[i]),
        "tk_spread":              float(tk_spread.iloc[i]),
        "cloud_thickness":        float(cloud_thickness.iloc[i]),
        "price_vs_cloud":         float(price_vs_cloud.iloc[i]),
        "cloud_twist":            float(cloud_twist.iloc[i]),
        "don_position":           float(don_position.iloc[i]),
        "don_width_z":            float(don_width_z.iloc[i]),
        "don_breakout":           float(don_breakout.iloc[i]),
        "momentum_decay":         float(momentum_decay.iloc[i]),
        "tension":                float(tension.iloc[i]),
        "ret_10_atr_norm":        float(ret_10_atr_norm.iloc[i]),
        "ret_10_z":               float(ret_10_z.iloc[i]),
        "slope_alignment":        float(slope_alignment.iloc[i]),
        "vol_displacement":       float(vol_displacement.iloc[i]),
        "_atr":                   float(atr.iloc[i]) if not np.isnan(atr.iloc[i]) else float(atr.dropna().iloc[-1]),
        "_context_close":         float(close.iloc[i]),
        "_di_balance":            float(di_balance.iloc[i]),
        "_trend_position":        float(trend_position.iloc[i]),
        "_ema_slope_norm":        float(ema_slope_norm.iloc[i]),
        "_mean_rev_stretch":      float(mean_reversion_stretch.iloc[i]),
        "_slope_alignment":       float(slope_alignment.iloc[i]),
        "_momentum_persistence":  float(momentum_persistence.iloc[i]),
    }
    return features


# ─────────────────────────────────────────────────────────────────────
# LABEL COMPUTATION — 6 heads derived from future 4 M15 candles
# ─────────────────────────────────────────────────────────────────────

def label_momentum_sample(features: dict, future_candles: list) -> tuple:
    atr = max(features["_atr"], EPS)
    ref = features["_context_close"]

    c1, c2 = future_candles[0], future_candles[1]
    c3 = future_candles[2] if len(future_candles) >= 3 else c2
    c4 = future_candles[3] if len(future_candles) >= 4 else c3

    h1_open  = c1["open"]
    h1_close = c4["close"]
    h1_high  = max(c["high"] for c in future_candles)
    h1_low   = min(c["low"]  for c in future_candles)
    h1_range = (h1_high - h1_low) / atr

    net_atr  = (h1_close - ref) / atr

    bodies = [abs(c["close"] - c["open"]) / atr for c in future_candles]
    avg_body = float(np.mean(bodies))

    bull_count = sum(1 for c in future_candles if c["close"] > c["open"])

    close_grav = (h1_close - h1_low) / (h1_range * atr + EPS)

    early_avg = float(np.mean(bodies[:2]))
    late_avg  = float(np.mean(bodies[2:]))
    accel_ratio = late_avg / (early_avg + EPS)

    if   net_atr > 0.50:  next_dir = 2
    elif net_atr < -0.50: next_dir = 0
    else:                  next_dir = 1

    if   avg_body > 1.5:  next_mag = 2
    elif avg_body > 0.5:  next_mag = 1
    else:                  next_mag = 0

    if avg_body < 0.25 and h1_range < 1.2:
        nxt_reg = 0
    elif bull_count == 2 and net_atr > 0.1:
        nxt_reg = 1
    elif bull_count <= 1 and net_atr < -0.1:
        nxt_reg = 1
    elif bull_count >= 3 and net_atr > 0.3:
        nxt_reg = 2 if h1_range < 3.0 else 3
    elif bull_count <= 1 and net_atr < -0.3:
        nxt_reg = 2 if h1_range < 3.0 else 3
    else:
        nxt_reg = 0

    if   h1_range > 3.0:   next_vol = 2
    elif h1_range < 1.2:   next_vol = 0
    else:                   next_vol = 1

    if   close_grav > 0.65 and net_atr > 0:   nxt_struct = 2
    elif close_grav < 0.35 and net_atr < 0:   nxt_struct = 0
    else:                                       nxt_struct = 1

    if   accel_ratio > 1.25:  nxt_mom = 2
    elif accel_ratio < 0.75:  nxt_mom = 0
    else:                      nxt_mom = 1

    return (int(next_dir), int(next_mag), int(nxt_reg),
            int(next_vol), int(nxt_struct), int(nxt_mom))



def context_features_momentum(df: pd.DataFrame, atr: float) -> dict:
    """4 features derived strictly from the last 16 M15 (effective context)."""
    w = df.iloc[-16:].reset_index(drop=True)
    close = w["close"].astype(float); high = w["high"].astype(float); low = w["low"].astype(float)
    dirs = (close.diff().fillna(0) > 0).astype(int)
    bull_ratio_16       = float(dirs.sum() / (len(dirs) + EPS))
    ret_series          = close.pct_change().fillna(0)
    trend_persistence_16= float(ret_series.autocorr(lag=1)) if len(ret_series) > 3 else 0.0
    if not np.isfinite(trend_persistence_16): trend_persistence_16 = 0.0
    # momentum acceleration: slope of 4-bar return over the window
    roll4 = close.pct_change(4).fillna(0)
    early4 = float(roll4.iloc[:8].mean()); late4 = float(roll4.iloc[8:].mean())
    momentum_acceleration_16 = float(np.clip((late4 - early4) / (atr + EPS), -3, 3))
    # net displacement vs range
    net = float(close.iloc[-1] - close.iloc[0])
    rng = float(high.max() - low.min())
    trend_displacement_16 = float(np.clip(net / (rng + EPS), -1, 1))
    return {"bull_ratio_16": bull_ratio_16, "trend_persistence_16": trend_persistence_16,
            "momentum_acceleration_16": momentum_acceleration_16, "trend_displacement_16": trend_displacement_16}

# ─────────────────────────────────────────────────────────────────────
# SEQUENCE GENERATOR
# ─────────────────────────────────────────────────────────────────────

def generate_sequence(base_price: float = BASE_PRICE) -> dict:
    all_m15, active_cond, future_cond, future_m15, ls = generate_market_history(base_price)
    df_full  = pd.DataFrame(all_m15)
    features = compute_momentum_features(df_full)
    atr      = max(features["_atr"], EPS)
    ctx      = context_features_momentum(df_full, atr)
    labels   = label_momentum_sample(features, future_m15)
    row = {k: v for k, v in features.items() if not k.startswith("_")}
    row.update(ctx)
    row["condition"]             = active_cond
    row["future_condition"]      = future_cond
    if EXPORT_METADATA:
        row["meta_trend_strength"]       = round(ls.trend_strength, 3)
        row["meta_trend_age"]            = ls.trend_age
        row["meta_vol_regime"]           = ls.vol_regime
        row["meta_compression_score"]    = round(ls.compression_score, 3)
        row["meta_compression_duration"] = ls.compression_duration
        row["meta_pullback_depth"]       = round(ls.pullback_depth, 3)
        row["meta_liquidity_pressure"]   = round(ls.liquidity_pressure, 3)
        row["meta_breakout_pressure"]    = round(ls.breakout_pressure, 3)
        row["meta_reversal_pressure"]    = round(ls.reversal_pressure, 3)
    row["next_direction"]        = labels[0]
    row["next_magnitude"]        = labels[1]
    row["next_regime_behaviour"] = labels[2]
    row["next_volatility"]       = labels[3]
    row["next_structural_bias"]  = labels[4]
    row["next_momentum_quality"] = labels[5]
    return row


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Generating {N_SAMPLES} sequences | v5 latent-state engine | momentum specialist")

    rows = []
    for _ in trange(N_SAMPLES, desc="momentum"):
        rows.append(generate_sequence())

    df_out = pd.DataFrame(rows)

    label_cols   = ["next_direction", "next_magnitude", "next_regime_behaviour",
                    "next_volatility", "next_structural_bias", "next_momentum_quality"]
    meta_cols    = ["condition", "future_condition"]
    feature_cols = [c for c in df_out.columns if c not in label_cols + meta_cols]

    df_out = df_out[meta_cols + feature_cols + label_cols]
    df_out.to_csv(OUTPUT_PATH, index=False)

    label_maps = {
        "next_direction":        {0:"bearish", 1:"flat", 2:"bullish"},
        "next_magnitude":        {0:"small", 1:"medium", 2:"large"},
        "next_regime_behaviour": {0:"chop", 1:"pullback", 2:"continuation", 3:"breakout"},
        "next_volatility":       {0:"contracting", 1:"normal", 2:"expanding"},
        "next_structural_bias":  {0:"bearish", 1:"neutral", 2:"bullish"},
        "next_momentum_quality": {0:"fading", 1:"stable", 2:"accelerating"},
    }
    print(f"\n✅ Saved → {OUTPUT_PATH}   shape={df_out.shape}")
    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")
    print("\nLabel distributions:")
    for lc in label_cols:
        counts = df_out[lc].value_counts().sort_index()
        named  = {label_maps[lc].get(k,k): f"{100*v/len(df_out):.1f}%" for k, v in counts.items()}
        print(f"  {lc:<25}: {named}")
