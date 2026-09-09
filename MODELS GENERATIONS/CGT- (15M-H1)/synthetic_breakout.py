# synthetic_breakout.py  ·  v5.1 (latent-state-aware / H1-prediction edition)
# ─────────────────────────────────────────────────────────────────────
# Generates training data for the Breakout / Squeeze model.
#
# NEW v5 architecture:
#   Market Condition
#   → Latent Market State  (compression_duration, breakout_pressure,
#                           compression_score, range_width, trend_age,
#                           volatility_regime, pullback_depth, etc.)
#   → State-Biased Transition  (breakout_pressure and compression_duration
#                               directly gate next-condition probabilities)
#   → State-Modulated H1 Structure
#   → M15 Expansion
#   → Breakout/Squeeze Features  (unchanged)
#   → Labels                     (unchanged)
#
# SPECIALIST weights: overweights RANGE_COMPRESSION / BREAKOUT_SETUP /
#                     BREAKOUT_CONFIRMED / FALSE_BREAKOUT
#
# Run:  python synthetic_breakout.py
# Out:  synthetic_breakout_dataset.csv
# ─────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import random

try:
    from tqdm import trange
except ImportError:
    def trange(n, **kw): return range(n)

N_SAMPLES   = 100000
OUTPUT_PATH = "synthetic_breakout_dataset.csv"
BASE_PRICE  = 1.5000
SEED        = 42
random.seed(SEED)
np.random.seed(SEED)
EPS = 1e-12
EXPORT_METADATA = True

# ─────────────────────────────────────────────────────────────────────
# MARKET CONDITION DEFINITIONS
# Specialist weights: breakout model overweights compression + breakout cycle
# ─────────────────────────────────────────────────────────────────────

CONDITIONS = {
    "STRONG_BULL":             {"weight": 0.02, "h1_seq": ["bull","bull","bull","bull"],                  "vol_mult": 1.3,  "body_scale": 0.0010},
    "STRONG_BEAR":             {"weight": 0.02, "h1_seq": ["bear","bear","bear","bear"],                  "vol_mult": 1.3,  "body_scale": 0.0010},
    "ACCELERATING_BULL":       {"weight": 0.02, "h1_seq": ["bull","bull_strong","bull_strong","bull_exp"],"vol_mult": 1.5,  "body_scale": 0.0012},
    "ACCELERATING_BEAR":       {"weight": 0.02, "h1_seq": ["bear","bear_strong","bear_strong","bear_exp"],"vol_mult": 1.5,  "body_scale": 0.0012},
    "MATURE_BULL":             {"weight": 0.02, "h1_seq": ["bull","bull","doji","bull_weak"],             "vol_mult": 1.0,  "body_scale": 0.0008},
    "MATURE_BEAR":             {"weight": 0.02, "h1_seq": ["bear","bear","doji","bear_weak"],             "vol_mult": 1.0,  "body_scale": 0.0008},
    "BULL_PULLBACK":           {"weight": 0.03, "h1_seq": ["bull","bull","bear","bear"],                  "vol_mult": 0.9,  "body_scale": 0.0007},
    "BEAR_PULLBACK":           {"weight": 0.03, "h1_seq": ["bear","bear","bull","bull"],                  "vol_mult": 0.9,  "body_scale": 0.0007},
    "BULL_PULLBACK_ENDING":    {"weight": 0.03, "h1_seq": ["bull","bear","bear_weak","doji"],             "vol_mult": 0.8,  "body_scale": 0.0006},
    "BEAR_PULLBACK_ENDING":    {"weight": 0.03, "h1_seq": ["bear","bull","bull_weak","doji"],             "vol_mult": 0.8,  "body_scale": 0.0006},
    "RANGE":                   {"weight": 0.06, "h1_seq": ["bull","bear","bull","bear"],                  "vol_mult": 0.7,  "body_scale": 0.0005},
    "RANGE_HIGH":              {"weight": 0.04, "h1_seq": ["bull","bull_weak","doji","bear_weak"],        "vol_mult": 0.7,  "body_scale": 0.0005},
    "RANGE_LOW":               {"weight": 0.04, "h1_seq": ["bear","bear_weak","doji","bull_weak"],        "vol_mult": 0.7,  "body_scale": 0.0005},
    "RANGE_COMPRESSION":       {"weight": 0.12, "h1_seq": ["doji","doji","doji","doji"],                 "vol_mult": 0.45, "body_scale": 0.0002},
    "BREAKOUT_SETUP_BULL":     {"weight": 0.10, "h1_seq": ["doji","doji","bull_weak","bull"],             "vol_mult": 0.8,  "body_scale": 0.0006},
    "BREAKOUT_SETUP_BEAR":     {"weight": 0.10, "h1_seq": ["doji","doji","bear_weak","bear"],             "vol_mult": 0.8,  "body_scale": 0.0006},
    "BREAKOUT_CONFIRMED_BULL": {"weight": 0.10, "h1_seq": ["doji","bull","bull_strong","bull_exp"],       "vol_mult": 2.2,  "body_scale": 0.0015},
    "BREAKOUT_CONFIRMED_BEAR": {"weight": 0.10, "h1_seq": ["doji","bear","bear_strong","bear_exp"],       "vol_mult": 2.2,  "body_scale": 0.0015},
    "FALSE_BREAKOUT_BULL":     {"weight": 0.07, "h1_seq": ["bull_strong","spike_up","rejection","bear"],  "vol_mult": 1.8,  "body_scale": 0.0011},
    "FALSE_BREAKOUT_BEAR":     {"weight": 0.07, "h1_seq": ["bear_strong","spike_dn","rejection","bull"],  "vol_mult": 1.8,  "body_scale": 0.0011},
    "EARLY_BULL_REVERSAL":     {"weight": 0.02, "h1_seq": ["bear","bear","doji","bull_strong"],           "vol_mult": 1.2,  "body_scale": 0.0009},
    "EARLY_BEAR_REVERSAL":     {"weight": 0.02, "h1_seq": ["bull","bull","doji","bear_strong"],           "vol_mult": 1.2,  "body_scale": 0.0009},
    "LIQUIDITY_SWEEP_HIGH":    {"weight": 0.02, "h1_seq": ["bull","bull","spike_up","bear_strong"],       "vol_mult": 2.0,  "body_scale": 0.0013},
    "LIQUIDITY_SWEEP_LOW":     {"weight": 0.02, "h1_seq": ["bear","bear","spike_dn","bull_strong"],       "vol_mult": 2.0,  "body_scale": 0.0013},
}

TRANSITIONS = {
    "STRONG_BULL":             {"STRONG_BULL":0.25,"ACCELERATING_BULL":0.15,"MATURE_BULL":0.25,"EARLY_BEAR_REVERSAL":0.10,"BULL_PULLBACK":0.25},
    "STRONG_BEAR":             {"STRONG_BEAR":0.25,"ACCELERATING_BEAR":0.15,"MATURE_BEAR":0.25,"EARLY_BULL_REVERSAL":0.10,"BEAR_PULLBACK":0.25},
    "ACCELERATING_BULL":       {"STRONG_BULL":0.15,"ACCELERATING_BULL":0.15,"LIQUIDITY_SWEEP_HIGH":0.25,"EARLY_BEAR_REVERSAL":0.25,"FALSE_BREAKOUT_BULL":0.20},
    "ACCELERATING_BEAR":       {"STRONG_BEAR":0.15,"ACCELERATING_BEAR":0.15,"LIQUIDITY_SWEEP_LOW":0.25,"EARLY_BULL_REVERSAL":0.25,"FALSE_BREAKOUT_BEAR":0.20},
    "MATURE_BULL":             {"BULL_PULLBACK":0.30,"EARLY_BEAR_REVERSAL":0.20,"RANGE":0.20,"STRONG_BULL":0.15,"RANGE_HIGH":0.15},
    "MATURE_BEAR":             {"BEAR_PULLBACK":0.30,"EARLY_BULL_REVERSAL":0.20,"RANGE":0.20,"STRONG_BEAR":0.15,"RANGE_LOW":0.15},
    "BULL_PULLBACK":           {"BULL_PULLBACK_ENDING":0.35,"STRONG_BULL":0.25,"RANGE":0.20,"EARLY_BEAR_REVERSAL":0.20},
    "BEAR_PULLBACK":           {"BEAR_PULLBACK_ENDING":0.35,"STRONG_BEAR":0.25,"RANGE":0.20,"EARLY_BULL_REVERSAL":0.20},
    "BULL_PULLBACK_ENDING":    {"STRONG_BULL":0.35,"ACCELERATING_BULL":0.20,"RANGE":0.25,"BEAR_PULLBACK":0.20},
    "BEAR_PULLBACK_ENDING":    {"STRONG_BEAR":0.35,"ACCELERATING_BEAR":0.20,"RANGE":0.25,"BULL_PULLBACK":0.20},
    "RANGE":                   {"RANGE":0.25,"RANGE_HIGH":0.10,"RANGE_LOW":0.10,"RANGE_COMPRESSION":0.25,"BREAKOUT_SETUP_BULL":0.15,"BREAKOUT_SETUP_BEAR":0.15},
    "RANGE_HIGH":              {"RANGE":0.15,"FALSE_BREAKOUT_BULL":0.25,"BREAKOUT_CONFIRMED_BULL":0.25,"EARLY_BEAR_REVERSAL":0.20,"LIQUIDITY_SWEEP_HIGH":0.15},
    "RANGE_LOW":               {"RANGE":0.15,"FALSE_BREAKOUT_BEAR":0.25,"BREAKOUT_CONFIRMED_BEAR":0.25,"EARLY_BULL_REVERSAL":0.20,"LIQUIDITY_SWEEP_LOW":0.15},
    "RANGE_COMPRESSION":       {"BREAKOUT_SETUP_BULL":0.28,"BREAKOUT_SETUP_BEAR":0.28,"RANGE":0.25,"RANGE_COMPRESSION":0.19},
    "BREAKOUT_SETUP_BULL":     {"BREAKOUT_CONFIRMED_BULL":0.40,"FALSE_BREAKOUT_BULL":0.22,"RANGE":0.20,"STRONG_BULL":0.18},
    "BREAKOUT_SETUP_BEAR":     {"BREAKOUT_CONFIRMED_BEAR":0.40,"FALSE_BREAKOUT_BEAR":0.22,"RANGE":0.20,"STRONG_BEAR":0.18},
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
# LATENT MARKET STATE ENGINE  (breakout-specialist version)
# Extra latent vars: range_width (tracks squeeze tightness),
#                    coil_turns  (bars at or near compression extremes)
# ─────────────────────────────────────────────────────────────────────

class _LS:
    __slots__ = ("trend_strength","trend_age","vol_regime",
                 "compression_score","compression_duration",
                 "pullback_depth","liquidity_pressure",
                 "breakout_pressure","reversal_pressure",
                 "range_width","coil_turns","_vrn")
    def __init__(self):
        self.trend_strength       = float(np.random.uniform(0.20, 0.65))
        self.trend_age            = int(np.random.randint(1, 6))
        self.vol_regime           = "normal"
        self.compression_score    = float(np.random.uniform(0.10, 0.40))
        self.compression_duration = 0
        self.pullback_depth       = 0.0
        self.liquidity_pressure   = float(np.random.uniform(0.0, 0.25))
        self.breakout_pressure    = 0.0
        self.reversal_pressure    = 0.0
        self.range_width          = float(np.random.uniform(0.4, 1.0))  # 0=tight 1=wide
        self.coil_turns           = 0   # consecutive bars hugging compression boundary
        self._vrn                 = 1

def _init_ls(cond):
    ls = _LS()
    if "COMPRESSION" in cond:
        ls.compression_score    = np.random.uniform(0.60, 0.95)
        ls.compression_duration = np.random.randint(6, 25)
        ls.breakout_pressure    = min(ls.compression_duration * 0.04, 0.90)
        ls.trend_strength       = np.random.uniform(0.00, 0.18)
        ls.range_width          = np.random.uniform(0.05, 0.30)
        ls.coil_turns           = ls.compression_duration
        ls._vrn = 0
    elif "BREAKOUT_SETUP" in cond:
        ls.compression_score    = np.random.uniform(0.35, 0.65)
        ls.compression_duration = np.random.randint(4, 15)
        ls.breakout_pressure    = np.random.uniform(0.40, 0.75)
        ls.range_width          = np.random.uniform(0.15, 0.45)
        ls.coil_turns           = np.random.randint(2, 8)
        ls._vrn = 0
    elif "BREAKOUT_CONFIRMED" in cond:
        ls.breakout_pressure    = np.random.uniform(0.70, 1.00)
        ls.compression_score    = 0.0
        ls.compression_duration = 0
        ls.trend_strength       = np.random.uniform(0.65, 0.95)
        ls.range_width          = np.random.uniform(0.60, 1.00)
        ls.coil_turns           = 0
        ls._vrn = 2
    elif "FALSE_BREAKOUT" in cond:
        ls.breakout_pressure    = np.random.uniform(0.30, 0.65)
        ls.compression_score    = np.random.uniform(0.10, 0.35)
        ls.liquidity_pressure   = np.random.uniform(0.45, 0.80)
        ls.range_width          = np.random.uniform(0.40, 0.75)
        ls._vrn = 2
    elif "RANGE" in cond:
        ls.compression_score    = np.random.uniform(0.20, 0.50)
        ls.trend_strength       = np.random.uniform(0.00, 0.30)
        ls.range_width          = np.random.uniform(0.30, 0.65)
        ls._vrn = 0
    elif any(k in cond for k in ("STRONG","ACCELERATING")):
        ls.trend_strength       = np.random.uniform(0.65, 1.00)
        ls.trend_age            = np.random.randint(2, 8)
        ls.range_width          = np.random.uniform(0.65, 1.00)
        ls._vrn = 2
    elif "MATURE" in cond:
        ls.trend_strength       = np.random.uniform(0.35, 0.65)
        ls.trend_age            = np.random.randint(10, 22)
        ls.reversal_pressure    = np.random.uniform(0.25, 0.55)
        ls._vrn = 1
    elif "SWEEP" in cond:
        ls.liquidity_pressure   = np.random.uniform(0.70, 1.00)
        ls.range_width          = np.random.uniform(0.55, 0.90)
        ls._vrn = 2
    ls.vol_regime = ["low","normal","high"][ls._vrn]
    return ls

def _step_ls(ls, cond):
    # trend_age
    if any(k in cond for k in ("STRONG","ACCELERATING","MATURE")):
        ls.trend_age = min(ls.trend_age + 1, 40)
    elif any(k in cond for k in ("REVERSAL","BREAKOUT_CONFIRMED")):
        ls.trend_age = 1

    # trend_strength
    if "ACCELERATING" in cond:
        ls.trend_strength = min(1.0, ls.trend_strength + np.random.uniform(0.05, 0.14))
    elif "MATURE" in cond:
        ls.trend_strength = max(0.05, ls.trend_strength - np.random.uniform(0.02, 0.07))
    elif "PULLBACK" in cond and "ENDING" not in cond:
        ls.trend_strength = max(0.05, ls.trend_strength - np.random.uniform(0.05, 0.12))
    elif "PULLBACK_ENDING" in cond:
        ls.trend_strength = min(1.0, ls.trend_strength + np.random.uniform(0.06, 0.13))
    elif "BREAKOUT_CONFIRMED" in cond:
        ls.trend_strength = np.random.uniform(0.65, 0.95)
    elif any(k in cond for k in ("RANGE","COMPRESSION")):
        ls.trend_strength = max(0.0, ls.trend_strength - np.random.uniform(0.03, 0.08))

    # compression + range_width + coil_turns — the core breakout dynamics
    if "COMPRESSION" in cond:
        ls.compression_score    = min(1.0, ls.compression_score + np.random.uniform(0.05, 0.11))
        ls.compression_duration += 1
        ls.range_width           = max(0.03, ls.range_width - np.random.uniform(0.04, 0.09))
        ls.coil_turns           += 1
        # breakout_pressure rises faster with longer compression
        age_boost = 0.03 + 0.005 * ls.compression_duration
        ls.breakout_pressure    = min(1.0, ls.breakout_pressure + age_boost)
    elif "RANGE" in cond and "COMPRESSION" not in cond:
        ls.compression_score    = min(0.80, ls.compression_score + np.random.uniform(0.02, 0.06))
        ls.compression_duration += 1
        ls.range_width           = float(np.clip(ls.range_width + np.random.uniform(-0.05, 0.05), 0.20, 0.70))
        ls.breakout_pressure    = min(1.0, ls.breakout_pressure + 0.015 * ls.compression_duration)
    elif "BREAKOUT_CONFIRMED" in cond:
        ls.compression_score    = max(0.0, ls.compression_score - np.random.uniform(0.30, 0.60))
        ls.compression_duration = 0
        ls.breakout_pressure    = max(0.0, ls.breakout_pressure - np.random.uniform(0.50, 0.90))
        ls.range_width          = min(1.0, ls.range_width + np.random.uniform(0.30, 0.60))
        ls.coil_turns           = 0
    elif "FALSE_BREAKOUT" in cond:
        ls.breakout_pressure    = max(0.0, ls.breakout_pressure - np.random.uniform(0.20, 0.45))
        ls.compression_score    = max(0.0, ls.compression_score - np.random.uniform(0.10, 0.25))
        ls.range_width          = float(np.clip(ls.range_width + np.random.uniform(-0.10, 0.20), 0.15, 0.75))
        ls.coil_turns           = max(0, ls.coil_turns - 3)
    elif any(k in cond for k in ("STRONG","ACCELERATING")):
        ls.compression_score    = max(0.0, ls.compression_score - np.random.uniform(0.15, 0.40))
        ls.compression_duration = 0
        ls.breakout_pressure    = max(0.0, ls.breakout_pressure - np.random.uniform(0.30, 0.70))
        ls.range_width          = min(1.0, ls.range_width + np.random.uniform(0.20, 0.45))
        ls.coil_turns           = 0

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

    # reversal_pressure
    age_f = min(ls.trend_age / 20.0, 1.0)
    if any(k in cond for k in ("MATURE","SWEEP","FALSE_BREAKOUT")):
        ls.reversal_pressure = min(1.0, ls.reversal_pressure + np.random.uniform(0.04, 0.11) + age_f * 0.04)
    elif any(k in cond for k in ("REVERSAL","BREAKOUT_CONFIRMED","PULLBACK_ENDING")):
        ls.reversal_pressure = max(0.0, ls.reversal_pressure - np.random.uniform(0.20, 0.50))

    # vol_regime
    if any(k in cond for k in ("ACCELERATING","BREAKOUT_CONFIRMED","SWEEP")):
        ls._vrn = min(2, ls._vrn + 1)
    elif "COMPRESSION" in cond:
        ls._vrn = max(0, ls._vrn - 1)
    elif "RANGE" in cond and np.random.random() < 0.30:
        ls._vrn = max(0, ls._vrn - 1)
    ls.vol_regime = ["low","normal","high"][ls._vrn]

    # clamp
    for attr in ("trend_strength","compression_score","breakout_pressure",
                 "pullback_depth","liquidity_pressure","reversal_pressure"):
        setattr(ls, attr, float(np.clip(getattr(ls, attr), 0.0, 1.0)))
    ls.range_width = float(np.clip(ls.range_width, 0.02, 1.0))
    return ls

def _next_cond(cond, ls):
    base = TRANSITIONS.get(cond, {})
    if not base:
        keys = list(CONDITIONS.keys())
        return random.choices(keys, weights=[CONDITIONS[k]["weight"] for k in keys], k=1)[0]

    w = {}
    for c, p in base.items():
        b = 1.0

        # breakout_pressure dominates when high
        if ls.breakout_pressure > 0.50:
            if any(k in c for k in ("BREAKOUT_CONFIRMED","BREAKOUT_SETUP","FALSE_BREAKOUT")):
                b *= 1.0 + ls.breakout_pressure * 3.0
            if "COMPRESSION" in c:
                b *= max(0.04, 1.0 - ls.breakout_pressure * 0.90)
            if "RANGE" in c and "COMPRESSION" not in c:
                b *= max(0.10, 1.0 - ls.breakout_pressure * 0.60)

        # compression_duration accelerates breakout probability
        if ls.compression_duration > 10:
            cd = ls.compression_duration
            if "BREAKOUT" in c:
                b *= 1.0 + min(cd * 0.06, 2.0)
            if "COMPRESSION" in c:
                b *= max(0.03, 1.0 - cd * 0.07)

        # high liquidity_pressure → false breakout or sweep
        if ls.liquidity_pressure > 0.60:
            if any(k in c for k in ("FALSE_BREAKOUT","SWEEP")):
                b *= 1.0 + ls.liquidity_pressure * 2.5

        # coil_turns: many coil turns → breakout more certain
        if ls.coil_turns > 8:
            if "BREAKOUT_CONFIRMED" in c:
                b *= 1.0 + ls.coil_turns * 0.04
            if "FALSE_BREAKOUT" in c:
                b *= max(0.20, 1.0 - ls.coil_turns * 0.03)

        # reversal_pressure
        if ls.reversal_pressure > 0.55:
            if any(k in c for k in ("REVERSAL","SWEEP","FALSE_BREAKOUT")):
                b *= 1.0 + ls.reversal_pressure * 1.8
            if any(k in c for k in ("STRONG","ACCELERATING")):
                b *= max(0.10, 1.0 - ls.reversal_pressure * 0.90)

        # deep pullback → pullback_ending
        if ls.pullback_depth > 0.55:
            if "PULLBACK_ENDING" in c:
                b *= 1.0 + ls.pullback_depth * 2.0
            if "PULLBACK" in c and "ENDING" not in c:
                b *= 0.35

        w[c] = max(1e-6, p * b)

    total = sum(w.values())
    keys  = list(w.keys())
    probs = [w[k] / total for k in keys]
    return random.choices(keys, weights=probs, k=1)[0]

def _modulate(cond, ls, base_bs, base_vm):
    vol_scale = {"low": 0.55, "normal": 1.00, "high": 1.45}[ls.vol_regime]
    bs = base_bs * vol_scale
    vm = base_vm * vol_scale

    body_taper  = float(np.clip(0.40 + 0.60 * ls.trend_strength, 0.22, 1.55))
    body_taper *= float(np.clip(1.0 - ls.trend_age / 40.0, 0.40, 1.20))

    wick_mult   = float(np.clip(1.0 + ls.trend_age / 30.0, 1.0, 1.60))
    wick_mult  *= float(np.clip(1.0 + ls.liquidity_pressure * 0.80, 1.0, 2.20))
    wick_mult  *= float(np.clip(1.0 + ls.reversal_pressure  * 0.55, 1.0, 1.80))

    # compression: squeeze body & vol hard; range_width modulates how tight
    if "COMPRESSION" in cond or "RANGE" in cond:
        squeeze_body = float(np.clip(1.0 - ls.compression_score * 0.70, 0.15, 1.0))
        squeeze_vm   = float(np.clip(1.0 - ls.compression_score * 0.55, 0.20, 1.0))
        # range_width makes compression even narrower when very tight
        rw_factor    = float(np.clip(ls.range_width + 0.10, 0.15, 1.0))
        bs *= squeeze_body * rw_factor
        vm *= squeeze_vm

    # breakout: vol explosion proportional to stored pressure
    if "BREAKOUT_CONFIRMED" in cond:
        bs *= 1.0 + ls.breakout_pressure * 1.00
        vm *= 1.0 + ls.breakout_pressure * 0.80
        # a longer squeeze → bigger bang
        if ls.compression_duration > 0:
            cd_bonus = min(ls.compression_duration * 0.03, 0.60)
            bs *= 1.0 + cd_bonus
            vm *= 1.0 + cd_bonus * 0.5

    # pullback modulation
    if "PULLBACK" in cond and "ENDING" not in cond:
        body_taper *= max(0.35, 1.0 - ls.pullback_depth * 0.50)
        wick_mult  *= 1.0 + ls.pullback_depth * 0.45

    bs = float(np.clip(bs, 0.0001, 0.0040))
    vm = float(np.clip(vm, 0.10,   6.00))
    return bs, vm, float(np.clip(wick_mult, 0.50, 3.50)), float(np.clip(body_taper, 0.18, 1.65))


# ─────────────────────────────────────────────────────────────────────
# H1 ARCHETYPE → 4 M15 EXPANSION  (wick_mult applied to all wick params)
# ─────────────────────────────────────────────────────────────────────

def _expand_h1_to_m15(archetype, base_price, body_scale, vol_mult, wick_mult=1.0):
    bs = body_scale
    vm = vol_mult
    wx = wick_mult
    p  = base_price

    def candle(o, c, wu, wl, vlo, vhi):
        h = max(o, c) + abs(float(np.random.normal(0, wu * wx)))
        l = min(o, c) - abs(float(np.random.normal(0, wl * wx)))
        h = max(h, o, c); l = min(l, o, c)
        return {"open": float(o), "high": float(h), "low": float(l),
                "close": float(c), "volume": int(np.random.uniform(vlo, vhi))}

    rn = lambda lo, hi: float(np.random.uniform(lo, hi))

    if archetype == "bull":
        c=[candle(p,p+rn(0.10,0.35)*bs, bs*0.12,bs*0.20,200*vm,600*vm)]
        c.append(candle(c[0]["close"],c[0]["close"]+rn(0.15,0.45)*bs,bs*0.10,bs*0.15,300*vm,800*vm))
        c.append(candle(c[1]["close"],c[1]["close"]+rn(0.20,0.50)*bs,bs*0.10,bs*0.12,400*vm,1000*vm))
        c.append(candle(c[2]["close"],c[2]["close"]+rn(0.25,0.60)*bs,bs*0.06,bs*0.10,500*vm,1200*vm))
    elif archetype == "bull_strong":
        c=[candle(p,p+rn(0.30,0.60)*bs, bs*0.06,bs*0.15,500*vm,1200*vm)]
        c.append(candle(c[0]["close"],c[0]["close"]+rn(0.40,0.70)*bs,bs*0.05,bs*0.10,600*vm,1500*vm))
        c.append(candle(c[1]["close"],c[1]["close"]+rn(0.50,0.90)*bs,bs*0.04,bs*0.09,700*vm,1800*vm))
        c.append(candle(c[2]["close"],c[2]["close"]+rn(0.60,1.00)*bs,bs*0.03,bs*0.07,800*vm,2100*vm))
    elif archetype == "bull_exp":
        c=[candle(p,p+rn(0.50,1.00)*bs, bs*0.04,bs*0.08,800*vm,2000*vm)]
        c.append(candle(c[0]["close"],c[0]["close"]+rn(0.80,1.40)*bs,bs*0.03,bs*0.07,1200*vm,2800*vm))
        c.append(candle(c[1]["close"],c[1]["close"]+rn(1.00,1.80)*bs,bs*0.03,bs*0.06,1500*vm,3500*vm))
        c.append(candle(c[2]["close"],c[2]["close"]+rn(1.20,2.00)*bs,bs*0.02,bs*0.05,2000*vm,5000*vm))
    elif archetype == "bull_weak":
        c=[candle(p,p+rn(0.05,0.18)*bs, bs*0.20,bs*0.30,130*vm,380*vm)]
        c.append(candle(c[0]["close"],c[0]["close"]+rn(-0.08,0.18)*bs,bs*0.25,bs*0.35,110*vm,340*vm))
        c.append(candle(c[1]["close"],c[1]["close"]+rn(0.05,0.18)*bs,bs*0.22,bs*0.32,120*vm,360*vm))
        c.append(candle(c[2]["close"],c[2]["close"]+rn(0.00,0.14)*bs, bs*0.25,bs*0.35,100*vm,310*vm))
    elif archetype == "bear":
        c=[candle(p,p-rn(0.10,0.35)*bs, bs*0.20,bs*0.12,200*vm,600*vm)]
        c.append(candle(c[0]["close"],c[0]["close"]-rn(0.15,0.45)*bs,bs*0.15,bs*0.10,300*vm,800*vm))
        c.append(candle(c[1]["close"],c[1]["close"]-rn(0.20,0.50)*bs,bs*0.12,bs*0.10,400*vm,1000*vm))
        c.append(candle(c[2]["close"],c[2]["close"]-rn(0.25,0.60)*bs,bs*0.10,bs*0.06,500*vm,1200*vm))
    elif archetype == "bear_strong":
        c=[candle(p,p-rn(0.30,0.60)*bs, bs*0.15,bs*0.06,500*vm,1200*vm)]
        c.append(candle(c[0]["close"],c[0]["close"]-rn(0.40,0.70)*bs,bs*0.10,bs*0.05,600*vm,1500*vm))
        c.append(candle(c[1]["close"],c[1]["close"]-rn(0.50,0.90)*bs,bs*0.09,bs*0.04,700*vm,1800*vm))
        c.append(candle(c[2]["close"],c[2]["close"]-rn(0.60,1.00)*bs,bs*0.07,bs*0.03,800*vm,2100*vm))
    elif archetype == "bear_exp":
        c=[candle(p,p-rn(0.50,1.00)*bs, bs*0.08,bs*0.04,800*vm,2000*vm)]
        c.append(candle(c[0]["close"],c[0]["close"]-rn(0.80,1.40)*bs,bs*0.07,bs*0.03,1200*vm,2800*vm))
        c.append(candle(c[1]["close"],c[1]["close"]-rn(1.00,1.80)*bs,bs*0.06,bs*0.03,1500*vm,3500*vm))
        c.append(candle(c[2]["close"],c[2]["close"]-rn(1.20,2.00)*bs,bs*0.05,bs*0.02,2000*vm,5000*vm))
    elif archetype == "bear_weak":
        c=[candle(p,p-rn(0.05,0.18)*bs, bs*0.30,bs*0.20,130*vm,380*vm)]
        c.append(candle(c[0]["close"],c[0]["close"]-rn(-0.08,0.18)*bs,bs*0.35,bs*0.25,110*vm,340*vm))
        c.append(candle(c[1]["close"],c[1]["close"]-rn(0.05,0.18)*bs,bs*0.32,bs*0.22,120*vm,360*vm))
        c.append(candle(c[2]["close"],c[2]["close"]-rn(0.00,0.14)*bs, bs*0.35,bs*0.25,100*vm,310*vm))
    elif archetype == "doji":
        c=[candle(p,p+rn(-0.12,0.12)*bs, bs*0.40,bs*0.40,120*vm,450*vm)]
        c.append(candle(c[0]["close"],c[0]["close"]+rn(-0.12,0.12)*bs,bs*0.40,bs*0.40,110*vm,420*vm))
        c.append(candle(c[1]["close"],c[1]["close"]+rn(-0.10,0.10)*bs,bs*0.38,bs*0.38,100*vm,400*vm))
        c.append(candle(c[2]["close"],c[2]["close"]+rn(-0.08,0.08)*bs,bs*0.35,bs*0.35,95*vm,380*vm))
    elif archetype == "spike_up":
        c=[candle(p,p+rn(0.40,0.80)*bs, bs*0.10,bs*0.05,600*vm,1500*vm)]
        c.append(candle(c[0]["close"],c[0]["close"]+rn(0.70,1.30)*bs,bs*0.20,bs*0.05,1200*vm,3200*vm))
        c.append(candle(c[1]["close"],c[1]["close"]+rn(0.20,0.60)*bs,bs*0.45,bs*0.10,800*vm,2000*vm))
        c.append(candle(c[2]["close"],c[2]["close"]+rn(-0.30,0.30)*bs,bs*0.55,bs*0.55,500*vm,1200*vm))
    elif archetype == "spike_dn":
        c=[candle(p,p-rn(0.40,0.80)*bs, bs*0.05,bs*0.10,600*vm,1500*vm)]
        c.append(candle(c[0]["close"],c[0]["close"]-rn(0.70,1.30)*bs,bs*0.05,bs*0.20,1200*vm,3200*vm))
        c.append(candle(c[1]["close"],c[1]["close"]-rn(0.20,0.60)*bs,bs*0.10,bs*0.45,800*vm,2000*vm))
        c.append(candle(c[2]["close"],c[2]["close"]+rn(-0.30,0.30)*bs,bs*0.55,bs*0.55,500*vm,1200*vm))
    elif archetype == "rejection":
        c=[candle(p,p+rn(0.30,0.60)*bs, bs*0.10,bs*0.10,400*vm,1000*vm)]
        c.append(candle(c[0]["close"],c[0]["close"]+rn(0.50,1.00)*bs,bs*0.55,bs*0.06,1000*vm,2800*vm))
        c.append(candle(c[1]["close"],c[1]["close"]-rn(0.50,1.10)*bs,bs*0.10,bs*0.20,900*vm,2500*vm))
        c.append(candle(c[2]["close"],c[2]["close"]-rn(0.30,0.70)*bs,bs*0.12,bs*0.15,600*vm,1500*vm))
    else:
        c=[]; cp=p
        for _ in range(4):
            nc=cp+rn(-0.10,0.10)*bs
            c.append(candle(cp,nc,bs*0.30,bs*0.30,200*vm,600*vm)); cp=nc

    return c, c[-1]["close"]


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
            m15s, price = _expand_h1_to_m15(arch, price, bs * bt, vm, wx)
            all_m15.extend(m15s)
        ls   = _step_ls(ls, cond)
        cond = _next_cond(cond, ls)

    active_condition = cond
    cfg              = CONDITIONS[active_condition]
    bs, vm, wx, bt   = _modulate(active_condition, ls, cfg["body_scale"], cfg["vol_mult"])
    for arch in cfg["h1_seq"]:
        m15s, price = _expand_h1_to_m15(arch, price, bs * bt, vm, wx)
        all_m15.extend(m15s)

    ls               = _step_ls(ls, active_condition)
    future_condition = _next_cond(active_condition, ls)
    fut_cfg          = CONDITIONS[future_condition]
    bs, vm, wx, bt   = _modulate(future_condition, ls, fut_cfg["body_scale"], fut_cfg["vol_mult"])

    future_m15 = []
    for arch in fut_cfg["h1_seq"][:1]:
        m15s, price = _expand_h1_to_m15(arch, price, bs * bt, vm, wx)
        future_m15.extend(m15s)
        if len(future_m15) >= 4: break
    while len(future_m15) < 4: future_m15.append(future_m15[-1])
    return all_m15, active_condition, future_condition, future_m15[:4], ls


# ─────────────────────────────────────────────────────────────────────
# FEATURE COMPUTATION — 32 BREAKOUT/SQUEEZE FEATURES  (unchanged)
# ─────────────────────────────────────────────────────────────────────

def _rz(s, w):
    return (s - s.rolling(w).mean()) / (s.rolling(w).std() + EPS)

def compute_breakout_features(df: pd.DataFrame):
    df = df.copy().reset_index(drop=True)

    df["range"]      = df["high"] - df["low"]
    df["body"]       = df["close"] - df["open"]
    df["abs_body"]   = df["body"].abs()
    df["dir"]        = np.sign(df["body"]).fillna(0)
    df["upper_wick"] = df["high"] - df[["open","close"]].max(axis=1)
    df["lower_wick"] = df[["open","close"]].min(axis=1) - df["low"]
    df["atr14"]      = df["range"].rolling(14, min_periods=1).mean()
    df["atr5"]       = df["range"].rolling(5,  min_periods=1).mean()

    for w in [8, 14, 20]:
        mid   = df["close"].rolling(w, min_periods=1).mean()
        std   = df["close"].rolling(w, min_periods=1).std().fillna(0)
        atr_w = df["range"].rolling(w, min_periods=1).mean()
        df[f"bb_upper_{w}"] = mid + 2 * std
        df[f"bb_lower_{w}"] = mid - 2 * std
        df[f"bb_width_{w}"] = 4 * std
        df[f"kc_width_{w}"] = 3 * atr_w
        df[f"kc_mid_{w}"]   = mid
        df[f"kc_upper_{w}"] = mid + 1.5 * atr_w
        df[f"kc_lower_{w}"] = mid - 1.5 * atr_w

    df["bb_width_z"]  = _rz(df["bb_width_14"], 100).fillna(0.0)
    df["kc_width_n"]  = df["kc_width_14"] / (df["atr14"] + EPS)

    sq8  = (df["bb_width_8"]  < df["kc_width_8"]).astype(int)
    sq14 = (df["bb_width_14"] < df["kc_width_14"]).astype(int)
    sq20 = (df["bb_width_20"] < df["kc_width_20"]).astype(int)
    df["squeeze_stack"] = (sq8 + sq14 + sq20).astype(float)
    df["kb_ratio"]      = df["bb_width_14"] / (df["kc_width_14"] + EPS)
    df["kb_slope"]      = _rz(df["kb_ratio"].diff(1).rolling(10, min_periods=1).mean(), 20).fillna(0.0)

    sq_active_s = (df["kb_ratio"] < 1.0).astype(int)
    cons_sq     = []
    cur = 0
    for s in sq_active_s.tolist():
        cur = cur + 1 if s == 1 else 0
        cons_sq.append(cur)
    df["consec_squeeze"]   = cons_sq
    df["sq_active"]        = sq_active_s
    avg_sq = max(float(pd.Series(cons_sq).rolling(100, min_periods=5).mean().iloc[-1]), 1.0)
    df["squeeze_maturity"] = (df["consec_squeeze"] / avg_sq).clip(0, 4)

    hv_pct                 = df["bb_width_14"].rank(pct=True).fillna(0.5)
    df["vol_potential_e"]  = ((1 - hv_pct) * (1 - df["kb_ratio"]).clip(0, 2)).clip(0, 2)

    don_high  = df["high"].rolling(20, min_periods=1).max()
    don_low   = df["low"].rolling(20, min_periods=1).min()
    don_mid   = (don_high + don_low) / 2
    don_width = (don_high - don_low) + EPS
    lean_raw  = ((df["close"] - don_mid) / don_width).rolling(3, min_periods=1).mean()
    df["sq_lean"] = (lean_raw * sq_active_s).fillna(0.0)

    df["atr_ratio"]    = df["atr5"] / (df["atr14"] + EPS)
    df["atr_ratio_z"]  = _rz(df["atr_ratio"], 50).fillna(0.0)
    df["exp_velocity"] = _rz(df["bb_width_14"].diff(3).fillna(0), 50).fillna(0.0)
    above_upper        = (df["close"] - df["bb_upper_14"]).clip(lower=0) / (df["atr14"] + EPS)
    below_lower        = (df["bb_lower_14"] - df["close"]).clip(lower=0) / (df["atr14"] + EPS)
    df["band_walk_bull"]  = above_upper.rolling(5, min_periods=1).sum()
    df["band_walk_bear"]  = below_lower.rolling(5, min_periods=1).sum()
    df["hv_percentile"]   = hv_pct

    bb_range             = (df["bb_upper_14"] - df["bb_lower_14"]) + EPS
    df["bb_position"]    = (df["close"] - df["bb_lower_14"]) / bb_range
    df["bb_pos_z"]       = _rz(df["bb_position"], 50).fillna(0.0)
    df["kc_position"]    = (df["close"] - df["kc_lower_14"]) / (df["kc_width_14"] + EPS)
    mid_bb14             = df["close"].rolling(14, min_periods=1).mean()
    df["mr_dist"]        = ((df["close"] - mid_bb14) / (df["atr14"] + EPS)).clip(-3, 3)
    df["coil_tightness"] = (1 - df["kb_ratio"]).clip(-1, 1)

    body_avg20           = df["abs_body"].rolling(20, min_periods=1).mean()
    df["body_comp_r"]    = (df["abs_body"] / (body_avg20 + EPS)).clip(0, 3)
    df["close_grav"]     = (df["close"] - df["low"]) / (df["range"] + EPS)
    vol_avg20            = df["volume"].rolling(20, min_periods=1).mean()
    df["vol_z20"]        = _rz(df["volume"], 20).fillna(0.0)
    df["prebreak_qual"]  = (df["close_grav"] * df["body_comp_r"] *
                             df["vol_z20"].clip(0, 3)).clip(-3, 3)
    df["intrabar_fill"]  = df["abs_body"] / (df["range"] + EPS)
    df["vol_sq_div"]     = (df["vol_z20"] * (df["squeeze_stack"] / 3.0)).clip(-3, 3)
    close_above          = (df["close"] > df["bb_upper_14"]).astype(float)
    close_below          = (df["close"] < df["bb_lower_14"]).astype(float)
    df["bkout_confirm"]  = ((close_above - close_below) *
                             (df["vol_z20"] + 1).clip(0, 3)).clip(-3, 3)
    df["don_bias"]       = ((df["close"] - don_mid) / don_width).clip(-1, 1)

    df["vol_z30"]   = _rz(df["volume"], 30).fillna(0.0)
    df["vol_z100"]  = _rz(df["volume"], min(100, len(df))).fillna(0.0)
    df["vol_trend"] = _rz(
        df["volume"].rolling(5, min_periods=1).mean() /
        (vol_avg20 + EPS), 30).fillna(0.0)

    last = df.iloc[-1]
    feat = {
        "bb_width_z":        float(last["bb_width_z"]),
        "kc_width_n":        float(last["kc_width_n"]),
        "squeeze_stack":     float(last["squeeze_stack"]),
        "kb_ratio":          float(last["kb_ratio"]),
        "kb_slope":          float(last["kb_slope"]),
        "squeeze_maturity":  float(last["squeeze_maturity"]),
        "vol_potential_e":   float(last["vol_potential_e"]),
        "sq_lean":           float(last["sq_lean"]),
        "atr_ratio_z":       float(last["atr_ratio_z"]),
        "exp_velocity":      float(last["exp_velocity"]),
        "band_walk_bull":    float(last["band_walk_bull"]),
        "band_walk_bear":    float(last["band_walk_bear"]),
        "hv_percentile":     float(last["hv_percentile"]),
        "bb_position":       float(last["bb_position"]),
        "bb_pos_z":          float(last["bb_pos_z"]),
        "kc_position":       float(last["kc_position"]),
        "mr_dist":           float(last["mr_dist"]),
        "coil_tightness":    float(last["coil_tightness"]),
        "body_comp_r":       float(last["body_comp_r"]),
        "prebreak_qual":     float(last["prebreak_qual"]),
        "intrabar_fill":     float(last["intrabar_fill"]),
        "vol_sq_div":        float(last["vol_sq_div"]),
        "bkout_confirm":     float(last["bkout_confirm"]),
        "don_bias":          float(last["don_bias"]),
        "vol_z30":           float(last["vol_z30"]),
        "vol_z100":          float(last["vol_z100"]),
        "vol_trend":         float(last["vol_trend"]),
        "_atr":              float(last["atr14"]) if float(last["atr14"]) > EPS else 0.0001,
        "_context_close":    float(last["close"]),
        "_sq_lean":          float(last["sq_lean"]),
        "_bkout_confirm":    float(last["bkout_confirm"]),
        "_kb_ratio":         float(last["kb_ratio"]),
    }
    return feat, df


# ─────────────────────────────────────────────────────────────────────
# LABEL COMPUTATION — 6 heads from future 4 M15 candles  (unchanged)
# ─────────────────────────────────────────────────────────────────────

def compute_labels(f: dict, future_candles: list) -> tuple:
    atr           = max(f["_atr"], EPS)
    context_close = f["_context_close"]

    c1 = future_candles[0]
    c2 = future_candles[1] if len(future_candles) >= 2 else c1
    c3 = future_candles[2] if len(future_candles) >= 3 else c2
    c4 = future_candles[3] if len(future_candles) >= 4 else c3

    h1_close = c4["close"]
    h1_high  = max(cx["high"] for cx in future_candles)
    h1_low   = min(cx["low"]  for cx in future_candles)
    h1_range = (h1_high - h1_low) / atr

    net_move_atr   = (h1_close - context_close) / atr
    bodies_atr     = [abs(cx["close"] - cx["open"]) / atr for cx in future_candles]
    avg_body_atr   = float(np.mean(bodies_atr))
    close_grav     = (h1_close - h1_low) / (h1_range * atr + EPS)

    bull_count = sum(1 for cx in future_candles if cx["close"] > cx["open"])

    c1_bull     = c1["close"] > c1["open"]
    c1_body_atr = bodies_atr[0]
    c1_uw       = (c1["high"]  - max(c1["open"], c1["close"])) / (c1["high"] - c1["low"] + EPS)
    c1_lw       = (min(c1["open"], c1["close"]) - c1["low"])   / (c1["high"] - c1["low"] + EPS)

    if   net_move_atr > 0.40:   next_dir = 2
    elif net_move_atr < -0.40:  next_dir = 0
    else:                        next_dir = 1

    if   avg_body_atr > 1.20:   next_mag = 2
    elif avg_body_atr > 0.40:   next_mag = 1
    else:                        next_mag = 0

    if bull_count >= 3:
        next_pat = 0 if net_move_atr > 0.1 else 2
    elif bull_count <= 1:
        next_pat = 0 if net_move_atr < -0.1 else 2
    elif c1_body_atr > 0.6:
        next_pat = 1
    else:
        next_pat = 2

    if   h1_range > 3.0:   next_vol = 2
    elif h1_range < 1.2:   next_vol = 0
    else:                   next_vol = 1

    if   close_grav > 0.67:  next_loc = 2
    elif close_grav < 0.33:  next_loc = 0
    else:                     next_loc = 1

    sq_lean    = f["_sq_lean"]
    bk_confirm = f["_bkout_confirm"]
    expected_bull = (sq_lean > 0.05 or bk_confirm > 0.3)
    expected_bear = (sq_lean < -0.05 or bk_confirm < -0.3)

    if expected_bull:
        if   net_move_atr > 0.70:   follow = 2
        elif net_move_atr > 0.15:   follow = 1
        else:                        follow = 0
    elif expected_bear:
        if   net_move_atr < -0.70:  follow = 2
        elif net_move_atr < -0.15:  follow = 1
        else:                        follow = 0
    else:
        if   avg_body_atr > 0.80:   follow = 2
        elif avg_body_atr > 0.30:   follow = 1
        else:                        follow = 0

    return (int(next_dir), int(next_mag), int(next_pat),
            int(next_vol), int(next_loc), int(follow))



def context_features_breakout(df: pd.DataFrame, atr: float) -> dict:
    """Breakout-specialist context features from last 16 M15."""
    w = df.iloc[-16:].reset_index(drop=True)
    rng  = (w["high"] - w["low"]).astype(float)
    body = (w["close"] - w["open"]).abs().astype(float)
    # compression_strength_16: how narrow are ranges vs their own mean
    rng_mean = float(rng.mean()); rng_std = float(rng.std())
    compression_strength_16 = float(np.clip(1.0 - rng_mean / (atr * 2 + EPS), 0, 1))
    # compression_duration_16: consecutive bars with below-avg range
    avg_rng = float(df["high"].sub(df["low"]).rolling(30, min_periods=1).mean().iloc[-1])
    narrow = (rng < avg_rng * 0.75).astype(int).tolist()
    cd = 0
    for v in reversed(narrow):
        if v: cd += 1
        else: break
    compression_duration_16 = float(cd)
    # breakout_pressure_16: body fill ratio trending upward into last bar
    br = body / (rng + EPS)
    early_br = float(br.iloc[:8].mean()); late_br = float(br.iloc[8:].mean())
    breakout_pressure_16 = float(np.clip((late_br - early_br + 0.5), 0, 1))
    # squeeze_count_16: bars where body < 20% of range
    squeeze_count_16 = float((br < 0.20).sum())
    return {"compression_strength_16": compression_strength_16,
            "compression_duration_16": compression_duration_16,
            "breakout_pressure_16": breakout_pressure_16,
            "squeeze_count_16": squeeze_count_16}

# ─────────────────────────────────────────────────────────────────────
# SEQUENCE GENERATOR
# ─────────────────────────────────────────────────────────────────────

def generate_sequence(base_price=BASE_PRICE):
    all_m15, active_cond, future_cond, future_m15, ls = generate_market_history(base_price)
    df_full = pd.DataFrame(all_m15)
    f, _    = compute_breakout_features(df_full)
    atr     = max(f["_atr"], EPS)
    ctx     = context_features_breakout(df_full, atr)
    labels  = compute_labels(f, future_m15)
    feat = {k: v for k, v in f.items() if not k.startswith("_")}
    row  = dict(feat); row.update(ctx)
    row["condition"]        = active_cond
    row["future_condition"] = future_cond
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
        row["meta_range_width"]          = round(ls.range_width, 3)
        row["meta_coil_turns"]           = ls.coil_turns
    (row["label_next_direction"], row["label_next_magnitude"],  row["label_next_pattern"],
     row["label_next_volatility"],row["label_next_close_loc"],  row["label_bk_follow_through"]) = labels
    return row


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Generating {N_SAMPLES} breakout samples | v5 latent-state engine | breakout specialist")
    rows = []
    for _ in trange(N_SAMPLES, desc="breakout"):
        rows.append(generate_sequence())

    df_out     = pd.DataFrame(rows)
    label_cols = [c for c in df_out.columns if c.startswith("label_")]
    meta_cols  = ["condition", "future_condition"]
    feat_cols  = [c for c in df_out.columns if c not in label_cols + meta_cols]
    df_out     = df_out[meta_cols + sorted(feat_cols) + label_cols]
    df_out.to_csv(OUTPUT_PATH, index=False)

    label_maps = {
        "label_next_direction":    {0:"bearish", 1:"flat", 2:"bullish"},
        "label_next_magnitude":    {0:"small", 1:"medium", 2:"large"},
        "label_next_pattern":      {0:"continuation", 1:"reversal", 2:"indecision"},
        "label_next_volatility":   {0:"contracting", 1:"normal", 2:"expanding"},
        "label_next_close_loc":    {0:"lower_third", 1:"mid", 2:"upper_third"},
        "label_bk_follow_through": {0:"no", 1:"partial", 2:"yes"},
    }
    print(f"\n✅  Saved → {OUTPUT_PATH}  shape={df_out.shape}")
    print(f"    Features : {len(feat_cols)}")
    for lc in label_cols:
        counts = df_out[lc].value_counts().sort_index()
        named  = {label_maps[lc].get(k,k): f"{100*v/len(df_out):.1f}%" for k, v in counts.items()}
        print(f"    {lc}: {named}")
