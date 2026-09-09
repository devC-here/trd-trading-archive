"""
market_physics_h1.py  — shared H1 FX market simulation engine
Imported by all 5 synthetic_*_h1_v2.py generators.

Key improvements over v3:
  1. GARCH(1,1)-style volatility clustering  — quiet → explosive regimes
  2. Session volume profile  — London/NY peaks, Asia thin
  3. Autocorrelated price path  — AR(1) momentum + mean-reversion
  4. Realistic wick mechanics  — liquidity-sweep wicks, reaction tails
  5. Overnight/news micro-gaps  — open ≠ prior close ~8% of bars
  6. 3-bar future window with path metrics for sensitivity labels
"""

import numpy as np
import pandas as pd
import random

EPS = 1e-12

# ── Session volume profile (hour → multiplier, London=1.0 baseline) ─
SESSION_VOL = {
    0: 0.30, 1: 0.25, 2: 0.22, 3: 0.20, 4: 0.22,   # deep Asia
    5: 0.30, 6: 0.45, 7: 0.65, 8: 1.00, 9: 1.20,   # pre/early London
   10: 1.15,11: 1.05,12: 0.85,13: 1.30,14: 1.40,   # London / NY open
   15: 1.20,16: 1.00,17: 0.80,18: 0.60,19: 0.45,   # NY afternoon
   20: 0.35,21: 0.30,22: 0.28,23: 0.28,             # NY close / late
}

def session_vol_mult(hour: int) -> float:
    return SESSION_VOL.get(hour % 24, 0.5)


# ── GARCH-like volatility state ───────────────────────────────────────
class VolState:
    """Persistent volatility state passed through candle generation."""
    def __init__(self, base_atr: float, omega=0.05, alpha=0.15, beta=0.80):
        self.sigma  = base_atr        # current vol
        self.base   = base_atr
        self.omega  = omega * base_atr
        self.alpha  = alpha
        self.beta   = beta

    def update(self, shock: float) -> float:
        """Update sigma given last price shock (raw pips). Returns new sigma."""
        self.sigma = (self.omega
                      + self.alpha * abs(shock)
                      + self.beta  * self.sigma)
        # clamp: never go below 30% or above 350% of base
        self.sigma = float(np.clip(self.sigma, self.base * 0.30, self.base * 3.50))
        return self.sigma


# ── Realistic H1 candle generator ────────────────────────────────────
def gen_candle_h1(price: float, vol_state: VolState, bias: float,
                  hour: int, base_vol: int,
                  wick_lo: float = 0.10, wick_hi: float = 0.55,
                  news_prob: float = 0.04) -> tuple:
    """
    Returns (candle_dict, new_close).
    Uses vol_state for GARCH body sizing.
    Injects liquidity-sweep wicks probabilistically.
    Applies session volume multiplier.
    """
    sigma = vol_state.sigma

    # Direction with persistence (bias carries directional momentum)
    d = 1 if random.random() < bias else -1

    # Body: gamma-distributed size, scaled by current vol
    body_size = float(np.random.gamma(shape=1.8, scale=sigma * 0.55))
    body_size  = max(body_size, sigma * 0.05)

    o = float(price)
    c = float(o + d * body_size)

    # Update vol state with this shock
    vol_state.update(body_size)

    # Wicks — base wick from wick_lo/wick_hi multiplier of body
    wk = float(body_size * np.random.uniform(wick_lo, wick_hi))
    tl = float(body_size * np.random.uniform(wick_lo, wick_hi))

    # Liquidity sweep wick (8% chance): spike beyond body in one direction
    if random.random() < 0.08:
        sweep_side = 1 if random.random() < 0.5 else -1
        sweep_size = float(np.random.exponential(sigma * 0.70))
        if sweep_side == 1:
            wk += sweep_size   # upper sweep
        else:
            tl += sweep_size   # lower sweep

    # News gap: open ≠ prior close (micro-gap, ~4% of bars)
    if random.random() < news_prob:
        gap_dir  = 1 if random.random() < 0.5 else -1
        gap_size = float(np.random.exponential(sigma * 0.40))
        o = float(price + gap_dir * gap_size)
        c = float(o + d * body_size)

    h = max(o, c) + wk
    l = min(o, c) - tl

    # Volume: session-scaled, GARCH-correlated (high vol → higher volume)
    vol_mult   = session_vol_mult(hour)
    vol_noise  = float(np.random.lognormal(0, 0.35))
    garch_mult = float(np.clip(vol_state.sigma / (vol_state.base + EPS), 0.3, 4.0))
    volume     = int(base_vol * vol_mult * vol_noise * garch_mult)
    volume     = max(volume, 50)

    candle = {"open": float(o), "high": float(h),
              "low": float(l),  "close": float(c),
              "volume": volume,  "hour": hour}
    return candle, float(c)


# ── AR(1) bias evolution ──────────────────────────────────────────────
def evolve_bias(bias: float, target_bias: float,
                ar_coef: float = 0.85, noise_std: float = 0.06) -> float:
    """
    AR(1) mean-reverting bias.  bias → target_bias with momentum.
    Injects random noise so the path isn't perfectly deterministic.
    """
    new_bias = ar_coef * bias + (1 - ar_coef) * target_bias + np.random.normal(0, noise_std)
    return float(np.clip(new_bias, 0.10, 0.90))


# ── 3-bar future path metrics ─────────────────────────────────────────
def future_path_metrics(future_candles: list, atr: float) -> dict:
    """
    Compute rich path metrics from 3 future candles.
    All values normalised by ATR.
    """
    c1 = future_candles[0]
    c2 = future_candles[1] if len(future_candles) >= 2 else c1
    c3 = future_candles[2] if len(future_candles) >= 3 else c2

    ref = c1["open"]   # reference = open of first future bar

    # Body sizes
    b1 = abs(c1["close"] - c1["open"]) / atr
    b2 = abs(c2["close"] - c2["open"]) / atr
    b3 = abs(c3["close"] - c3["open"]) / atr

    # Net moves
    net1 = (c1["close"] - ref) / atr
    net2 = (c2["close"] - ref) / atr
    net3 = (c3["close"] - ref) / atr

    # Directional consistency
    d1 = np.sign(c1["close"] - c1["open"])
    d2 = np.sign(c2["close"] - c2["open"])
    d3 = np.sign(c3["close"] - c3["open"])

    # Path high/low
    path_high = max(c1["high"], c2["high"], c3["high"])
    path_low  = min(c1["low"],  c2["low"],  c3["low"])
    path_range = (path_high - path_low) / atr

    # Max adverse / favourable excursion (relative to net3 direction)
    if net3 >= 0:
        mae = (ref - path_low)  / atr   # max adverse for a long
        mfe = (path_high - ref) / atr   # max favourable
    else:
        mae = (path_high - ref) / atr
        mfe = (ref - path_low)  / atr

    # Pullback depth: did price retrace after c1?
    if d1 == 1:
        pullback = (c1["high"] - min(c2["low"], c3["low"])) / atr
    else:
        pullback = (max(c2["high"], c3["high"]) - c1["low"]) / atr
    pullback = max(0.0, pullback)

    # Close gravity of c3
    c3_range = c3["high"] - c3["low"] + EPS
    close_grav = (c3["close"] - c3["low"]) / c3_range

    avg_body = (b1 + b2 + b3) / 3

    return {
        "net1": net1, "net2": net2, "net3": net3,
        "b1": b1, "b2": b2, "b3": b3, "avg_body": avg_body,
        "d1": d1, "d2": d2, "d3": d3,
        "path_range": path_range,
        "mae": mae, "mfe": mfe,
        "pullback": pullback,
        "close_grav": close_grav,
        "path_high_atr": (path_high - ref) / atr,
        "path_low_atr":  (path_low  - ref) / atr,
    }
