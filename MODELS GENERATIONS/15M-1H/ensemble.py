# ensemble.py
# Ensemble function library for the 5-Specialist EUR/USD M15 system.
# Import this into any script — no MT5 dependency, no side effects.
#
# Usage:
#   from ensemble import run_ensemble, score_trade, print_report, lot_size_label
#
# ─────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────
# Gate thresholds  (sourced from the trading manual — do NOT change)
# ─────────────────────────────────────────────────────────────────────

MOMENTUM_CONF_THRESHOLD = 0.60   # Gate 2: min directional confidence
STRUCTURE_BUY_MAX       = +0.30  # Gate 3: BUY blocked if E_s  >  +0.30
STRUCTURE_SELL_MIN      = -0.30  # Gate 3: SELL blocked if E_s < -0.30
VOLUME_MIN_INTENSITY    =  1.80  # Gate 4: institutional participation floor
GEOMETRY_BUY_MAX        = -0.15  # Gate 5: BUY needs R_p  <= -0.15
GEOMETRY_SELL_MIN       = +0.15  # Gate 5: SELL needs R_p >= +0.15

# Score thresholds for position sizing
SCORE_SKIP = 40
SCORE_HALF = 60
SCORE_FULL = 80


# ─────────────────────────────────────────────────────────────────────
# score_trade()
# ─────────────────────────────────────────────────────────────────────

def score_trade(m1_result: dict, m2_result: dict, m3_result: dict,
                m4_result: dict, m5_result: dict, direction: str) -> int:
    """
    Returns a signal quality score 0–100 after all 5 gates have passed.

    Weights (from the trading manual):
      Model 1 — momentum conviction   0–25 pts
      Model 3 — breakout strength     0–20 pts
      Model 4 — volume intensity      0–30 pts  ← highest weight
      Model 2 — structural position   0–15 pts
      Model 5 — geometry clarity       0–10 pts

    Position sizing:
      0–39  → skip (weak signal despite gate passage)
      40–59 → 0.5× standard lot
      60–79 → 1.0× standard lot
      80–100→ 1.5× standard lot (high conviction)
    """
    score = 0.0

    # Model 1 — momentum conviction (0–25 pts)
    p_dir = (m1_result["direction"]["probs"].get("bullish", 0.0)
             if direction == "BUY"
             else m1_result["direction"]["probs"].get("bearish", 0.0))
    score += min(25.0, (p_dir - 0.60) / 0.40 * 25.0)

    # Model 3 — breakout strength (0–20 pts)
    p_brk = m3_result["regime"]["p_breakout"]
    thr   = m3_result["regime"].get("threshold", 0.12)
    score += min(20.0, (p_brk - thr) / max(1.0 - thr, 1e-12) * 20.0)

    # Model 4 — volume intensity (0–30 pts)
    intensity = m4_result["volume_intensity"]["I_c"]
    score += min(30.0, (intensity - 1.8) / 3.2 * 30.0)

    # Model 2 — structural position (0–15 pts)
    s = m2_result["structural_position"]["E_s"]
    if direction == "BUY":
        score += min(15.0, (0.30 - s) / 1.30 * 15.0)
    else:
        score += min(15.0, (s + 0.30) / 1.30 * 15.0)

    # Model 5 — geometry clarity (0–10 pts)
    rp = m5_result["rejection_polarity"]["R_p"]
    if direction == "BUY":
        score += min(10.0, abs(min(rp, -0.15) + 0.15) / 0.85 * 10.0)
    else:
        score += min(10.0, abs(max(rp, +0.15) - 0.15) / 0.85 * 10.0)

    return int(round(max(0.0, min(100.0, score))))


def lot_size_label(score: int) -> str:
    """Maps a quality score to a position sizing recommendation."""
    if score < SCORE_SKIP: return "SKIP (score too low)"
    if score < SCORE_HALF: return "0.5× standard lot"
    if score < SCORE_FULL: return "1.0× standard lot"
    return "1.5× standard lot (HIGH CONVICTION)"


# ─────────────────────────────────────────────────────────────────────
# run_ensemble_from_results()  ← primary entry point
# ─────────────────────────────────────────────────────────────────────

def run_ensemble_from_results(r1: dict, r2: dict, r3: dict,
                               r4: dict, r5: dict) -> dict:
    """
    Run the 5-gate cascade on already-computed result dicts.
    Use this when each model has called predict(symbol) independently
    (each fetching its own timeframe / lookback).

    Parameters
    ----------
    r1 : result from MomentumPredictor.predict()
    r2 : result from StructurePredictor.predict()
    r3 : result from VolatilityPredictor.predict()
    r4 : result from VolumePredictor.predict()
    r5 : result from GeometryPredictor.predict()
    """
    # Extract key values
    trade_enabled = r3.get("trade_enabled", False)
    p_bullish     = r1["direction"]["probs"].get("bullish", 0.0)
    p_bearish     = r1["direction"]["probs"].get("bearish", 0.0)
    direction     = "BUY" if p_bullish >= p_bearish else "SELL"
    dir_confirmed = max(p_bullish, p_bearish) >= MOMENTUM_CONF_THRESHOLD
    es            = r2["structural_position"]["E_s"]
    ic            = r4["volume_intensity"]["I_c"]
    rp            = r5["rejection_polarity"]["R_p"]

    def _hold(reason, score=0):
        return {"signal": "HOLD", "reason": reason, "score": score,
                "lot_size": "—", "direction": direction,
                "model_1": r1, "model_2": r2, "model_3": r3,
                "model_4": r4, "model_5": r5}

    # Gate 1 — Volatility (Model 3)
    if not trade_enabled:
        return _hold(f"GATE_1_FAILED · chop detected "
                     f"(p_breakout={r3['regime']['p_breakout']:.4f} < threshold)")

    # Gate 2 — Direction / Momentum (Model 1)
    if not dir_confirmed:
        return _hold(f"GATE_2_FAILED · momentum ambiguous "
                     f"(p_bull={p_bullish:.4f}, p_bear={p_bearish:.4f})")

    # Gate 3 — Structural Location (Model 2)
    if direction == "BUY"  and es > STRUCTURE_BUY_MAX:
        return _hold(f"GATE_3_FAILED · BUY near resistance "
                     f"(E_s={es:+.4f} > +{STRUCTURE_BUY_MAX})")
    if direction == "SELL" and es < STRUCTURE_SELL_MIN:
        return _hold(f"GATE_3_FAILED · SELL near support "
                     f"(E_s={es:+.4f} < {STRUCTURE_SELL_MIN:+.2f})")

    # Gate 4 — Volume Conviction (Model 4)
    if ic < VOLUME_MIN_INTENSITY:
        return _hold(f"GATE_4_FAILED · low institutional volume "
                     f"(I_c={ic:.4f} < {VOLUME_MIN_INTENSITY})")

    # Gate 5 — Geometry Confirmation (Model 5)
    if direction == "BUY"  and rp > GEOMETRY_BUY_MAX:
        return _hold(f"GATE_5_FAILED · no lower-wick absorption "
                     f"(R_p={rp:+.4f} > {GEOMETRY_BUY_MAX})")
    if direction == "SELL" and rp < GEOMETRY_SELL_MIN:
        return _hold(f"GATE_5_FAILED · no upper-wick rejection "
                     f"(R_p={rp:+.4f} < +{GEOMETRY_SELL_MIN})")

    # All gates passed — score the trade
    quality = score_trade(r1, r2, r3, r4, r5, direction)
    if quality < SCORE_SKIP:
        return _hold(f"SCORE_TOO_LOW · all gates passed but score={quality} < {SCORE_SKIP}",
                     score=quality)

    return {"signal": direction, "reason": "ALL_GATES_PASSED",
            "score": quality, "lot_size": lot_size_label(quality),
            "direction": direction,
            "model_1": r1, "model_2": r2, "model_3": r3,
            "model_4": r4, "model_5": r5}


# ─────────────────────────────────────────────────────────────────────
# run_ensemble()  ← convenience wrapper for df-based usage / backtesting
# ─────────────────────────────────────────────────────────────────────

def run_ensemble(df, m1, m2, m3, m4, m5) -> dict:
    """
    Convenience wrapper: runs predict_from_df on each model then calls
    run_ensemble_from_results. Useful for backtesting or CSV-based runs
    where you supply a shared DataFrame.
    """
    r1 = m1.predict_from_df(df)
    r2 = m2.predict_from_df(df)
    r3 = m3.predict_from_df(df)
    r4 = m4.predict_from_df(df)
    r5 = m5.predict_from_df(df)
    return run_ensemble_from_results(r1, r2, r3, r4, r5)


# ─────────────────────────────────────────────────────────────────────
# print_report()
# ─────────────────────────────────────────────────────────────────────

def print_report(result: dict) -> None:
    """Pretty-print the ensemble result to stdout."""
    r1, r2, r3, r4, r5 = (result[f"model_{i}"] for i in range(1, 6))
    sep = "─" * 58

    print(f"\n{sep}")
    print(f"  ENSEMBLE SIGNAL  →  {result['signal']}")
    print(sep)

    p_b = r1["direction"]["probs"].get("bullish", 0.0)
    p_s = r1["direction"]["probs"].get("bearish", 0.0)
    m1_tag = ("✓ BUY"  if p_b >= MOMENTUM_CONF_THRESHOLD else
              "✓ SELL" if p_s >= MOMENTUM_CONF_THRESHOLD else "✗ AMBIGUOUS")
    m3_tag = "✓ OPEN" if r3.get("trade_enabled", False) else "✗ BLOCKED"
    ic     = r4["volume_intensity"]["I_c"]
    m4_tag = "✓ CONFIRMED" if ic >= VOLUME_MIN_INTENSITY else "✗ LOW FLOW"

    print(f"  Gate 1 · Volatility (M3)   {m3_tag:<14} "
          f"p_breakout={r3['regime']['p_breakout']:.4f}")
    print(f"  Gate 2 · Momentum  (M1)   {m1_tag:<14} "
          f"p_bull={p_b:.4f}  p_bear={p_s:.4f}")
    print(f"  Gate 3 · Structure (M2)   "
          f"E_s={r2['structural_position']['E_s']:+.4f}  "
          f"{r2['structural_position']['zone']}")
    print(f"  Gate 4 · Volume    (M4)   {m4_tag:<14} I_c={ic:.4f}")
    print(f"  Gate 5 · Geometry  (M5)   "
          f"R_p={r5['rejection_polarity']['R_p']:+.4f}  "
          f"{r5['rejection_polarity']['interpretation']}")
    print(sep)

    if result["signal"] != "HOLD":
        print(f"  Quality Score  →  {result['score']} / 100")
        print(f"  Position Size  →  {result['lot_size']}")
    else:
        print(f"  Blocked        →  {result['reason']}")
    print(sep + "\n")
