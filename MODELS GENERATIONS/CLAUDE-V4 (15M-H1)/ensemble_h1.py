# ensemble_h1.py
# Weighted ensemble for 5 H1 specialist models.
# Combines model outputs into a single directional signal with confidence.
#
# Rules derived from live calibration:
#   1. Base weight = model importance × model confidence × consistency multiplier
#   2. trade_active=False suppresses that model's weight
#   3. VWAP + Micro agreement = strong prior (+bonus)
#   4. Breakout override only fires if VWAP or Micro agrees with it
#   5. Collective risk-off (3+ inactive): if inactive models all agree on same
#      direction → issue that direction at LOW confidence rather than pure FLAT
#   6. Net weighted edge must exceed threshold for a directional signal
# ─────────────────────────────────────────────────────────────────────
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Literal

Signal = Literal["bullish", "bearish", "flat"]

# ─────────────────────────────────────────────────────────────────────
# Tunable constants
# ─────────────────────────────────────────────────────────────────────

EDGE_THRESHOLD = 0.12        # minimum net weighted edge for directional signal

MODEL_IMPORTANCE = {
    "microstructure": 1.30,
    "vwap":           1.25,
    "breakout":       1.10,
    "momentum":       1.00,
    "candle":         0.90,
}

CONSISTENCY_BOOST   = 0.10
CONSISTENCY_PENALTY = 0.08

VWAP_MICRO_BONUS = 0.20

BREAKOUT_OVERRIDE_THRESHOLD = 0.85

TRADE_INACTIVE_SUPPRESSION = 0.60   # weight multiplier when trade_active=False

# When collective risk-off fires + inactive models all agree:
# emit their direction at this confidence cap instead of FLAT
RISK_OFF_CONSENSUS_CONF = 0.35


# ─────────────────────────────────────────────────────────────────────
# Data container
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ModelOutput:
    name:         str
    signal:       Signal
    conf:         float
    mag:          str
    range_type:   str
    loc:          str
    body:         str
    mom:          str
    trade_active: bool


def from_predict(name: str, r: dict) -> ModelOutput:
    """Build ModelOutput from a deploy script predict() return dict."""
    return ModelOutput(
        name         = name,
        signal       = r["direction"]["label"],
        conf         = r["direction"]["confidence"],
        mag          = r["magnitude"]["label"],
        range_type   = r["range_type"]["label"],
        loc          = r["close_loc"]["label"],
        body         = r["body_ratio"]["label"],
        mom          = r["momentum"]["label"],
        trade_active = r["trade_active"],
    )


# ─────────────────────────────────────────────────────────────────────
# Internal consistency scorer
# ─────────────────────────────────────────────────────────────────────

def _consistency_multiplier(m: ModelOutput) -> float:
    """
    How well do aux heads (loc, body, mom) support the stated direction?
    Returns a multiplier [0.5, ~1.4] applied to that model's weight.
    """
    if m.signal == "flat":
        return 1.0

    bull  = m.signal == "bullish"
    score = 0.0

    if   bull and m.loc == "upper":      score += CONSISTENCY_BOOST
    elif bull and m.loc == "lower":      score -= CONSISTENCY_PENALTY
    elif not bull and m.loc == "lower":  score += CONSISTENCY_BOOST
    elif not bull and m.loc == "upper":  score -= CONSISTENCY_PENALTY

    if   m.body == "strong":  score += CONSISTENCY_BOOST
    elif m.body == "weak":    score -= CONSISTENCY_PENALTY

    if   bull and m.mom == "building":      score += CONSISTENCY_BOOST
    elif bull and m.mom == "fading":        score -= CONSISTENCY_PENALTY
    elif not bull and m.mom == "fading":    score += CONSISTENCY_BOOST
    elif not bull and m.mom == "building":  score -= CONSISTENCY_PENALTY

    return max(0.5, 1.0 + score)


# ─────────────────────────────────────────────────────────────────────
# Ensemble engine
# ─────────────────────────────────────────────────────────────────────

@dataclass
class EnsembleResult:
    signal:        Signal
    confidence:    float
    bull_score:    float
    bear_score:    float
    flat_score:    float
    trade_active:  bool
    reason:        str
    model_weights: dict = field(default_factory=dict)


def ensemble(outputs: list[ModelOutput]) -> EnsembleResult:
    by_name = {m.name: m for m in outputs}

    # ── Step 1: effective weight per model ────────────────────────
    eff_weights: dict[str, float] = {}
    for m in outputs:
        w  = MODEL_IMPORTANCE.get(m.name, 1.0)
        w *= m.conf
        w *= _consistency_multiplier(m)
        if not m.trade_active:
            w *= TRADE_INACTIVE_SUPPRESSION
        eff_weights[m.name] = w

    # ── Step 2: directional scores ────────────────────────────────
    bull_score = sum(eff_weights[m.name] for m in outputs if m.signal == "bullish")
    bear_score = sum(eff_weights[m.name] for m in outputs if m.signal == "bearish")
    flat_score = sum(eff_weights[m.name] for m in outputs if m.signal == "flat")

    def _norm():
        t = bull_score + bear_score + flat_score + 1e-10
        return bull_score / t, bear_score / t
    bull_w, bear_w = _norm()

    # ── Step 3: VWAP + Micro agreement bonus ─────────────────────
    vwap  = by_name.get("vwap")
    micro = by_name.get("microstructure")
    vwap_micro_dir = None

    if vwap and micro and vwap.signal == micro.signal and vwap.signal != "flat":
        vwap_micro_dir = vwap.signal
        if vwap.signal == "bullish": bull_score += VWAP_MICRO_BONUS
        else:                        bear_score += VWAP_MICRO_BONUS
        bull_w, bear_w = _norm()

    # ── Step 4: Breakout override ─────────────────────────────────
    bk = by_name.get("breakout")
    breakout_override = False
    if bk and bk.signal != "flat" and bk.conf >= BREAKOUT_OVERRIDE_THRESHOLD:
        vwap_agrees  = vwap  and vwap.signal  == bk.signal
        micro_agrees = micro and micro.signal == bk.signal
        if vwap_agrees or micro_agrees:
            breakout_override = True
            extra = eff_weights["breakout"] * 0.40
            if bk.signal == "bullish": bull_score += extra
            else:                      bear_score += extra
            bull_w, bear_w = _norm()

    # ── Step 5: Collective risk-off check ─────────────────────────
    inactive_models     = [m for m in outputs if not m.trade_active]
    inactive_count      = len(inactive_models)
    collective_risk_off = inactive_count >= 3

    # Key fix: if inactive models all agree on the SAME direction
    # → that IS a signal (at reduced confidence), not pure flat
    inactive_consensus_dir: Signal | None = None
    if collective_risk_off:
        inactive_signals = {m.signal for m in inactive_models if m.signal != "flat"}
        if len(inactive_signals) == 1:
            inactive_consensus_dir = inactive_signals.pop()

    # ── Step 6: final signal ──────────────────────────────────────
    net_edge     = abs(bull_w - bear_w)
    raw_winner: Signal = "bullish" if bull_w > bear_w else "bearish"
    reasons      = []

    bull_models = [m.name for m in outputs if m.signal == "bullish"]
    bear_models = [m.name for m in outputs if m.signal == "bearish"]
    reasons.append(f"bull={bull_models} bear={bear_models}")
    reasons.append(f"edge={net_edge:.3f} (bull_w={bull_w:.3f} bear_w={bear_w:.3f})")

    if vwap_micro_dir:
        reasons.append(f"VWAP+Micro agree={vwap_micro_dir} (+{VWAP_MICRO_BONUS})")
    if breakout_override:
        reasons.append(f"breakout override ACTIVE conf={bk.conf:.2%}")

    # Determine final signal
    if collective_risk_off and inactive_consensus_dir:
        # Inactive majority all agree → emit their direction at capped confidence
        final_signal: Signal  = inactive_consensus_dir
        final_confidence      = RISK_OFF_CONSENSUS_CONF
        ensemble_trade_active = False   # risk-off consensus = signal only, no trade
        reasons.append(
            f"risk-off consensus: {inactive_count} inactive models all={inactive_consensus_dir}"
            f" → low-conf signal (conf capped at {RISK_OFF_CONSENSUS_CONF})")

    elif collective_risk_off and not inactive_consensus_dir:
        # Inactive models disagree among themselves → genuinely ambiguous
        final_signal          = "flat"
        final_confidence      = 0.0
        ensemble_trade_active = False
        reasons.append(f"collective risk-off ({inactive_count}/5 inactive, no consensus) → FLAT")

    elif net_edge < EDGE_THRESHOLD:
        final_signal          = "flat"
        final_confidence      = round(net_edge, 4)
        ensemble_trade_active = False
        reasons.append(f"edge below threshold ({net_edge:.3f} < {EDGE_THRESHOLD}) → FLAT")

    else:
        final_signal          = raw_winner
        final_confidence      = round(net_edge, 4)
        ensemble_trade_active = net_edge >= EDGE_THRESHOLD + 0.05

    return EnsembleResult(
        signal        = final_signal,
        confidence    = final_confidence,
        bull_score    = round(bull_score, 4),
        bear_score    = round(bear_score, 4),
        flat_score    = round(flat_score, 4),
        trade_active  = ensemble_trade_active,
        reason        = " | ".join(reasons),
        model_weights = {k: round(v, 4) for k, v in eff_weights.items()},
    )


# ─────────────────────────────────────────────────────────────────────
# Logging helper
# ─────────────────────────────────────────────────────────────────────

def log_ensemble(result: EnsembleResult) -> None:
    logging.info("=" * 80)
    logging.info(f"  ENSEMBLE  →  {result.signal.upper():<10}"
                 f"  conf={result.confidence:.3f}"
                 f"  trade_active={result.trade_active}")
    logging.info(f"  scores    bull={result.bull_score:.4f}"
                 f"  bear={result.bear_score:.4f}"
                 f"  flat={result.flat_score:.4f}")
    logging.info(f"  reason    {result.reason}")
    logging.info("  weights:")
    for name, w in result.model_weights.items():
        logging.info(f"    {name:<16} → {w:.4f}")
    logging.info("=" * 80)
