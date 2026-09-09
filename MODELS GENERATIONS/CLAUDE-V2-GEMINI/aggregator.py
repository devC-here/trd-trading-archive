"""
aggregator.py — Gemini-powered ensemble aggregator (FINAL CLEAN VERSION)

- Uses new google-genai SDK
- Uses structured + probability-aware prompt
- Safe fallback (never crashes trading loop)
- Same output format as before
"""

import logging
from google import genai

logger = logging.getLogger(__name__)

# =========================
# CONFIG
# =========================
client = genai.Client(api_key="AIzaSyCraR3GRXVHLhYTzLLZozhZfT-qESjtmFo")

MODEL_NAME = "gemini-2.5-flash-lite"


# =========================
# SYSTEM PROMPT
# =========================
SYSTEM_PROMPT = """
You are an expert trading decision system for a 5-model ensemble:
Breakout, Momentum, Microstructure, VWAP, Candle.

Return ONLY ONE word: BUY, SELL, or HOLD.

Rules:
- Use probability strength (not just labels)
- Strong agreement → BUY/SELL
- Strong single model dominance is valid
- Weak/conflicting signals → HOLD
- Follow-through = strong confirmation
- Avoid trading in uncertainty/chop
"""


# =========================
# HELPERS
# =========================
def _get_probs(d, key):
    val = d.get(key, {})
    if isinstance(val, dict):
        return val.get("probs", {})
    return {}

def _fmt_probs(p):
    if not p:
        return "none"
    return ", ".join(f"{k}={round(v,3)}" for k, v in p.items())

def _normalize_vwap(signal):
    if signal in ("below_away", "rejection"):
        return "bearish"
    elif signal in ("above_away", "reclaim_hold"):
        return "bullish"
    return "neutral"


# =========================
# BUILD PROMPT
# =========================
def _build_signal_text(raw):

    b  = raw.get("breakout", {})
    mo = raw.get("momentum", {})
    mc = raw.get("microstructure", {})
    vw = raw.get("vwap", {})
    cd = raw.get("candle", {})

    return f"""
BREAKOUT:
direction_probs: {_fmt_probs(_get_probs(b, 'next_direction'))}
confidence: {b.get('confidence', 0)}
pattern: {b.get('next_pattern', {}).get('label')}
follow_through: {b.get('follow_through')}

MOMENTUM:
direction_probs: {_fmt_probs(_get_probs(mo, 'next_direction'))}
confidence: {mo.get('_summary', {}).get('signal_conf', 0)}
regime: {mo.get('_summary', {}).get('regime')}
structure: {mo.get('_summary', {}).get('structure')}
momentum_quality: {mo.get('_summary', {}).get('momentum')}

MICROSTRUCTURE:
flow_probs: {_fmt_probs(_get_probs(mc, 'next_flow_dir'))}
confidence: {mc.get('confidence', 0)}
intent: {mc.get('next_bar_intent', {}).get('label')}
range_type: {mc.get('next_range_type', {}).get('label')}
follow: {mc.get('follow')}

VWAP:
bias: {_normalize_vwap(vw.get('signal'))}
confidence: {vw.get('confidence', 0)}
interaction: {vw.get('vwap_interaction', {}).get('label')}
follow_through: {vw.get('follow_through')}

CANDLE:
direction_probs: {_fmt_probs(_get_probs(cd, 'next_direction'))}
confidence: {cd.get('confidence', 0)}
pattern: {cd.get('next_pattern', {}).get('label')}
continuation: {cd.get('continuation')}
"""


# =========================
# GEMINI DECISION
# =========================
def _gemini_decision(signal_text):

    prompt = f"{SYSTEM_PROMPT}\n\nDATA:\n{signal_text}"

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        text = (response.text or "").strip().upper()

        if "BUY" in text:
            return "BUY"
        elif "SELL" in text:
            return "SELL"
        else:
            return "HOLD"

    except Exception as e:
        logger.error("Gemini error: %s", str(e))
        return "HOLD"


# =========================
# CONFIDENCE (UNCHANGED STYLE)
# =========================
def _confidence_score(raw):

    confirm_map = {
        "yes": 2, "confirmed": 2, "partial": 1,
        "continuation": 2, "stable": 1, "accelerating": 2,
        "fading": -2, "failed": -2, "no": 0, "indecision": 0,
    }

    score = 0

    mo = raw.get("momentum") or {}
    if mo.get("trade_active"):
        score += confirm_map.get(
            mo.get("_summary", {}).get("momentum"), 0
        )

    b = raw.get("breakout") or {}
    if b.get("trade_active"):
        score += confirm_map.get(b.get("follow_through"), 0)

    mc = raw.get("microstructure") or {}
    if mc.get("trade_active"):
        score += confirm_map.get(mc.get("follow"), 0)

    vw = raw.get("vwap") or {}
    if vw.get("trade_active"):
        score += confirm_map.get(vw.get("follow_through"), 0)

    cd = raw.get("candle") or {}
    if cd.get("trade_active"):
        score += confirm_map.get(cd.get("continuation"), 0)

    return max(0.0, min(1.0, score / 10))


# =========================
# MAIN FUNCTION
# =========================
def aggregate_predictions(raw: dict, min_conf: float = 0.0):

    signal_text = _build_signal_text(raw)

    decision = _gemini_decision(signal_text)
    conf = _confidence_score(raw)

    logger.info("[Gemini] decision=%s | conf=%.2f", decision, conf)

    # Safety: avoid trading on weak/confused signals
    if decision == "HOLD" or conf < min_conf:
        return {"signal": None, "confidence": conf, "trade_active": False}

    signal = "bullish" if decision == "BUY" else "bearish"

    return {
        "signal": signal,
        "confidence": conf,
        "trade_active": True
    }