   
import logging
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# =========================
# CONFIG
# =========================
# Using the recommended model for reasoning and complex structured tasks
MODEL_NAME = "gemini-2.5-flash-lite" 
client = genai.Client(api_key="AIzaSyCraR3GRXVHLhYTzLLZozhZfT-qESjtmFo")

# =========================
# STRUCTURAL PROTOCOL (Pydantic Schema)
# =========================
class EnsembleDecision(BaseModel):
    decision: str = Field(description="Must be exactly one of: BUY, SELL, HOLD")
    primary_bias: str = Field(description="Overall structural market bias: BULLISH, BEARISH, or NEUTRAL")
    conviction_score: float = Field(description="Internal model confidence rating from 0.0 to 1.0")
    reasoning_summary: str = Field(description="A concise 1-sentence synthesis of why this choice was made across the timesteps.")

# =========================
# SYSTEM PROMPT
# =========================
SYSTEM_PROMPT = """
You are an advanced, institutional-grade trading consensus engine mapping a 5-model ensemble over time.
Sub-models: Breakout, Momentum, Microstructure, VWAP, Candle.

You will receive historical data chunks ordered chronologically from past to present (t-3 -> t-2 -> t-1 -> t0).
t0 represents the current/fresh execution block.

Core Directives:
1. Prioritize Trend Transitions: Analyze how confidence shifts from past blocks to t0. A dying trend (declining confidence) combined with an emerging opposite flow is highly predictive.
2. Order Flow > Macro Trailing Labels: When high-level macro models (Momentum/Candle) show weak/flat responses, defer to Microstructure and VWAP lead indicators.
3. Handle VWAP Rejection Contextually: If price action pulls back to VWAP from above and shows a 'rejection', this is a bullish defense of value, NOT a bearish signal.
4. Volatility Check: High conflict across all sub-models with falling confidence is uncertainty. Outout HOLD.
"""

# =========================
# HELPERS & NORMALIZATION
# =========================
def _get_probs(d: dict, key: str) -> dict:
    val = d.get(key, {})
    if isinstance(val, dict):
        return val.get("probs", {})
    return {}

def _fmt_probs(p: dict) -> str:
    if not p:
        return "none"
    return ", ".join(f"{k}={round(v,3)}" for k, v in p.items())

def _contextual_vwap_bias(vwap_data: dict) -> str:
    """Fixes the flawed logic by interpreting VWAP direction alongside interaction type."""
    signal = vwap_data.get('signal', '').lower()
    interaction = vwap_data.get('vwap_interaction', {}).get('label', '').lower()
    
    if "above_away" in signal or "reclaim" in signal:
        return "bullish"
    if "below_away" in signal:
        # A rejection while below away means it tried to break up but failed (Bearish)
        # A rejection while pulling back from above away means support held (Bullish)
        if "rejection" in interaction:
            return "bullish_defense" 
        return "bearish"
    return "neutral"

# =========================
# TIME-SERIES PROMPT BUILDER
# =========================
def _build_chronological_prompt(history_payload: List[Dict[str, Any]]) -> str:
    """
    Accepts a list of raw state dicts ordered chronologically:
    Index 0 = oldest (t-3), Index -1 = newest/current (t0).
    """
    prompt_blocks = []
    
    # Enforce chronological ordering map visually for the LLM
    total_steps = len(history_payload)
    for i, raw in enumerate(history_payload):
        # Calculate steps relative to current (t0)
        time_label = f"t{- (total_steps - 1 - i)}" if i < total_steps - 1 else "t0 (CURRENT_STATE)"
        
        b  = raw.get("breakout", {})
        mo = raw.get("momentum", {})
        mc = raw.get("microstructure", {})
        vw = raw.get("vwap", {})
        cd = raw.get("candle", {})
        
        block = f"""
=== TIMESTEP: {time_label} ===
[BREAKOUT MODEL]
  Direction Probs: {_fmt_probs(_get_probs(b, 'next_direction'))}
  Confidence: {b.get('confidence', 0)}% | Pattern: {b.get('next_pattern', {}).get('label')}
  Follow Through: {b.get('follow_through')} | Active: {b.get('trade_active')}

[MOMENTUM MODEL]
  Direction Probs: {_fmt_probs(_get_probs(mo, 'next_direction'))}
  Confidence: {mo.get('_summary', {}).get('signal_conf', 0)}%
  Regime: {mo.get('_summary', {}).get('regime')} | Quality: {mo.get('_summary', {}).get('momentum')}

[MICROSTRUCTURE MODEL]
  Flow Probs: {_fmt_probs(_get_probs(mc, 'next_flow_dir'))}
  Confidence: {mc.get('confidence', 0)}% | Intent: {mc.get('next_bar_intent', {}).get('label')}
  Follow Profile: {mc.get('follow')}

[VWAP MODEL]
  Contextual Bias: {_contextual_vwap_bias(vw)}
  Confidence: {vw.get('confidence', 0)}% | Interaction: {vw.get('vwap_interaction', {}).get('label')}
  Institutional Follow: {vw.get('follow_through')}

[CANDLE SEQUENCE MODEL]
  Direction Probs: {_fmt_probs(_get_probs(cd, 'next_direction'))}
  Confidence: {cd.get('confidence', 0)}% | Structure: {cd.get('continuation')}
"""
        prompt_blocks.append(block)
        
    return "\n".join(prompt_blocks)

# =========================
# GEMINI ENGINE WITH STRUCTURED OUTPUT
# =========================
def _gemini_structured_decision(signal_text: str) -> EnsembleDecision:
    prompt = f"DATA TIMELINE:\n{signal_text}"
    
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=EnsembleDecision,
                temperature=0.1, # Keep it deterministic for trading consistency
            ),
        )
        # Parse the structured JSON response automatically back into our Pydantic schema
        return EnsembleDecision.model_validate_json(response.text)
        
    except Exception as e:
        logger.error("Structured Gemini execution failed: %s", str(e))
        # Hard failure safety valve
        return EnsembleDecision(
            decision="HOLD", 
            primary_bias="NEUTRAL", 
            conviction_score=0.0, 
            reasoning_summary=f"System Exception Handled: {str(e)}"
        )

# =========================
# RUNTIME ENTRY POINT
# =========================
def aggregate_predictions_timeseries(history_payload: List[Dict[str, Any]], min_conf: float = 0.40) -> dict:
    """
    Pass a list of your parsed sub-model frames over time.
    Example: [t_minus_3_dict, t_minus_2_dict, t_minus_1_dict, current_t0_dict]
    """
    if not history_payload:
        return {"signal": None, "confidence": 0.0, "trade_active": False, "meta": "Empty history payload passed."}
        
    # 1. Compile the time sequence string
    signal_text = _build_chronological_prompt(history_payload)
    
    # 2. Get Structured Engine Prediction
    structured_res = _gemini_structured_decision(signal_text)
    
    decision = structured_res.decision.upper()
    conf = structured_res.conviction_score
    
    logger.info("[Ensemble AI Engine] Decision: %s | Bias: %s | Conviction: %.2f", 
                decision, structured_res.primary_bias, conf)
    logger.info("[AI Reasoning]: %s", structured_res.reasoning_summary)
    
    # 3. Execution Rule Validation
    if decision == "HOLD" or conf < min_conf:
        return {
            "signal": None, 
            "confidence": conf, 
            "trade_active": False,
            "reason": structured_res.reasoning_summary
        }
        
    return {
        "signal": "bullish" if decision == "BUY" else "bearish",
        "confidence": conf,
        "trade_active": True,
        "reason": structured_res.reasoning_summary
    }