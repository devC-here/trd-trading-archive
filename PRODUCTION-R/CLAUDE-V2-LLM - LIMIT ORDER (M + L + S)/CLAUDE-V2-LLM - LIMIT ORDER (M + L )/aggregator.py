"""
aggregator.py  —  Ensemble aggregator using the fine-tuned TradingExpert model.

The fine-tuned model (BUY / SELL / HOLD) replaces the hand-coded direction /
kill-switch logic.  The raw prediction dicts from all 5 deploy scripts are
translated into the exact prompt format the model was trained on, then the
model's single-word answer is returned.

The original confidence score (rule-based, 0-1) is kept alongside so the
rest of the pipeline (main.py, sltp, placer) stays unchanged.

Assumes the 4 adapter files sit in the same directory as this file:
    adapter_config.json
    adapter_model.safetensors
    tokenizer.json
    tokenizer_config.json
"""

import os
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logger = logging.getLogger(__name__)

# ── adapter files live next to this script ────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
ADAPTER_DIR  = _HERE                                        # same directory
BASE_MODEL   = "HuggingFaceTB/SmolLM2-135M-Instruct"

SYSTEM_PROMPT = (
    "You are a professional trading signal interpreter for a 5-model ensemble system "
    "(Breakout, Candle, Microstructure, Momentum, VWAP). "
    "Given the model output labels for the current bar, respond with exactly one word: "
    "BUY, SELL, or HOLD."
)

# ── singleton — loaded once, reused forever ───────────────────────────────────
_model     = None
_tokenizer = None
_device    = None


def _load_once():
    global _model, _tokenizer, _device

    if _model is not None:
        return

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("[TradingExpert] Loading model on %s …", str(_device).upper())

    # find base model in local HF cache — no internet needed
    from huggingface_hub import constants as hf_constants
    folder    = "models--" + BASE_MODEL.replace("/", "--")
    snap_root = os.path.join(hf_constants.HF_HUB_CACHE, folder, "snapshots")
    local_base = BASE_MODEL
    if os.path.isdir(snap_root):
        snaps = sorted(os.listdir(snap_root))
        if snaps:
            local_base = os.path.join(snap_root, snaps[-1])

    base = AutoModelForCausalLM.from_pretrained(
        local_base,
        dtype=torch.float32,
        trust_remote_code=True,
        local_files_only=True,
    )
    _model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    _model.to(_device)
    _model.eval()

    _tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_DIR, trust_remote_code=True
    )

    logger.info("[TradingExpert] Model ready.")


# ── label extractor helpers ───────────────────────────────────────────────────

def _get(d: dict, key: str, default: str = "flat") -> str:
    """Safely pull a label from a raw predictor output dict."""
    val = d.get(key)
    if val is None:
        return default
    # deploy scripts return either {"label": "..."} or plain strings
    if isinstance(val, dict):
        return val.get("label", default)
    return str(val)


def _build_signal_text(raw: dict) -> str:
    """Convert the raw 5-model output dict into the prompt the model was trained on."""

    b  = raw.get("breakout")       or {}
    cd = raw.get("candle")         or {}
    mc = raw.get("microstructure") or {}
    mo = raw.get("momentum")       or {}
    vw = raw.get("vwap")           or {}

    return (
        f"BREAKOUT  - "
        f"direction:{_get(b,  'next_direction',      'flat')} | "
        f"magnitude:{_get(b,  'next_magnitude',      'small')} | "
        f"pattern:{_get(b,    'next_pattern',        'indecision')} | "
        f"volatility:{_get(b, 'next_volatility',     'normal')} | "
        f"close_loc:{_get(b,  'next_close_loc',      'mid')} | "
        f"follow_through:{_get(b, 'bk_follow_through', 'no')}\n"

        f"CANDLE    - "
        f"direction:{_get(cd, 'next_direction',      'doji')} | "
        f"body_size:{_get(cd, 'next_body_size',      'small')} | "
        f"pattern:{_get(cd,   'next_pattern',        'inside_doji')} | "
        f"seq_continuation:{_get(cd, 'seq_continuation', 'indecision')} | "
        f"close_grav:{_get(cd,'next_close_grav',     'mid')} | "
        f"volatility:{_get(cd,'next_volatility',     'normal')}\n"

        f"MICRO     - "
        f"flow_dir:{_get(mc,  'next_flow_dir',       'contested')} | "
        f"bar_intent:{_get(mc,'next_bar_intent',     'absorption')} | "
        f"magnitude:{_get(mc, 'next_magnitude',      'thin')} | "
        f"rejection:{_get(mc, 'next_rejection',      'none')} | "
        f"momentum_follow:{_get(mc, 'momentum_follow', 'partial')} | "
        f"range_type:{_get(mc,'next_range_type',     'normal')}\n"

        f"MOMENTUM  - "
        f"direction:{_get(mo, 'next_direction',      'flat')} | "
        f"magnitude:{_get(mo, 'next_magnitude',      'small')} | "
        f"regime:{_get(mo,    'next_regime_behaviour','chop')} | "
        f"volatility:{_get(mo,'next_volatility',     'normal')} | "
        f"structural_bias:{_get(mo, 'next_structural_bias', 'neutral')} | "
        f"momentum_quality:{_get(mo, 'next_momentum_quality', 'stable')}\n"

        f"VWAP      - "
        f"vwap_dir:{_get(vw,  'next_vwap_dir',       'oscillate_near')} | "
        f"vol_char:{_get(vw,  'next_vol_char',        'stable')} | "
        f"interaction:{_get(vw,'vwap_interaction',   'test_return')} | "
        f"inst_signal:{_get(vw,'next_inst_signal',   'neutral')} | "
        f"close_location:{_get(vw,'next_close_location','mid')} | "
        f"follow_through:{_get(vw,'vwap_follow_through','no')}"
    )


def _model_decision(signal_text: str) -> str:
    """Run the fine-tuned model and return 'BUY', 'SELL', or 'HOLD'."""
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
        f"<|im_start|>user\n{signal_text}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = _tokenizer(prompt, return_tensors="pt").to(_device)
    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=_tokenizer.eos_token_id,
        )
    decoded = _tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    word = decoded.split()[0].upper() if decoded.split() else "HOLD"
    return word if word in ("BUY", "SELL", "HOLD") else "HOLD"


# ── confidence score (rule-based, kept for sltp / position sizing) ────────────

def _confidence_score(raw: dict) -> float:
    """
    Original rule-based confirmation score (0–1).
    Used by main.py for position sizing / logging — not for direction.
    """
    confirm_map = {
        "yes": 2, "confirmed": 2, "partial": 1,
        "continuation": 2, "stable": 1, "accelerating": 2,
        "fading": -2, "failed": -2, "no": 0, "indecision": 0,
    }
    score = 0

    mo = raw.get("momentum") or {}
    if mo.get("trade_active"):
        score += confirm_map.get(_get(mo, "next_momentum_quality"), 0)

    b = raw.get("breakout") or {}
    if b.get("trade_active"):
        score += confirm_map.get(_get(b, "bk_follow_through"), 0)

    mc = raw.get("microstructure") or {}
    if mc.get("trade_active"):
        score += confirm_map.get(_get(mc, "momentum_follow"), 0)

    vw = raw.get("vwap") or {}
    if vw.get("trade_active"):
        score += confirm_map.get(_get(vw, "vwap_follow_through"), 0)

    cd = raw.get("candle") or {}
    if cd.get("trade_active"):
        score += confirm_map.get(_get(cd, "seq_continuation"), 0)

    return max(0.0, min(1.0, score / 10))


# ── public API ────────────────────────────────────────────────────────────────

def aggregate_predictions(raw: dict, min_conf: float = 0.0) -> dict:
    """
    Drop-in replacement for the original aggregator.

    Uses the fine-tuned model for BUY / SELL / HOLD decision.
    Returns the same dict shape as before:
        {"signal": "bullish"|"bearish"|None, "confidence": float, "trade_active": bool}
    """
    _load_once()

    signal_text = _build_signal_text(raw)
    decision    = _model_decision(signal_text)
    conf        = _confidence_score(raw)

    logger.info(
        "[TradingExpert] model=%s | conf=%.2f | prompt_snippet=%.80s …",
        decision, conf, signal_text.replace("\n", " | ")
    )

    if decision == "HOLD" or conf < min_conf:
        return {"signal": None, "confidence": conf, "trade_active": False}

    signal = "bullish" if decision == "BUY" else "bearish"
    return {"signal": signal, "confidence": conf, "trade_active": True}
