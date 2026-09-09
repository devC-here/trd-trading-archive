"""
===============================================================================
  TRADING ENSEMBLE EXPERT — Inference / Integration Script
  Model : ./smollm2-trading-expert  (LoRA adapters, ~4 MB)
  Base  : HuggingFaceTB/SmolLM2-135M-Instruct (loaded from local cache)
  Output: "BUY" | "SELL" | "HOLD"

  Usage:
      from trading_inference import TradingExpert
      expert = TradingExpert()
      signal = expert.predict(bk, cd, mc, mo, vw)   # pass your model dicts
      print(signal)  # "BUY" / "SELL" / "HOLD"
===============================================================================
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — adjust these paths if needed
# ─────────────────────────────────────────────────────────────────────────────

ADAPTER_DIR = "./smollm2-trading-expert"   # your trained adapter folder
BASE_MODEL  = "HuggingFaceTB/SmolLM2-135M-Instruct"

SYSTEM_PROMPT = (
    "You are a professional trading signal interpreter for a 5-model ensemble system "
    "(Breakout, Candle, Microstructure, Momentum, VWAP). "
    "Given the model output labels for the current bar, respond with exactly one word: "
    "BUY, SELL, or HOLD."
)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER — find model in local HF cache (no internet needed)
# ─────────────────────────────────────────────────────────────────────────────

def _find_cached_model(model_id: str) -> str:
    from huggingface_hub import constants as hf_constants
    folder     = "models--" + model_id.replace("/", "--")
    snap_root  = os.path.join(hf_constants.HF_HUB_CACHE, folder, "snapshots")
    if os.path.isdir(snap_root):
        snaps = sorted(os.listdir(snap_root))
        if snaps:
            return os.path.join(snap_root, snaps[-1])
    return model_id   # fallback — try HF Hub


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────

class TradingExpert:
    """
    Drop-in trading signal predictor.

    Quick start:
        expert = TradingExpert()
        signal = expert.predict(bk, cd, mc, mo, vw)

    Each argument is a dict with the model's head outputs.
    See predict() docstring for exact keys.
    """

    def __init__(self, adapter_dir: str = ADAPTER_DIR, verbose: bool = True):
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)

        if verbose:
            print(f"Loading TradingExpert on {device_str.upper()}...")

        local_path = _find_cached_model(BASE_MODEL)
        base = AutoModelForCausalLM.from_pretrained(
            local_path,
            dtype=torch.float32,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model = PeftModel.from_pretrained(base, adapter_dir)
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter_dir,           # tokenizer was saved here during training
            trust_remote_code=True,
        )

        if verbose:
            print("Ready.\n")

    # ── low-level: raw string input ───────────────────────────────────────────

    def predict_raw(self, signal_text: str) -> str:
        """
        Pass a pre-formatted signal string (5 lines, one per model).
        Returns 'BUY', 'SELL', or 'HOLD'.
        """
        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
            f"<|im_start|>user\n{signal_text}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        decoded = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        # extract first word and uppercase
        word = decoded.split()[0].upper() if decoded.split() else "HOLD"
        return word if word in ("BUY", "SELL", "HOLD") else "HOLD"

    # ── high-level: dict input (plug into your existing system) ──────────────

    def predict(self,
                breakout:       dict,
                candle:         dict,
                microstructure: dict,
                momentum:       dict,
                vwap:           dict) -> str:
        """
        Pass the output dicts from your 5 deployed models.

        Expected keys per model:
          breakout       : next_direction, next_magnitude, next_pattern,
                           next_volatility, next_close_loc, bk_follow_through
          candle         : next_direction, next_body_size, next_pattern,
                           seq_continuation, next_close_grav, next_volatility
          microstructure : next_flow_dir, next_bar_intent, next_magnitude,
                           next_rejection, momentum_follow, next_range_type
          momentum       : next_direction, next_magnitude, next_regime_behaviour,
                           next_volatility, next_structural_bias, next_momentum_quality
          vwap           : next_vwap_dir, next_vol_char, vwap_interaction,
                           next_inst_signal, next_close_location, vwap_follow_through

        Returns: "BUY" | "SELL" | "HOLD"
        """
        signal_text = (
            f"BREAKOUT  - "
            f"direction:{breakout.get('next_direction','flat')} | "
            f"magnitude:{breakout.get('next_magnitude','small')} | "
            f"pattern:{breakout.get('next_pattern','indecision')} | "
            f"volatility:{breakout.get('next_volatility','normal')} | "
            f"close_loc:{breakout.get('next_close_loc','mid')} | "
            f"follow_through:{breakout.get('bk_follow_through','no')}\n"

            f"CANDLE    - "
            f"direction:{candle.get('next_direction','doji')} | "
            f"body_size:{candle.get('next_body_size','small')} | "
            f"pattern:{candle.get('next_pattern','inside_doji')} | "
            f"seq_continuation:{candle.get('seq_continuation','indecision')} | "
            f"close_grav:{candle.get('next_close_grav','mid')} | "
            f"volatility:{candle.get('next_volatility','normal')}\n"

            f"MICRO     - "
            f"flow_dir:{microstructure.get('next_flow_dir','contested')} | "
            f"bar_intent:{microstructure.get('next_bar_intent','absorption')} | "
            f"magnitude:{microstructure.get('next_magnitude','thin')} | "
            f"rejection:{microstructure.get('next_rejection','none')} | "
            f"momentum_follow:{microstructure.get('momentum_follow','partial')} | "
            f"range_type:{microstructure.get('next_range_type','normal')}\n"

            f"MOMENTUM  - "
            f"direction:{momentum.get('next_direction','flat')} | "
            f"magnitude:{momentum.get('next_magnitude','small')} | "
            f"regime:{momentum.get('next_regime_behaviour','chop')} | "
            f"volatility:{momentum.get('next_volatility','normal')} | "
            f"structural_bias:{momentum.get('next_structural_bias','neutral')} | "
            f"momentum_quality:{momentum.get('next_momentum_quality','stable')}\n"

            f"VWAP      - "
            f"vwap_dir:{vwap.get('next_vwap_dir','oscillate_near')} | "
            f"vol_char:{vwap.get('next_vol_char','stable')} | "
            f"interaction:{vwap.get('vwap_interaction','test_return')} | "
            f"inst_signal:{vwap.get('next_inst_signal','neutral')} | "
            f"close_location:{vwap.get('next_close_location','mid')} | "
            f"follow_through:{vwap.get('vwap_follow_through','no')}"
        )
        return self.predict_raw(signal_text)


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST — run this file directly to verify everything works
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    expert = TradingExpert()

    # ── Example 1: plug in real dict outputs from your deploy scripts ─────────
    bk = {"next_direction": "bullish", "next_magnitude": "large",
          "next_pattern": "continuation", "next_volatility": "expanding",
          "next_close_loc": "upper_third", "bk_follow_through": "yes"}

    cd = {"next_direction": "bullish", "next_body_size": "large",
          "next_pattern": "engulf", "seq_continuation": "continuation",
          "next_close_grav": "upper_third", "next_volatility": "expansion"}

    mc = {"next_flow_dir": "bullish", "next_bar_intent": "impulse",
          "next_magnitude": "strong", "next_rejection": "none",
          "momentum_follow": "confirmed", "next_range_type": "outside_expand"}

    mo = {"next_direction": "bullish", "next_magnitude": "large",
          "next_regime_behaviour": "breakout", "next_volatility": "expanding",
          "next_structural_bias": "bullish", "next_momentum_quality": "accelerating"}

    vw = {"next_vwap_dir": "above_away", "next_vol_char": "surge",
          "vwap_interaction": "reclaim_hold", "next_inst_signal": "accumulation",
          "next_close_location": "upper_third", "vwap_follow_through": "confirmed"}

    signal = expert.predict(bk, cd, mc, mo, vw)
    print(f"Strong bullish bar  →  {signal}")   # expected: BUY

    # ── Example 2: chop / no trade ────────────────────────────────────────────
    bk2 = {"next_direction": "flat",    "next_magnitude": "small",
           "next_pattern": "indecision","next_volatility": "contracting",
           "next_close_loc": "mid",     "bk_follow_through": "no"}

    cd2 = {"next_direction": "doji",    "next_body_size": "small",
           "next_pattern": "inside_doji","seq_continuation": "indecision",
           "next_close_grav": "mid",    "next_volatility": "compression"}

    mc2 = {"next_flow_dir": "contested","next_bar_intent": "absorption",
           "next_magnitude": "thin",    "next_rejection": "partial",
           "momentum_follow": "failed", "next_range_type": "inside_compress"}

    mo2 = {"next_direction": "flat",    "next_magnitude": "small",
           "next_regime_behaviour": "chop","next_volatility": "contracting",
           "next_structural_bias": "neutral","next_momentum_quality": "fading"}

    vw2 = {"next_vwap_dir": "oscillate_near","next_vol_char": "declining",
           "vwap_interaction": "test_return","next_inst_signal": "neutral",
           "next_close_location": "mid","vwap_follow_through": "no"}

    signal2 = expert.predict(bk2, cd2, mc2, mo2, vw2)
    print(f"Chop / no edge bar  →  {signal2}")  # expected: HOLD

    # ── Example 3: raw string (if you prefer to format the prompt yourself) ───
    raw = (
        "BREAKOUT  - direction:bearish | magnitude:large | pattern:continuation | "
        "volatility:expanding | close_loc:lower_third | follow_through:yes\n"
        "CANDLE    - direction:bearish | body_size:large | pattern:engulf | "
        "seq_continuation:continuation | close_grav:lower_third | volatility:expansion\n"
        "MICRO     - flow_dir:bearish | bar_intent:impulse | magnitude:strong | "
        "rejection:none | momentum_follow:confirmed | range_type:outside_expand\n"
        "MOMENTUM  - direction:bearish | magnitude:large | regime:breakout | "
        "volatility:expanding | structural_bias:bearish | momentum_quality:accelerating\n"
        "VWAP      - vwap_dir:below_away | vol_char:surge | interaction:rejection | "
        "inst_signal:distribution | close_location:lower_third | follow_through:confirmed"
    )
    signal3 = expert.predict_raw(raw)
    print(f"Strong bearish bar  →  {signal3}")  # expected: SELL
