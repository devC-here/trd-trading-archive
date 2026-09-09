# override_engine.py
def check_override_conditions(momentum, breakout, micro, vwap, candle):
    conditions = []

    # --- CATEGORY: Microstructure Traps & Bursts (Recalibrated) ---
    if micro["micro_risk_zone"] == 0:
        return {
            "condition_id": 1,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 0.95,
            "entry_viability": "low",
            "supporting_factors": ["❌ Trap zone detected (microstructure) — avoid in calm EURUSD"]
        }

    if micro["burst_type"] == 1 and momentum["entry_bias"] == 1 and vwap["vwap_zone"] == 2:
        return {
            "condition_id": 2,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.7,
            "entry_viability": "medium",
            "supporting_factors": ["🚀 Micro burst + bullish bias + VWAP support"]
        }

    if micro["burst_type"] == 1 and momentum["entry_bias"] == 0 and vwap["vwap_zone"] == 0:
        return {
            "condition_id": 3,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.7,
            "entry_viability": "medium",
            "supporting_factors": ["🚨 Micro burst + bearish bias + VWAP resistance"]
        }

    # --- CATEGORY: Squeeze & Breakout Alignment (Recalibrated) ---
    if breakout["squeeze_state"] == 1:
        if breakout["breakout_bias"] == 2 and momentum["entry_bias"] == 1:
            return {
                "condition_id": 4,
                "final_bias": "long",
                "suggested_trade_direction": "Buy",
                "confidence": 0.8,
                "entry_viability": "medium",
                "supporting_factors": ["⚡ Breakout squeeze + bullish momentum confirmation"]
            }
        elif breakout["breakout_bias"] == 0 and momentum["entry_bias"] == 0:
            return {
                "condition_id": 5,
                "final_bias": "short",
                "suggested_trade_direction": "Sell",
                "confidence": 0.8,
                "entry_viability": "medium",
                "supporting_factors": ["⚡ Breakout squeeze + bearish momentum confirmation"]
            }

    # --- CATEGORY: Momentum Conflicts (Recalibrated) ---
    if momentum["entry_bias"] == 1 and vwap["vwap_zone"] == 0 and vwap["volume_pressure"] == 0:
        return {
            "condition_id": 6,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 0.95,
            "entry_viability": "low",
            "supporting_factors": ["🧊 Bullish momentum trapped under VWAP + no volume (sideways drift)"]
        }

    if momentum["entry_bias"] == 1 and candle["sequence_type"] == 2:
        return {
            "condition_id": 7,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 0.9,
            "entry_viability": "low",
            "supporting_factors": ["⚠️ Bullish momentum vs bearish candle conflict"]
        }

    # --- CATEGORY: High Confidence Alignments (Recalibrated) ---
    if (momentum["entry_bias"] == 1 and
        breakout["breakout_bias"] == 2 and
        candle["sequence_type"] in [1, 3] and candle["candle_dominance"] == 2):
        return {
            "condition_id": 8,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.8,
            "entry_viability": "high",
            "supporting_factors": ["🔥 Triple alignment (long) — toned down for EURUSD calm"]
        }

    if (momentum["entry_bias"] == 0 and
        breakout["breakout_bias"] == 0 and
        candle["sequence_type"] in [2, 3] and candle["candle_dominance"] == 0):
        return {
            "condition_id": 9,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.8,
            "entry_viability": "high",
            "supporting_factors": ["🔥 Triple alignment (short) — toned down for EURUSD calm"]
        }

    # --- CATEGORY: VWAP Conflict or Support (Recalibrated) ---
    if vwap["vwap_zone"] == 2 and candle["sequence_type"] == 2 and candle["candle_dominance"] == 0:
        return {
            "condition_id": 10,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.75,
            "entry_viability": "medium",
            "supporting_factors": ["📉 VWAP rejection + bearish candle (less forceful in calm)"]
        }

    if vwap["vwap_zone"] == 0 and candle["sequence_type"] == 1 and micro["tail_pressure_direction"] == 2:
        return {
            "condition_id": 11,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.7,
            "entry_viability": "medium",
            "supporting_factors": ["📈 Pullback + bullish wick reversal near VWAP"]
        }

    # --- CATEGORY: Indecision + Neutrality (Recalibrated) ---
    if vwap["vwap_zone"] == 1 and candle["pattern_strength"] == 0 and micro["micro_risk_zone"] == 1:
        return {
            "condition_id": 12,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 1.0,
            "entry_viability": "low",
            "supporting_factors": ["🚫 Neutral VWAP + weak candle + micro indecision — strict no_trade"]
        }

    # --- CATEGORY: Long Bias Conditions (Recalibrated) ---
    if (momentum["entry_bias"] == 1 and breakout["breakout_bias"] == 2 and candle["sequence_type"] == 1):
        return {
            "condition_id": 13,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.75,
            "entry_viability": "medium",
            "supporting_factors": ["⬆️ Momentum + breakout + bullish continuation (softened for calm)"]
        }

    if (breakout["squeeze_state"] == 1 and breakout["breakout_bias"] == 2 and micro["burst_type"] == 1):
        return {
            "condition_id": 14,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.8,
            "entry_viability": "medium",
            "supporting_factors": ["💥 Breakout squeeze + long bias + micro burst (less weight)"]
        }

    if (candle["sequence_type"] == 1 and vwap["vwap_zone"] == 2 and vwap["volume_pressure"] == 2):
        return {
            "condition_id": 15,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.8,
            "entry_viability": "high",
            "supporting_factors": ["📊 Bullish reversal + VWAP above + high volume"]
        }

    if (candle["sequence_type"] == 3 and momentum["entry_bias"] == 1 and vwap["vwap_zone"] == 2):
        return {
            "condition_id": 16,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.7,
            "entry_viability": "medium",
            "supporting_factors": ["📈 Continuation + momentum up + VWAP support"]
        }

    if (candle["sequence_type"] == 1 and micro["tail_pressure_direction"] == 2 and micro["micro_risk_zone"] == 2):
        return {
            "condition_id": 17,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.75,
            "entry_viability": "medium",
            "supporting_factors": ["🔄 Bullish reversal + micro tail support + liquidity imbalance"]
        }

    if (vwap["vwap_zone"] == 2 and vwap["volume_pressure"] == 2 and candle["candle_dominance"] == 2):
        return {
            "condition_id": 18,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.8,
            "entry_viability": "high",
            "supporting_factors": ["💪 VWAP support + high volume + bullish candle"]
        }

    if (momentum["entry_bias"] == 1 and candle["sequence_type"] == 3 and micro["tail_pressure_direction"] == 2):
        return {
            "condition_id": 19,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.75,
            "entry_viability": "medium",
            "supporting_factors": ["🚀 Momentum up + continuation + bullish tail pressure"]
        }

    if (momentum["momentum_state"] == 0 and candle["sequence_type"] == 1 and vwap["vwap_zone"] == 2):
        return {
            "condition_id": 20,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.7,
            "entry_viability": "medium",
            "supporting_factors": ["📉 Oversold exhaustion + bullish engulfing + VWAP re-entry"]
        }
        # Momentum slope zone 2 + continuation + volume spike
    if (momentum["momentum_slope_zone"] == 2 and candle["sequence_type"] == 3 and vwap["volume_pressure"] == 2):
        return {
            "condition_id": 21,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.75,
            "entry_viability": "medium",
            "supporting_factors": ["📈 Strong momentum slope + continuation + volume spike (softer in EURUSD)"]
        }

    # Tail-dominant bar + continuation + microstructure support
    if (candle["candle_dominance"] == 2 and candle["sequence_type"] == 3 and micro["micro_risk_zone"] == 2):
        return {
            "condition_id": 22,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.7,
            "entry_viability": "medium",
            "supporting_factors": ["🕯️ Tail-dominant bar + continuation + micro support"]
        }

    # Double confirmation: bullish candle + bullish VWAP
    if (candle["candle_dominance"] == 2 and vwap["vwap_zone"] == 2):
        return {
            "condition_id": 23,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.8,
            "entry_viability": "high",
            "supporting_factors": ["✅ Bullish candle + VWAP confirmation (EURUSD steady bias)"]
        }

    # Breakout bias up + strong candle + micro burst
    if (breakout["breakout_bias"] == 2 and candle["pattern_strength"] == 2 and micro["burst_type"] == 1):
        return {
            "condition_id": 24,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.8,
            "entry_viability": "medium",
            "supporting_factors": ["⚡ Breakout up + strong candle + micro burst (less forceful for calm)"]
        }

    # Candle dominance + VWAP above + tail pressure up
    if (candle["candle_dominance"] == 2 and vwap["vwap_zone"] == 2 and micro["tail_pressure_direction"] == 2):
        return {
            "condition_id": 25,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.75,
            "entry_viability": "medium",
            "supporting_factors": ["💪 Bullish dominance + VWAP support + tail pressure up"]
        }

    # Momentum low state but slope strong + confirmed candle
    if (momentum["momentum_state"] == 0 and momentum["momentum_slope_zone"] == 2 and candle["candle_dominance"] == 2):
        return {
            "condition_id": 26,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.65,
            "entry_viability": "medium",
            "supporting_factors": ["📈 Recovery: momentum low but slope strong + bullish candle"]
        }

    # Bullish candle + squeeze released + micro confirm
    if (candle["candle_dominance"] == 2 and breakout["squeeze_state"] == 2 and micro["micro_risk_zone"] == 2):
        return {
            "condition_id": 27,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.75,
            "entry_viability": "medium",
            "supporting_factors": ["✅ Squeeze release + bullish candle + micro confirm"]
        }

    # --- CATEGORY: Short Bias Conditions (Recalibrated) ---

    # Momentum down + breakout down + bearish continuation
    if (momentum["entry_bias"] == 0 and breakout["breakout_bias"] == 0 and candle["sequence_type"] == 2):
        return {
            "condition_id": 28,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.75,
            "entry_viability": "medium",
            "supporting_factors": ["⬇️ Momentum down + breakout down + bearish continuation"]
        }

    # Breakout squeeze + short bias + micro burst
    if (breakout["squeeze_state"] == 1 and breakout["breakout_bias"] == 0 and micro["burst_type"] == 1):
        return {
            "condition_id": 29,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.8,
            "entry_viability": "medium",
            "supporting_factors": ["💥 Bearish squeeze + short bias + micro burst"]
        }

    # Bearish reversal + VWAP below + volume climax
    if (candle["sequence_type"] == 2 and vwap["vwap_zone"] == 0 and vwap["volume_pressure"] == 2):
        return {
            "condition_id": 30,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.75,
            "entry_viability": "medium",
            "supporting_factors": ["📊 Bearish reversal + VWAP resistance + volume climax"]
        }

    # Continuation + momentum down + VWAP resistance
    if (candle["sequence_type"] == 3 and momentum["entry_bias"] == 0 and vwap["vwap_zone"] == 0):
        return {
            "condition_id": 31,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.7,
            "entry_viability": "medium",
            "supporting_factors": ["📉 Continuation + bearish momentum + VWAP resistance"]
        }

    # Bearish reversal + tail down + liquidity drain
    if (candle["sequence_type"] == 2 and micro["tail_pressure_direction"] == 0 and micro["micro_risk_zone"] == 0):
        return {
            "condition_id": 32,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.75,
            "entry_viability": "medium",
            "supporting_factors": ["🔄 Bearish reversal + tail down + liquidity drain"]
        }

    # VWAP below + volume pressure + bearish candle
    if (vwap["vwap_zone"] == 0 and vwap["volume_pressure"] == 2 and candle["candle_dominance"] == 0):
        return {
            "condition_id": 33,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.8,
            "entry_viability": "high",
            "supporting_factors": ["💪 VWAP resistance + high volume + bearish candle"]
        }

    # Momentum down + continuation + bearish tail
    if (momentum["entry_bias"] == 0 and candle["sequence_type"] == 3 and micro["tail_pressure_direction"] == 0):
        return {
            "condition_id": 34,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.75,
            "entry_viability": "medium",
            "supporting_factors": ["🚀 Bearish momentum + continuation + bearish tail pressure"]
        }

    # Overbought + bearish engulfing + VWAP rejection
    if (momentum["momentum_state"] == 2 and candle["sequence_type"] == 2 and vwap["vwap_zone"] == 0):
        return {
            "condition_id": 35,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.7,
            "entry_viability": "medium",
            "supporting_factors": ["📈 Overbought + bearish engulfing + VWAP rejection"]
        }

    # Strong momentum slope down + continuation + volume
    if (momentum["momentum_slope_zone"] == 0 and candle["sequence_type"] == 3 and vwap["volume_pressure"] == 2):
        return {
            "condition_id": 36,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.8,
            "entry_viability": "high",
            "supporting_factors": ["📉 Strong bearish slope + continuation + high volume"]
        }

    # Wick-dominant bar + continuation + weak micro
    if (candle["candle_dominance"] == 0 and candle["sequence_type"] == 3 and micro["micro_risk_zone"] == 0):
        return {
            "condition_id": 37,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.7,
            "entry_viability": "medium",
            "supporting_factors": ["🕯️ Wick-dominant bearish bar + continuation + weak micro"]
        }

    # Double confirmation: bearish candle + bearish VWAP
    if (candle["candle_dominance"] == 0 and vwap["vwap_zone"] == 0):
        return {
            "condition_id": 38,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.8,
            "entry_viability": "high",
            "supporting_factors": ["✅ Bearish candle + VWAP confirmation"]
        }

    # Breakout down + strong candle + micro burst
    if (breakout["breakout_bias"] == 0 and candle["pattern_strength"] == 2 and micro["burst_type"] == 1):
        return {
            "condition_id": 39,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.8,
            "entry_viability": "medium",
            "supporting_factors": ["⚡ Breakout down + strong candle + micro burst (tempered for calm)"]
        }

    # Bearish dominance + VWAP below + tail down
    if (candle["candle_dominance"] == 0 and vwap["vwap_zone"] == 0 and micro["tail_pressure_direction"] == 0):
        return {
            "condition_id": 40,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.75,
            "entry_viability": "medium",
            "supporting_factors": ["💪 Bearish dominance + VWAP resistance + bearish tail pressure"]
        }

    # Momentum top reversal + strong candle + micro support
    if (momentum["momentum_state"] == 2 and candle["pattern_strength"] == 2 and micro["micro_risk_zone"] == 2):
        return {
            "condition_id": 41,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.7,
            "entry_viability": "medium",
            "supporting_factors": ["📉 Top reversal + bearish candle + weak micro support"]
        }

    # Bearish squeeze release + high volume + rejection wick
    if (breakout["squeeze_state"] == 2 and vwap["volume_pressure"] == 2 and candle["candle_dominance"] == 0):
        return {
            "condition_id": 42,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.8,
            "entry_viability": "high",
            "supporting_factors": ["✅ Bearish squeeze release + volume climax + rejection wick"]
        }

    # --- CATEGORY: Hold/No Trade Conditions (Recalibrated) ---

    if (momentum["entry_bias"] == 1 and candle["sequence_type"] == 2 and vwap["vwap_zone"] == 0):
        return {
            "condition_id": 43,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 0.95,
            "entry_viability": "low",
            "supporting_factors": ["🚫 Disagreement: bullish momentum vs bearish candle + VWAP below"]
        }

    if (breakout["breakout_bias"] == 1 and candle["pattern_strength"] == 0):
        return {
            "condition_id": 44,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 1.0,
            "entry_viability": "low",
            "supporting_factors": ["🤔 Indecision: flat breakout + weak candle"]
        }

    if (vwap["vwap_zone"] == 1 and vwap["volume_pressure"] == 1 and candle["sequence_type"] == 0):
        return {
            "condition_id": 45,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 1.0,
            "entry_viability": "low",
            "supporting_factors": ["🚫 Neutral VWAP + neutral volume + neutral candle — strong no_trade"]
        }
        # Momentum up + no candle strength + micro neutral
    if (momentum["entry_bias"] == 1 and candle["pattern_strength"] == 0 and micro["micro_risk_zone"] == 1):
        return {
            "condition_id": 46,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 1.0,
            "entry_viability": "low",
            "supporting_factors": ["⚠️ Conflict: Momentum up but no candle strength & micro neutral (EURUSD often stalls here)"]
        }

    # Breakout valid but candle weak & micro disagrees
    if (breakout["breakout_bias"] in [0, 2] and candle["pattern_strength"] == 0 and micro["micro_risk_zone"] == 0):
        return {
            "condition_id": 47,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 1.0,
            "entry_viability": "low",
            "supporting_factors": ["❌ Breakout bias but weak candle & micro disagree — typical false move in EURUSD"]
        }

    # VWAP unclear + tail pressure mixed + neutral candle
    if (vwap["vwap_zone"] == 1 and micro["tail_pressure_direction"] == 1 and candle["sequence_type"] == 0):
        return {
            "condition_id": 48,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 1.0,
            "entry_viability": "low",
            "supporting_factors": ["🚫 Unclear VWAP + mixed tail pressure + neutral candle — strong indecision zone"]
        }

    # Breakout up vs. bearish candle & tail down
    if (breakout["breakout_bias"] == 2 and candle["sequence_type"] == 2 and micro["tail_pressure_direction"] == 0):
        return {
            "condition_id": 49,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 1.0,
            "entry_viability": "low",
            "supporting_factors": ["⚠️ Conflict: Breakout bias up vs. bearish candle & bearish tail pressure"]
        }

    # Momentum down vs. bullish reversal + VWAP reversion
    if (momentum["entry_bias"] == 0 and candle["sequence_type"] == 1 and vwap["vwap_zone"] == 1):
        return {
            "condition_id": 50,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 1.0,
            "entry_viability": "low",
            "supporting_factors": ["🔄 Conflict: Momentum down vs. bullish reversal near VWAP reversion"]
        }

    # All major models misaligned
    if (momentum["entry_bias"] != breakout["breakout_bias"] and
        momentum["entry_bias"] != candle["candle_dominance"] and
        breakout["breakout_bias"] != candle["candle_dominance"] and
        vwap["vwap_zone"] not in [0, 2]):
        return {
            "condition_id": 51,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 1.0,
            "entry_viability": "low",
            "supporting_factors": ["🚫 Major models misaligned — EURUSD ranging"]
        }

    # Volume collapse + wick-dominant candle + VWAP middle
    if (vwap["volume_pressure"] == 0 and candle["candle_dominance"] == 0 and vwap["vwap_zone"] == 1):
        return {
            "condition_id": 52,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 1.0,
            "entry_viability": "low",
            "supporting_factors": ["📉 Volume collapse + wick candle + VWAP mid — dead zone"]
        }

    # Reversal candle vs. strong opposing momentum
    if ((candle["sequence_type"] == 1 and momentum["momentum_slope_zone"] == 0) or
        (candle["sequence_type"] == 2 and momentum["momentum_slope_zone"] == 2)):
        return {
            "condition_id": 53,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 1.0,
            "entry_viability": "low",
            "supporting_factors": ["⚠️ Reversal attempt blocked by opposing momentum slope"]
        }

    # Micro neutral + neutral candle + VWAP neutral
    if (micro["micro_risk_zone"] == 1 and candle["sequence_type"] == 0 and vwap["vwap_zone"] == 1):
        return {
            "condition_id": 54,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 1.0,
            "entry_viability": "low",
            "supporting_factors": ["🤝 Triple neutrality: micro + candle + VWAP"]
        }

    # Breakout bias but weak candle
    if (breakout["breakout_bias"] in [0, 2] and candle["pattern_strength"] == 0):
        return {
            "condition_id": 55,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 1.0,
            "entry_viability": "low",
            "supporting_factors": ["📉 Breakout bias but weak candle — unreliable"]
        }

    # General conflict: mismatched signals / low confidence
    if (momentum["entry_bias"] != breakout["breakout_bias"] or
        momentum["entry_bias"] != micro["burst_type"] or
        (vwap["vwap_zone"] == 1 and candle["pattern_strength"] == 0)):
        return {
            "condition_id": 56,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 1.0,
            "entry_viability": "low",
            "supporting_factors": ["🚫 Conflicting major signals — avoid trade"]
        }

    # Bearish pattern failed to confirm (extra cautious)
    if (
        candle["sequence_type"] == 4 and
        candle["candle_dominance"] == 0 and
        vwap["vwap_zone"] == 0 and
        momentum["momentum_state"] != 0
    ):
        return {
            "condition_id": 57,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 1.0,
            "entry_viability": "low",
            "supporting_factors": ["⚠️ Bearish pattern failed — momentum does not confirm"]
        }

    # --- NEW CALM-SESSION FILTERS ---

    # Fake breakout: squeeze release but VWAP neutral
    if (breakout["squeeze_state"] == 2 and vwap["vwap_zone"] == 1 and candle["pattern_strength"] == 1):
        return {
            "condition_id": 58,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 1.0,
            "entry_viability": "low",
            "supporting_factors": ["⚠️ Fake breakout: squeeze release but VWAP neutral"]
        }

    # Range lock: low volatility + micro neutral + VWAP mid
    if (breakout["range"] < breakout["range_20"] * 0.5 and 
       micro["micro_risk_zone"] == 1 and 
       vwap["vwap_zone"] == 1):
       return {
        "condition_id": 59,
        "final_bias": "no_trade",
        "suggested_trade_direction": "none",
        "confidence": 1.0,
        "entry_viability": "low",
        "supporting_factors": ["🔒 Range lock: very low volatility zone"]
       }

        # Volatility trap: ATR collapse but momentum bias present
    

    # VWAP fake rejection: price below VWAP but volume weak
    if (vwap["vwap_zone"] == 0 and vwap["volume_pressure"] == 0 and candle["sequence_type"] in [1, 2]):
        return {
            "condition_id": 61,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 1.0,
            "entry_viability": "low",
            "supporting_factors": ["❌ VWAP rejection attempt but no volume confirmation"]
        }

    # Late-session exhaustion: bias up but micro + volume collapse
    if (momentum["entry_bias"] == 1 and micro["micro_risk_zone"] == 1 and vwap["volume_pressure"] == 0):
        return {
            "condition_id": 62,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 1.0,
            "entry_viability": "low",
            "supporting_factors": ["🌙 Late-session exhaustion: bullish bias but no liquidity"]
        }

    # Inside-bar compression: neutral candle + neutral VWAP + micro indecision
    if (candle["sequence_type"] == 0 and vwap["vwap_zone"] == 1 and micro["micro_risk_zone"] == 1):
        return {
            "condition_id": 63,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 1.0,
            "entry_viability": "low",
            "supporting_factors": ["📏 Inside-bar compression: neutral stack across models"]
        }

    # Cross-confirmation failure: momentum + breakout agree but VWAP opposite
    if ((momentum["entry_bias"] == breakout["breakout_bias"]) and
        ((momentum["entry_bias"] == 1 and vwap["vwap_zone"] == 0) or 
         (momentum["entry_bias"] == 0 and vwap["vwap_zone"] == 2))):
        return {
            "condition_id": 64,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 1.0,
            "entry_viability": "low",
            "supporting_factors": ["🚫 Cross-confirmation failed: VWAP disagrees with momentum+breakout"]
        }
        # Strong breakout + volume confirmation (EURUSD calm breakout check)
    if (breakout["breakout_bias"] == 2 and vwap["volume_pressure"] == 2 and candle["sequence_type"] in [1, 3]):
        return {
            "condition_id": 65,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.95,
            "entry_viability": "high",
            "supporting_factors": ["⚡ Confirmed breakout up + strong volume + bullish candle"]
        }

    # VWAP rebound long (price below VWAP, reversal candle, micro burst up)
    if (vwap["vwap_zone"] == 0 and candle["sequence_type"] == 1 and micro["burst_type"] == 1):
        return {
            "condition_id": 66,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.9,
            "entry_viability": "high",
            "supporting_factors": ["📈 VWAP rebound: bullish reversal from below VWAP with micro burst"]
        }

    # Microstructure wick trap → long (long wick rejection + bullish micro)
    if (candle["candle_dominance"] == 2 and micro["tail_pressure_direction"] == 2 and vwap["vwap_zone"] in [1, 2]):
        return {
            "condition_id": 67,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.85,
            "entry_viability": "medium",
            "supporting_factors": ["🕯️ Bullish wick trap + micro tail confirmation + VWAP not bearish"]
        }

    # Failed breakdown recovery (fakeout short, price reclaims VWAP)
    if (breakout["breakout_bias"] == 0 and vwap["vwap_zone"] == 2 and candle["sequence_type"] == 1):
        return {
            "condition_id": 68,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.9,
            "entry_viability": "high",
            "supporting_factors": ["🔄 Failed breakdown → VWAP reclaim bullish reversal"]
        }

    # Compression breakout short (tight range + bearish candle + micro burst down)
    if (breakout["squeeze_state"] == 1 and breakout["breakout_bias"] == 0 and micro["burst_type"] == 1 and candle["sequence_type"] == 2):
        return {
            "condition_id": 69,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.9,
            "entry_viability": "high",
            "supporting_factors": ["📉 Compression resolved bearish: squeeze + micro burst down + bearish candle"]
        }
