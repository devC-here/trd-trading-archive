# override_engine.py

def check_override_conditions(momentum, breakout, micro, vwap, candle):
    conditions = []

    # --- CATEGORY: Microstructure Traps & Bursts (Original) ---
    if micro["micro_risk_zone"] == 0:
        return {
            "condition_id": 1,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 0.0,
            "entry_viability": "low",
            "supporting_factors": ["❌ Trap zone detected (microstructure)"]
        }

    if micro["burst_type"] == 1 and momentum["entry_bias"] == 1:
        return {
            "condition_id": 2,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.9,
            "entry_viability": "high",
            "supporting_factors": ["🚀 Microstructure burst + bullish bias"]
        }

    if micro["burst_type"] == 1 and momentum["entry_bias"] == 0:
        return {
            "condition_id": 3,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.9,
            "entry_viability": "high",
            "supporting_factors": ["🚨 Microstructure burst + bearish bias"]
        }

    # --- CATEGORY: Squeeze & Breakout Alignment (Original) ---
    if breakout["squeeze_state"] == 1:
        if breakout["breakout_bias"] == 2: # Assuming 2 means long bias for breakout
            return {
                "condition_id": 4,
                "final_bias": "long",
                "suggested_trade_direction": "Buy",
                "confidence": 1.0,
                "entry_viability": "high",
                "supporting_factors": ["⚡ Breakout squeeze — forced long"]
            }
        elif breakout["breakout_bias"] == 0: # Assuming 0 means short bias for breakout
            return {
                "condition_id": 5,
                "final_bias": "short",
                "suggested_trade_direction": "Sell",
                "confidence": 1.0,
                "entry_viability": "high",
                "supporting_factors": ["⚡ Breakout squeeze — forced short"]
            }

    # --- CATEGORY: Momentum Conflicts (Original) ---
    if momentum["entry_bias"] == 1 and vwap["vwap_zone"] == 0 and vwap["volume_pressure"] == 0:
        return {
            "condition_id": 6,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 0.0,
            "entry_viability": "low",
            "supporting_factors": ["🧊 Bullish momentum trapped under VWAP with no volume"]
        }

    if momentum["entry_bias"] == 1 and candle["sequence_type"] == 2: # Assuming 2 is a bearish candle sequence
        return {
            "condition_id": 7,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 0.0,
            "entry_viability": "low",
            "supporting_factors": ["⚠️ Momentum vs bearish candle conflict"]
        }

    # --- CATEGORY: High Confidence Alignments (Original) ---
    if (momentum["entry_bias"] == 1 and
        breakout["breakout_bias"] == 2 and # Assuming 2 is bullish breakout bias
        candle["sequence_type"] in [1, 3] and candle["candle_dominance"] == 2): # Assuming 1,3 are bullish/continuation and 2 is bullish dominance
        return {
            "condition_id": 8,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.95,
            "entry_viability": "high",
            "supporting_factors": ["🔥 Triple alignment: Momentum + Breakout + Candle"]
        }

    if (momentum["entry_bias"] == 0 and
        breakout["breakout_bias"] == 0 and # Assuming 0 is bearish breakout bias
        candle["sequence_type"] in [2, 3] and candle["candle_dominance"] == 0): # Assuming 2,3 are bearish/continuation and 0 is bearish dominance
        return {
            "condition_id": 9,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.95,
            "entry_viability": "high",
            "supporting_factors": ["🔥 Triple alignment: Momentum + Breakout + Candle (short)"]
        }

    # --- CATEGORY: VWAP Conflict or Support (Original) ---
    if vwap["vwap_zone"] == 2 and candle["sequence_type"] == 2 and candle["candle_dominance"] == 0:
        return {
            "condition_id": 10,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.8,
            "entry_viability": "high",
            "supporting_factors": ["📉 VWAP rejection + bearish candle"]
        }

    if vwap["vwap_zone"] == 0 and candle["sequence_type"] == 1 and micro["tail_pressure_direction"] == 2: # Assuming 0 for VWAP zone is below and 2 for tail pressure is bullish
        return {
            "condition_id": 11,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.75,
            "entry_viability": "medium",
            "supporting_factors": ["📈 Deep pullback + bullish reversal wick"]
        }

    # --- CATEGORY: Indecision + Neutrality (Original) ---
    if vwap["vwap_zone"] == 1 and candle["pattern_strength"] == 0 and micro["micro_risk_zone"] == 1:
        return {
            "condition_id": 12,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 0.0,
            "entry_viability": "low",
            "supporting_factors": ["🚫 Neutral VWAP + weak candle + indecision"]
        }


    # --- NEW CATEGORY: Long Bias Conditions ---

    # Momentum up + breakout up + bullish continuation
    if (momentum["entry_bias"] == 1 and breakout["breakout_bias"] == 2 and candle["sequence_type"] == 1):
        return {
            "condition_id": 13,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.9,
            "entry_viability": "high",
            "supporting_factors": ["⬆️ Momentum up + breakout up + bullish continuation"]
        }

    # Breakout squeeze + long bias + microstructure burst zone
    if (breakout["squeeze_state"] == 1 and breakout["breakout_bias"] == 2 and micro["burst_type"] == 1):
        return {
            "condition_id": 14,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.95,
            "entry_viability": "high",
            "supporting_factors": ["💥 Breakout squeeze + long bias + micro burst"]
        }

    # Candle bullish reversal + VWAP above + volume pressure (assuming volume_pressure 2 is high bullish)
    if (candle["sequence_type"] == 1 and vwap["vwap_zone"] == 2 and vwap["volume_pressure"] == 2):
        return {
            "condition_id": 15,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.85,
            "entry_viability": "high",
            "supporting_factors": ["📊 Candle bullish reversal + VWAP above + high volume pressure"]
        }

    # Continuation pattern + momentum up + VWAP support (assuming sequence_type 3 is continuation, vwap_zone 2 is support)
    if (candle["sequence_type"] == 3 and momentum["entry_bias"] == 1 and vwap["vwap_zone"] == 2):
        return {
            "condition_id": 16,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.8,
            "entry_viability": "medium",
            "supporting_factors": ["📈 Continuation pattern + momentum up + VWAP support"]
        }

    # Reversal pattern + micro tail support + liquidity imbalance (assuming candle seq 1 is reversal, tail_pressure_direction 2 is support, micro_risk_zone 2 is imbalance)
    if (candle["sequence_type"] == 1 and micro["tail_pressure_direction"] == 2 and micro["micro_risk_zone"] == 2):
        return {
            "condition_id": 17,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.85,
            "entry_viability": "high",
            "supporting_factors": ["🔄 Reversal pattern + micro tail support + liquidity imbalance"]
        }

    # VWAP zone 2 + volume pressure 2 + bullish candle (assuming candle_dominance 2 is bullish)
    if (vwap["vwap_zone"] == 2 and vwap["volume_pressure"] == 2 and candle["candle_dominance"] == 2):
        return {
            "condition_id": 18,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.9,
            "entry_viability": "high",
            "supporting_factors": ["💪 VWAP strong support + high volume + bullish candle"]
        }

    # Momentum up + candle strong continuation + burst tail (assuming sequence_type 3 is continuation, tail_pressure_direction 2 is burst tail)
    if (momentum["entry_bias"] == 1 and candle["sequence_type"] == 3 and micro["tail_pressure_direction"] == 2):
        return {
            "condition_id": 19,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.85,
            "entry_viability": "high",
            "supporting_factors": ["🚀 Momentum up + strong continuation + burst tail"]
        }

    # Oversold exhaustion + strong engulfing + VWAP re-entry (assuming momentum low state for oversold, candle sequence 1 for engulfing, vwap_zone 2 for re-entry)
    if (momentum["momentum_state"] == 0 and candle["sequence_type"] == 1 and vwap["vwap_zone"] == 2):
        return {
            "condition_id": 20,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.8,
            "entry_viability": "medium",
            "supporting_factors": ["📉 Oversold exhaustion + engulfing + VWAP re-entry"]
        }

    # Momentum slope zone 2 + continuation + volume spike (assuming momentum_slope_zone 2 for strong up, sequence_type 3 for continuation, volume_pressure 2 for spike)
    if (momentum["momentum_slope_zone"] == 2 and candle["sequence_type"] == 3 and vwap["volume_pressure"] == 2):
        return {
            "condition_id": 21,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.9,
            "entry_viability": "high",
            "supporting_factors": ["📈 Strong momentum slope + continuation + volume spike"]
        }

    # Tail-dominant bar + sequence type 3 + microstructure support (assuming candle_dominance 2 for tail-dominant, micro_risk_zone 2 for support)
    if (candle["candle_dominance"] == 2 and candle["sequence_type"] == 3 and micro["micro_risk_zone"] == 2):
        return {
            "condition_id": 22,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.85,
            "entry_viability": "high",
            "supporting_factors": ["🕯️ Tail-dominant bar + continuation + micro support"]
        }

    # Double confirmation: bullish candle + bullish VWAP (assuming candle_dominance 2 is bullish candle, vwap_zone 2 is bullish VWAP)
    if (candle["candle_dominance"] == 2 and vwap["vwap_zone"] == 2):
        return {
            "condition_id": 23,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.9,
            "entry_viability": "high",
            "supporting_factors": ["✅ Double confirmation: bullish candle + bullish VWAP"]
        }

    # Breakout bias up + candle strength 2 + burst type 1 (assuming candle_strength 2 is strong)
    if (breakout["breakout_bias"] == 2 and candle["pattern_strength"] == 2 and micro["burst_type"] == 1):
        return {
            "condition_id": 24,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.95,
            "entry_viability": "high",
            "supporting_factors": ["⚡ Breakout up + strong candle + micro burst"]
        }

    # Candle dominance 2 + VWAP above + tail pressure up (assuming tail_pressure_direction 2 is up)
    if (candle["candle_dominance"] == 2 and vwap["vwap_zone"] == 2 and micro["tail_pressure_direction"] == 2):
        return {
            "condition_id": 25,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.9,
            "entry_viability": "high",
            "supporting_factors": ["💪 Candle dominance + VWAP above + tail pressure up"]
        }

    # Momentum low state but slope strong + confirmed candle (assuming momentum_state 0 is low, momentum_slope_zone 2 is strong, candle_dominance 2 is confirmed)
    if (momentum["momentum_state"] == 0 and momentum["momentum_slope_zone"] == 2 and candle["candle_dominance"] == 2):
        return {
            "condition_id": 26,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.8,
            "entry_viability": "medium",
            "supporting_factors": ["📈 Momentum low but slope strong + confirmed candle"]
        }

    # Candle bullish + squeeze released + micro confirm (assuming candle_dominance 2 is bullish, squeeze_state 2 is released, micro_risk_zone 2 is confirm)
    if (candle["candle_dominance"] == 2 and breakout["squeeze_state"] == 2 and micro["micro_risk_zone"] == 2):
        return {
            "condition_id": 27,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.9,
            "entry_viability": "high",
            "supporting_factors": ["✅ Bullish candle + squeeze released + micro confirm"]
        }


    # --- NEW CATEGORY: Short Bias Conditions ---

    # Momentum down + breakout down + bearish continuation
    if (momentum["entry_bias"] == 0 and breakout["breakout_bias"] == 0 and candle["sequence_type"] == 2): # Assuming 2 is bearish continuation
        return {
            "condition_id": 28,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.9,
            "entry_viability": "high",
            "supporting_factors": ["⬇️ Momentum down + breakout down + bearish continuation"]
        }

    # Breakout squeeze + short bias + microstructure burst zone
    if (breakout["squeeze_state"] == 1 and breakout["breakout_bias"] == 0 and micro["burst_type"] == 1): # Assuming 1 for burst type applies to both directions
        return {
            "condition_id": 29,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.95,
            "entry_viability": "high",
            "supporting_factors": ["💥 Breakout squeeze + short bias + micro burst"]
        }

    # Candle bearish reversal + VWAP below + volume climax (assuming sequence_type 2 is bearish reversal, vwap_zone 0 is below, volume_pressure 2 is climax)
    if (candle["sequence_type"] == 2 and vwap["vwap_zone"] == 0 and vwap["volume_pressure"] == 2):
        return {
            "condition_id": 30,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.85,
            "entry_viability": "high",
            "supporting_factors": ["📊 Candle bearish reversal + VWAP below + volume climax"]
        }

    # Continuation pattern + momentum down + VWAP resistance (assuming sequence_type 3 is continuation, vwap_zone 0 is resistance)
    if (candle["sequence_type"] == 3 and momentum["entry_bias"] == 0 and vwap["vwap_zone"] == 0):
        return {
            "condition_id": 31,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.8,
            "entry_viability": "medium",
            "supporting_factors": ["📉 Continuation pattern + momentum down + VWAP resistance"]
        }

    # Reversal pattern + micro tail pressure down + liquidity drain (assuming candle seq 2 is reversal, tail_pressure_direction 0 is down, micro_risk_zone 0 is liquidity drain)
    if (candle["sequence_type"] == 2 and micro["tail_pressure_direction"] == 0 and micro["micro_risk_zone"] == 0):
        return {
            "condition_id": 32,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.85,
            "entry_viability": "high",
            "supporting_factors": ["🔄 Reversal pattern + micro tail pressure down + liquidity drain"]
        }

    # VWAP zone 0 + volume pressure 2 + bearish candle (assuming candle_dominance 0 is bearish)
    if (vwap["vwap_zone"] == 0 and vwap["volume_pressure"] == 2 and candle["candle_dominance"] == 0):
        return {
            "condition_id": 33,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.9,
            "entry_viability": "high",
            "supporting_factors": ["💪 VWAP strong resistance + high volume + bearish candle"]
        }

    # Momentum down + candle strong continuation + burst tail (assuming sequence_type 3 is continuation, tail_pressure_direction 0 is burst tail)
    if (momentum["entry_bias"] == 0 and candle["sequence_type"] == 3 and micro["tail_pressure_direction"] == 0):
        return {
            "condition_id": 34,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.85,
            "entry_viability": "high",
            "supporting_factors": ["🚀 Momentum down + strong continuation + burst tail"]
        }

    # Overbought spike + strong bearish engulfing + VWAP rejection (assuming momentum high state for overbought, candle sequence 2 for engulfing, vwap_zone 0 for rejection)
    if (momentum["momentum_state"] == 2 and candle["sequence_type"] == 2 and vwap["vwap_zone"] == 0):
        return {
            "condition_id": 35,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.8,
            "entry_viability": "medium",
            "supporting_factors": ["📈 Overbought spike + engulfing + VWAP rejection"]
        }

    # Momentum slope zone 2 + continuation short + high volume (assuming momentum_slope_zone 0 for strong down, sequence_type 3 for continuation, volume_pressure 2 for high volume)
    if (momentum["momentum_slope_zone"] == 0 and candle["sequence_type"] == 3 and vwap["volume_pressure"] == 2):
        return {
            "condition_id": 36,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.9,
            "entry_viability": "high",
            "supporting_factors": ["📉 Strong momentum slope down + continuation + high volume"]
        }

    # Wick-dominant bar + sequence type 3 + microstructure weak (assuming candle_dominance 0 for wick-dominant, micro_risk_zone 0 for weak)
    if (candle["candle_dominance"] == 0 and candle["sequence_type"] == 3 and micro["micro_risk_zone"] == 0):
        return {
            "condition_id": 37,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.85,
            "entry_viability": "high",
            "supporting_factors": ["🕯️ Wick-dominant bar + continuation + micro weak"]
        }

    # Double confirmation: bearish candle + bearish VWAP (assuming candle_dominance 0 is bearish candle, vwap_zone 0 is bearish VWAP)
    if (candle["candle_dominance"] == 0 and vwap["vwap_zone"] == 0):
        return {
            "condition_id": 38,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.9,
            "entry_viability": "high",
            "supporting_factors": ["✅ Double confirmation: bearish candle + bearish VWAP"]
        }

    # Breakout bias down + candle strength 2 + burst type 1 (assuming candle_strength 2 is strong)
    if (breakout["breakout_bias"] == 0 and candle["pattern_strength"] == 2 and micro["burst_type"] == 1):
        return {
            "condition_id": 39,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.95,
            "entry_viability": "high",
            "supporting_factors": ["⚡ Breakout down + strong candle + micro burst"]
        }

    # Candle dominance 0 + VWAP below + tail pressure down (assuming tail_pressure_direction 0 is down)
    if (candle["candle_dominance"] == 0 and vwap["vwap_zone"] == 0 and micro["tail_pressure_direction"] == 0):
        return {
            "condition_id": 40,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.9,
            "entry_viability": "high",
            "supporting_factors": ["💪 Candle dominance + VWAP below + tail pressure down"]
        }

    # Momentum top reversal + candle strength 2 + micro support (assuming momentum_state 2 is top reversal, candle_strength 2 is strong, micro_risk_zone 2 is support)
    if (momentum["momentum_state"] == 2 and candle["pattern_strength"] == 2 and micro["micro_risk_zone"] == 2):
        return {
            "condition_id": 41,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.8,
            "entry_viability": "medium",
            "supporting_factors": ["📉 Momentum top reversal + strong candle + micro support"]
        }

    # Bearish squeeze release + high climax volume + rejection wick (assuming squeeze_state 2 is released, volume_pressure 2 is high climax, candle_dominance 0 is rejection wick)
    if (breakout["squeeze_state"] == 2 and vwap["volume_pressure"] == 2 and candle["candle_dominance"] == 0):
        return {
            "condition_id": 42,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.9,
            "entry_viability": "high",
            "supporting_factors": ["✅ Bearish squeeze release + high volume + rejection wick"]
        }

    # --- NEW CATEGORY: Hold/No Trade Conditions ---

    # Micro risk zone = 0 (trap zone) - Already exists, but kept for consistency
    # Condition 1 is the micro risk zone trap.

    # Momentum long + candle bearish + VWAP below — disagreement (assuming momentum 1 is long, candle 2 is bearish, VWAP 0 is below)
    if (momentum["entry_bias"] == 1 and candle["sequence_type"] == 2 and vwap["vwap_zone"] == 0):
        return {
            "condition_id": 43,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 0.0,
            "entry_viability": "low",
            "supporting_factors": ["🚫 Disagreement: Momentum long vs. bearish candle & VWAP below"]
        }

    # Breakout flat + no bias + weak candle — indecision (assuming breakout_bias 1 is flat, candle pattern_strength 0 is weak)
    if (breakout["breakout_bias"] == 1 and candle["pattern_strength"] == 0):
        return {
            "condition_id": 44,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 0.0,
            "entry_viability": "low",
            "supporting_factors": ["🤔 Indecision: Breakout flat + no bias + weak candle"]
        }

    # VWAP zone 1 + volume neutral + sequence type 0 (assuming VWAP 1 is neutral, volume_pressure 1 is neutral, sequence_type 0 is neutral)
    if (vwap["vwap_zone"] == 1 and vwap["volume_pressure"] == 1 and candle["sequence_type"] == 0):
        return {
            "condition_id": 45,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 0.0,
            "entry_viability": "low",
            "supporting_factors": ["🚫 Neutrality: VWAP + volume + candle sequence"]
        }

    # Momentum up + no candle strength + micro neutral (assuming momentum 1 is up, candle pattern_strength 0 is weak, micro_risk_zone 1 is neutral)
    if (momentum["entry_bias"] == 1 and candle["pattern_strength"] == 0 and micro["micro_risk_zone"] == 1):
        return {
            "condition_id": 46,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 0.0,
            "entry_viability": "low",
            "supporting_factors": ["⚠️ Conflict: Momentum up but no candle strength & micro neutral"]
        }

    # Breakout valid but candle is weak & micro disagrees (assuming breakout_bias 2 or 0 is valid, candle pattern_strength 0 is weak, micro_risk_zone 0 for disagreement)
    if (breakout["breakout_bias"] in [0, 2] and candle["pattern_strength"] == 0 and micro["micro_risk_zone"] == 0):
        return {
            "condition_id": 47,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 0.0,
            "entry_viability": "low",
            "supporting_factors": ["❌ Breakout valid but candle weak & micro disagrees"]
        }

    # VWAP unclear + tail pressure mixed + sequence type 0 (assuming VWAP 1 is unclear, tail_pressure_direction 1 is mixed, sequence_type 0 is neutral)
    if (vwap["vwap_zone"] == 1 and micro["tail_pressure_direction"] == 1 and candle["sequence_type"] == 0):
        return {
            "condition_id": 48,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 0.0,
            "entry_viability": "low",
            "supporting_factors": ["🚫 Unclear: VWAP + tail pressure mixed + neutral candle"]
        }

    # Breakout bias up + candle bearish + micro tail pressure down (assuming breakout_bias 2 is up, candle sequence_type 2 is bearish, tail_pressure_direction 0 is down)
    if (breakout["breakout_bias"] == 2 and candle["sequence_type"] == 2 and micro["tail_pressure_direction"] == 0):
        return {
            "condition_id": 49,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 0.0,
            "entry_viability": "low",
            "supporting_factors": ["⚠️ Conflict: Breakout up vs. bearish candle & tail pressure down"]
        }

    # Momentum down + candle bullish reversal + VWAP reversion zone (assuming momentum 0 is down, candle sequence_type 1 is bullish reversal, vwap_zone 1 is reversion)
    if (momentum["entry_bias"] == 0 and candle["sequence_type"] == 1 and vwap["vwap_zone"] == 1):
        return {
            "condition_id": 50,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 0.0,
            "entry_viability": "low",
            "supporting_factors": ["🔄 Conflict: Momentum down vs. bullish reversal & VWAP reversion"]
        }

    # All 5 models misaligned directionally (this is a general condition, hard to capture precisely without more specific data points for each model's "alignment" state)
    # For now, a simplified interpretation assuming major indicators are conflicting.
    if (momentum["entry_bias"] != breakout["breakout_bias"] and
        momentum["entry_bias"] != candle["candle_dominance"] and
        breakout["breakout_bias"] != candle["candle_dominance"] and
        vwap["vwap_zone"] not in [0, 2]): # Check if VWAP is neutral/unclear
        return {
            "condition_id": 51,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 0.0,
            "entry_viability": "low",
            "supporting_factors": ["🚫 All major models misaligned directionally"]
        }


    # Volume collapse + wick-dominant candle + VWAP middle (assuming volume_pressure 0 is collapse, candle_dominance 0 is wick-dominant, vwap_zone 1 is middle)
    if (vwap["volume_pressure"] == 0 and candle["candle_dominance"] == 0 and vwap["vwap_zone"] == 1):
        return {
            "condition_id": 52,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 0.0,
            "entry_viability": "low",
            "supporting_factors": ["📉 Volume collapse + wick-dominant candle + VWAP middle"]
        }

    # Reversal candle but momentum opposes strongly (assuming candle seq 1 or 2, momentum_slope_zone 0 or 2 strongly opposes)
    if ((candle["sequence_type"] == 1 and momentum["momentum_slope_zone"] == 0) or
        (candle["sequence_type"] == 2 and momentum["momentum_slope_zone"] == 2)):
        return {
            "condition_id": 53,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 0.0,
            "entry_viability": "low",
            "supporting_factors": ["⚠️ Reversal candle vs. strong opposing momentum"]
        }

    # Microstructure neutral + sequence type 0 + VWAP neutral (assuming micro_risk_zone 1, sequence_type 0, vwap_zone 1)
    if (micro["micro_risk_zone"] == 1 and candle["sequence_type"] == 0 and vwap["vwap_zone"] == 1):
        return {
            "condition_id": 54,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 0.0,
            "entry_viability": "low",
            "supporting_factors": ["🤝 Neutrality: Microstructure + candle sequence + VWAP"]
        }

    # Breakout entry viable but candle strength weak (assuming breakout_bias in [0,2] and candle pattern_strength 0)
    if (breakout["breakout_bias"] in [0, 2] and candle["pattern_strength"] == 0):
        return {
            "condition_id": 55,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 0.0,
            "entry_viability": "low",
            "supporting_factors": ["📉 Breakout viable but candle strength weak"]
        }

    # Multiple model conflicts + confidence low (< 0.6) - This needs a more holistic calculation, hardcoding a confidence check here
    # For now, a general catch-all for conflicting major signals.
    if (momentum["entry_bias"] != breakout["breakout_bias"] or
        momentum["entry_bias"] != micro["burst_type"] or # Assuming burst_type 1 indicates strong direction
        (vwap["vwap_zone"] == 1 and candle["pattern_strength"] == 0)):
         # Simplified conflict: If major indicators point opposite or are neutral/weak
         # This needs refinement based on how confidence is derived in your full system.
        return {
            "condition_id": 56,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 0.0,
            "entry_viability": "low",
            "supporting_factors": ["🚫 Multiple model conflicts / low confidence signals"]
        }

    # NEW CONDITION: Bearish pattern failed to confirm — momentum doesn't agree
    if (
        candle["sequence_type"] == 4 and  # bearish continuation (assuming 4 specifically means bearish continuation)
        candle["candle_dominance"] == 0 and
        vwap["vwap_zone"] == 0 and
        momentum["momentum_state"] != 0  # not in bearish momentum anymore (assuming 0 is bearish momentum state)
    ):
        return {
            "condition_id": 57, # Assigning the next available ID
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 0.0,
            "entry_viability": "low",
            "supporting_factors": ["⚠️ Bearish pattern failed to confirm — momentum doesn't agree"]
        }

    # No override triggered
    return None