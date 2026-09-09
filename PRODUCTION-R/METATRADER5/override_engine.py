# override_engine.py

def check_override_conditions(momentum, breakout, micro, vwap, candle):
    conditions = []

    # ---- EURUSD guardrails (shared flags) ----
    v_above = vwap.get("is_above_vwap", 0) == 1
    v_below = not v_above
    vol_ok = vwap.get("volume_pressure", 0) == 2
    v_bear = vwap.get("vwap_zone", 1) == 2 or v_below
    v_bull = vwap.get("vwap_zone", 1) == 0 or v_above

    momo_up = momentum.get("momentum_state", 1) == 2
    momo_dn = momentum.get("momentum_state", 1) == 1
    momo_slope_up = momentum.get("momentum_slope_zone", 1) == 2
    momo_slope_dn = momentum.get("momentum_slope_zone", 1) == 1

    micro_burst = micro.get("burst_type", 0) == 1
    micro_safe = micro.get("micro_risk_zone", 1) == 2   # 2 = supportive / not-trap in your schema

    c_dom_bull = candle.get("candle_dominance", 1) == 2
    c_dom_bear = candle.get("candle_dominance", 1) == 0
    c_seq_cont = candle.get("sequence_type", 1) == 3    # your "continuation" tag
    c_seq_chop = candle.get("sequence_type", 1) == 2
    c_strong = candle.get("pattern_strength", 1) == 2

    brk_squeeze = breakout.get("squeeze_state", 0) == 1  # just released
    brk_expanded = breakout.get("squeeze_state", 0) == 2  # in expansion
    brk_long = breakout.get("breakout_bias", 1) == 2
    brk_short = breakout.get("breakout_bias", 1) == 0

    # --- early neutralization (place near the top) ---
    if c_seq_chop and not micro_safe and not vol_ok:
        return {
            "condition_id": 44,  # keep your existing neutral IDs
            "final_bias": "neutral",
            "suggested_trade_direction": "Hold",
            "confidence": 0.9,
            "entry_viability": "low",
            "supporting_factors": ["⏸️ Chop + weak micro + no volume"]
        }
    if (not momo_up and not momo_dn) and (not c_dom_bull and not c_dom_bear) and (not v_above and not v_below):
        return {
            "condition_id": 45,
            "final_bias": "neutral",
            "suggested_trade_direction": "Hold",
            "confidence": 0.9,
            "entry_viability": "low",
            "supporting_factors": ["⏸️ No directional consensus (momo/candle/VWAP)"]
        }

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

    # --- CATEGORY: Squeeze & Breakout Alignment (Updated) ---
    if brk_squeeze and brk_long and v_bull:
        return {
            "condition_id": 4,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.9,
            "entry_viability": "medium",
            "supporting_factors": ["⚡ Breakout squeeze + VWAP alignment (long)"]
        }
    if brk_squeeze and brk_short and v_bear:
        return {
            "condition_id": 5,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.9,
            "entry_viability": "medium",
            "supporting_factors": ["⚡ Breakout squeeze + VWAP alignment (short)"]
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

    if momentum["entry_bias"] == 1 and candle["sequence_type"] == 2:
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
        breakout["breakout_bias"] == 2 and
        candle["sequence_type"] in [1, 3] and candle["candle_dominance"] == 2):
        return {
            "condition_id": 8,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.95,
            "entry_viability": "high",
            "supporting_factors": ["🔥 Triple alignment: Momentum + Breakout + Candle"]
        }

    if (momentum["entry_bias"] == 0 and
        breakout["breakout_bias"] == 0 and
        candle["sequence_type"] in [2, 3] and candle["candle_dominance"] == 0):
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

    if vwap["vwap_zone"] == 0 and candle["sequence_type"] == 1 and micro["tail_pressure_direction"] == 2:
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

    # --- NEW CATEGORY: Long Bias Conditions (Updated) ---

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

    # Breakout squeeze + long bias + microstructure burst zone (Updated)
    if brk_squeeze and brk_long and micro_burst and vol_ok:
        return {
            "condition_id": 14,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.95,
            "entry_viability": "high",
            "supporting_factors": ["💥 Breakout + long bias + micro burst + volume"]
        }

    # Momentum-only guard (Updated)
    if momo_up and (v_bull or micro_burst) and micro_safe:
        return {
            "condition_id": 15,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.75,
            "entry_viability": "medium",
            "supporting_factors": ["📊 Momentum up + (VWAP or micro) + safe zone"]
        }

    # Continuation pattern + momentum up + VWAP support
    if (candle["sequence_type"] == 3 and momentum["entry_bias"] == 1 and vwap["vwap_zone"] == 2):
        return {
            "condition_id": 16,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.8,
            "entry_viability": "medium",
            "supporting_factors": ["📈 Continuation pattern + momentum up + VWAP support"]
        }

    # Reversal pattern + micro tail support + liquidity imbalance
    if (candle["sequence_type"] == 1 and micro["tail_pressure_direction"] == 2 and micro["micro_risk_zone"] == 2):
        return {
            "condition_id": 17,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.85,
            "entry_viability": "high",
            "supporting_factors": ["🔄 Reversal pattern + micro tail support + liquidity imbalance"]
        }

    # VWAP zone 2 + volume pressure 2 + bullish candle
    if (vwap["vwap_zone"] == 2 and vwap["volume_pressure"] == 2 and candle["candle_dominance"] == 2):
        return {
            "condition_id": 18,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.9,
            "entry_viability": "high",
            "supporting_factors": ["💪 VWAP strong support + high volume + bullish candle"]
        }

    # Momentum up + candle strong continuation + burst tail
    if (momentum["entry_bias"] == 1 and candle["sequence_type"] == 3 and micro["tail_pressure_direction"] == 2):
        return {
            "condition_id": 19,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.85,
            "entry_viability": "high",
            "supporting_factors": ["🚀 Momentum up + strong continuation + burst tail"]
        }

    # Oversold exhaustion + strong engulfing + VWAP re-entry
    if (momentum["momentum_state"] == 0 and candle["sequence_type"] == 1 and vwap["vwap_zone"] == 2):
        return {
            "condition_id": 20,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.8,
            "entry_viability": "medium",
            "supporting_factors": ["📉 Oversold exhaustion + engulfing + VWAP re-entry"]
        }

    # Momentum slope zone 2 + continuation + volume spike
    if (momentum["momentum_slope_zone"] == 2 and candle["sequence_type"] == 3 and vwap["volume_pressure"] == 2):
        return {
            "condition_id": 21,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.9,
            "entry_viability": "high",
            "supporting_factors": ["📈 Strong momentum slope + continuation + volume spike"]
        }

    # Candle continuation (Updated)
    if c_dom_bull and c_seq_cont and micro_safe and v_bull:
        return {
            "condition_id": 22,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.85,
            "entry_viability": "high",
            "supporting_factors": ["🕯️ Bull continuation + micro support + VWAP"]
        }

    # Double confirmation: bullish candle + bullish VWAP
    if (candle["candle_dominance"] == 2 and vwap["vwap_zone"] == 2):
        return {
            "condition_id": 23,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.9,
            "entry_viability": "high",
            "supporting_factors": ["✅ Double confirmation: bullish candle + bullish VWAP"]
        }

    # Breakout bias up + candle strength 2 + burst type 1
    if (breakout["breakout_bias"] == 2 and candle["pattern_strength"] == 2 and micro["burst_type"] == 1):
        return {
            "condition_id": 24,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.95,
            "entry_viability": "high",
            "supporting_factors": ["⚡ Breakout up + strong candle + micro burst"]
        }

    # Candle dominance 2 + VWAP above + tail pressure up
    if (candle["candle_dominance"] == 2 and vwap["vwap_zone"] == 2 and micro["tail_pressure_direction"] == 2):
        return {
            "condition_id": 25,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.9,
            "entry_viability": "high",
            "supporting_factors": ["💪 Candle dominance + VWAP above + tail pressure up"]
        }

    # Momentum low state but slope strong + confirmed candle
    if (momentum["momentum_state"] == 0 and momentum["momentum_slope_zone"] == 2 and candle["candle_dominance"] == 2):
        return {
            "condition_id": 26,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.8,
            "entry_viability": "medium",
            "supporting_factors": ["📈 Momentum low but slope strong + confirmed candle"]
        }

    # Candle bullish + squeeze released + micro confirm
    if (candle["candle_dominance"] == 2 and breakout["squeeze_state"] == 2 and micro["micro_risk_zone"] == 2):
        return {
            "condition_id": 27,
            "final_bias": "long",
            "suggested_trade_direction": "Buy",
            "confidence": 0.9,
            "entry_viability": "high",
            "supporting_factors": ["✅ Bullish candle + squeeze released + micro confirm"]
        }

    # --- NEW CATEGORY: Short Bias Conditions (Updated) ---

    # Momentum-only guard (Updated)
    if momo_dn and (v_bear or micro_burst) and micro_safe:
        return {
            "condition_id": 28,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.75,
            "entry_viability": "medium",
            "supporting_factors": ["📊 Momentum down + (VWAP or micro) + safe zone"]
        }

    # Breakout squeeze + short bias + microstructure burst zone (Updated)
    if brk_squeeze and brk_short and micro_burst and vol_ok:
        return {
            "condition_id": 29,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.95,
            "entry_viability": "high",
            "supporting_factors": ["💥 Breakout + short bias + micro burst + volume"]
        }

    # Candle bearish reversal + VWAP below + volume climax
    if (candle["sequence_type"] == 2 and vwap["vwap_zone"] == 0 and vwap["volume_pressure"] == 2):
        return {
            "condition_id": 30,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.85,
            "entry_viability": "high",
            "supporting_factors": ["📊 Candle bearish reversal + VWAP below + volume climax"]
        }

    # Continuation pattern + momentum down + VWAP resistance
    if (candle["sequence_type"] == 3 and momentum["entry_bias"] == 0 and vwap["vwap_zone"] == 0):
        return {
            "condition_id": 31,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.8,
            "entry_viability": "medium",
            "supporting_factors": ["📉 Continuation pattern + momentum down + VWAP resistance"]
        }

    # Reversal pattern + micro tail pressure down + liquidity drain
    if (candle["sequence_type"] == 2 and micro["tail_pressure_direction"] == 0 and micro["micro_risk_zone"] == 0):
        return {
            "condition_id": 32,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.85,
            "entry_viability": "high",
            "supporting_factors": ["🔄 Reversal pattern + micro tail pressure down + liquidity drain"]
        }

    # VWAP zone 0 + volume pressure 2 + bearish candle
    if (vwap["vwap_zone"] == 0 and vwap["volume_pressure"] == 2 and candle["candle_dominance"] == 0):
        return {
            "condition_id": 33,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.9,
            "entry_viability": "high",
            "supporting_factors": ["💪 VWAP strong resistance + high volume + bearish candle"]
        }

    # Momentum down + candle strong continuation + burst tail
    if (momentum["entry_bias"] == 0 and candle["sequence_type"] == 3 and micro["tail_pressure_direction"] == 0):
        return {
            "condition_id": 34,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.85,
            "entry_viability": "high",
            "supporting_factors": ["🚀 Momentum down + strong continuation + burst tail"]
        }

    # Overbought spike + strong bearish engulfing + VWAP rejection (Updated)
    if momo_up and c_seq_chop and v_bear:
        return {
            "condition_id": 35,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.7,
            "entry_viability": "medium",
            "supporting_factors": ["📈 Overbought/weak structure + VWAP rejection (filtered)"]
        }

    # Momentum slope zone 2 + continuation short + high volume
    if (momentum["momentum_slope_zone"] == 0 and candle["sequence_type"] == 3 and vwap["volume_pressure"] == 2):
        return {
            "condition_id": 36,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.9,
            "entry_viability": "high",
            "supporting_factors": ["📉 Strong momentum slope down + continuation + high volume"]
        }

    # Wick-dominant bar + sequence type 3 + microstructure weak
    if (candle["candle_dominance"] == 0 and candle["sequence_type"] == 3 and micro["micro_risk_zone"] == 0):
        return {
            "condition_id": 37,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.85,
            "entry_viability": "high",
            "supporting_factors": ["🕯️ Wick-dominant bar + continuation + micro weak"]
        }

    # Double confirmation: bearish candle + bearish VWAP
    if (candle["candle_dominance"] == 0 and vwap["vwap_zone"] == 0):
        return {
            "condition_id": 38,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.9,
            "entry_viability": "high",
            "supporting_factors": ["✅ Double confirmation: bearish candle + bearish VWAP"]
        }

    # Breakout bias down + candle strength 2 + burst type 1
    if (breakout["breakout_bias"] == 0 and candle["pattern_strength"] == 2 and micro["burst_type"] == 1):
        return {
            "condition_id": 39,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.95,
            "entry_viability": "high",
            "supporting_factors": ["⚡ Breakout down + strong candle + micro burst"]
        }

    # Candle dominance 0 + VWAP below + tail pressure down
    if (candle["candle_dominance"] == 0 and vwap["vwap_zone"] == 0 and micro["tail_pressure_direction"] == 0):
        return {
            "condition_id": 40,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.9,
            "entry_viability": "high",
            "supporting_factors": ["💪 Candle dominance + VWAP below + tail pressure down"]
        }

    # Momentum top reversal + strong candle + VWAP agreement (Updated)
    if momo_up and c_strong and micro_safe and v_bear:
        return {
            "condition_id": 41,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.7,
            "entry_viability": "medium",
            "supporting_factors": ["📉 Momentum top + strong candle + VWAP agreement"]
        }

    # Late continuation (Updated)
    if brk_expanded and vol_ok and c_dom_bear and micro_burst:
        return {
            "condition_id": 42,
            "final_bias": "short",
            "suggested_trade_direction": "Sell",
            "confidence": 0.8,
            "entry_viability": "medium",
            "supporting_factors": ["✅ Expansion + volume + bearish candle + micro confirmation"]
        }
    
    # --- NEW CATEGORY: Hold/No Trade Conditions ---

    # Micro risk zone = 0 (trap zone) - Already exists, but kept for consistency
    # Condition 1 is the micro risk zone trap.

    # Momentum long + candle bearish + VWAP below — disagreement
    if (momentum["entry_bias"] == 1 and candle["sequence_type"] == 2 and vwap["vwap_zone"] == 0):
        return {
            "condition_id": 43,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 0.0,
            "entry_viability": "low",
            "supporting_factors": ["🚫 Disagreement: Momentum long vs. bearish candle & VWAP below"]
        }

    # Breakout flat + no bias + weak candle — indecision
    if (breakout["breakout_bias"] == 1 and candle["pattern_strength"] == 0):
        return {
            "condition_id": 44,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 0.0,
            "entry_viability": "low",
            "supporting_factors": ["🤔 Indecision: Breakout flat + no bias + weak candle"]
        }

    # VWAP zone 1 + volume neutral + sequence type 0
    if (vwap["vwap_zone"] == 1 and vwap["volume_pressure"] == 1 and candle["sequence_type"] == 0):
        return {
            "condition_id": 45,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 0.0,
            "entry_viability": "low",
            "supporting_factors": ["🚫 Neutrality: VWAP + volume + candle sequence"]
        }

    # Momentum up + no candle strength + micro neutral
    if (momentum["entry_bias"] == 1 and candle["pattern_strength"] == 0 and micro["micro_risk_zone"] == 1):
        return {
            "condition_id": 46,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 0.0,
            "entry_viability": "low",
            "supporting_factors": ["⚠️ Conflict: Momentum up but no candle strength & micro neutral"]
        }

    # Breakout valid but candle is weak & micro disagrees
    if (breakout["breakout_bias"] in [0, 2] and candle["pattern_strength"] == 0 and micro["micro_risk_zone"] == 0):
        return {
            "condition_id": 47,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 0.0,
            "entry_viability": "low",
            "supporting_factors": ["❌ Breakout valid but candle weak & micro disagrees"]
        }

    # VWAP unclear + tail pressure mixed + sequence type 0
    if (vwap["vwap_zone"] == 1 and micro["tail_pressure_direction"] == 1 and candle["sequence_type"] == 0):
        return {
            "condition_id": 48,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 0.0,
            "entry_viability": "low",
            "supporting_factors": ["🚫 Unclear: VWAP + tail pressure mixed + neutral candle"]
        }

    # Breakout bias up + candle bearish + micro tail pressure down
    if (breakout["breakout_bias"] == 2 and candle["sequence_type"] == 2 and micro["tail_pressure_direction"] == 0):
        return {
            "condition_id": 49,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 0.0,
            "entry_viability": "low",
            "supporting_factors": ["⚠️ Conflict: Breakout up vs. bearish candle & tail pressure down"]
        }

    # Momentum down + candle bullish reversal + VWAP reversion zone
    if (momentum["entry_bias"] == 0 and candle["sequence_type"] == 1 and vwap["vwap_zone"] == 1):
        return {
            "condition_id": 50,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 0.0,
            "entry_viability": "low",
            "supporting_factors": ["🔄 Conflict: Momentum down vs. bullish reversal & VWAP reversion"]
        }

    # All 5 models misaligned directionally
    if (momentum["entry_bias"] != breakout["breakout_bias"] and
        momentum["entry_bias"] != candle["candle_dominance"] and
        breakout["breakout_bias"] != candle["candle_dominance"] and
        vwap["vwap_zone"] not in [0, 2]):
        return {
            "condition_id": 51,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 0.0,
            "entry_viability": "low",
            "supporting_factors": ["🚫 All major models misaligned directionally"]
        }

    # Volume collapse + wick-dominant candle + VWAP middle
    if (vwap["volume_pressure"] == 0 and candle["candle_dominance"] == 0 and vwap["vwap_zone"] == 1):
        return {
            "condition_id": 52,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 0.0,
            "entry_viability": "low",
            "supporting_factors": ["📉 Volume collapse + wick-dominant candle + VWAP middle"]
        }

    # Reversal candle but momentum opposes strongly
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

    # Microstructure neutral + sequence type 0 + VWAP neutral
    if (micro["micro_risk_zone"] == 1 and candle["sequence_type"] == 0 and vwap["vwap_zone"] == 1):
        return {
            "condition_id": 54,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 0.0,
            "entry_viability": "low",
            "supporting_factors": ["🤝 Neutrality: Microstructure + candle sequence + VWAP"]
        }

    # Breakout entry viable but candle strength weak
    if (breakout["breakout_bias"] in [0, 2] and candle["pattern_strength"] == 0):
        return {
            "condition_id": 55,
            "final_bias": "no_trade",
            "suggested_trade_direction": "none",
            "confidence": 0.0,
            "entry_viability": "low",
            "supporting_factors": ["📉 Breakout viable but candle strength weak"]
        }

    # Multiple model conflicts + confidence low (< 0.6)
    if (momentum["entry_bias"] != breakout["breakout_bias"] or
        momentum["entry_bias"] != micro["burst_type"] or
        (vwap["vwap_zone"] == 1 and candle["pattern_strength"] == 0)):
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