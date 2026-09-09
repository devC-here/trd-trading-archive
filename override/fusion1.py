from override_enginesell import check_override_conditions

def fuse_model_outputs(momentum, breakout, micro, vwap, candle):
    reasons = []
    score = 0.0
    votes = []

    # ----------------------------------
    # 🔒 STEP 1: CHECK OVERRIDE CONDITIONS
    # ----------------------------------
    override_result = check_override_conditions(momentum, breakout, micro, vwap, candle)
    if override_result is not None:
        return override_result  # Shortcut if override matched

    # ----------------------------------
    # ✅ STEP 2: NORMAL FUSION SCORING
    # ----------------------------------

    # Momentum
    if momentum["entry_bias"] == 1:
        votes.append("long")
        score += 1.5
        reasons.append("Momentum favors long")
    elif momentum["entry_bias"] == 0:
        votes.append("short")
        score += 1.5
        reasons.append("Momentum favors short")

    # Breakout
    if breakout["entry_viability"] == 2:
        score += 1.2
        reasons.append("Breakout confirmed")
    elif breakout["entry_viability"] == 1:
        score += 0.5
        reasons.append("Possible breakout condition")

    if breakout["breakout_bias"] == 2:
        votes.append("long")
        reasons.append("Breakout bias long")
    elif breakout["breakout_bias"] == 0:
        votes.append("short")
        reasons.append("Breakout bias short")

    # Microstructure
    if micro["micro_risk_zone"] == 2:
        score += 0.8
        reasons.append("Microstructure burst zone")

    # VWAP
    if vwap["vwap_zone"] == 2 and vwap["volume_pressure"] == 2:
        votes.append("long")
        score += 1.0
        reasons.append("VWAP bullish zone with volume")
    elif vwap["vwap_zone"] == 0 and vwap["volume_pressure"] == 2:
        votes.append("short")
        score += 0.6
        reasons.append("VWAP bearish zone with climax volume")
    elif vwap["vwap_zone"] == 1:
        score += 0.3
        reasons.append("VWAP neutrality")

    # Candle pattern
    if candle["sequence_type"] == 1 and candle["candle_dominance"] == 2:
        votes.append("long")
        score += 0.8
        reasons.append("Bullish reversal")
    elif candle["sequence_type"] == 2 and candle["candle_dominance"] == 0:
        votes.append("short")
        score += 0.8
        reasons.append("Bearish reversal")
    elif candle["sequence_type"] == 3:
        if candle["candle_dominance"] == 2:
            votes.append("long")
            score += 0.6
            reasons.append("Bullish continuation")
        elif candle["candle_dominance"] == 0:
            votes.append("short")
            score += 0.6
            reasons.append("Bearish continuation")

    # ----------------------------
    # 🧠 FINAL DECISION
    # ----------------------------
    long_votes = votes.count("long")
    short_votes = votes.count("short")

    if long_votes > short_votes:
        bias = "long"
        suggested = "Buy"
    elif short_votes > long_votes:
        bias = "short"
        suggested = "Sell"
    else:
        bias = "no_trade"
        suggested = "none"
        reasons.append("No strong vote majority")

    confidence = min(1.0, round(score / 5.0, 3))
    if score >= 3:
        viability = "high"
    elif score >= 2:
        viability = "medium"
    else:
        viability = "low"

    return {
        "final_bias": bias,
        "suggested_trade_direction": suggested,
        "confidence": confidence,
        "entry_viability": viability,
        "supporting_factors": reasons
    }
