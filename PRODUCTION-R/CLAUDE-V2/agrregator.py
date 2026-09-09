def aggregate_predictions(raw, min_conf=0.0):

    direction_score = 0
    confirmation_score = 0
    active_models = 0
    kill_switch = False

    # direction mapping
    dir_map = {
        "bullish": 1,
        "above_away": 1,
        "bearish": -1,
        "below_away": -1,
        "flat": 0,
        "contested": 0,
        "oscillate_near": 0
    }

    # confirmation weights
    confirm_map = {
        "yes": 2,
        "confirmed": 2,
        "partial": 1,
        "continuation": 2,
        "stable": 1,
        "accelerating": 2,
        "fading": -2,
        "failed": -2,
        "no": 0,
        "indecision": 0
    }

    # -------------------------
    # MOMENTUM
    # -------------------------
    m = raw.get("momentum")
    if m and m.get("trade_active"):
        active_models += 1

        direction = m["next_direction"]["label"]
        direction_score += dir_map.get(direction, 0)

        q = m["next_momentum_quality"]["label"]
        confirmation_score += confirm_map.get(q, 0)

        regime = m["next_regime_behaviour"]["label"]

        if regime == "chop" or q == "fading":
            kill_switch = True

    # -------------------------
    # BREAKOUT
    # -------------------------
    b = raw.get("breakout")
    if b and b.get("trade_active"):
        active_models += 1

        direction = b["next_direction"]["label"]
        direction_score += dir_map.get(direction, 0)

        ft = b["bk_follow_through"]["label"]
        confirmation_score += confirm_map.get(ft, 0)

        if b["next_pattern"]["label"] == "reversal":
            kill_switch = True

    # -------------------------
    # MICROSTRUCTURE
    # -------------------------
    micro = raw.get("microstructure")
    if micro and micro.get("trade_active"):
        active_models += 1

        direction = micro["next_flow_dir"]["label"]
        direction_score += dir_map.get(direction, 0)

        mf = micro["momentum_follow"]["label"]
        confirmation_score += confirm_map.get(mf, 0)

        if micro["next_bar_intent"]["label"] == "absorption":
            kill_switch = True

        if mf == "failed":
            kill_switch = True

    # -------------------------
    # VWAP
    # -------------------------
    v = raw.get("vwap")
    if v and v.get("trade_active"):
        active_models += 1

        direction = v["next_vwap_dir"]["label"]
        direction_score += dir_map.get(direction, 0)

        vf = v["vwap_follow_through"]["label"]
        confirmation_score += confirm_map.get(vf, 0)

        if v["vwap_interaction"]["label"] == "rejection":
            kill_switch = True

    # -------------------------
    # CANDLE
    # -------------------------
    c = raw.get("candle")
    if c and c.get("trade_active"):
        active_models += 1

        direction = c["next_direction"]["label"]
        direction_score += dir_map.get(direction, 0)

        seq = c["seq_continuation"]["label"]
        confirmation_score += confirm_map.get(seq, 0)

    # -------------------------
    # MASTER GATE
    # -------------------------
    if active_models <= 1:
        return {"signal": None, "confidence": 0.0, "trade_active": False}

    if kill_switch:
        return {"signal": None, "confidence": 0.0, "trade_active": False}

    # -------------------------
    # DIRECTION DECISION
    # -------------------------
    if direction_score >= 3:
        signal = "bullish"
    elif direction_score >= 1:
        signal = "bullish"
    elif direction_score <= -3:
        signal = "bearish"
    elif direction_score <= -1:
        signal = "bearish"
    else:
        signal = None

    # -------------------------
    # CONFIDENCE
    # -------------------------
    conf = max(0.0, min(1.0, confirmation_score / 10))

    if conf < min_conf:
        signal = None

    return {
        "signal": signal,
        "confidence": conf,
        "trade_active": signal is not None
    }