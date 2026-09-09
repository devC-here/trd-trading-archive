def elaborate_market_conditions(predictions_data: dict) -> dict:
    """
    Elaborates on market conditions by converting raw prediction outputs
    into human-readable descriptions based on the provided interpretation guides.
    This version handles inputs where timeframe predictions are nested
    under keys like 'M15_predictions'.

    Args:
        predictions_data (dict): The full dictionary containing predictions,
                                 which may include a 'trade_magic_number'
                                 and nested timeframe predictions
                                 (e.g., 'M15_predictions').

    Returns:
        dict: A dictionary containing elaborated market conditions for each
              timeframe and a consolidated summary.
    """
    elaborated_conditions = {}

    # Mappings extracted from info.txt
    frama_trend_map = {
        0: "Downtrend",
        1: "Sideways (Consolidation)",
        2: "Uptrend"
    }
    #
    frama_strength_map = {
        0: "Weak (Trend forming or fading)",
        1: "Medium (Reliable but needs confirmation)",
        1: "Strong (Moving with momentum)"
    }
    #
    volatility_bucket_frama_map = {
        0: "Low Volatility (Calm market)",
        1: "Medium Volatility (Balanced)",
        2: "High Volatility (Fast market)"
    }
    #
    position_relative_frama_map = {
        0: "Below FRAMA (Bearish pressure)",
        1: "Near FRAMA (Market equilibrium)",
        2: "Above FRAMA (Bullish sentiment)"
    }
    #
    future_sim_direction_frama_map = {
        0: "Down (Expect short-term drop)",
        1: "Up (Expect short-term rally)"
    }

    #
    vwrsi_trend_map = {
        0: "Downtrend",
        1: "Sideways",
        2: "Uptrend"
    }
    #
    vwrsi_strength_map = {
        0: "Weak",
        1: "Strong"
    }
    #
    volatility_bucket_vwrsi_map = {
        0: "Low Volatility",
        1: "Moderate",
        2: "High",
        3: "Extreme"
    }
    #
    position_relative_vwrsi_map = {
        0: "Below VW-RSI",
        1: "Near VW-RSI",
        2: "Above VW-RSI"
    }
    #
    future_sim_direction_vwrsi_map = {
        0: "Downward",
        1: "Neutral",
        2: "Upward"
    }

    #
    candlestick_pattern_map = {
        0: "Doji (Indecision)",
        1: "Hammer (Bullish reversal)",
        2: "Engulfing (Strong reversal)",
        3: "Shooting Star (Bearish reversal)",
        4: "Marubozu (Strong trend continuation)",
        5: "Inverted Hammer (Potential reversal)",
        6: "Spinning Top (Market indecision)"
    }
    #
    candlestick_direction_map = {
        0: "Sell (Likely price drop)",
        1: "Buy (Likely price rise)",
        2: "No Trade (Unclear/Neutral expectation)"
    }

    #
    adx_regime_map = {
        0: "Weak trend (Range-trading / caution)",
        1: "Moderate trend (Confirm with other signals)",
        2: "Strong trend (Trend-following OK)"
    }
    #
    cci_zone_map = {
        0: "Oversold (< -100) (Consider Buy)",
        1: "Neutral (Wait or confirm others)",
        2: "Overbought (> 100) (Consider Sell)"
    }
    #
    macd_cross_flag_map = {
        0: "Bearish Cross (Sell Signal)",
        1: "No Cross (Neutral)",
        2: "Bullish Cross (Buy Signal)"
    }

    #
    boll_position_class_map = {
        "lower-out": "Price broken below lower band (Oversold / potential bounce)",
        "lower-in": "Price re-entered from lower edge (Reversal forming / weak downtrend)",
        "mean": "Near mid-band (Market indecision)",
        "upper-in": "Re-entered from above (Distribution forming / weakening bullishness)",
        "upper-out": "Price broken above upper band (Overbought / potential top)"
    }
    # (numerical map for 'boll' raw predictions)
    boll_position_class_raw_map = {
        0: "Price is outside the lower band (oversold)",
        1: "Price is near lower band but still inside",
        2: "Near the middle band (20-period SMA)",
        3: "Price is inside upper band",
        4: "Price is outside upper band (overbought)"
    }

    #
    squeeze_state_map = {
        "squeeze": "Bands are very narrow (Market is coiling — breakout likely)",
        "stable": "Normal range (Regular price movement)",
        "expanding": "Bands are widening (Volatility expansion — momentum surge likely)"
    }
    # (numerical map for 'boll' raw predictions)
    squeeze_state_raw_map = {
        0: "Very low volatility (tight bands) -> potential breakout",
        1: "Normal volatility",
        2: "Bands widening -> volatility increasing"
    }

    #
    trend_regime_map = {
        "no_trend": "Flat / range-bound (Avoid trend trades)",
        "weak_trend": "Mild direction (Cautious entry zone)",
        "strong_trend": "Conviction movement (Trend-following trades active)"
    }
    # (numerical map for 'boll' raw predictions)
    trend_regime_raw_map = {
        0: "Market is flat or ranging",
        1: "Trend beginning or uncertain",
        2: "Strong directional move"
    }

    #
    momentum_zone_map = {
        "bearish": "RSI < 40 (Downward bias)",
        "neutral": "RSI 40–60 (Unclear)",
        "bullish": "RSI > 60 (Upward bias)"
    }
    # (numerical map for 'boll' raw predictions)
    momentum_zone_raw_map = {
        0: "RSI < 40",
        1: "RSI between 40–60",
        2: "RSI > 60"
    }

    #
    reversal_signal_map = {
        "none": "No reversal setup",
        "reversal_confirmed": "Reversal signal present (band incursion + MACD shift)"
    }
    # (numerical map for 'boll' raw predictions)
    reversal_signal_raw_map = {
        0: "No reversal signal currently",
        1: "Reversal condition detected (MACD cross + band breach)"
    }

    #
    rejection_flag_map = {
        "no_rejection": "No rejection wick present (Breakout may be real)",
        "rejection_confirmed": "Wick / tail formed at band (Weak breakout — possible fade setup)"
    }
    # (numerical map for 'boll' raw predictions - derived from context)
    rejection_flag_raw_map = {
        0: "No rejection wick present",
        1: "Wick / tail formed at band"
    }

    #
    compression_event_map = {
        "no_compression": "No compression (Market moving normally)",
        "compression_confirmed": "Recent tight consolidation (Likely to break soon)"
    }
    # (numerical map for 'boll' raw predictions)
    compression_event_raw_map = {
        0: "Normal or expanding range",
        1: "Market becoming quiet/tight"
    }

    # Adjusted to extract data from keys like "M15_predictions"
    timeframe_keys = {"M15": "M15_predictions", "M30": "M30_predictions", "H1": "H1_predictions"}

    for tf, tf_key in timeframe_keys.items():
        tf_preds = predictions_data.get(tf_key, {})
        if not tf_preds:
            elaborated_conditions[tf] = {"status": "No data available for this timeframe."}
            continue

        tf_elaborated = {}

        # --- FRAMA Elaborations ---
        frama_data = tf_preds.get("frama", {})
        tf_elaborated["FRAMA_Overall_Trend"] = frama_trend_map.get(frama_data.get("frama_trend"), "Unknown")
        tf_elaborated["FRAMA_Trend_Strength"] = frama_strength_map.get(frama_data.get("frama_strength"), "Unknown")
        tf_elaborated["FRAMA_Volatility_Level"] = volatility_bucket_frama_map.get(frama_data.get("volatility_bucket"), "Unknown")
        tf_elaborated["FRAMA_Price_Relative_To_Line"] = position_relative_frama_map.get(frama_data.get("position_relative"), "Unknown")
        tf_elaborated["FRAMA_Predicted_Near_Term_Direction"] = future_sim_direction_frama_map.get(frama_data.get("future_sim_direction"), "Unknown")

        # --- VWRSI Elaborations ---
        vwrsi_data = tf_preds.get("vwrsi", {})
        tf_elaborated["VWRSI_Trend"] = vwrsi_trend_map.get(vwrsi_data.get("vwrsi_trend"), "Unknown")
        tf_elaborated["VWRSI_Strength"] = vwrsi_strength_map.get(vwrsi_data.get("vwrsi_strength"), "Unknown")
        tf_elaborated["VWRSI_Volatility_Level"] = volatility_bucket_vwrsi_map.get(vwrsi_data.get("volatility_bucket"), "Unknown")
        tf_elaborated["VWRSI_Price_Relative_To_Line"] = position_relative_vwrsi_map.get(vwrsi_data.get("position_relative"), "Unknown")
        tf_elaborated["VWRSI_Predicted_Future_Direction"] = future_sim_direction_vwrsi_map.get(vwrsi_data.get("future_sim_direction"), "Unknown")


        # --- Candlestick Elaborations ---
        candlestick_data = tf_preds.get("candlestick", {})
        # Map string directions to numerical for consistency with map, then back to string
        candlestick_direction_raw = candlestick_data.get("direction")
        candlestick_direction_num = None
        if candlestick_direction_raw == "Sell":
            candlestick_direction_num = 0
        elif candlestick_direction_raw == "Buy":
            candlestick_direction_num = 1
        elif candlestick_direction_raw == "No Trade":
            candlestick_direction_num = 2

        tf_elaborated["Candlestick_Pattern"] = candlestick_pattern_map.get(candlestick_data.get("pattern"), "N/A or Unknown Pattern")
        tf_elaborated["Candlestick_Direction_Prediction"] = candlestick_direction_map.get(candlestick_direction_num, "N/A or No Trade")
        tf_elaborated["Candlestick_Confidence"] = f"{candlestick_data.get('confidence', 0.0) * 100:.2f}%"

        # --- ADX-CCI-MACD Elaborations ---
        adx_cci_macd_data = tf_preds.get("adx_cci_macd", {})
        tf_elaborated["ADX_Regime"] = adx_regime_map.get(adx_cci_macd_data.get("adx_regime"), "Unknown")
        tf_elaborated["CCI_Zone"] = cci_zone_map.get(adx_cci_macd_data.get("cci_zone"), "Unknown")
        tf_elaborated["MACD_Cross_Flag"] = macd_cross_flag_map.get(adx_cci_macd_data.get("macd_cross_flag"), "Unknown")

        # --- Bollinger Bands (position_bollinger - 'advanced') Elaborations ---
        boll_adv_preds = tf_preds.get("position_bollinger", {}).get("predictions", {})
        tf_elaborated["BollAdv_Position_Class"] = boll_position_class_map.get(boll_adv_preds.get("boll_position_class"), "Unknown")
        tf_elaborated["BollAdv_Squeeze_State"] = squeeze_state_map.get(boll_adv_preds.get("squeeze_state"), "Unknown")
        tf_elaborated["BollAdv_Trend_Regime"] = trend_regime_map.get(boll_adv_preds.get("trend_regime"), "Unknown")
        tf_elaborated["BollAdv_Momentum_Zone"] = momentum_zone_map.get(boll_adv_preds.get("momentum_zone"), "Unknown")
        tf_elaborated["BollAdv_Reversal_Signal"] = reversal_signal_map.get(boll_adv_preds.get("reversal_signal"), "Unknown")
        tf_elaborated["BollAdv_Rejection_Flag"] = rejection_flag_map.get(boll_adv_preds.get("rejection_flag"), "Unknown")
        tf_elaborated["BollAdv_Compression_Event"] = compression_event_map.get(boll_adv_preds.get("compression_event"), "Unknown")

        # --- Bollinger Bands (boll - 'raw' with numerical classes) Elaborations ---
        boll_raw_preds = tf_preds.get("boll", {})
        tf_elaborated["BollRaw_Position_Class"] = boll_position_class_raw_map.get(boll_raw_preds.get("boll_position_class", {}).get("class"), "Unknown")
        tf_elaborated["BollRaw_Squeeze_State"] = squeeze_state_raw_map.get(boll_raw_preds.get("squeeze_state", {}).get("class"), "Unknown")
        tf_elaborated["BollRaw_Trend_Regime"] = trend_regime_raw_map.get(boll_raw_preds.get("trend_regime", {}).get("class"), "Unknown")
        tf_elaborated["BollRaw_Momentum_Zone"] = momentum_zone_raw_map.get(boll_raw_preds.get("momentum_zone", {}).get("class"), "Unknown")
        tf_elaborated["BollRaw_Reversal_Signal"] = reversal_signal_raw_map.get(boll_raw_preds.get("reversal_signal", {}).get("class"), "Unknown")
        tf_elaborated["BollRaw_Rejection_Flag"] = rejection_flag_raw_map.get(boll_raw_preds.get("rejection_flag", {}).get("class"), "Unknown")
        tf_elaborated["BollRaw_Compression_Event"] = compression_event_raw_map.get(boll_raw_preds.get("compression_event", {}).get("class"), "Unknown")

        elaborated_conditions[tf] = tf_elaborated

    # --- Consolidated Summary based on Multi-Timeframe Use ---
    high_level_summary = "Overall Market Summary:\n"

    h1_context = elaborated_conditions.get("H1", {})
    m30_trigger = elaborated_conditions.get("M30", {})
    m15_immediate = elaborated_conditions.get("M15", {})

    # Extract human-readable values for summary
    h1_trend = h1_context.get("BollAdv_Trend_Regime", "Unknown")
    h1_momentum = h1_context.get("BollAdv_Momentum_Zone", "unknown")
    h1_squeeze = h1_context.get("BollAdv_Squeeze_State", "unknown")
    high_level_summary += f"  1H Context: Trend is '{h1_trend}' with '{h1_momentum}' momentum. Volatility state is '{h1_squeeze}'.\n"

    m30_trend = m30_trigger.get("BollAdv_Trend_Regime", "Unknown")
    m30_momentum = m30_trigger.get("BollAdv_Momentum_Zone", "unknown")
    m30_squeeze = m30_trigger.get("BollAdv_Squeeze_State", "unknown")
    m30_reversal = m30_trigger.get("BollAdv_Reversal_Signal", "unknown")
    high_level_summary += f"  30M Trigger: Trend is '{m30_trend}' with '{m30_momentum}' momentum. Volatility state is '{m30_squeeze}'. Reversal signal: '{m30_reversal}'.\n"

    m15_trend = m15_immediate.get("BollAdv_Trend_Regime", "Unknown")
    m15_momentum = m15_immediate.get("BollAdv_Momentum_Zone", "unknown")
    m15_squeeze = m15_immediate.get("BollAdv_Squeeze_State", "unknown")
    high_level_summary += f"  15M Immediate: Trend is '{m15_trend}' with '{m15_momentum}' momentum. Volatility state is '{m15_squeeze}'.\n"

    # Example Strategy application from info.txt
    if "strong_trend" in h1_context.get("BollAdv_Trend_Regime", "") and \
       "bullish" in h1_context.get("BollAdv_Momentum_Zone", "") and \
       "squeeze" in m30_trigger.get("BollAdv_Squeeze_State", "") and \
       "Reversal signal present" in m30_trigger.get("BollAdv_Reversal_Signal", "") and \
       "lower-in" in m30_trigger.get("BollAdv_Position_Class", ""):
        high_level_summary += "  **Consolidated Insight:** Strong bullish context with a 30M squeeze and reversal from lower-in, suggesting a potential breakout buy opportunity.\n"
    elif "strong_trend" in h1_context.get("BollAdv_Trend_Regime", "") and \
         "bearish" in h1_context.get("BollAdv_Momentum_Zone", "") and \
         "squeeze" in m30_trigger.get("BollAdv_Squeeze_State", "") and \
         "Reversal signal present" in m30_trigger.get("BollAdv_Reversal_Signal", "") and \
         "upper-in" in m30_trigger.get("BollAdv_Position_Class", ""):
        high_level_summary += "  **Consolidated Insight:** Strong bearish context with a 30M squeeze and reversal from upper-in, suggesting a potential breakout sell opportunity.\n"
    else:
        high_level_summary += "  **Consolidated Insight:** No specific multi-timeframe strategy pattern detected at this moment. Analyze individual timeframe conditions for further action.\n"


    elaborated_conditions["Consolidated_Summary"] = high_level_summary
    return elaborated_conditions