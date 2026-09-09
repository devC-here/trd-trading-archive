# fusion_engine.py - Rebuilt for Robustness

import json # For pretty printing elaborated conditions if needed for debugging

def get_final_trading_signal(predictions_by_model):
    """
    Applies a robust hierarchical fusion logic to produce a final trading signal.
    Prioritizes strong, clear signals and falls back to a weighted voting system.

    :param predictions_by_model: Dict[str, dict] – model name to its prediction output.
                                 Expected keys: "frama", "vw_rsi", "candlestick",
                                 "adx_cci_macd", "boll" (for advanced), "position_bollinger" (for raw).
    :return: str – "Buy", "Sell", or "Neutral"
    """
    print("\n--- Fusion Engine: Starting Signal Aggregation ---")
    
    # --- 1. Safely Extract and Standardize Model Predictions ---
    # Use .get() with empty dict as default to prevent KeyError if a predictor is missing
    frama_preds = predictions_by_model.get("frama", {})
    vw_rsi_preds = predictions_by_model.get("vw_rsi", {})
    candle_preds = predictions_by_model.get("candlestick", {})
    adx_cci_macd_preds = predictions_by_model.get("adx_cci_macd", {})
    # Assuming "boll" from main4.py is the advanced Bollinger, and "position_bollinger" is the raw/rule-based one
    boll_adv_preds = predictions_by_model.get("boll", {})
    boll_raw_preds = predictions_by_model.get("position_bollinger", {})

    # --- Helper to get standardized signal values from raw predictions ---
    def get_frama_trend(preds):
        # 0: Downtrend, 1: Sideways, 2: Uptrend
        trend = preds.get("frama_trend")
        if trend == 2: return "Buy"
        if trend == 0: return "Sell"
        return "Neutral" # Or "No Signal"

    def get_frama_future_direction(preds):
        # 0: Down, 1: Up
        direction = preds.get("future_sim_direction")
        if direction == 1: return "Buy"
        if direction == 0: return "Sell"
        return "Neutral"

    def get_frama_volatility_bucket(preds):
        # 0: Low, 1: Medium, 2: High
        return preds.get("volatility_bucket")

    def get_vwrsi_trend(preds):
        # 0: Downtrend, 1: Sideways, 2: Uptrend
        trend = preds.get("vwrsi_trend")
        if trend == 2: return "Buy"
        if trend == 0: return "Sell"
        return "Neutral"

    def get_vwrsi_future_direction(preds):
        # 0: Downward, 1: Neutral, 2: Upward
        direction = preds.get("future_sim_direction")
        if direction == 2: return "Buy"
        if direction == 0: return "Sell"
        return "Neutral"

    def get_candle_direction(preds):
        return preds.get("direction", "No Trade")

    def get_candle_confidence(preds):
        return preds.get("confidence", 0.0)

    def get_adx_macd_cross(preds):
        # 0: Bearish Cross (Sell), 1: No Cross (Neutral), 2: Bullish Cross (Buy)
        cross = preds.get("macd_cross_flag")
        if cross == 2: return "Buy"
        if cross == 0: return "Sell"
        return "Neutral"

    def get_adx_regime(preds):
        # 0: Weak trend, 1: Moderate trend, 2: Strong trend
        return preds.get("adx_regime")

    def get_boll_adv_reversal_signal(preds):
        # 0: No reversal, 1: Reversal signal present
        return preds.get("reversal_signal")

    def get_boll_adv_momentum_zone(preds):
        # "bearish", "neutral", "bullish"
        return preds.get("momentum_zone")

    def get_boll_adv_rejection_flag(preds):
        # 0: No rejection, 1: Rejection present
        return preds.get("rejection_flag")

    def get_boll_adv_position_class(preds):
        # "lower-out", "lower-in", "mean", "upper-in", "upper-out"
        return preds.get("boll_position_class")

    def get_boll_adv_compression_event(preds):
        # 0: No compression, 1: Compression present
        return preds.get("compression_event")

    def get_boll_adv_squeeze_state(preds):
        # "squeeze", "stable", "expanding"
        return preds.get("squeeze_state")

    def get_boll_raw_position_class(preds):
        # "lower-out", "lower-in", "mean", "upper-in", "upper-out"
        return preds.get("boll_position_class")

    def get_boll_raw_squeeze_state(preds):
        # "squeeze", "stable", "expanding"
        return preds.get("squeeze_state")
    
    # --- 2. Priority Overrides (Early Exits for Strong Signals) ---
    # These rules are designed to capture high-conviction scenarios.
    # Order matters: more specific/critical rules should come first.

    # Candlestick high confidence signal
    candle_dir = get_candle_direction(candle_preds)
    candle_conf = get_candle_confidence(candle_preds)
    if candle_dir in ["Buy", "Sell"] and candle_conf > 0.75:
        print(f"Fusion Rule: High Confidence Candlestick Signal ({candle_dir}, {candle_conf:.2f})")
        return candle_dir

    # Strong Trend with MACD Cross (ADX/MACD/CCI)
    adx_macd_cross = get_adx_macd_cross(adx_cci_macd_preds)
    adx_regime = get_adx_regime(adx_cci_macd_preds)
    if adx_macd_cross == "Buy" and adx_regime == 2: # Strong trend (2) and Bullish Cross (Buy)
        print("Fusion Rule: Strong Trend & Bullish MACD Cross")
        return "Buy"
    if adx_macd_cross == "Sell" and adx_regime == 2: # Strong trend (2) and Bearish Cross (Sell)
        print("Fusion Rule: Strong Trend & Bearish MACD Cross")
        return "Sell"

    # Reversal Signal with Momentum Confirmation (Bollinger Advanced)
    boll_adv_reversal = get_boll_adv_reversal_signal(boll_adv_preds)
    boll_adv_momentum = get_boll_adv_momentum_zone(boll_adv_preds)
    if boll_adv_reversal == 1: # Reversal signal present
        if boll_adv_momentum == "bullish":
            print("Fusion Rule: Bollinger Advanced Reversal + Bullish Momentum")
            return "Buy"
        if boll_adv_momentum == "bearish":
            print("Fusion Rule: Bollinger Advanced Reversal + Bearish Momentum")
            return "Sell"

    # Rejection at Bollinger Band Extremes (Bollinger Advanced)
    boll_adv_rejection = get_boll_adv_rejection_flag(boll_adv_preds)
    boll_adv_pos_class = get_boll_adv_position_class(boll_adv_preds)
    if boll_adv_rejection == 1 and boll_adv_pos_class in ["upper-out", "lower-out"]:
        if boll_adv_pos_class == "upper-out":
            print("Fusion Rule: Bollinger Advanced Rejection at Upper-Out (Sell)")
            return "Sell"
        if boll_adv_pos_class == "lower-out":
            print("Fusion Rule: Bollinger Advanced Rejection at Lower-Out (Buy)")
            return "Buy"

    # Volatility Expansion with FRAMA Future Direction (Frama & Bollinger Advanced)
    frama_vol_bucket = get_frama_volatility_bucket(frama_preds)
    boll_adv_squeeze = get_boll_adv_squeeze_state(boll_adv_preds)
    frama_future_dir = get_frama_future_direction(frama_preds)
    if frama_vol_bucket == 2 and boll_adv_squeeze == "expanding": # High Volatility & Expanding Bands
        if frama_future_dir == "Buy":
            print("Fusion Rule: High Volatility Expansion + FRAMA Future Buy")
            return "Buy"
        elif frama_future_dir == "Sell":
            print("Fusion Rule: High Volatility Expansion + FRAMA Future Sell")
            return "Sell"
        # If future_sim_direction is Neutral or None, this rule won't trigger a trade.

    # Compression leading to breakout with FRAMA Future Direction (Bollinger Advanced & Frama)
    boll_adv_compression = get_boll_adv_compression_event(boll_adv_preds)
    if boll_adv_compression == 1 and boll_adv_squeeze == "squeeze": # Compression & Squeeze
        if frama_future_dir == "Buy":
            print("Fusion Rule: Bollinger Advanced Compression/Squeeze + FRAMA Future Buy")
            return "Buy"
        elif frama_future_dir == "Sell":
            print("Fusion Rule: Bollinger Advanced Compression/Squeeze + FRAMA Future Sell")
            return "Sell"

    # VW-RSI Strong Trend and Future Direction Alignment
    vwrsi_trend = get_vwrsi_trend(vw_rsi_preds)
    vwrsi_future_dir = get_vwrsi_future_direction(vw_rsi_preds)
    if vwrsi_trend == "Buy" and vwrsi_future_dir == "Buy": # Uptrend & Upward prediction
        print("Fusion Rule: VW-RSI Strong Uptrend & Future Upward")
        return "Buy"
    if vwrsi_trend == "Sell" and vwrsi_future_dir == "Sell": # Downtrend & Downward prediction
        print("Fusion Rule: VW-RSI Strong Downtrend & Future Downward")
        return "Sell"

    # CORRECTED: Bollinger Raw Breakout Continuation (NOT Reversal)
    boll_raw_pos_class = get_boll_raw_position_class(boll_raw_preds)
    boll_raw_squeeze = get_boll_raw_squeeze_state(boll_raw_preds)
    if boll_raw_pos_class == "lower-out" and boll_raw_squeeze == "expanding":
        # Price broke below lower band, and bands are expanding -> continuation of downtrend (Sell)
        print("Fusion Rule: CORRECTED Bollinger Raw Lower-Out & Expanding (Sell)")
        return "Sell"
    if boll_raw_pos_class == "upper-out" and boll_raw_squeeze == "expanding":
        # Price broke above upper band, and bands are expanding -> continuation of uptrend (Buy)
        print("Fusion Rule: CORRECTED Bollinger Raw Upper-Out & Expanding (Buy)")
        return "Buy"

    print("--- Fusion Engine: No Priority Override Triggered. Proceeding to Weighted Vote. ---")

    # --- 3. Fallback Weighted Voting System ---
    # Assign weights to signals. Higher weight = more influential.
    # You can tune these weights based on backtesting/performance.
    buy_votes = 0
    sell_votes = 0

    # Weights for different signal types (example values, tune as needed)
    WEIGHT_STRONG_TREND = 2
    WEIGHT_MOMENTUM = 1.5
    WEIGHT_REVERSAL = 2
    WEIGHT_CANDLE_CONF = 1.0 # Base weight for confident candle signals
    WEIGHT_VOLATILITY = 1
    WEIGHT_BREAKOUT = 2

    # FRAMA Signals
    frama_trend = get_frama_trend(frama_preds)
    if frama_trend == "Buy":
        buy_votes += WEIGHT_STRONG_TREND
        print(f"  Vote: FRAMA Trend Buy (+{WEIGHT_STRONG_TREND})")
    elif frama_trend == "Sell":
        sell_votes += WEIGHT_STRONG_TREND
        print(f"  Vote: FRAMA Trend Sell (+{WEIGHT_STRONG_TREND})")

    frama_future_dir = get_frama_future_direction(frama_preds)
    if frama_future_dir == "Buy":
        buy_votes += WEIGHT_MOMENTUM
        print(f"  Vote: FRAMA Future Direction Buy (+{WEIGHT_MOMENTUM})")
    elif frama_future_dir == "Sell":
        sell_votes += WEIGHT_MOMENTUM
        print(f"  Vote: FRAMA Future Direction Sell (+{WEIGHT_MOMENTUM})")

    # VW-RSI Signals
    vwrsi_trend = get_vwrsi_trend(vw_rsi_preds)
    if vwrsi_trend == "Buy":
        buy_votes += WEIGHT_STRONG_TREND
        print(f"  Vote: VW-RSI Trend Buy (+{WEIGHT_STRONG_TREND})")
    elif vwrsi_trend == "Sell":
        sell_votes += WEIGHT_STRONG_TREND
        print(f"  Vote: VW-RSI Trend Sell (+{WEIGHT_STRONG_TREND})")

    vwrsi_future_dir = get_vwrsi_future_direction(vw_rsi_preds)
    if vwrsi_future_dir == "Buy":
        buy_votes += WEIGHT_MOMENTUM
        print(f"  Vote: VW-RSI Future Direction Buy (+{WEIGHT_MOMENTUM})")
    elif vwrsi_future_dir == "Sell":
        sell_votes += WEIGHT_MOMENTUM
        print(f"  Vote: VW-RSI Future Direction Sell (+{WEIGHT_MOMENTUM})")

    # ADX/MACD/CCI Signals
    adx_macd_cross = get_adx_macd_cross(adx_cci_macd_preds)
    if adx_macd_cross == "Buy":
        buy_votes += WEIGHT_MOMENTUM
        print(f"  Vote: ADX/MACD/CCI MACD Cross Buy (+{WEIGHT_MOMENTUM})")
    elif adx_macd_cross == "Sell":
        sell_votes += WEIGHT_MOMENTUM
        print(f"  Vote: ADX/MACD/CCI MACD Cross Sell (+{WEIGHT_MOMENTUM})")
    
    adx_regime = get_adx_regime(adx_cci_macd_preds)
    if adx_regime == 2: # Strong trend
        if adx_macd_cross == "Buy": # Add extra weight if strong trend and bullish cross
            buy_votes += 0.5 # Small bonus
            print(f"  Vote: ADX Strong Trend Bonus Buy (+0.5)")
        elif adx_macd_cross == "Sell": # Add extra weight if strong trend and bearish cross
            sell_votes += 0.5 # Small bonus
            print(f"  Vote: ADX Strong Trend Bonus Sell (+0.5)")


    # Candlestick Signals (lower confidence threshold for voting)
    candle_dir = get_candle_direction(candle_preds)
    candle_conf = get_candle_confidence(candle_preds)
    if candle_dir == "Buy" and candle_conf > 0.6:
        buy_votes += WEIGHT_CANDLE_CONF
        print(f"  Vote: Candlestick Buy (Conf: {candle_conf:.2f}) (+{WEIGHT_CANDLE_CONF})")
    elif candle_dir == "Sell" and candle_conf > 0.6:
        sell_votes += WEIGHT_CANDLE_CONF
        print(f"  Vote: Candlestick Sell (Conf: {candle_conf:.2f}) (+{WEIGHT_CANDLE_CONF})")

    # Bollinger Advanced Signals
    boll_adv_momentum = get_boll_adv_momentum_zone(boll_adv_preds)
    if boll_adv_momentum == "bullish":
        buy_votes += WEIGHT_MOMENTUM
        print(f"  Vote: Bollinger Advanced Bullish Momentum (+{WEIGHT_MOMENTUM})")
    elif boll_adv_momentum == "bearish":
        sell_votes += WEIGHT_MOMENTUM
        print(f"  Vote: Bollinger Advanced Bearish Momentum (+{WEIGHT_MOMENTUM})")
    
    boll_adv_reversal = get_boll_adv_reversal_signal(boll_adv_preds)
    if boll_adv_reversal == 1:
        # Reversal signals are high impact, but need direction.
        # If momentum is aligned, it's already covered. Add a small directional push if not.
        if boll_adv_momentum == "bullish": # Already added, but for clarity
            pass
        elif boll_adv_momentum == "bearish": # Already added, but for clarity
            pass
        else: # If neutral momentum, but reversal signal is present, give a slight nudge based on position
            boll_adv_pos_class = get_boll_adv_position_class(boll_adv_preds)
            if boll_adv_pos_class == "lower-out" or boll_adv_pos_class == "lower-in":
                buy_votes += (WEIGHT_REVERSAL / 2) # Half weight for less clear direction
                print(f"  Vote: Bollinger Advanced Reversal (Lower) (+{WEIGHT_REVERSAL/2})")
            elif boll_adv_pos_class == "upper-out" or boll_adv_pos_class == "upper-in":
                sell_votes += (WEIGHT_REVERSAL / 2)
                print(f"  Vote: Bollinger Advanced Reversal (Upper) (+{WEIGHT_REVERSAL/2})")

    # Bollinger Raw Signals (for voting, less priority than overrides)
    boll_raw_pos_class = get_boll_raw_position_class(boll_raw_preds)
    if boll_raw_pos_class == "lower-out" or boll_raw_pos_class == "lower-in":
        buy_votes += WEIGHT_VOLATILITY # Price is low, potential bounce
        print(f"  Vote: Bollinger Raw Lower Position (+{WEIGHT_VOLATILITY})")
    elif boll_raw_pos_class == "upper-out" or boll_raw_pos_class == "upper-in":
        sell_votes += WEIGHT_VOLATILITY # Price is high, potential pullback
        print(f"  Vote: Bollinger Raw Upper Position (+{WEIGHT_VOLATILITY})")

    print(f"\n--- Fusion Engine: Total Votes ---")
    print(f"  Buy Votes: {buy_votes:.2f}")
    print(f"  Sell Votes: {sell_votes:.2f}")

    # Final Decision based on Weighted Votes
    if buy_votes > sell_votes:
        print("--- Fusion Engine: Final Decision: BUY (Weighted Vote) ---")
        return "Buy"
    elif sell_votes > buy_votes:
        print("--- Fusion Engine: Final Decision: SELL (Weighted Vote) ---")
        return "Sell"
    else:
        print("--- Fusion Engine: Final Decision: NEUTRAL (Equal Votes or No Votes) ---")
        return "Neutral"