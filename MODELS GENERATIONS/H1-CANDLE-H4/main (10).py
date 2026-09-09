import MetaTrader5 as mt5
import logging
import time
from datetime import datetime
from collections import deque

# ===============================
# Predictor imports
# ===============================
from momentum_deploy import MomentumPredictor
from breakout_deploy import BreakoutPredictor
from microstructure_deploy import MicrostructurePredictor
from vwap_deploy import VwapPredictor
from candle_deploy import CandlePredictor

# ===============================
# Utility imports
# ===============================
from sltp import compute_sltp
from placer import execute_automated_trade, get_m30_calibrated
# Import the updated time-series aggregator function
from aggregator import aggregate_predictions_timeseries 

# ===============================
# Config
# ===============================
SYMBOL = "BTCUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
MAX_HISTORY_DEPTH = 4  # Keeps tracking from t-3 down to t0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

PREDICTORS = {}
# Thread-safe rolling buffer to store historical sub-model states
STATE_HISTORY = deque(maxlen=MAX_HISTORY_DEPTH) 


# ------------------------------------------------------------
# Symbol check
# ------------------------------------------------------------
def ensure_symbol(symbol):
    info = mt5.symbol_info(symbol)

    if info is None:
        logging.error("Symbol not found: %s", symbol)
        return False

    if not info.visible:
        mt5.symbol_select(symbol, True)

    return True


# ------------------------------------------------------------
# Predictor initialization
# ------------------------------------------------------------
def initialize_predictors():
    global PREDICTORS

    PREDICTORS = {
        "momentum": MomentumPredictor(),
        "breakout": BreakoutPredictor(),
        "microstructure": MicrostructurePredictor(),
        "vwap": VwapPredictor(),
        "candle": CandlePredictor(),
    }

    logging.info("All predictors initialized.")

    # Warm up the Gemini Engine with a mock time-series payload
    logging.info("Loading TradingExpert model via Ensemble AI Engine...")
    mock_state = {k: {"trade_active": False} for k in PREDICTORS}
    aggregate_predictions_timeseries([mock_state])
    logging.info("TradingExpert model ready.")


# ------------------------------------------------------------
# Time restriction
# ------------------------------------------------------------
def is_restricted_time(now):
    now_mins = now.hour * 60 + now.minute

    start_times = [1100, 330, 810, 1110]
    duration = 90

    for start in start_times:
        if start <= now_mins < start + duration:
            return True

    return False


# ------------------------------------------------------------
# Prediction step
# ------------------------------------------------------------
def get_raw_predictions(symbol, timeframe):
    """
    Fetches predictions from all sub-models and formats them strictly
    to match the structure expected by aggregator.py's prompt builder.
    """
    raw = {}

    for name, model in PREDICTORS.items():
        # Execute individual model predictions
        r = model.predict(symbol, timeframe)
        
        # We save the raw dictionary returned by the models into their namespaces
        # so aggregator.py can safely parse inner fields like 'next_direction', etc.
        raw[name] = r

        # Extract high-level metrics for standard logging output
        s = r.get("_summary", {})
        signal = r.get("signal") or s.get("signal") or s.get("label")
        conf = r.get("confidence", s.get("signal_conf", s.get("confidence", 0)))
        active = r.get("trade_active", s.get("trade_active", False))

        logging.info(
            "%s → signal=%s | conf=%.2f | active=%s",
            name.upper(), signal, conf, active
        )

    return raw


# ------------------------------------------------------------
# Close position
# ------------------------------------------------------------
def close_position(pos):
    close_type = (
        mt5.ORDER_TYPE_SELL
        if pos.type == mt5.ORDER_TYPE_BUY
        else mt5.ORDER_TYPE_BUY
    )

    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": pos.symbol,
        "volume": pos.volume,
        "type": close_type,
        "position": pos.ticket,
        "deviation": 50,
        "magic": 10000,
        "comment": "Auto-close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    result = mt5.order_send(req)

    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        logging.info("Closed position ticket=%d", pos.ticket)
        return True

    logging.warning("Close failed ticket=%d", pos.ticket)
    return False


# ------------------------------------------------------------
# Close profitable trades
# ------------------------------------------------------------
def close_profitable_trades(symbol):
    positions = mt5.positions_get(symbol=symbol) or []

    for pos in positions:
        if pos.profit > 0:
            logging.info(
                "Closing profitable trade ticket=%d profit=%.2f",
                pos.ticket,
                pos.profit
            )
            close_position(pos)


# ------------------------------------------------------------
# Update SLTP
# ------------------------------------------------------------
def update_existing_trade_sltp(pos):
    tick = mt5.symbol_info_tick(SYMBOL)

    if not tick:
        return False

    estimate = compute_sltp(SYMBOL, TIMEFRAME)

    if estimate is None:
        return False

    target = estimate
    draw = estimate * 0.9

    if pos.type == mt5.ORDER_TYPE_BUY:
        sl = tick.ask - draw
        tp = tick.ask + target
    else:
        sl = tick.bid + draw
        tp = tick.bid - target

    req = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": SYMBOL,
        "position": pos.ticket,
        "sl": round(sl, 5),
        "tp": round(tp, 5),
    }

    mt5.order_send(req)

    logging.info(
        "Updated SLTP ticket=%d SL=%.5f TP=%.5f",
        pos.ticket,
        sl,
        tp
    )

    return True


# ------------------------------------------------------------
# Place trade
# ------------------------------------------------------------
def place_new_trade(direction):
    tick = mt5.symbol_info_tick(SYMBOL)

    if not tick:
        logging.error("No tick data.")
        return False

    estimate = compute_sltp(SYMBOL, TIMEFRAME)

    if estimate is None:
        return False

    target = estimate * 0.9
    draw = estimate * 0.7

    if direction == "Buy":
        sl = tick.ask - draw
        tp = tick.ask + target
    else:
        sl = tick.bid + draw
        tp = tick.bid - target

    return execute_automated_trade(
        SYMBOL,
        direction,
        sl=round(sl, 5),
        tp=round(tp, 5)
    )


# ------------------------------------------------------------
# Trading cycle
# ------------------------------------------------------------
def run_single_cycle():
    now = datetime.now()

    logging.info("=== M30 Cycle @ %s ===", now)

    restricted = is_restricted_time(now)

    # 1. Get current (t0) features from all sub-models
    current_raw = get_raw_predictions(SYMBOL, TIMEFRAME)

    # 2. Append to rolling history queue
    STATE_HISTORY.append(current_raw)

    # 3. Pass the entire chronological list (up to 4 steps) to the engine
    # Convert deque to standard list so it's safely processed
    history_payload = list(STATE_HISTORY)
    
    logging.info("Sending sequence payload to Gemini. Sequence length: %d", len(history_payload))
    agg = aggregate_predictions_timeseries(history_payload, min_conf=0.40)

    decision = agg.get("signal")
    confidence = agg.get("confidence", 0)
    reason = agg.get("reason", "No reason provided.")

    logging.info(
        "AGGREGATOR RESPONSE → decision=%s | confidence=%.2f",
        decision,
        confidence
    )
    logging.info("AGGREGATOR REASONING → %s", reason)

    if decision is None:
        logging.info("Decision = HOLD")
        return

    if decision == "bullish":
        decision = "Buy"
    elif decision == "bearish":
        decision = "Sell"
    else:
        return

    positions = mt5.positions_get(symbol=SYMBOL) or []

    buys = [p for p in positions if p.type == mt5.ORDER_TYPE_BUY]
    sells = [p for p in positions if p.type == mt5.ORDER_TYPE_SELL]

    same = buys if decision == "Buy" else sells
    opposite = sells if decision == "Buy" else buys

    if same:
        logging.info(
            "Existing %s position found → updating SLTP",
            decision
        )
        for pos in same:
            update_existing_trade_sltp(pos)
        return

    for pos in opposite:
        close_position(pos)

    if not restricted:
        placed = place_new_trade(decision)
        logging.info("Trade placed: %s", placed)
    else:
        logging.info("Trade blocked due to restriction window.")


# ------------------------------------------------------------
# Scheduler
# ------------------------------------------------------------
def main():
    if not mt5.initialize():
        logging.error("MT5 init failed")
        return

    if not ensure_symbol(SYMBOL):
        return

    initialize_predictors()

    logging.info("M30 calibrated scheduler started.")

    try:
        while True:
            sleep_sec = get_m30_calibrated(SYMBOL, offset=+1)

            if not sleep_sec or sleep_sec <= 0:
                time.sleep(1)
                continue

            early_sleep = max(0, sleep_sec - 2)

            logging.info(
                "Sleeping %.3f seconds (early wake).",
                early_sleep
            )

            time.sleep(early_sleep)

            logging.info("Closing profitable trades.")
            close_profitable_trades(SYMBOL)

            time.sleep(2)

            run_single_cycle()

    except KeyboardInterrupt:
        logging.info("Stopped by user.")

    finally:
        mt5.shutdown()
        logging.info("MT5 shutdown.")


if __name__ == "__main__":
    main()