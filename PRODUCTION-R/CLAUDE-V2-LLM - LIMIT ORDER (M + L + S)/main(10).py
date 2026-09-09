import MetaTrader5 as mt5 
import logging
import time
from datetime import datetime

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
from placer import execute_automated_trade, get_m30_calibrated, close_profitable_trades , update_existing_trade_sltp , close_position 
from aggregator import aggregate_predictions  # renamed from agrregator

# ===============================
# Config
# ===============================
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M15

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

PREDICTORS = {}


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

    # warm up the fine-tuned model now so first cycle has no delay
    logging.info("Loading TradingExpert model ...")
    aggregate_predictions({k: {"trade_active": False} for k in PREDICTORS})
    logging.info("TradingExpert model ready.")


# ------------------------------------------------------------
# Time restriction
# ------------------------------------------------------------
def is_restricted_time(now):

    now_mins = now.hour * 60 + now.minute

    start_times = [90, 330, 810, 1110]
    duration = 90

    for start in start_times:
        if start <= now_mins < start + duration:
            return True

    return False


# ------------------------------------------------------------
# Prediction step
# ------------------------------------------------------------
def get_raw_predictions(symbol, timeframe):

    raw = {}

    for name, model in PREDICTORS.items():

        r = model.predict(symbol, timeframe)
        s = r.get("_summary", {})

        signal = r.get("signal") or s.get("signal") or s.get("label")
        conf = r.get("confidence", s.get("signal_conf", s.get("confidence", 0)))
        active = r.get("trade_active", s.get("trade_active", False))

        raw[name] = {
            "signal": signal,
            "confidence": conf,
            "trade_active": active
        }

        logging.info(
            "%s → signal=%s | conf=%.2f | active=%s",
            name.upper(), signal, conf, active
        )

    return raw
# ------------------------------------------------------------
# Close position
# ------------------------------------------------------------

# ------------------------------------------------------------
# Place trade
# ------------------------------------------------------------
def place_new_trade(direction):

    tick = mt5.symbol_info_tick(SYMBOL)

    if not tick:
        logging.error("No tick data.")
        return False

    spread = tick.ask - tick.bid
    estimate = compute_sltp(SYMBOL, TIMEFRAME)

    return execute_automated_trade(
        SYMBOL,
        direction,
        estimate 
    )


# ------------------------------------------------------------
# Trading cycle
# ------------------------------------------------------------
def run_single_cycle():

    # --- Calibration moved here ---
    sleep_sec = get_m30_calibrated(SYMBOL, offset= 0)

    if not sleep_sec or sleep_sec <= 0:
        time.sleep(1)
        return

    early_sleep = max(0, sleep_sec - 2)

    logging.info("Sleeping %.3f seconds (early wake).", early_sleep)
    time.sleep(early_sleep)

    logging.info("Closing profitable trades.")
    close_profitable_trades(SYMBOL)

    time.sleep(2)

    # --- Original logic starts here ---
    now = datetime.now()
    logging.info("=== M30 Cycle @ %s ===", now)

    restricted = is_restricted_time(now)

    raw = get_raw_predictions(SYMBOL, TIMEFRAME)
    agg = aggregate_predictions(raw)

    decision = agg.get("signal")
    confidence = agg.get("confidence", 0)

    logging.info(
        "AGGREGATOR → decision=%s | confidence=%.2f",
        decision,
        confidence
    )

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
            update_existing_trade_sltp(pos , SYMBOL ,TIMEFRAME)
        return

    for pos in opposite:
        close_position(pos)

    if not restricted:
        placed = place_new_trade(decision)
        logging.info("Trade placed: %s", placed)
    else:
        logging.info("Trade blocked due to restriction window.")
        
def main():

    if not mt5.initialize():
        logging.error("MT5 init failed")
        return

    if not ensure_symbol(SYMBOL):
        return

    initialize_predictors()

    logging.info("M30 scheduler started.")

    try:
        while True:
            run_single_cycle()

    except KeyboardInterrupt:
        logging.info("Stopped by user.")

    finally:
        mt5.shutdown()
        logging.info("MT5 shutdown.")

if __name__ == "__main__":
    main()
