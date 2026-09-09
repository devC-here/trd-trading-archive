import MetaTrader5 as mt5
import logging
import time
from datetime import datetime

# ===============================
# Predictor imports (UNCHANGED)
# ===============================
from momentumdeploy import MomentumPredictor
from breakoutdeploy import BreakoutPredictor
from microstructure import MicrostructurePredictor
from VWdeploy import VwapVolumePredictor
from candledeploy import CandlestickSequencePredictor

# ===============================
# Condition engine (UNCHANGED)
# ===============================
from code1 import evaluate_market_state, trading_conditions

# ===============================
# Trade execution + scheduler (UNCHANGED)
# ===============================
from placer import (
    execute_automated_trade,
    get_m30_calibrated,
    prev_candle_extremes,
    place_stop_order,
    clear_orphaned_stops
)

# ===============================
# CONFIG
# ===============================

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY"]   # ← ADD / REMOVE WHAT YOU WANT
CALIBRATION_SYMBOL = SYMBOLS[0]            # ← Used ONLY for timing sync
TIMEFRAME = mt5.TIMEFRAME_M30

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

PREDICTORS = {}

# ------------------------------------------------------------
# Initialization
# ------------------------------------------------------------
def initialize_predictors():
    global PREDICTORS
    PREDICTORS = {
        "momentum": MomentumPredictor(),
        "breakout": BreakoutPredictor(),
        "micro": MicrostructurePredictor(),
        "vwap": VwapVolumePredictor(),
        "candle": CandlestickSequencePredictor(),
    }
    logging.info("All predictors initialized.")

# ------------------------------------------------------------
# Time restriction (UNCHANGED)
# ------------------------------------------------------------
def is_restricted_time(now: datetime) -> bool:
    if 18 <= now.hour <= 19:
        return True
    if (now.hour == 12 and now.minute >= 30) or now.hour == 13:
        return True
    if (now.hour == 0 and now.minute >= 30) or now.hour == 1:
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
# Position close helper
# ------------------------------------------------------------
def close_position(pos, max_retries=3):
    for _ in range(max_retries):
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

        mt5.order_send(req)
        time.sleep(0.5)

        if not mt5.positions_get(ticket=pos.ticket):
            return True

    return False

# ------------------------------------------------------------
# Trade placement (NOW SYMBOL-AWARE)
# ------------------------------------------------------------
def place_new_trade(symbol, direction):
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        logging.warning("[%s] No tick.", symbol)
        return False

    current_price = tick.ask if direction == "Buy" else tick.bid

    if direction == "Buy":
        tp = round(tick.ask + (tick.ask * 0.00017), 5)
    else:
        tp = round(tick.bid - (tick.ask * 0.00017), 5)

    result = execute_automated_trade(symbol, direction, sl=None, tp=tp)

    if not result:
        return False

    clear_orphaned_stops(symbol)

    levels = prev_candle_extremes(symbol)
    if not levels:
        return True

    for level in [
        levels["upper_80"],
        levels["upper_110"],
        levels["lower_80"],
        levels["lower_110"],
    ]:
        if level > current_price:
            place_stop_order(symbol, "buy_stop", level)
        elif level < current_price:
            place_stop_order(symbol, "sell_stop", level)

    return True

# ------------------------------------------------------------
# Process ONE symbol per cycle
# ------------------------------------------------------------
def process_symbol(symbol, restricted):
    logging.info("Processing %s", symbol)

    clear_orphaned_stops(symbol)

    raw = get_raw_predictions(symbol, TIMEFRAME)

    condition = evaluate_market_state(
        momentum=raw["momentum"],
        breakout=raw["breakout"],
        micro=raw["micro"],
        vwap=raw["vwap"],
        candle=raw["candle"],
        conditions=trading_conditions
    )

    if condition["final_bias"] == "long" and condition["entry_viability"] == "high":
        decision = "Buy"
    elif condition["final_bias"] == "short" and condition["entry_viability"] == "high":
        decision = "Sell"
    else:
        decision = "Hold"

    if decision == "Hold":
        return

    positions = mt5.positions_get(symbol=symbol) or []
    buys = [p for p in positions if p.type == mt5.ORDER_TYPE_BUY]
    sells = [p for p in positions if p.type == mt5.ORDER_TYPE_SELL]

    same = buys if decision == "Buy" else sells
    opposite = sells if decision == "Buy" else buys

    if same:
        return

    for pos in opposite:
        close_position(pos)

    if not restricted:
        place_new_trade(symbol, decision)

# ------------------------------------------------------------
# One M30 Cycle → All Symbols
# ------------------------------------------------------------
def run_single_cycle():
    now = datetime.now()
    logging.info("=== M30 Cycle @ %s ===", now)

    restricted = is_restricted_time(now)

    for symbol in SYMBOLS:
        try:
            process_symbol(symbol, restricted)
        except Exception as e:
            logging.exception("Error on %s: %s", symbol, e)

# ------------------------------------------------------------
# MAIN LOOP (Single Calibration Symbol)
# ------------------------------------------------------------
def main():
    if not mt5.initialize():
        logging.error("MT5 init failed.")
        return

    initialize_predictors()
    logging.info("Multi-symbol engine started.")

    try:
        while True:
            sleep_sec = get_m30_calibrated(CALIBRATION_SYMBOL, offset=2)

            if sleep_sec is None or sleep_sec <= 0:
                logging.warning("Invalid sleep. Retry 1s.")
                time.sleep(1)
                continue

            logging.info("Sleeping %.2f sec (calibrated on %s).",
                         sleep_sec, CALIBRATION_SYMBOL)

            time.sleep(sleep_sec)
            run_single_cycle()

    except KeyboardInterrupt:
        logging.info("Stopped by user.")
    finally:
        mt5.shutdown()
        logging.info("MT5 shutdown.")

# ------------------------------------------------------------
if __name__ == "__main__":
    main()
