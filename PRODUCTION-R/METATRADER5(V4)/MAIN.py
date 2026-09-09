import MetaTrader5 as mt5
import logging
import time
from datetime import datetime

# ===============================
# Predictor imports
# ===============================
from momentumdeploy import MomentumPredictor
from breakoutdeploy import BreakoutPredictor
from microstructure import MicrostructurePredictor
from VWdeploy import VwapVolumePredictor
from candledeploy import CandlestickSequencePredictor

# ===============================
# Condition engine
# ===============================
from code1 import evaluate_market_state, trading_conditions

# ===============================
# Trade execution + scheduler
# ===============================
from placer import (
    execute_automated_trade,
    get_m30_calibrated,
    prev_candle_extremes,
    place_stop_order,
    clear_orphaned_stops
)

# ===============================
# Config
# ===============================
SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M5

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
# Time restriction
# ------------------------------------------------------------
def is_restricted_time(now: datetime) -> bool:
    if 18 <= now.hour <= 19:
        return True
    if (now.hour == 6 and now.minute >= 30) or now.hour == 8:
        return True
    if (now.hour == 0 and now.minute >= 30) or now.hour == 12:
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
# Trade helpers
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
# Trade placement (rebuild)
# ------------------------------------------------------------
def place_new_trade(direction):
    tick = mt5.symbol_info_tick(SYMBOL)
    if not tick:
        logging.error("No tick data.")
        return False

    current_price = tick.ask if direction == "Buy" else tick.bid

    if direction == "Buy":
        tp = round(tick.ask + (tick.ask * 0.00017), 5)
    else:
        tp = round(tick.bid - (tick.ask * 0.00017), 5)

    result = execute_automated_trade(
        SYMBOL,
        direction,
        sl=None,
        tp=tp
    )

    # 🚨 Only continue if trade was actually placed
    if not result:
        return result

    # ✅ Now clear previous orphan stops
    clear_orphaned_stops(SYMBOL)

    levels = prev_candle_extremes(SYMBOL)
    if not levels:
        return result

    candidate_levels = [
        levels["upper_80"],
        levels["upper_110"],
        levels["lower_80"],
        levels["lower_110"],
    ]

    for level in candidate_levels:
        if level > current_price:
            place_stop_order(SYMBOL, "buy_stop", level)
        elif level < current_price:
            place_stop_order(SYMBOL, "sell_stop", level)

    return result

# ------------------------------------------------------------
# Single execution cycle
# ------------------------------------------------------------
def run_single_cycle():
    now = datetime.now()
    logging.info("=== M30 Cycle @ %s ===", now)

    # Clear orphan stops every cycle
    clear_orphaned_stops(SYMBOL)

    restricted = is_restricted_time(now)
    if restricted:
        logging.info("Restricted time: placements blocked.")

    raw = get_raw_predictions(SYMBOL, TIMEFRAME)

    condition = evaluate_market_state(
        momentum=raw["momentum"],
        breakout=raw["breakout"],
        micro=raw["micro"],
        vwap=raw["vwap"],
        candle=raw["candle"],
        conditions=trading_conditions
    )

    logging.info(
        "CODE=%s | ID=%s | BIAS=%s | CONF=%.2f | %s",
        condition["condition_code"],
        condition["condition_id"],
        condition["final_bias"],
        condition["confidence"],
        condition["meaning"]
    )

    if condition["final_bias"] == "long" and condition["entry_viability"] == "high":
        decision = "Buy"
    elif condition["final_bias"] == "short" and condition["entry_viability"] == "high":
        decision = "Sell"
    else:
        decision = "Hold"

    positions = mt5.positions_get(symbol=SYMBOL) or []
    buys = [p for p in positions if p.type == mt5.ORDER_TYPE_BUY]
    sells = [p for p in positions if p.type == mt5.ORDER_TYPE_SELL]

    if decision == "Hold":
        logging.info("Decision = HOLD")
        return

    same = buys if decision == "Buy" else sells
    opposite = sells if decision == "Buy" else buys

    if same:
        logging.info("Same direction position exists. Skipping entry.")
        return

    for pos in opposite:
        close_position(pos)

    if not restricted:
        placed = place_new_trade(decision)
        logging.info("Trade placed: %s", placed)
    else:
        logging.info("Placement skipped due to restriction.")


# ------------------------------------------------------------
# Scheduler
# ------------------------------------------------------------
def main():
    if not mt5.initialize():
        logging.error("MT5 init failed.")
        return

    initialize_predictors()
    logging.info("M30 calibrated sleep scheduler started.")

    try:
        while True:
            sleep_sec = get_m30_calibrated(SYMBOL, offset=2)

            if sleep_sec is None or sleep_sec <= 0:
                logging.warning("Invalid sleep time. Retrying in 1s.")
                time.sleep(1)
                continue

            logging.info("Sleeping %.3f seconds until next M30 candle.", sleep_sec)
            time.sleep(sleep_sec)

            run_single_cycle()

    except KeyboardInterrupt:
        logging.info("Stopped by user.")
    finally:
        mt5.shutdown()
        logging.info("MT5 shutdown.")


if __name__ == "__main__":
    main()
