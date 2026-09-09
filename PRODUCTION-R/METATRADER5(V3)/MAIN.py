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
# Estimate Utility Import
# ===============================
from sltp import compute_sltp

# ===============================
# Condition engine
# ===============================
from code1 import evaluate_market_state, trading_conditions

# ===============================
# Trade execution + scheduler
# ===============================
from placer import execute_automated_trade, get_m30_calibrated

# ===============================
# Config
# ===============================
SYMBOL = "BTCUSD"
TIMEFRAME = mt5.TIMEFRAME_H1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ------------------------------------------------------------
# Symbol check
# ------------------------------------------------------------
def ensure_symbol(symbol):
    info = mt5.symbol_info(symbol)
    if info is None:
        logging.error("Symbol %s not found in terminal.", symbol)
        return False
    if not info.visible:
        mt5.symbol_select(symbol, True)
    return True

# ------------------------------------------------------------
# Initialization
# ------------------------------------------------------------
def initialize_predictors():
    global PREDICTORS
    PREDICTORS = {
        "momentum": MomentumPredictor(),
        "breakout": BreakoutPredictor(),
        "micro":    MicrostructurePredictor(),
        "vwap":     VwapVolumePredictor(),
        "candle":   CandlestickSequencePredictor(),
    }
    logging.info("All predictors initialized.")

# ------------------------------------------------------------
# Time restriction (TRADE GATE ONLY)
# ------------------------------------------------------------
def is_restricted_time(now: datetime) -> bool:
    now_mins = now.hour * 60 + now.minute
    start_times = [290, 530, 650, 1250]
    duration = 50
    for start in start_times:
        if start <= now_mins < (start + duration):
            return True
    return False

# ------------------------------------------------------------
# Prediction step
# ------------------------------------------------------------
def get_predictions(symbol, timeframe):
    raw = {}

    for name, model in PREDICTORS.items():
        try:
            raw[name] = model.predict(symbol, timeframe)

            s = raw[name].get("_summary", {})

            logging.info(
                "%s -> %s %.2f%%",
                name.upper(),
                s.get("signal", s.get("label", "?")),
                float(s.get("signal_conf", s.get("confidence", 0)))
            )

        except Exception as e:
            logging.error("%s failed: %s", name.upper(), e)
            return None

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
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       pos.symbol,
            "volume":       pos.volume,
            "type":         close_type,
            "position":     pos.ticket,
            "deviation":    50,
            "magic":        10000,
            "comment":      "Auto-close",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        mt5.order_send(req)
        time.sleep(0.5)
        if not mt5.positions_get(ticket=pos.ticket):
            return True
    return False

# ------------------------------------------------------------
# Pre-cycle: close all profitable trades
# ------------------------------------------------------------
def close_profitable_trades(symbol):
    positions = mt5.positions_get(symbol=symbol) or []
    for pos in positions:
        if pos.profit > 0:
            logging.info(
                "Pre-cycle close: ticket=%d type=%s profit=%.2f",
                pos.ticket,
                "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL",
                pos.profit,
            )
            result = close_position(pos)
            if result:
                logging.info("Closed profitable trade: ticket=%d", pos.ticket)
            else:
                logging.warning("Failed to close profitable trade: ticket=%d", pos.ticket)

# ------------------------------------------------------------
# Trade placement
# ------------------------------------------------------------
def place_new_trade(direction):
    tick = mt5.symbol_info_tick(SYMBOL)
    if not tick:
        logging.error("No tick data.")
        return False
    try:
        estimate = compute_sltp(SYMBOL, TIMEFRAME)
        if estimate is None:
            logging.error("Estimate returned invalid value.")
            return False

        target = estimate * 0.9
        draw   = estimate * 0.45

        logging.info(
            "Estimate Used | EST=%.6f | TP=%.6f | SL=%.6f",
            estimate, target, draw
        )

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
    except Exception as e:
        logging.error("Failed to place trade: %s", e)
        return False

# ------------------------------------------------------------
# Single execution cycle
# ------------------------------------------------------------
def run_single_cycle():
    now = datetime.now()
    logging.info("=== M30 Cycle @ %s ===", now)

    restricted = is_restricted_time(now)
    if restricted:
        logging.info("Restricted time: placements blocked.")

    if not mt5.initialize():
        logging.error("MT5 init failed")
        return

    if not ensure_symbol(SYMBOL):
        return

    raw = get_predictions(SYMBOL, TIMEFRAME)
    if not raw:
        return

    condition = evaluate_market_state(
        momentum=raw["momentum"],
        breakout=raw["breakout"],
        micro=raw["micro"],
        vwap=raw["vwap"],
        candle=raw["candle"],
        conditions=trading_conditions
    )

    logging.info(
        "CODE=%s | ID=%s | BIAS=%s | CONF=%.2f | CONSENSUS=%s | %s",
        condition["condition_code"],
        condition.get("condition_id"),
        condition["final_bias"],
        condition["confidence"],
        condition.get("consensus_used", False),
        condition["meaning"]
    )

    if condition["final_bias"] == "long":
        decision = "Buy"
    elif condition["final_bias"] == "short":
        decision = "Sell"
    else:
        logging.info("Decision = HOLD")
        return

    positions = mt5.positions_get(symbol=SYMBOL) or []
    buys  = [p for p in positions if p.type == mt5.ORDER_TYPE_BUY]
    sells = [p for p in positions if p.type == mt5.ORDER_TYPE_SELL]

    same     = buys  if decision == "Buy" else sells
    opposite = sells if decision == "Buy" else buys

    # Same direction exists: hold, do not touch SL/TP
    if same:
        logging.info(
            "Same direction trade exists (%d position(s)). Holding — SL/TP unchanged.",
            len(same)
        )
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

    if not ensure_symbol(SYMBOL):
        return

    initialize_predictors()
    logging.info("M30 calibrated scheduler started.")

    try:
        while True:
            if not mt5.initialize():
                logging.warning("MT5 re-init failed in scheduler loop. Retrying in 1s.")
                time.sleep(1)
                continue

            sleep_sec = get_m30_calibrated(SYMBOL, offset=-4)

            if sleep_sec is None or sleep_sec <= 0:
                logging.warning("Invalid sleep time. Retrying in 1s.")
                time.sleep(1)
                continue

            early_sleep = sleep_sec - 2
            if early_sleep > 0:
                logging.info("Sleeping %.3f seconds (waking 2s early for profit close).", early_sleep)
                time.sleep(early_sleep)
            else:
                logging.info("Not enough time for early wake; skipping pre-close this cycle.")

            logging.info("Pre-cycle: closing all profitable trades on %s.", SYMBOL)
            close_profitable_trades(SYMBOL)

            logging.info("Sleeping 2 seconds before main cycle.")
            time.sleep(2)

            run_single_cycle()

    except KeyboardInterrupt:
        logging.info("Stopped by user.")
    finally:
        mt5.shutdown()
        logging.info("MT5 shutdown.")

if __name__ == "__main__":
    main()