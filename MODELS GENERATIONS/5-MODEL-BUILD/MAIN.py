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
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M12

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
    start_times = [90, 330, 810, 1110]
    duration = 90
    for start in start_times:
        if start <= now_mins < (start + duration):
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
# Modify SL/TP of an existing position
# ------------------------------------------------------------
def modify_position_sltp(pos, sl, tp, max_retries=3):
    for attempt in range(max_retries):
        if not mt5.initialize():
            logging.warning("MT5 re-init failed on SL/TP modify attempt %d", attempt + 1)
            time.sleep(0.5)
            continue
        req = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "symbol":   pos.symbol,
            "position": pos.ticket,
            "sl":       round(sl, 5),
            "tp":       round(tp, 5),
        }
        result = mt5.order_send(req)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logging.info(
                "SL/TP updated: ticket=%d SL=%.5f TP=%.5f",
                pos.ticket, sl, tp
            )
            return True
        else:
            logging.warning(
                "SL/TP modify attempt %d failed for ticket=%d retcode=%s",
                attempt + 1, pos.ticket,
                result.retcode if result else "None"
            )
        time.sleep(0.5)
    logging.error("Failed to modify SL/TP for ticket=%d after %d attempts", pos.ticket, max_retries)
    return False

# ------------------------------------------------------------
# Recalc and update SL/TP for existing same-direction trade
# ------------------------------------------------------------
def update_existing_trade_sltp(pos):
    tick = mt5.symbol_info_tick(SYMBOL)
    if not tick:
        logging.error("No tick data for SL/TP update.")
        return False
    try:
        estimate = compute_sltp(SYMBOL, TIMEFRAME)
        if estimate is None:
            logging.error("Estimate returned None, cannot update SL/TP.")
            return False

        target = estimate * 0.8
        draw   = estimate * 1.2

        if pos.type == mt5.ORDER_TYPE_BUY:
            sl = tick.ask - draw
            tp = tick.ask + target
        else:
            sl = tick.bid + draw
            tp = tick.bid - target

        logging.info(
            "Updating SL/TP for existing ticket=%d | EST=%.6f | new SL=%.5f | new TP=%.5f",
            pos.ticket, estimate, sl, tp
        )
        return modify_position_sltp(pos, sl, tp)
    except Exception as e:
        logging.error("Failed to update SL/TP for ticket=%d: %s", pos.ticket, e)
        return False

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

        target = estimate * 0.8
        draw   = estimate * 1.2

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
        "CODE=%s | ID=%s | BIAS=%s | CONF=%.2f | CONSENSUS=%s | %s",
        condition["condition_code"],
        condition.get("condition_id"),
        condition["final_bias"],
        condition["confidence"],
        condition.get("consensus_used", False),
        condition["meaning"]
    )

    # Decision: trust final_bias directly
    # code1 already handles: best match first, majority of top 5 as fallback if best is hold/no_trade
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

    # Same direction exists: update SL/TP instead of placing new
    if same:
        logging.info(
            "Same direction trade exists (%d position(s)). Updating SL/TP.",
            len(same)
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

            # Wake up 2s early, close profitable trades, then sleep 2s before main cycle
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