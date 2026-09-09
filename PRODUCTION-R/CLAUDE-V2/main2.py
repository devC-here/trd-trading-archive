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
from sltp_test import compute_all_estimates
from placer import execute_automated_trade, get_m30_calibrated
from agrregator import aggregate_predictions


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
# TP band validation
# ------------------------------------------------------------
def tp_inside_higher_tf_band(symbol, tp_distance):

    estimates = compute_all_estimates(symbol)

    if estimates is None:
        return False

    m15_est, m30_est, h1_est = estimates

    if m30_est is None or h1_est is None:
        return False

    band = min(m30_est, h1_est)

    if tp_distance <= band:
        return True

    logging.info(
        "TP rejected %.6f outside higher TF band %.6f",
        tp_distance,
        band
    )

    return False


# ------------------------------------------------------------
# Update SLTP
# ------------------------------------------------------------
def update_existing_trade_sltp(pos):

    tick = mt5.symbol_info_tick(SYMBOL)

    if not tick:
        return False

    estimates = compute_all_estimates(SYMBOL)

    if estimates is None:
        return False

    m15_est, m30_est, h1_est = estimates

    target = m15_est
    draw = m15_est * 0.9

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

    spread = tick.ask - tick.bid

    if spread > 0.0003:
        logging.info("Spread too high: %.5f", spread)
        return False

    estimates = compute_all_estimates(SYMBOL)

    if estimates is None:
        return False

    m15_est, m30_est, h1_est = estimates

    target = m15_est * 1
    draw = m15_est * 0.8

    # ------------------------------------------------
    # NEW CHECK → TP inside higher timeframe band
    # ------------------------------------------------
    if not tp_inside_higher_tf_band(SYMBOL, target):

        logging.info("Trade rejected due to higher TF band rule.")
        return False

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