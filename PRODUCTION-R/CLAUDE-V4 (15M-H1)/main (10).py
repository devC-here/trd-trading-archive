import MetaTrader5 as mt5
import logging
import time
from datetime import datetime

# ===============================
# Predictor imports
# ===============================
from deploy_momentum_h1       import MomentumPredictor
from deploy_breakout_h1       import BreakoutPredictor
from deploy_microstructure_h1 import MicrostructurePredictor
from deploy_vwap_h1           import VwapPredictor
from deploy_candle_h1         import CandlePredictor

# ===============================
# Utility imports
# ===============================
from sltp    import compute_sltp
from placer  import execute_automated_trade, get_m30_calibrated
from ensemble_h1 import from_predict, ensemble, log_ensemble

# ===============================
# Config
# ===============================
SYMBOL    = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_H1

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PREDICTORS = {}


def ensure_symbol(symbol):
    info = mt5.symbol_info(symbol)
    if info is None:
        logging.error("Symbol not found: %s", symbol); return False
    if not info.visible:
        mt5.symbol_select(symbol, True)
    return True


def initialize_predictors():
    global PREDICTORS
    PREDICTORS = {
        "momentum":       MomentumPredictor(),
        "breakout":       BreakoutPredictor(),
        "microstructure": MicrostructurePredictor(),
        "vwap":           VwapPredictor(),
        "candle":         CandlePredictor(),
    }
    logging.info("All predictors initialized.")


def is_restricted_time(now):
    now_mins   = now.hour * 60 + now.minute
    start_times = [1100, 330, 810, 1110]
    duration    = 90
    return any(s <= now_mins < s + duration for s in start_times)


def get_raw_predictions(symbol, timeframe):
    model_outputs = []
    for name, model in PREDICTORS.items():
        try:
            r = model.predict(symbol, timeframe)
            if "error" in r:
                logging.warning("%s → ERROR: %s", name.upper(), r["error"])
                continue
            s = r.get("_summary", {})
            signal = r.get("signal") or s.get("signal") or s.get("label")
            conf   = r.get("confidence", s.get("signal_conf", s.get("confidence", 0)))
            active = r.get("trade_active", s.get("trade_active", False))
            logging.info("%s → signal=%s | conf=%.2f | active=%s", name.upper(), signal, conf, active)
            model_outputs.append(from_predict(name, r))
        except Exception as e:
            logging.error("%s → FAILED: %s", name.upper(), e)
    return model_outputs


def close_position(pos):
    close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    req = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": pos.symbol,
        "volume": pos.volume, "type": close_type, "position": pos.ticket,
        "deviation": 50, "magic": 10000, "comment": "Auto-close",
        "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_FOK,
    }
    result = mt5.order_send(req)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        logging.info("Closed position ticket=%d", pos.ticket); return True
    logging.warning("Close failed ticket=%d", pos.ticket); return False


def close_profitable_trades(symbol):
    for pos in (mt5.positions_get(symbol=symbol) or []):
        if pos.profit > 0:
            logging.info("Closing profitable trade ticket=%d profit=%.2f", pos.ticket, pos.profit)
            close_position(pos)


def update_existing_trade_sltp(pos):
    tick = mt5.symbol_info_tick(SYMBOL)
    if not tick: return False
    estimate = compute_sltp(SYMBOL, TIMEFRAME)
    if estimate is None: return False
    target = estimate; draw = estimate * 0.9
    if pos.type == mt5.ORDER_TYPE_BUY:
        sl, tp = tick.ask - draw, tick.ask + target
    else:
        sl, tp = tick.bid + draw, tick.bid - target
    mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "symbol": SYMBOL,
                    "position": pos.ticket, "sl": round(sl, 5), "tp": round(tp, 5)})
    logging.info("Updated SLTP ticket=%d SL=%.5f TP=%.5f", pos.ticket, sl, tp)
    return True


def place_new_trade(direction):
    tick = mt5.symbol_info_tick(SYMBOL)
    if not tick: logging.error("No tick data."); return False
    estimate = compute_sltp(SYMBOL, TIMEFRAME)
    if estimate is None: return False
    target = estimate * 1.1; draw = estimate * 0.55
    if direction == "Buy":
        sl, tp = tick.ask - draw, tick.ask + target
    else:
        sl, tp = tick.bid + draw, tick.bid - target
    return execute_automated_trade(SYMBOL, direction, sl=round(sl, 5), tp=round(tp, 5))


def run_single_cycle():
    now = datetime.now()
    logging.info("=== M30 Cycle @ %s ===", now)
    restricted   = is_restricted_time(now)
    model_outputs = get_raw_predictions(SYMBOL, TIMEFRAME)

    if len(model_outputs) < 3:
        logging.warning("Not enough model outputs for ensemble (need ≥ 3)."); return

    agg      = ensemble(model_outputs)
    log_ensemble(agg)

    decision   = agg.signal
    confidence = agg.confidence
    logging.info("ENSEMBLE → decision=%s | confidence=%.2f", decision, confidence)

    if decision is None: logging.info("Decision = HOLD"); return
    if decision == "bullish":  decision = "Buy"
    elif decision == "bearish": decision = "Sell"
    else: return

    positions = mt5.positions_get(symbol=SYMBOL) or []
    buys  = [p for p in positions if p.type == mt5.ORDER_TYPE_BUY]
    sells = [p for p in positions if p.type == mt5.ORDER_TYPE_SELL]
    same     = buys  if decision == "Buy"  else sells
    opposite = sells if decision == "Buy"  else buys

    if same:
        logging.info("Existing %s position found → updating SLTP", decision)
        for pos in same: update_existing_trade_sltp(pos)
        return

    for pos in opposite: close_position(pos)

    if not restricted:
        logging.info("Trade placed: %s", place_new_trade(decision))
    else:
        logging.info("Trade blocked due to restriction window.")


def main():
    if not mt5.initialize(): logging.error("MT5 init failed"); return
    if not ensure_symbol(SYMBOL): return
    initialize_predictors()
    logging.info("M30 calibrated scheduler started.")
    try:
        while True:
            sleep_sec = get_m30_calibrated(SYMBOL, offset=+1)
            if not sleep_sec or sleep_sec <= 0: time.sleep(1); continue
            early_sleep = max(0, sleep_sec - 2)
            logging.info("Sleeping %.3f seconds (early wake).", early_sleep)
            time.sleep(early_sleep)
            logging.info("Closing profitable trades.")
            close_profitable_trades(SYMBOL)
            time.sleep(2)
            run_single_cycle()
    except KeyboardInterrupt:
        logging.info("Stopped by user.")
    finally:
        mt5.shutdown(); logging.info("MT5 shutdown.")


if __name__ == "__main__":
    main()
