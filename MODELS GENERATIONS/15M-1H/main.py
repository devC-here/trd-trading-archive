import MetaTrader5 as mt5
import logging
import time
from datetime import datetime

# ===============================
# Specialist predictor imports
# ===============================
from deploy_model1_momentum   import MomentumPredictor
from deploy_model2_structure  import StructurePredictor
from deploy_model3_volatility import VolatilityPredictor
from deploy_model4_volume     import VolumePredictor
from deploy_model5_geometry   import GeometryPredictor

# ===============================
# Ensemble library
# ===============================
from ensemble import run_ensemble_from_results, print_report

# ===============================
# Utility imports
# ===============================
from sltp   import compute_sltp
from placer import execute_automated_trade, get_m30_calibrated

# ===============================
# Config
# ===============================
SYMBOL         = "EURUSD"
DATA_TIMEFRAME = mt5.TIMEFRAME_M15  # all 5 specialists run on M15
SLTP_TF        = mt5.TIMEFRAME_H1   # SL/TP sizing matches the H1 cycle clock

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PREDICTORS = {}


# ===============================
# Helpers
# ===============================

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
        "m1": MomentumPredictor (model_path="model1_best.pth",    meta_path="model1_meta.pkl"),
        "m2": StructurePredictor(model_path="model2_best.pth",    scaler_path="model2_scaler.pkl"),
        "m3": VolatilityPredictor(model_path="model3_best.pth",   meta_path="model3_meta.pkl"),
        "m4": VolumePredictor   (ensemble_path="model4_ensemble.pkl"),
        "m5": GeometryPredictor (ensemble_path="model5_ensemble.pkl"),
    }
    logging.info("All 5 specialist predictors initialized.")


def is_restricted_time(now):
    now_mins    = now.hour * 60 + now.minute
    start_times = [1100, 330, 810, 1110]
    duration    = 90
    return any(s <= now_mins < s + duration for s in start_times)




# ===============================
# Trade management (unchanged)
# ===============================

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
    estimate = compute_sltp(SYMBOL, SLTP_TF)
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
    estimate = compute_sltp(SYMBOL, SLTP_TF)
    if estimate is None: return False
    target = estimate * 1.1; draw = estimate * 0.55
    if direction == "Buy":
        sl, tp = tick.ask - draw, tick.ask + target
    else:
        sl, tp = tick.bid + draw, tick.bid - target
    return execute_automated_trade(SYMBOL, direction, sl=round(sl, 5), tp=round(tp, 5))


# ===============================
# Main cycle
# ===============================

def run_single_cycle():
    now = datetime.now()
    logging.info("=== Cycle @ %s ===", now)
    restricted = is_restricted_time(now)

    # Each specialist fetches M15 bars — timeframe passed explicitly,
    # overriding whatever default is baked into each deploy script
    results = {}
    for key, model in PREDICTORS.items():
        try:
            r = model.predict(SYMBOL, DATA_TIMEFRAME)
            if "error" in r:
                logging.warning("%s → %s", key.upper(), r["error"])
                results[key] = None
            else:
                results[key] = r
        except Exception as e:
            logging.error("%s → FAILED: %s", key.upper(), e)
            results[key] = None

    if any(v is None for v in results.values()):
        logging.warning("One or more models failed — skipping cycle."); return

    # Run ensemble gate cascade on the collected results
    result = run_ensemble_from_results(
        results["m1"], results["m2"], results["m3"],
        results["m4"], results["m5"]
    )
    print_report(result)

    signal = result["signal"]        # "BUY" | "SELL" | "HOLD"
    score  = result["score"]         # 0–100
    reason = result["reason"]

    logging.info("ENSEMBLE → signal=%s | score=%d | reason=%s", signal, score, reason)

    if signal == "HOLD":
        logging.info("Decision = HOLD — no action."); return

    # Map to placer convention
    decision = "Buy" if signal == "BUY" else "Sell"

    positions = mt5.positions_get(symbol=SYMBOL) or []
    buys      = [p for p in positions if p.type == mt5.ORDER_TYPE_BUY]
    sells     = [p for p in positions if p.type == mt5.ORDER_TYPE_SELL]
    same      = buys  if decision == "Buy"  else sells
    opposite  = sells if decision == "Buy"  else buys

    if same:
        logging.info("Existing %s position — updating SLTP.", decision)
        for pos in same: update_existing_trade_sltp(pos)
        return

    for pos in opposite: close_position(pos)

    if not restricted:
        logging.info("Placing trade: %s (score=%d, lot=%s)",
                     decision, score, result["lot_size"])
        place_new_trade(decision)
    else:
        logging.info("Trade blocked — restricted time window.")


# ===============================
# Entry point
# ===============================

def main():
    if not mt5.initialize(): logging.error("MT5 init failed"); return
    if not ensure_symbol(SYMBOL): return
    initialize_predictors()
    logging.info("Ensemble bot started. data_tf=M15 | sltp_tf=H1 | cycle=H1 calibrated")
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
        for p in PREDICTORS.values(): p.shutdown()
        mt5.shutdown()
        logging.info("MT5 shutdown.")


if __name__ == "__main__":
    main()