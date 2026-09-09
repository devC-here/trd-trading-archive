import MetaTrader5 as mt5
import logging
import time
from datetime import datetime

# Predictor imports
from momentumdeploy import MomentumPredictor
from breakoutdeploy import BreakoutPredictor
from microstructure import MicrostructurePredictor
from VWdeploy import VwapVolumePredictor as VWAPPredictor
from candledeploy import CandlestickSequencePredictor
from fusion import fuse_model_outputs as combine_model_outputs

# SLTP predictor
from sltp import predict_size

# Trading execution
try:
    from placer import execute_automated_trade
except ImportError:
    execute_automated_trade = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SYMBOL = "EURUSD"
PREDICTORS = {}

# ------------------------------------------------------------
# Front-run timing controls (LIVE TUNABLE)
# ------------------------------------------------------------
FRONTRUN_MINUTES = {14, 29, 44, 59}
FRONTRUN_SECOND = 59
FRONTRUN_MS = 300     # tune later: 900, 980, 995 etc.

# ------------------------------------------------------------
# Initialization
# ------------------------------------------------------------
def initialize_predictors():
    global PREDICTORS
    PREDICTORS = {
        "momentum": MomentumPredictor(),
        "breakout": BreakoutPredictor(),
        "micro": MicrostructurePredictor(),
        "vwap": VWAPPredictor(),
        "candle": CandlestickSequencePredictor(),
    }
    logging.info("All predictors loaded.")

# ------------------------------------------------------------
# Prediction + Fusion
# ------------------------------------------------------------
def get_fused_prediction(symbol, timeframe):
    results = {}
    for key, model in PREDICTORS.items():
        results[key] = model.predict(symbol, timeframe)
        logging.info(f"{key.upper()} → {results[key]}")

    fusion = combine_model_outputs(
        results["momentum"],
        results["breakout"],
        results["micro"],
        results["vwap"],
        results["candle"],
    )
    logging.info("Fusion Output → %s", fusion)
    return fusion, results

# ------------------------------------------------------------
# Time restriction checks
# ------------------------------------------------------------
def is_restricted_time(now: datetime) -> bool:
    if 18 <= now.hour <= 19:
        return True
    if (now.hour == 12 and now.minute >= 30) or now.hour == 13:
        return True
    if (now.hour == 0 and now.minute >= 0) or now.hour == 1:
        return True
    return False

# ------------------------------------------------------------
# Front-run fire gate (ms precision)
# ------------------------------------------------------------
def is_frontrun_fire(now: datetime, last_fired_minute: int) -> bool:
    if now.minute not in FRONTRUN_MINUTES:
        return False
    if now.second != FRONTRUN_SECOND:
        return False
    if now.microsecond < FRONTRUN_MS * 1000:
        return False
    if now.minute == last_fired_minute:
        return False
    return True

# ------------------------------------------------------------
# Decision engine
# ------------------------------------------------------------
def make_trade_decision():
    timeframe = mt5.TIMEFRAME_M15
    fusion_output, raw = get_fused_prediction(SYMBOL, timeframe)

    if fusion_output.get("final_bias") == "long" and fusion_output.get("entry_viability") == "high":
        return "Buy", fusion_output, raw
    if fusion_output.get("final_bias") == "short" and fusion_output.get("entry_viability") == "high":
        return "Sell", fusion_output, raw

    return "Hold", fusion_output, raw

# ------------------------------------------------------------
# Close position helper
# ------------------------------------------------------------
def close_position(pos):
    close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
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

# ------------------------------------------------------------
# Place trade (dynamic SL/TP, RR 1:1)
# ------------------------------------------------------------
def place_new_trade(decision):
    if not execute_automated_trade:
        logging.warning("Executor unavailable.")
        return False

    tick = mt5.symbol_info_tick(SYMBOL)
    if not tick:
        logging.error("Tick data unavailable.")
        return False

    try:
        size = predict_size(SYMBOL, mt5.TIMEFRAME_M15)

        if decision == "Buy":
            entry = tick.ask
            tp = round(entry + size, 5)
            sl = round(entry - size, 5)
        else:
            entry = tick.bid
            tp = round(entry - size, 5)
            sl = round(entry + size, 5)

        logging.info(
            "SLTP → size=%.5f | SL=%s | TP=%s",
            size, sl, tp
        )

    except Exception as e:
        logging.error("SLTP prediction failed: %s", e)
        return False

    return execute_automated_trade(SYMBOL, decision, sl=sl, tp=tp)

# ------------------------------------------------------------
# Main loop (front-runner)
# ------------------------------------------------------------
def main_loop():
    if not mt5.initialize():
        logging.error("MT5 init failed.")
        return

    initialize_predictors()
    last_fired_minute = -1

    try:
        while True:
            now = datetime.now()

            if is_frontrun_fire(now, last_fired_minute):
                last_fired_minute = now.minute

                logging.info(
                    "=== Front-run cycle @ %s.%03d ===",
                    now.strftime("%Y-%m-%d %H:%M:%S"),
                    now.microsecond // 1000
                )

                restricted = is_restricted_time(now)
                decision, fusion, _ = make_trade_decision()
                logging.info("Decision → %s | Confidence → %s",
                             decision, fusion.get("confidence"))

                positions = mt5.positions_get(symbol=SYMBOL) or []
                buys = [p for p in positions if p.type == mt5.ORDER_TYPE_BUY]
                sells = [p for p in positions if p.type == mt5.ORDER_TYPE_SELL]

                if decision == "Hold":
                    time.sleep(1)
                    continue

                if decision == "Buy" and buys:
                    time.sleep(1)
                    continue
                if decision == "Sell" and sells:
                    time.sleep(1)
                    continue

                opposites = sells if decision == "Buy" else buys
                for pos in opposites:
                    close_position(pos)

                if not restricted:
                    place_new_trade(decision)
                else:
                    logging.info("Restricted time – placement skipped.")

            time.sleep(0.001)  # high-resolution polling

    except KeyboardInterrupt:
        logging.info("Stopped by user.")
    finally:
        mt5.shutdown()
        logging.info("MT5 shutdown complete.")

if __name__ == "__main__":
    main_loop()
