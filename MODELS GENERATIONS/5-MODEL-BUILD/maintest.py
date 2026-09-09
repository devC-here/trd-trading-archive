import json
import logging
import time
from datetime import datetime

import MetaTrader5 as mt5

from momentumdeploy import MomentumPredictor
from breakoutdeploy import BreakoutPredictor
from microstructure import MicrostructurePredictor
from VWdeploy import VwapVolumePredictor
from candledeploy import CandlestickSequencePredictor
from code1 import evaluate_market_state, trading_conditions
from placer import get_m30_calibrated

SYMBOLS   = ["EURUSD", "XAUUSD", "USDJPY", "DXYUSD"]   # change USDX to your broker's DXY symbol if needed
TIMEFRAME = mt5.TIMEFRAME_M15
OUTPUT_FILE = "predictions.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PREDICTORS = {}


def ensure_symbol(symbol):
    info = mt5.symbol_info(symbol)
    if info is None:
        logging.error("Symbol %s not found.", symbol)
        return False
    if not info.visible:
        mt5.symbol_select(symbol, True)
    return True


def initialize_predictors():
    global PREDICTORS
    PREDICTORS = {
        "momentum": MomentumPredictor(),
        "breakout": BreakoutPredictor(),
        "micro":    MicrostructurePredictor(),
        "vwap":     VwapVolumePredictor(),
        "candle":   CandlestickSequencePredictor(),
    }
    logging.info("Predictors initialized.")


def predict_symbol(symbol):
    if not ensure_symbol(symbol):
        return None

    raw = {name: model.predict(symbol, TIMEFRAME) for name, model in PREDICTORS.items()}

    condition = evaluate_market_state(
        momentum=raw["momentum"],
        breakout=raw["breakout"],
        micro=raw["micro"],
        vwap=raw["vwap"],
        candle=raw["candle"],
        conditions=trading_conditions
    )

    return {
        "timestamp":      datetime.now().isoformat(),
        "raw":            raw,
        "condition_code": condition["condition_code"],
        "condition_id":   condition.get("condition_id"),
        "final_bias":     condition["final_bias"],
        "confidence":     condition["confidence"],
        "consensus_used": condition.get("consensus_used", False),
        "meaning":        condition["meaning"],
    }


def run_cycle():
    if not mt5.initialize():
        logging.error("MT5 init failed.")
        return

    logging.info("=== Prediction Cycle @ %s ===", datetime.now())

    results = {}
    for symbol in SYMBOLS:
        result = predict_symbol(symbol)
        if result:
            results[symbol] = result
            logging.info("[%s] BIAS=%s | CONF=%.2f | %s",
                         symbol, result["final_bias"], result["confidence"], result["meaning"])

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    logging.info("Predictions saved to %s", OUTPUT_FILE)


def main():
    if not mt5.initialize():
        logging.error("MT5 init failed.")
        return

    initialize_predictors()
    logging.info("Scheduler started.")

    try:
        while True:
            if not mt5.initialize():
                logging.warning("MT5 re-init failed. Retrying in 1s.")
                time.sleep(1)
                continue

            sleep_sec = get_m30_calibrated("EURUSD", offset=-4)

            if not sleep_sec or sleep_sec <= 0:
                logging.warning("Invalid sleep time. Retrying in 1s.")
                time.sleep(1)
                continue

            logging.info("Sleeping %.3f seconds until next candle.", sleep_sec)
            time.sleep(sleep_sec)

            run_cycle()

    except KeyboardInterrupt:
        logging.info("Stopped by user.")
    finally:
        mt5.shutdown()
        logging.info("MT5 shutdown.")


if __name__ == "__main__":
    main()