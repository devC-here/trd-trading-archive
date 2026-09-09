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
# Config
# ===============================
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_H1

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
# Run predictors
# ------------------------------------------------------------
def run_predictions(symbol, timeframe):

    now = datetime.now()

    logging.info("========================================")
    logging.info("Prediction Cycle @ %s", now)
    logging.info("========================================")

    for name, model in PREDICTORS.items():

        try:

            result = model.predict(symbol, timeframe)

            logging.info("MODEL → %s", name.upper())
            logging.info("OUTPUT → %s", result)

        except Exception as e:

            logging.error(
                "Prediction failed for %s: %s",
                name,
                str(e)
            )

        logging.info("----------------------------------------")


# ------------------------------------------------------------
# Scheduler
# ------------------------------------------------------------
def main():

    if not mt5.initialize():
        logging.error("MT5 initialization failed.")
        return

    if not ensure_symbol(SYMBOL):
        return

    initialize_predictors()

    logging.info("Model runner started.")

    try:

        while True:

            run_predictions(SYMBOL, TIMEFRAME)

            # run every 30 seconds
            time.sleep(30)

    except KeyboardInterrupt:

        logging.info("Stopped by user.")

    finally:

        mt5.shutdown()
        logging.info("MT5 shutdown.")


if __name__ == "__main__":
    main()