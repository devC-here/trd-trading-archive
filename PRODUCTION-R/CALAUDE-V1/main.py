import MetaTrader5 as mt5
import logging
import time

# ===============================
# Predictor imports
# ===============================
from momentum_deploy import MomentumPredictor
from breakout_deploy import BreakoutPredictor
from microstructure_deploy import MicrostructurePredictor
from vwap_deploy import VwapPredictor
from candle_deploy import CandlePredictor

# ===============================
# Calibration import (RESTORED)
# ===============================
from placer import get_m30_calibrated

# ===============================
# Config
# ===============================
SYMBOLS = ["XAUUSD", "USDJPY", "EURUSD", "DXYUSD"]
TIMEFRAME = mt5.TIMEFRAME_M15

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ------------------------------------------------------------
def ensure_symbol(symbol):
    info = mt5.symbol_info(symbol)
    if info is None:
        logging.error("Symbol %s not found.", symbol)
        return False
    if not info.visible:
        mt5.symbol_select(symbol, True)
    return True

# ------------------------------------------------------------
def initialize_predictors():
    return {
        "momentum": MomentumPredictor(),
        "breakout": BreakoutPredictor(),
        "micro":    MicrostructurePredictor(),
        "vwap":     VwapPredictor(),
        "candle":   CandlePredictor(),
    }

# ------------------------------------------------------------
def get_predictions(symbol, timeframe, predictors):
    results = {}
    for name, model in predictors.items():
        try:
            results[name] = model.predict(symbol, timeframe)
        except Exception as e:
            results[name] = f"ERROR: {e}"
    return results

# ------------------------------------------------------------
def main():
    if not mt5.initialize():
        logging.error("MT5 initialization failed.")
        return

    predictors = initialize_predictors()
    logging.info("Server-calibrated prediction scheduler started.")

    try:
        while True:
            # --- Calibration using broker server time ---
            sleep_sec = get_m30_calibrated(SYMBOLS[0], offset=-3)

            if sleep_sec is None or sleep_sec <= 0:
                logging.warning("Invalid calibration sleep time. Retrying...")
                time.sleep(1)
                continue

            logging.info("Sleeping %.3f seconds (server-calibrated).", sleep_sec)
            time.sleep(sleep_sec)

            print("\n" + "="*60)
            print("NEW CALIBRATED CYCLE")
            print("="*60)

            for symbol in SYMBOLS:
                if not ensure_symbol(symbol):
                    continue

                print(f"\nSYMBOL: {symbol}")
                print("-"*40)

                preds = get_predictions(symbol, TIMEFRAME, predictors)

                for name, value in preds.items():
                    print(f"{name.upper():<12} → {value}")

    except KeyboardInterrupt:
        logging.info("Stopped by user.")
    finally:
        mt5.shutdown()
        logging.info("MT5 shutdown.")

if __name__ == "__main__":
    main()