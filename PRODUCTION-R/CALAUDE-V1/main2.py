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
# Condition Engine (RESTORED)
# ===============================
from code1_updated import evaluate_market_state, trading_conditions

# ===============================
# Calibration import (RESTORED)
# ===============================
from placer import get_m30_calibrated

# ===============================
# Config
# ===============================
SYMBOLS = ["EURUSD"]
TIMEFRAME = mt5.TIMEFRAME_M15

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ------------------------------------------------------------
# Ensure symbol available
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
# Initialize predictors
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
# Get raw predictions
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
# Single symbol cycle (NO TRADING)
# ------------------------------------------------------------
def process_symbol(symbol, predictors):
    if not ensure_symbol(symbol):
        return

    raw = get_raw_predictions(symbol, TIMEFRAME, predictors)

    condition = evaluate_market_state(
        momentum=raw["momentum"],
        breakout=raw["breakout"],
        micro=raw["micro"],
        vwap=raw["vwap"],
        candle=raw["candle"],
        conditions=trading_conditions
    )

    print("\n" + "=" * 60)
    print(f"SYMBOL: {symbol}")
    print("=" * 60)

    print("\n--- RAW MODEL OUTPUTS ---")
    for name, value in raw.items():
        print(f"{name.upper():<12} → {value}")

    print("\n--- CONDITION ENGINE OUTPUT ---")
    print(f"CODE        → {condition['condition_code']}")
    print(f"ID          → {condition.get('condition_id')}")
    print(f"FINAL BIAS  → {condition['final_bias']}")
    print(f"CONFIDENCE  → {condition['confidence']:.2f}")
    print(f"CONSENSUS   → {condition.get('consensus_used', False)}")
    print(f"MEANING     → {condition['meaning']}")

# ------------------------------------------------------------
# Main calibrated scheduler
# ------------------------------------------------------------
def main():
    if not mt5.initialize():
        logging.error("MT5 initialization failed.")
        return

    predictors = initialize_predictors()
    logging.info("Multi-symbol calibrated prediction engine started.")

    try:
        while True:
            sleep_sec = get_m30_calibrated(SYMBOLS[0], offset=-4) 

            if sleep_sec is None or sleep_sec <= 0:
                logging.warning("Invalid calibration time. Retrying...")
                time.sleep(1)
                continue

            logging.info("Sleeping %.3f seconds (server calibrated).", sleep_sec)
            time.sleep(sleep_sec)

            print("\n" + "#" * 70)
            print(f"NEW CALIBRATED CYCLE @ {datetime.now()}")
            print("#" * 70)

            for symbol in SYMBOLS:
                process_symbol(symbol, predictors)

    except KeyboardInterrupt:
        logging.info("Stopped by user.")
    finally:
        mt5.shutdown()
        logging.info("MT5 shutdown.")

if __name__ == "__main__":
    main()