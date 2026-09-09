import MetaTrader5 as mt5
import logging
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
# Condition engine
# ===============================
from code1 import evaluate_market_state, trading_conditions

# ===============================
# Config
# ===============================
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_H1   # ⚠️ Use lower TF for 1–2 candle prediction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ===============================
# MT5 Helpers
# ===============================
def initialize_mt5():
    if not mt5.initialize():
        logging.error("MT5 initialization failed")
        return False
    return True

def shutdown_mt5():
    mt5.shutdown()
    logging.info("MT5 shutdown complete")

def ensure_symbol(symbol):
    info = mt5.symbol_info(symbol)
    if info is None:
        logging.error("Symbol %s not found", symbol)
        return False
    if not info.visible:
        mt5.symbol_select(symbol, True)
    return True

# ===============================
# Predictor Engine
# ===============================
class PredictionEngine:
    def __init__(self):
        self.models = {
            "momentum": MomentumPredictor(),
            "breakout": BreakoutPredictor(),
            "micro":    MicrostructurePredictor(),
            "vwap":     VwapVolumePredictor(),
            "candle":   CandlestickSequencePredictor(),
        }
        logging.info("Predictors initialized")

    def get_predictions(self, symbol, timeframe):
        results = {}
        for name, model in self.models.items():
            try:
                pred = model.predict(symbol, timeframe)
                results[name] = pred
                logging.info(f"{name.upper()} → {pred}")
            except Exception as e:
                logging.error(f"{name} failed: {e}")
                results[name] = None
        return results

# ===============================
# Decision Engine
# ===============================
def compute_decision(predictions):
    condition = evaluate_market_state(
        momentum=predictions["momentum"],
        breakout=predictions["breakout"],
        micro=predictions["micro"],
        vwap=predictions["vwap"],
        candle=predictions["candle"],
        conditions=trading_conditions
    )

    logging.info(
        "CODE=%s | BIAS=%s | CONF=%.2f | %s",
        condition["condition_code"],
        condition["final_bias"],
        condition["confidence"],
        condition["meaning"]
    )

    if condition["final_bias"] == "long":
        decision = "Buy"
    elif condition["final_bias"] == "short":
        decision = "Sell"
    else:
        decision = "Hold"

    return {
        "decision": decision,
        "confidence": condition["confidence"],
        "condition_code": condition["condition_code"],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# ===============================
# MAIN EXECUTION (SINGLE RUN)
# ===============================
def run_prediction():
    logging.info("=== RUNNING SINGLE PREDICTION ===")

    if not initialize_mt5():
        return None

    if not ensure_symbol(SYMBOL):
        shutdown_mt5()
        return None

    engine = PredictionEngine()

    try:
        preds = engine.get_predictions(SYMBOL, TIMEFRAME)

        result = compute_decision(preds)

        logging.info(f"FINAL DECISION → {result['decision']}")
        logging.info(f"CONFIDENCE → {result['confidence']:.2f}")

        return result

    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        return None

    finally:
        shutdown_mt5()

# ===============================
# ENTRY POINT
# ===============================
if __name__ == "__main__":
    output = run_prediction()
    print("\n=== OUTPUT ===")
    print(output)