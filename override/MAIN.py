import MetaTrader5 as mt5
import logging
import time
from datetime import datetime

# -------------------------------
# Predictor imports
# -------------------------------
from momentumdeploy import MomentumPredictor
from breakoutdeploy import BreakoutPredictor
from microstructure import MicrostructurePredictor
from VWdeploy import VwapVolumePredictor as VWAPPredictor
from candledeploy import CandlestickSequencePredictor

# Two engines under test
from fusion import fuse_model_outputs as engine_A
from fusion1 import fuse_model_outputs as engine_B

# -------------------------------
# Logging config
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M15

PREDICTORS = {}

# ============================================================
# INITIALIZATION
# ============================================================
def initialize_predictors():
    global PREDICTORS
    PREDICTORS = {
        "momentum": MomentumPredictor(),
        "breakout": BreakoutPredictor(),
        "micro": MicrostructurePredictor(),
        "vwap": VWAPPredictor(),
        "candle": CandlestickSequencePredictor(),
    }
    logging.info("Predictors initialized.")

# ============================================================
# RUN ALL MODELS
# ============================================================
def run_models(symbol, timeframe):
    outputs = {}
    for name, model in PREDICTORS.items():
        outputs[name] = model.predict(symbol, timeframe)
        logging.info("%-10s → %s", name.upper(), outputs[name])
    return outputs

# ============================================================
# NORMALIZE ENGINE OUTPUT (CRITICAL)
# ============================================================
def normalize_result(raw, engine_name):
    """
    Forces condition_id to ALWAYS exist.
    This is NON-NEGOTIABLE for sentiment debugging.
    """
    if raw is None:
        return {
            "engine": engine_name,
            "condition_id": -1,
            "final_bias": "no_trade",
            "entry_viability": "low",
            "confidence": 0.0,
            "supporting_factors": ["Engine returned None"]
        }

    result = dict(raw)

    # FORCE required fields
    result["engine"] = engine_name
    result["condition_id"] = result.get("condition_id", 0)
    result["final_bias"] = result.get("final_bias", "no_trade")
    result["entry_viability"] = result.get("entry_viability", "low")
    result["confidence"] = result.get("confidence", 0.0)
    result["supporting_factors"] = result.get("supporting_factors", [])

    return result

# ============================================================
# LOG ENGINE RESULT (ALWAYS PRINTS condition_id)
# ============================================================
def log_engine(res):
    logging.info(
        "[%s] condition_id=%s | bias=%s | viability=%s | confidence=%.3f",
        res["engine"],
        res["condition_id"],
        res["final_bias"],
        res["entry_viability"],
        res["confidence"],
    )

    for reason in res["supporting_factors"]:
        logging.info("[%s] ↳ %s", res["engine"], reason)

# ============================================================
# MAIN LOOP (DATA + SENTIMENT ONLY)
# ============================================================
def main_loop():
    logging.info("Connecting to MT5 (DATA-ONLY MODE)...")
    if not mt5.initialize():
        logging.error("MT5 initialization failed.")
        return

    initialize_predictors()

    last_minute = -1

    try:
        while True:
            now = datetime.now()
            minute = now.minute

            # Run strictly on 00, 15, 30, 45
            if minute % 15 == 0 and minute != last_minute:
                last_minute = minute

                logging.info("============================================================")
                logging.info("🕒 New sentiment cycle @ %s", now.strftime("%Y-%m-%d %H:%M:%S"))

                # ----------------------------------------------------
                # 1. Run predictors (shared inputs)
                # ----------------------------------------------------
                model_data = run_models(SYMBOL, TIMEFRAME)

                # ----------------------------------------------------
                # 2. Run both engines
                # ----------------------------------------------------
                raw_A = engine_A(
                    model_data["momentum"],
                    model_data["breakout"],
                    model_data["micro"],
                    model_data["vwap"],
                    model_data["candle"],
                )

                raw_B = engine_B(
                    model_data["momentum"],
                    model_data["breakout"],
                    model_data["micro"],
                    model_data["vwap"],
                    model_data["candle"],
                )

                res_A = normalize_result(raw_A, "ENGINE_A")
                res_B = normalize_result(raw_B, "ENGINE_B")

                # ----------------------------------------------------
                # 3. Log results (condition_id GUARANTEED)
                # ----------------------------------------------------
                log_engine(res_A)
                log_engine(res_B)

                # ----------------------------------------------------
                # 4. Explicit divergence detection
                # ----------------------------------------------------
                if res_A["final_bias"] != res_B["final_bias"]:
                    logging.warning(
                        "⚠️ SENTIMENT DIVERGENCE | "
                        "A=%s (ID %s) vs B=%s (ID %s)",
                        res_A["final_bias"],
                        res_A["condition_id"],
                        res_B["final_bias"],
                        res_B["condition_id"],
                    )
                else:
                    logging.info(
                        "✅ Sentiment agreement → %s (A:%s | B:%s)",
                        res_A["final_bias"],
                        res_A["condition_id"],
                        res_B["condition_id"],
                    )

                time.sleep(60)

            else:
                time.sleep(1)

    except KeyboardInterrupt:
        logging.info("User stopped the system.")
    finally:
        mt5.shutdown()
        logging.info("MT5 shutdown complete.")

# ============================================================
if __name__ == "__main__":
    main_loop()
