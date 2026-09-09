import logging
import os
import time
from datetime import datetime

from breakoutdeploy import BreakoutPredictor
from candledeploy import CandlestickSequencePredictor
from coindcx_plugin import CoinDCXCredentialsError, CoinDCXFuturesPlugin, decision_to_side, position_side
from fusion import fuse_model_outputs as combine_model_outputs
from microstructure import MicrostructurePredictor
from momentumdeploy import MomentumPredictor
from VWdeploy import VwapVolumePredictor as VWAPPredictor


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SYMBOL = os.getenv("COINDCX_SYMBOL", "B-BTC_USDT")
TIMEFRAME = os.getenv("COINDCX_TIMEFRAME", "15")
QUANTITY = os.getenv("COINDCX_QUANTITY")
PREDICTORS = {}


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


def get_fused_prediction(symbol, timeframe):
    results = {}
    try:
        for key, model in PREDICTORS.items():
            results[key] = model.predict(symbol, timeframe)
            logging.info("%s -> %s", key.upper(), results[key])

        fusion = combine_model_outputs(
            results["momentum"],
            results["breakout"],
            results["micro"],
            results["vwap"],
            results["candle"],
        )
        logging.info("Fusion Output -> %s", fusion)
        return fusion, results
    except Exception as exc:
        logging.exception("Fusion error: %s", exc)
        return {
            "final_bias": "neutral",
            "entry_viability": "no",
            "confidence": 0.0,
            "supporting_factors": [str(exc)],
        }, results


def make_trade_decision():
    fusion_output, raw = get_fused_prediction(SYMBOL, TIMEFRAME)
    direction = fusion_output.get("final_bias")
    viability = fusion_output.get("entry_viability")
    if direction == "long" and viability == "high":
        return "Buy", fusion_output, raw
    if direction == "short" and viability == "high":
        return "Sell", fusion_output, raw
    return "Hold", fusion_output, raw


def is_restricted_time(now: datetime) -> bool:
    if 18 <= now.hour <= 19:
        return True
    if (now.hour == 12 and now.minute >= 30) or now.hour == 13:
        return True
    if (now.hour == 0 and now.minute >= 30) or now.hour == 1:
        return True
    return False


def sync_positions_and_execute(plugin: CoinDCXFuturesPlugin, decision: str):
    desired_side = decision_to_side(decision)
    if desired_side is None:
        logging.info("Hold - no CoinDCX action taken.")
        return None

    if plugin.config.dry_run:
        result = plugin.execute_trade(SYMBOL, decision, quantity=QUANTITY)
        logging.info("CoinDCX dry-run order preview: %s", result)
        return result

    quantity = QUANTITY or plugin.config.default_quantity
    if not quantity:
        raise ValueError("COINDCX_QUANTITY is required before changing live CoinDCX positions.")

    positions = plugin.get_positions(SYMBOL)
    active_positions = [pos for pos in positions if position_side(pos)]
    same_positions = [pos for pos in active_positions if position_side(pos) == desired_side]
    opposite_positions = [pos for pos in active_positions if position_side(pos) != desired_side]

    if same_positions:
        logging.info("Existing CoinDCX %s position found. Skipping duplicate entry.", desired_side)
        return {"skipped": True, "reason": "same_direction_position", "positions": same_positions}

    for position in opposite_positions:
        logging.info("Exiting opposite CoinDCX position: %s", position)
        plugin.exit_position(position)

    result = plugin.execute_trade(SYMBOL, decision, quantity=quantity)
    logging.info("CoinDCX order result: %s", result)
    return result


def main_loop():
    plugin = CoinDCXFuturesPlugin()
    logging.info(
        "Starting CoinDCX runner for %s on %s minute candles. Live trading: %s",
        SYMBOL,
        TIMEFRAME,
        not plugin.config.dry_run,
    )
    initialize_predictors()

    last_minute = -1
    try:
        while True:
            now = datetime.now()
            minute = now.minute
            if minute % 15 == 0 and minute != last_minute:
                last_minute = minute
                logging.info("=== New CoinDCX cycle at %s ===", now.strftime("%Y-%m-%d %H:%M:%S"))

                decision, fusion, _raw_models = make_trade_decision()
                logging.info("Decision: %s | Fusion confidence: %s", decision, fusion.get("confidence"))

                if is_restricted_time(now):
                    logging.info("Restricted time active -> skipping new CoinDCX placement.")
                    time.sleep(60)
                    continue

                try:
                    sync_positions_and_execute(plugin, decision)
                except CoinDCXCredentialsError as exc:
                    logging.error("CoinDCX credentials missing: %s", exc)

                time.sleep(60)
            else:
                time.sleep(1)
    except KeyboardInterrupt:
        logging.info("User stopped the CoinDCX runner.")


if __name__ == "__main__":
    main_loop()
