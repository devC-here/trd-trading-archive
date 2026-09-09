import MetaTrader5 as mt5
import logging
import os
import time
from datetime import datetime

# Predictor imports
from momentumdeploy import MomentumPredictor
from breakoutdeploy import BreakoutPredictor
from microstructure import MicrostructurePredictor
from VWdeploy import VwapVolumePredictor as VWAPPredictor
from candledeploy import CandlestickSequencePredictor
from fusion import fuse_model_outputs as combine_model_outputs

# Trading execution (user provided function)
try:
    from placer import execute_automated_trade
except ImportError:
    execute_automated_trade = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SYMBOL = os.getenv("MT5_SYMBOL", "BTCUSD")
DATA_SYMBOL = os.getenv("COINDCX_SYMBOL", "B-BTC_USDT")
DATA_TIMEFRAME = os.getenv("COINDCX_TIMEFRAME", "15")
PREDICTORS = {}

# ------------------------------------------------------------
# Initialization
# ------------------------------------------------------------
def initialize_predictors():
    global PREDICTORS
    try:
        PREDICTORS = {
            "momentum": MomentumPredictor(),
            "breakout": BreakoutPredictor(),
            "micro": MicrostructurePredictor(),
            "vwap": VWAPPredictor(),
            "candle": CandlestickSequencePredictor(),
        }
        logging.info("All predictors loaded.")
    except Exception as e:
        logging.error("Predictor initialization failed: %s", e)
        raise

# ------------------------------------------------------------
# Prediction + Fusion
# ------------------------------------------------------------
def get_fused_prediction(symbol, timeframe):
    results = {}
    try:
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
    except Exception as e:
        logging.error("Fusion error: %s", e)
        return {
            "final_bias": "neutral",
            "entry_viability": "no",
            "confidence": 0.0,
            "supporting_factors": [str(e)]
        }, results

# ------------------------------------------------------------
# Time restriction checks
# ------------------------------------------------------------
def is_restricted_time(now: datetime) -> bool:
    """
    Return True if *placing new trades* should be blocked.
    Closing existing positions is always allowed.
    """
    # 18:00–19:59 block
    if 18 <= now.hour <= 19:
        return True
    # 12:30–13:59 block
    if (now.hour == 12 and now.minute >= 30) or now.hour == 13:
        return True
    # 00:30–01:59 block
    if (now.hour == 0 and now.minute >= 30) or now.hour == 1:
        return True
    return False

# ------------------------------------------------------------
# Decision engine
# ------------------------------------------------------------
def make_trade_decision():
    fusion_output, raw = get_fused_prediction(DATA_SYMBOL, DATA_TIMEFRAME)
    direction = fusion_output.get("final_bias")
    viability = fusion_output.get("entry_viability")
    if direction == "long" and viability == "high":
        return "Buy", fusion_output, raw
    elif direction == "short" and viability == "high":
        return "Sell", fusion_output, raw
    return "Hold", fusion_output, raw

# ------------------------------------------------------------
# MT5 helpers: close positions, place trade
# ------------------------------------------------------------
def close_position(pos, max_retries: int = 3, sleep_between: float = 1.0):
    """
    Close single position 'pos'. Retries on failure.
    Returns True if closed (or not found), False otherwise.
    """
    for attempt in range(1, max_retries + 1):
        try:
            # to close a BUY -> send SELL; to close SELL -> send BUY
            close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": close_type,
                "position": pos.ticket,
                "deviation": 50,
                "magic": 10000,
                "comment": f"Auto-close (attempt {attempt})",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            res = mt5.order_send(req)
            logging.info("Close attempt %d for pos %s → %s", attempt, pos.ticket, res)
            # Check retcode success (10009/10008/10004/etc vary by broker). We'll treat deal!=0 or order!=0 as attempt made.
            if hasattr(res, "retcode") and res.retcode == 10009 or getattr(res, "deal", 0) != 0:
                # success condition may vary by broker; still, poll to confirm position gone
                time.sleep(0.5)
                updated = mt5.positions_get(ticket=pos.ticket)
                if not updated:
                    return True
            # Small pause before retrying
        except Exception as e:
            logging.error("Exception when closing pos %s: %s", pos.ticket, e)
        time.sleep(sleep_between)
    # final check: if position still exists, return False
    still = mt5.positions_get(ticket=pos.ticket)
    if still:
        logging.warning("Position %s still exists after retries.", pos.ticket)
        return False
    return True

def place_new_trade(decision):
    """
    Uses execute_automated_trade wrapper provided by user.
    Returns truthy success flag or False if not placed.
    """
    if not execute_automated_trade:
        logging.warning("execute_automated_trade is not available; skipping place.")
        return False

    tick = mt5.symbol_info_tick(SYMBOL)
    if not tick:
        logging.error("Tick data unavailable; cannot place trade.")
        return False

    # example TP 3 pips as before (adjust according to symbol decimal)
    if decision == "Buy":
        tp = round(tick.ask + 22.559, 5)
    else:
        tp = round(tick.bid - 22.559, 5)

    try:
        success = execute_automated_trade(SYMBOL, decision, sl=None, tp=tp)
        logging.info("Placed new trade: %s, TP=%s, success=%s", decision, tp, success)
        return success
    except Exception as e:
        logging.error("Error placing trade: %s", e)
        return False

# ------------------------------------------------------------
# Main loop: rebuilt with requested behavior
# ------------------------------------------------------------
def main_loop():
    logging.info("Connecting to MT5...")
    if not mt5.initialize():
        logging.error("MT5 initialization failed.")
        return

    logging.info("Initializing predictors...")
    initialize_predictors()

    last_minute = -1
    try:
        while True:
            now = datetime.now()
            minute = now.minute

            # Trigger exactly on 00,15,30,45
            if minute % 15 == 0 and minute != last_minute:
                last_minute = minute
                logging.info("=== New cycle at %s ===", now.strftime("%Y-%m-%d %H:%M:%S"))

                restricted_mode = is_restricted_time(now)
                if restricted_mode:
                    logging.info("⚠️ Restricted time active → New trade placement will be blocked; closures allowed.")

                # Get model decision
                decision, fusion, raw_models = make_trade_decision()
                logging.info("Decision: %s | Fusion confidence: %s", decision, fusion.get("confidence", None))

                # If no executor available, still perform closes (if required) but cannot place new trades
                can_place = (execute_automated_trade is not None) and (not restricted_mode)

                # Fetch positions for SYMBOL
                positions = mt5.positions_get(symbol=SYMBOL) or []
                # Determine existing directions in simpler form
                buy_positions = [p for p in positions if p.type == mt5.ORDER_TYPE_BUY]
                sell_positions = [p for p in positions if p.type == mt5.ORDER_TYPE_SELL]

                # Map decision to expected type
                if decision == "Buy":
                    desired_type = mt5.ORDER_TYPE_BUY
                    opposite_list = sell_positions
                    same_list = buy_positions
                elif decision == "Sell":
                    desired_type = mt5.ORDER_TYPE_SELL
                    opposite_list = buy_positions
                    same_list = sell_positions
                else:  # Hold
                    desired_type = None
                    opposite_list = []
                    same_list = buy_positions + sell_positions  # any existing

                # ----------------------------
                # 1) If Hold -> nothing unless you want to close? (we keep positions)
                # ----------------------------
                if decision == "Hold":
                    logging.info("Hold - no action taken on positions.")
                    time.sleep(60)
                    continue

                # ----------------------------
                # 2) If same direction exists -> keep it (do nothing)
                # ----------------------------
                if same_list:
                    logging.info("Existing position(s) in desired direction found (%d). Keeping them; no new entry.",
                                 len(same_list))
                    # We do not place a new trade if same direction exists.
                    time.sleep(60)
                    continue

                # ----------------------------
                # 3) Opposite positions exist -> close them (always allowed)
                # ----------------------------
                if opposite_list:
                    logging.info("Opposite position(s) found (%d). Closing them before any placement.", len(opposite_list))
                    closed_all = True
                    for pos in opposite_list:
                        ok = close_position(pos)
                        if not ok:
                            closed_all = False
                            logging.warning("Failed to fully close position %s; continuing to next but placement may be affected.", pos.ticket)
                    # refresh positions after attempting closure
                    time.sleep(0.5)
                    positions_after = mt5.positions_get(symbol=SYMBOL) or []
                    # ensure no same-direction accidentally exists:
                    same_after = [p for p in positions_after if (p.type == desired_type)]
                    if same_after:
                        logging.info("After closure attempts, found %d same-direction positions; will NOT place another entry.", len(same_after))
                        time.sleep(60)
                        continue
                else:
                    logging.info("No opposite positions present; continuing to placement check.")

                # ----------------------------
                # 4) Place new trade only if allowed by time and executor available
                # ----------------------------
                if can_place:
                    logging.info("Placing new trade (allowed by time & executor).")
                    placed = place_new_trade(decision)
                    if not placed:
                        logging.error("Failed to place new trade for %s.", decision)
                else:
                    if execute_automated_trade is None:
                        logging.info("Executor not available -> skipping placement (closures were performed if needed).")
                    elif restricted_mode:
                        logging.info("Restricted mode -> skipping placement (closures were performed if needed).")

                time.sleep(60)

            else:
                # poll every second until next 15-min tick
                time.sleep(1)

    except KeyboardInterrupt:
        logging.info("User stopped the bot.")
    finally:
        mt5.shutdown()
        logging.info("MT5 shutdown complete.")

if __name__ == "__main__":
    main_loop()
