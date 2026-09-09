import MetaTrader5 as mt5
import logging
import time
from datetime import datetime, timedelta

from momentumdeploy import MomentumPredictor
from breakoutdeploy import BreakoutPredictor
from microstructure import MicrostructurePredictor
from VWdeploy import VwapVolumePredictor as VWAPPredictor
from candledeploy import CandlestickSequencePredictor
from fusion import fuse_model_outputs as combine_model_outputs

try:
    from placer import execute_automated_trade
except ImportError:
    execute_automated_trade = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PREDICTORS = {}
SYMBOL = "XAUUSDm"

def initialize_predictors():
    global PREDICTORS
    PREDICTORS = {
        "momentum": MomentumPredictor(),
        "breakout": BreakoutPredictor(),
        "microstructure": MicrostructurePredictor(),
        "vwap": VWAPPredictor(),
        "candle": CandlestickSequencePredictor()
    }
    logging.info("All 5 predictors initialized.")

def get_fused_prediction(symbol, timeframe):
    results = {}
    try:
        results["momentum"] = PREDICTORS["momentum"].predict(symbol, timeframe)
        results["breakout"] = PREDICTORS["breakout"].predict(symbol, timeframe)
        results["micro"] = PREDICTORS["microstructure"].predict(symbol, timeframe)
        results["vwap"] = PREDICTORS["vwap"].predict(symbol, timeframe)
        results["candle"] = PREDICTORS["candle"].predict(symbol, timeframe)

        for name, pred in results.items():
            logging.info(f"📊 {name.capitalize()} Prediction: {pred}")

        fusion = combine_model_outputs(
            results["momentum"],
            results["breakout"],
            results["micro"],
            results["vwap"],
            results["candle"]
        )

        logging.info(f"🧫 Fusion Output: {fusion}")
        return fusion, results

    except Exception as e:
        logging.error(f"Fusion error: {e}")
        return {"final_bias": "neutral", "entry_viability": "no", "confidence": 0.0, "supporting_factors": [str(e)]}, results

def is_restricted_time(now):
    # Block 18:00–21:59
    if 18 <= now.hour <= 19:
        return datetime(now.year, now.month, now.day, 20, 0)
    # Block 12:30–13:59
    if (now.hour == 12 and now.minute >= 30) or (now.hour == 13):
        return datetime(now.year, now.month, now.day, 14, 0)
    if (now.hour == 0 and now.minute >= 30 ) or (now.hour == 1):
        return datetime (now.year,now.month,now.day,2,0 )
    return None

def make_trade_decision():
    now = datetime.now()
    timeframe = mt5.TIMEFRAME_M15
    label = "M15"

    fusion_output, raw = get_fused_prediction(SYMBOL, timeframe)
    direction = fusion_output["final_bias"]
    viability = fusion_output["entry_viability"]
    confidence = fusion_output["confidence"]

    if direction == "long" and viability == "high":
        decision = "Buy"
    elif direction == "short" and viability == "high":
        decision = "Sell"
    else:
        decision = "Hold"

    logging.info(f"[{label}] Final decision: {decision}, confidence: {confidence}")
    return decision, fusion_output, raw, label

def main_loop():
    print('Starting')
    print('Connecting to MetaTrader5...')
    if not mt5.initialize():
        logging.error("MT5 initialization failed.")
        return
    print('Connected to MetaTrader5')
    print('Initializing predictors...')
    initialize_predictors()

    last_checked_minute = -1

    try:
        while True:
            now = datetime.now()
            minute = now.minute

            if minute % 15 == 0 and minute != last_checked_minute:
                last_checked_minute = minute

                # ⛔ Restriction Check
                restricted_until = is_restricted_time(now)
                if restricted_until:
                    wait_seconds = (restricted_until - now).total_seconds()
                    logging.info(f"⛔ Trading halted until {restricted_until.strftime('%H:%M')}")
                    time.sleep(wait_seconds)
                    continue

                # ✅ Proceed to Decision
                decision, fusion, raw, tf = make_trade_decision()

                if decision in ["Buy", "Sell"] and execute_automated_trade:
                    current_positions = mt5.positions_get(symbol=SYMBOL)
                    desired_type = mt5.ORDER_TYPE_BUY if decision == "Buy" else mt5.ORDER_TYPE_SELL
                    opposite_type = mt5.ORDER_TYPE_SELL if decision == "Buy" else mt5.ORDER_TYPE_BUY

                    has_same_trade = False
                    opposite_tickets = []

                    if current_positions:
                        for pos in current_positions:
                            if pos.type == desired_type:
                                has_same_trade = True
                            elif pos.type == opposite_type:
                                opposite_tickets.append(pos.ticket)

                    for ticket in opposite_tickets:
                        pos = mt5.positions_get(ticket=ticket)[0]
                        close_request = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": SYMBOL,
                            "volume": pos.volume,
                            "type": desired_type,
                            "position": ticket,
                            "deviation": 20,
                            "magic": 123456,
                            "comment": "Auto-close opposite trade",
                            "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": mt5.ORDER_FILLING_IOC,
                        }
                        result = mt5.order_send(close_request)
                        logging.info(f"✅ Closed opposite position {ticket}: {result}")

                    if not has_same_trade:
                        tick = mt5.symbol_info_tick(SYMBOL)
                        if not tick:
                            logging.error("❌ Failed to fetch tick data for TP.")
                        else:
                            tp = round(tick.ask + 0.00030, 5) if decision == "Buy" else round(tick.bid - 0.00030, 5)
                            sl = None
                            success = execute_automated_trade(SYMBOL, decision, sl=sl, tp=tp)
                            logging.info(f"✅ Trade executed: {decision}, success: {success}")
                    else:
                        logging.info(f"ℹ️ Position already open in {decision} direction. No new trade.")
                else:
                    logging.info("No trade executed. Either Hold or trading disabled.")

                time.sleep(60)  # Ensure next 15-min round isn't triggered too soon

            else:
                time.sleep(1)

    except KeyboardInterrupt:
        logging.info("Stopped by user.")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main_loop()
