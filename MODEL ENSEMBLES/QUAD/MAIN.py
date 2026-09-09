import logging
import time
from datetime import datetime
import MetaTrader5 as mt5
from adxdeploy import TradingPredictor as ADXPredictor
from bollingerdeploy import TradingPredictor as BollingerPredictor
from candledeploy import TradingPredictor as CandlePredictor
from placer import execute_automated_trade
from positionploy import TradingPredictor as PositionPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Initialize MetaTrader 5
if not mt5.initialize():
    logging.error("Failed to initialize MetaTrader 5. Exiting.")
    exit()

# Initialize predictors
candle_predictor = CandlePredictor()
adx_predictor = ADXPredictor()
bollinger_predictor = BollingerPredictor()
position_predictor = PositionPredictor()

active_trades = {}
TIMEFRAMES = {
    "00:02": mt5.TIMEFRAME_M30,
    "30:02": mt5.TIMEFRAME_H1,
    "30:02": mt5.TIMEFRAME_M30,
}
last_checked_minute = None


def get_candle_data(symbol, timeframe=mt5.TIMEFRAME_M15, count=2):
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None or len(rates) < 2:
            return None, None, None
        last_candle = rates[-2]
        return last_candle[2], last_candle[3], last_candle
    except Exception as e:
        logging.warning(f"Error fetching candle data for {symbol}: {e}")
        return None, None, None


def fuse_decision(candle_result, adx_result, boll_result, position_result):
    """
    Fuse decisions from different predictors with priority:
    1. If PositionPredictor and BollingerPredictor agree on a trade direction, prioritize it.
    2. If ADXPredictor provides a clear signal, use it.
    3. If BollingerPredictor provides a clear signal, use it.
    4. Default to CandlePredictor's decision.
    """
    decisions = {
        "candle": candle_result["decision"],
        "adx": adx_result["final_decision"],
        "bollinger": boll_result["final_decision"],
        "position": position_result["final_decision"],
    }

    logging.info(
        f"Decisions - Candle: {decisions['candle']}, ADX: {decisions['adx']}, "
        f"Bollinger: {decisions['bollinger']}, Position: {decisions['position']}"
    )

    # Check for override condition
    position_decision = decisions["position"]
    bollinger_decision = decisions["bollinger"]

    if "Upper Out" in position_decision and bollinger_decision == "Sell":
        logging.info("Override: Strong SELL signal from Position and Bollinger.")
        return {
            "symbol": position_result.get("symbol", candle_result["symbol"]),
            "decision": "Sell",
            "confidence": min(
                position_result.get("confidence", 1.0),
                boll_result.get("confidence", 1.0),
            ),
        }

    if "Lower Out" in position_decision and bollinger_decision == "Buy":
        logging.info("Override: Strong BUY signal from Position and Bollinger.")
        return {
            "symbol": position_result.get("symbol", candle_result["symbol"]),
            "decision": "Buy",
            "confidence": min(
                position_result.get("confidence", 1.0),
                boll_result.get("confidence", 1.0),
            ),
        }

    # Fallback to other predictors
    if decisions["candle"] == "No Trade" and bollinger_decision in ["Buy", "Sell"]:
        return {
            "symbol": boll_result.get("symbol", candle_result["symbol"]),
            "decision": bollinger_decision,
            "confidence": boll_result.get("confidence", 1.0),
        }

    if decisions["adx"] in ["Buy", "Sell"]:
        return {
            "symbol": adx_result.get("symbol", candle_result["symbol"]),
            "decision": decisions["adx"],
            "confidence": adx_result.get("confidence", 1.0),
        }

    return candle_result


def trigger_functions():
    global active_trades, last_checked_minute
    logging.info("Auto-Trader Started...")

    while True:
        now = datetime.now()
        current_time = now.strftime("%M:%S")

        if now.minute != last_checked_minute and current_time in TIMEFRAMES:
            last_checked_minute = now.minute

            try:
                timeframe = TIMEFRAMES[current_time]
                candle = candle_predictor.get_most_confident_prediction(timeframe)
                adx = adx_predictor.get_final_decision()
                boll = bollinger_predictor.get_final_decision()
                position = position_predictor.get_final_decision()

                final = fuse_decision(candle, adx, boll, position)
                logging.info(f"Final Decision: {final}")

                symbol = final["symbol"]
                decision = final["decision"]
                confidence = final["confidence"]

                if decision == "No Trade" or confidence < 0.5:
                    continue

                high, low, _ = get_candle_data(symbol, timeframe)
                positions = mt5.positions_get(symbol=symbol)

                if symbol in active_trades and not positions:
                    del active_trades[symbol]

                if symbol in active_trades and active_trades[symbol] != decision:
                    continue

                if decision in ["Buy", "Sell"]:
                    if execute_automated_trade(symbol, decision):
                        logging.info(f"{symbol} - {decision} ({confidence:.2%})")
                        active_trades[symbol] = decision
                    else:
                        logging.error(f"Trade failed for {symbol}")

            except Exception as e:
                logging.error(f"Error: {e}")

        time.sleep(0.5)


if __name__ == "__main__":
    trigger_functions()
