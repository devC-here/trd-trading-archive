from candledeploy import TradingPredictor as CandlePredictor
from adxdeploy import TradingPredictor as ADXPredictor
from datetime import datetime
import time
from placer import execute_automated_trade
import MetaTrader5 as mt5

# Initialize MetaTrader 5
def initialize_mt5(retries=5):
    for attempt in range(retries):
        if mt5.initialize():
            return True
        print(f"❌ MT5 initialization failed. Retrying ({attempt + 1}/{retries})...")
        time.sleep(5)
    print("❌ Failed to initialize MT5 after multiple attempts. Exiting.")
    return False

if not initialize_mt5():
    exit()

# Initialize predictors
candle_predictor = CandlePredictor()
adx_predictor = ADXPredictor()

active_trades = {}

TIMEFRAMES = {
    "00:02": mt5.TIMEFRAME_M30,
    "30:02": mt5.TIMEFRAME_H1,
    "30:02": mt5.TIMEFRAME_M30,
}
last_checked_minute = None

def get_current_price(symbol):
    try:
        tick = mt5.symbol_info_tick(symbol)
        return tick.bid if tick else None
    except Exception as e:
        print(f"⚠️ Error fetching price for {symbol}: {e}")
        return None

def get_candle_data(symbol, timeframe=mt5.TIMEFRAME_M15, count=2):
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None or len(rates) < 2:
            return None, None, None
        last_completed_candle = rates[-2]
        return last_completed_candle[2], last_completed_candle[3], last_completed_candle
    except Exception as e:
        print(f"⚠️ Error fetching candle data for {symbol}: {e}")
        return None, None, None

def fuse_decision(candle_result, adx_result):
    # If ADX is "No Trade", fallback to candlestick
    if adx_result["final_decision"] == "No Trade":
        return candle_result
    # Otherwise follow ADX decision only
    return {
        "symbol": adx_result.get("symbol", candle_result["symbol"]),
        "decision": adx_result["final_decision"],
        "confidence": adx_result.get("confidence", 1.0)
    }

def trigger_functions():
    global active_trades, last_checked_minute
    print(f"🔹 [{datetime.now()}] Auto-Trader Started...")

    while True:
        now = datetime.now()
        current_time = now.strftime('%M:%S')

        try:
            if now.minute != last_checked_minute and current_time in TIMEFRAMES:
                last_checked_minute = now.minute

                timeframe = TIMEFRAMES[current_time]
                candle_result = candle_predictor.get_most_confident_prediction(timeframe)
                adx_result = adx_predictor.get_final_decision()
                final_decision = fuse_decision(candle_result, adx_result)

                print(f"🔍 [{now}] Final Decision: {final_decision}")
                symbol, decision, confidence = final_decision["symbol"], final_decision["decision"], final_decision["confidence"]

                if decision == "No Trade" or confidence < 0.5:
                    print(f"🔹 [{now}] No strong trade signal for {symbol}. Skipping.")
                    continue

                high, low, last_completed_candle = get_candle_data(symbol, timeframe=mt5.TIMEFRAME_M15)
                positions = mt5.positions_get(symbol=symbol)

                if symbol in active_trades and not positions:
                    print(f"🟡 [{now}] No active positions found for {symbol}. Resetting active_trade.")
                    del active_trades[symbol]

                if symbol in active_trades and active_trades[symbol] != decision:
                    print(f"⚠️ [{now}] Trade blocked: Current: {active_trades[symbol]}, Signal: {decision}")
                    continue

                if decision in ["Buy", "Sell"]:
                    success = execute_automated_trade(symbol, decision)
                    if success:
                        print(f"✅ [{now}] Trade Executed: {symbol} - {decision} with confidence {confidence}")
                        active_trades[symbol] = decision
                    else:
                        print(f"❌ [{now}] Trade execution failed for {symbol}.")
                else:
                    print(f"🔹 [{now}] Prediction received: {symbol} - {decision}, skipping execution.")

        except Exception as e:
            print(f"❌ [{now}] Error: {e}")

        time.sleep(0.5)

if __name__ == "__main__":
    trigger_functions()
