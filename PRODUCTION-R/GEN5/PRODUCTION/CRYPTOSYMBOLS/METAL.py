from deploy import TradingPredictor
from datetime import datetime
import time
from placer import execute_automated_trade
import MetaTrader5 as mt5

# Initialize MetaTrader 5
def initialize_mt5(retries=5):
    """Attempts to initialize MT5 with retry logic."""
    for attempt in range(retries):
        if mt5.initialize():
            return True
        print(f"❌ MT5 initialization failed. Retrying ({attempt + 1}/{retries})...")
        time.sleep(5)
    print("❌ Failed to initialize MT5 after multiple attempts. Exiting.")
    return False

if not initialize_mt5():
    exit()

# Initialize predictor
predictor = TradingPredictor()
active_trades = {}  # Dictionary to track active trades by symbol

TIMEFRAMES = {
    "00:02": mt5.TIMEFRAME_M30,
    "15:02": mt5.TIMEFRAME_M15,
    "30:02": mt5.TIMEFRAME_H1,
    "45:02": mt5.TIMEFRAME_M15,
}
last_checked_minute = None  # Prevent duplicate executions

def get_current_price(symbol):
    """Fetches the latest market price for a given symbol."""
    try:
        tick = mt5.symbol_info_tick(symbol)
        return tick.bid if tick else None
    except Exception as e:
        print(f"⚠️ Error fetching price for {symbol}: {e}")
        return None

def get_candle_data(symbol, timeframe=mt5.TIMEFRAME_M15, count=2):
    """Fetches recent price data and returns the last completed candle."""
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None or len(rates) < 2:
            return None, None, None  # No sufficient data available

        last_completed_candle = rates[-2]  # Use the second-last candle (completed one)
        return last_completed_candle[2], last_completed_candle[3], last_completed_candle  # High, Low, and full candle data
    except Exception as e:
        print(f"⚠️ Error fetching candle data for {symbol}: {e}")
        return None, None, None

def trigger_functions():
    global active_trades, last_checked_minute
    print(f"🔹 [{datetime.now()}] Auto-Trader Started...")

    while True:
        now = datetime.now()
        current_time = now.strftime('%M:%S')

        try:
            if now.minute != last_checked_minute and current_time in TIMEFRAMES:
                last_checked_minute = now.minute  # Update minute tracker

                timeframe = TIMEFRAMES[current_time]
                top_predictions = predictor.get_most_confident_predictions(timeframe, top_n=4)
                
                if not top_predictions:
                    print(f"🔹 [{now}] No valid trade signals. Skipping this cycle.")
                    continue

                for prediction in top_predictions:
                    symbol, decision, confidence = prediction["symbol"], prediction["decision"], prediction["confidence"]

                    if decision == "No Trade" or confidence < 0.5:
                        print(f"🔹 [{now}] No strong trade signal for {symbol}. Skipping.")
                        continue

                    # Fetch last completed candle
                    high, low, last_completed_candle = get_candle_data(symbol, timeframe=mt5.TIMEFRAME_M15)

                    # Fetch open positions before placing a trade
                    positions = mt5.positions_get(symbol=symbol)

                    # Reset active trade for symbol if no open positions are found
                    if symbol in active_trades and not positions:
                        print(f"🟡 [{now}] No active positions found for {symbol}. Resetting active_trade.")
                        del active_trades[symbol]

                    # Prevent opposite trade for the same symbol
                    if symbol in active_trades and active_trades[symbol] != decision:
                        print(f"⚠️ [{now}] Trade blocked: Current: {active_trades[symbol]}, Signal: {decision}")
                        continue

                    # Execute trade
                    if decision in ["Buy", "Sell"]:
                        success = execute_automated_trade(symbol, decision)
                        if success:
                            print(f"✅ [{now}] Trade Executed: {symbol} - {decision} with confidence {confidence}")
                            active_trades[symbol] = decision  # Track active trade
                        else:
                            print(f"❌ [{now}] Trade execution failed for {symbol}.")
                    else:
                        print(f"🔹 [{now}] Prediction received: {symbol} - {decision}, skipping execution.")

        except Exception as e:
            print(f"❌ [{now}] Error: {e}")

        time.sleep(0.5)  # Reduce CPU usage

if __name__ == "__main__":
    trigger_functions()
