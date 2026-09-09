from deploy import TradingPredictor
from datetime import datetime
import time
from placer import execute_automated_trade
import MetaTrader5 as mt5

# Initialize MetaTrader 5

def initialize_mt5(retries=5):
    for i in range(retries):
        if mt5.initialize():
            return True
        print(f"❌ MT5 initialization failed. Retrying... ({i+1}/{retries})")
        time.sleep(5)
    print("🚨 Critical Error: MT5 failed to initialize.")
    return False

if not initialize_mt5():
    exit()

# Parameters for Adaptive Fibonacci
N_CANDLES = 50  # Lookback period for high/low range
TOLERANCE = 0.025  # 2.5% tolerance
FIB_BASE_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786]  # Default Fibonacci ratios

# Initialize predictor
predictor = TradingPredictor()
active_trade = None
trade_blocked = False  # Track if trade blocking has been logged

SYMBOL = "GBPJPY"
TIMEFRAMES = {"00:02": "30M", "15:02": "15M", "30:02": "1H", "45:02": "15M"}

def get_current_price(symbol=SYMBOL):
    """Fetches the latest market price for GBPJPY."""
    try:
        info = mt5.symbol_info(symbol)
        if not info or not info.visible:
            print(f"⚠️ Symbol {symbol} not available in MT5. Enabling it...")
            mt5.symbol_select(symbol, True)
        tick = mt5.symbol_info_tick(symbol)
        return tick.bid if tick else None
    except Exception as e:
        print(f"⚠️ Error fetching price: {e}")
        return None

def get_candle_data(symbol=SYMBOL, timeframe=mt5.TIMEFRAME_M15, count=2):
    """Fetch recent price data and return the last completed candle."""
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None or len(rates) < 2:
            return None, None, None  # No sufficient data available

        last_completed_candle = rates[-2]  # Use the second-last candle (completed one)
        return last_completed_candle[2], last_completed_candle[3], last_completed_candle  # High, Low, and full candle data
    except Exception as e:
        print(f"⚠️ Error fetching candle data: {e}")
        return None, None, None

def trigger_functions():
    global active_trade, trade_blocked
    print(f"🔹 [{datetime.now()}] GBPJPY Auto-Trader Started...")

    while True:
        now = datetime.now()
        current_time = now.strftime('%M:%S')

        if now.second == 2 and current_time in TIMEFRAMES:
            timeframe = TIMEFRAMES[current_time]

            try:
                prediction = {
                    "30M": predictor.get_predictions_30M,
                    "15M": predictor.get_predictions_15M,
                    "1H": predictor.get_predictions_1H
                }.get(timeframe, lambda: None)()

                if prediction and "decision" in prediction:
                    signal = prediction["decision"]

                    high, low, last_completed_candle = get_candle_data(
                        timeframe=mt5.TIMEFRAME_M15 if timeframe == "15M" else 
                        mt5.TIMEFRAME_M30 if timeframe == "30M" else 
                        mt5.TIMEFRAME_H1
                    )

                    positions = mt5.positions_get(symbol=SYMBOL)

                    if not positions:
                        trade_blocked = False
                    
                    if positions and not trade_blocked:
                        print(f"⚠️ [{now}] Trade blocked due to ongoing position: {positions[0].type}")
                        trade_blocked = True  # Ensure it prints only once
                        continue  # Stop execution

                    if signal in ["Buy", "Sell"]:
                        success = execute_automated_trade(signal)

                        if success:
                            print(f"✅ [{now}] GBPJPY Trade Executed: {signal} ({timeframe})")
                            active_trade = {"signal": signal, "entry_price": get_current_price()}
                            trade_blocked = False  # Reset block flag after new trade execution
                        else:
                            print(f"❌ [{now}] Trade execution failed.")
                            active_trade = None  
                    else:
                        print(f"🔹 [{now}] GBPJPY Prediction received: {signal}, skipping execution.")

            except Exception as e:
                print(f"❌ [{now}] GBPJPY Error: {e}")

        time.sleep(1)  # Reduce CPU usage

if __name__ == "__main__":
    trigger_functions()
