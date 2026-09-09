from deploy import TradingPredictor
from datetime import datetime
import time
from placer import execute_automated_trade
import MetaTrader5 as mt5

SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M15

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

def get_positions(symbol):
    """Return current open positions for symbol."""
    try:
        positions = mt5.positions_get(symbol=symbol)
        return positions if positions else []
    except Exception as e:
        print(f"⚠️ Error fetching positions: {e}")
        return []

def close_positions(symbol, direction_to_keep):
    """Close all trades that are not in the desired direction."""
    positions = get_positions(symbol)
    for pos in positions:
        if direction_to_keep == "Buy" and pos.type == 1:  # 1 = Sell
            mt5.Close(pos.ticket)
        elif direction_to_keep == "Sell" and pos.type == 0:  # 0 = Buy
            mt5.Close(pos.ticket)

def trigger_functions():
    print(f"🔹 [{datetime.now()}] EURUSDm Auto-Trader Started (15M only)...")

    last_checked_time = None

    while True:
        now = datetime.now()
        current_time = now.strftime('%M:%S')

        try:
            # Every 15 minutes at second 02
            if current_time.endswith("02") and current_time != last_checked_time and now.minute % 15 == 0:
                last_checked_time = current_time

                prediction = predictor.get_most_confident_prediction(TIMEFRAME)
                if not prediction or "symbol" not in prediction:
                    print(f"⚠️ [{now}] Invalid prediction: {prediction}")
                    continue

                symbol, decision, confidence = prediction["symbol"], prediction["decision"], prediction["confidence"]

                # Only trade EURUSDm
                if symbol != SYMBOL:
                    continue

                print(f"🔍 [{now}] Prediction: {symbol} - {decision} ({confidence:.2f})")

                if decision == "No Trade" or confidence < 0.5:
                    continue

                # Handle positions
                positions = get_positions(symbol)
                if positions:
                    pos_type = positions[0].type  # 0=Buy, 1=Sell
                    if (decision == "Buy" and pos_type == 0) or (decision == "Sell" and pos_type == 1):
                        print(f"🟢 [{now}] Already in {decision} position. Keeping it.")
                        continue
                    else:
                        print(f"🔄 [{now}] Opposite trade found. Closing old position(s).")
                        close_positions(symbol, decision)

                # Place new trade
                success = execute_automated_trade(symbol, decision)
                if success:
                    print(f"✅ [{now}] Trade executed: {symbol} - {decision} ({confidence:.2f})")
                else:
                    print(f"❌ [{now}] Trade execution failed for {symbol}.")

        except Exception as e:
            print(f"❌ [{now}] Error: {e}")

        time.sleep(1)

if __name__ == "__main__":
    trigger_functions()
