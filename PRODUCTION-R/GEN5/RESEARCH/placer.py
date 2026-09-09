import MetaTrader5 as mt5
from datetime import datetime
import math

def initialize_mt5():
    """Initialize MT5 connection if not already connected."""
    if not mt5.initialize():
        print("MT5 initialization failed!", mt5.last_error())
        return False
    return True

def get_free_margin():
    """Retrieve free margin available in the trading account."""
    account_info = mt5.account_info()
    if account_info is None:
        print("Failed to retrieve account info.", mt5.last_error())
        return None
    return account_info.margin_free  # Use free margin instead of total balance

def get_lot_constraints(symbol):
    """Retrieve min lot size and lot step for the given symbol."""
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"⚠️ Failed to retrieve symbol info for {symbol}.", mt5.last_error())
        return 0.01, 0.01  # Default values for Forex pairs
    return symbol_info.volume_min, symbol_info.volume_step

def calculate_lot_size(free_margin, symbol):
    """Calculate appropriate lot size based on free margin, while following broker constraints."""
    min_lot, lot_step = get_lot_constraints(symbol)
    symbol_info = mt5.symbol_info(symbol)

    if not symbol_info:
        print(f"{datetime.now()} - ❌ Failed to retrieve symbol info for {symbol}")
        return min_lot  # Fallback to min lot

    margin_per_lot = symbol_info.margin_initial or 150  # Use fallback if unavailable

    # Debugging Info
    print(f"{datetime.now()} - 📊 Free Margin: {free_margin}, Margin Per Lot: {margin_per_lot}")

    # Ensure lot calculation does not exceed limits
    raw_lot_size = (free_margin / margin_per_lot) * 0.5  # 50% risk allocation
    lot_size = math.floor(raw_lot_size / lot_step) * lot_step  # Align to lot step

    # Ensure the final lot size is within the broker's range
    final_lot_size = max(min(lot_size, symbol_info.volume_max), min_lot)

    print(f"{datetime.now()} - 🔹 Final Lot Size for {symbol}: {final_lot_size}")
    return final_lot_size

def has_open_position(symbol, trade_type):
    """Check if there is an existing position in the opposite direction."""
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return False
    for position in positions:
        if (trade_type == "buy" and position.type == mt5.ORDER_TYPE_SELL) or \
           (trade_type == "sell" and position.type == mt5.ORDER_TYPE_BUY):
            print(f"⚠️ Open opposite trade found for {symbol}. Blocking trade.")
            return True
    return False

def place_trade(symbol, trade_type):
    """Place a buy or sell order on MT5."""
    if not initialize_mt5():
        return False
    
    if has_open_position(symbol, trade_type):
        print(f"❌ Trade blocked: Opposite position already exists for {symbol}.")
        return False
    
    free_margin = get_free_margin()
    if free_margin is None:
        return False
    
    lot_size = calculate_lot_size(free_margin, symbol)
    if lot_size <= 0:
        print("❌ Invalid lot size calculated.")
        return False

    trade_action = mt5.ORDER_TYPE_BUY if trade_type.lower() == "buy" else mt5.ORDER_TYPE_SELL
    
    # Ensure symbol is active for trading
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info or not symbol_info.visible:
        print(f"⚠️ Symbol {symbol} is not available for trading.")
        return False
    
    tick_info = mt5.symbol_info_tick(symbol)
    if tick_info is None:
        print(f"❌ Failed to retrieve tick info for {symbol}.", mt5.last_error())
        return False
    
    price = tick_info.ask if trade_type.lower() == "buy" else tick_info.bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": trade_action,
        "price": price,
        "deviation": 10,
        "magic": 0,
        "comment": "Auto Trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    print(f"📤 Sending trade request: {request}")
    result = mt5.order_send(request)
    
    if result is None:
        print(f"❌ Order send failed. Error: {mt5.last_error()}")
    elif result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"✅ Trade placed: {trade_type.upper()} {lot_size} lots at {price}")
    else:
        print(f"❌ Trade failed, error code: {result.retcode}")
    
    return result

def execute_automated_trade(symbol, prediction):
    """Executes trade based on model prediction."""
    if not prediction:  # Handles None or empty string
        print("❌ No valid prediction received, skipping trade.")
        return None  # Exit safely

    prediction = prediction.lower()

    if prediction == "buy":
        print(f"{datetime.now()} - 📈 Prediction: BUY, placing trade on {symbol}...")
        return place_trade(symbol, "buy")
    
    if prediction == "sell":
        print(f"{datetime.now()} - 📉 Prediction: SELL, placing trade on {symbol}...")
        return place_trade(symbol, "sell")
    
    if prediction == "no trade":
        print(f"{datetime.now()} - 🛑 Prediction: NO TRADE, skipping execution.")
        return None  # Do nothing
    
    print("❌ Invalid prediction received, skipping trade.")
    return None

# === Debugging ===
if __name__ == "__main__":
    test_symbol = "EURUSD"
    test_prediction = "buy"

    initialize_mt5()  # Ensure MT5 is initialized

    free_margin = get_free_margin()
    print(f"🔹 Free Margin: {free_margin}")

    min_lot, lot_step = get_lot_constraints(test_symbol)
    print(f"🔹 Min Lot: {min_lot}, Lot Step: {lot_step}")

    calculated_lot = calculate_lot_size(free_margin, test_symbol)
    print(f"🔹 Calculated Lot: {calculated_lot}")

    execute_automated_trade(test_symbol, test_prediction)
