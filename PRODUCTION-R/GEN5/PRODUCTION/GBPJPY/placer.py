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
        print(f"Failed to retrieve symbol info for {symbol}.", mt5.last_error())
        return 0.01, 0.01  # Default values for Forex pairs
    return symbol_info.volume_min, symbol_info.volume_step

def calculate_lot_size(free_margin, symbol):
    """Calculate lot size based on free margin while adhering to symbol constraints."""
    min_lot, lot_step = get_lot_constraints(symbol)
    margin_per_lot = mt5.symbol_info(symbol).margin_initial or 200  # Approximate margin for GBP/JPY
    lot_size = (free_margin / margin_per_lot) * 0.1  # Adjusted for GBP/JPY trading
    lot_size = math.floor(lot_size / lot_step) * lot_step  # Always rounds down
    return max(lot_size, min_lot)

def place_trade(symbol, trade_type):
    """Place a buy or sell order on MT5."""
    if not initialize_mt5():
        return False
    
    free_margin = get_free_margin()
    if free_margin is None:
        return False
    
    lot_size = calculate_lot_size(free_margin, symbol)
    if lot_size <= 0:
        print("Invalid lot size calculated.")
        return False

    trade_action = mt5.ORDER_TYPE_BUY if trade_type.lower() == "buy" else mt5.ORDER_TYPE_SELL
    
    # Ensure symbol is active for trading
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info or not symbol_info.visible:
        print(f"Symbol {symbol} is not available for trading.")
        return False
    
    tick_info = mt5.symbol_info_tick(symbol)
    if tick_info is None:
        print(f"Failed to retrieve tick info for {symbol}.", mt5.last_error())
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
    
    print(f"Sending trade request: {request}")
    result = mt5.order_send(request)
    
    if result is None:
        print(f"Order send failed. Error: {mt5.last_error()}")
    elif result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"Trade placed: {trade_type.upper()} {lot_size} lots at {price}")
    else:
        print(f"Trade failed, error code: {result.retcode}")
    
    return result

def execute_automated_trade(prediction, symbol="GBPJPY"):
    """Executes trade based on model prediction."""
    if not prediction:  # Handles None or empty string
        print("No valid prediction received, skipping trade.")
        return None  # Exit safely

    prediction = prediction.lower()

    if prediction == "buy":
        print(f"{datetime.now()}: Prediction received: BUY, placing trade on {symbol}...")
        return place_trade(symbol, "buy")
    
    if prediction == "sell":
        print(f"{datetime.now()}: Prediction received: SELL, placing trade on {symbol}...")
        return place_trade(symbol, "sell")
    
    if prediction == "no trade":
        print(f"{datetime.now()}: Prediction received: NO TRADE, skipping execution.")
        return None  # Do nothing
    
    print("Invalid prediction received, skipping trade.")
    return None
