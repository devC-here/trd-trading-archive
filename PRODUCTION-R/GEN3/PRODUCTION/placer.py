import MetaTrader5 as mt5

def initialize_mt5():
    """Initialize MT5 connection if not already connected."""
    if not mt5.initialize():
        print("MT5 initialization failed!", mt5.last_error())
        return False
    return True

def get_capital():
    """Retrieve account balance."""
    account_info = mt5.account_info()
    if account_info is None:
        print("Failed to retrieve account info.", mt5.last_error())
        return None
    return account_info.balance

def get_lot_constraints(symbol):
    """Retrieve min lot size and lot step for the given symbol."""
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Failed to retrieve symbol info for {symbol}.", mt5.last_error())
        return 0.01, 0.01
    return symbol_info.volume_min, symbol_info.volume_step

def calculate_lot_size(capital, symbol):
    """Calculate lot size based on capital while adhering to symbol constraints."""
    min_lot, lot_step = get_lot_constraints(symbol)
    lot_size = (capital / 10) * 0.01
    lot_size = round(lot_size / lot_step) * lot_step  # Align with step size
    return max(lot_size, min_lot)

def place_trade(symbol, trade_type):
    """Place a buy or sell order on MT5."""
    if not initialize_mt5():
        return False
    
    capital = get_capital()
    if capital is None:
        return False
    
    lot_size = calculate_lot_size(capital, symbol)
    if lot_size <= 0:
        print("Invalid lot size calculated.")
        return False

    trade_action = mt5.ORDER_TYPE_BUY if trade_type == "buy" else mt5.ORDER_TYPE_SELL
    
    # Ensure symbol is active for trading
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info or not symbol_info.trade_mode:
        print(f"Symbol {symbol} not available for trading.")
        return False
    
    tick_info = mt5.symbol_info_tick(symbol)
    if tick_info is None:
        print(f"Failed to retrieve tick info for {symbol}.", mt5.last_error())
        return False
    
    price = tick_info.ask if trade_type == "buy" else tick_info.bid

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

def execute_automated_trade(prediction, symbol="XAUUSD"):
    """Executes trade based on model prediction."""
    if not prediction:  # Handles None or empty string
        print("No valid prediction received, skipping trade.")
        return None  # Exit safely

    prediction = prediction.lower()

    if prediction == "buy":
        print(f"Prediction received: BUY, placing trade on {symbol}...")
        return place_trade(symbol, "buy")
    
    if prediction == "sell":
        print(f"Prediction received: SELL, placing trade on {symbol}...")
        return place_trade(symbol, "sell")
    
    if prediction == "no trade":
        print("Prediction received: NO TRADE, skipping execution.")
        return None  # Do nothing
    
    print("Invalid prediction received, skipping trade.")
    return None
