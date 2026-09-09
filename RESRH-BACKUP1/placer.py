import MetaTrader5 as mt5
from datetime import datetime
import math
import time

def initialize_mt5():
    """Initialize MT5 connection if not already connected."""
    if not mt5.initialize():
        print("MT5 initialization failed!", mt5.last_error())
        return False
    return True


def get_account_balance():
    """Retrieve account balance."""
    account_info = mt5.account_info()
    if account_info is None:
        print("Failed to retrieve account info.", mt5.last_error())
        return None
    return account_info.balance


def calculate_lot_size(balance):
    """
    Calculate lot size:
    - 0.01 lots per $40 balance
    - Minimum lot size = 0.01
    """
    if balance is None:
        return 0.01  # fallback minimum lot

    lot_units = math.floor(balance /100)
    lot_size = lot_units * 0.01
    lot_size = max(lot_size, 0.01)  # Ensure at least 0.01 lots

    print(f"💰 Account Balance: {balance}, Calculated Lot Size: {lot_size}")
    return lot_size


def has_open_position(symbol, trade_type):
    """Check if there is an existing opposite position."""
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return False
    for position in positions:
        if (trade_type == "buy" and position.type == mt5.ORDER_TYPE_SELL) or \
           (trade_type == "sell" and position.type == mt5.ORDER_TYPE_BUY):
            print(f"⚠️ Open opposite trade found for {symbol}. Blocking trade.")
            return True
    return False


def place_trade(symbol, trade_type, sl=None, tp=None):
    """Place a buy or sell order on MT5 with optional SL/TP."""
    if not initialize_mt5():
        return False

    # Ensure symbol is visible
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None or not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            print(f"❌ Failed to select symbol {symbol}")
            return False

    if has_open_position(symbol, trade_type):
        print(f"❌ Trade blocked: Opposite position already exists for {symbol}.")
        return False

    balance = get_account_balance()
    if balance is None:
        return False

    lot_size = calculate_lot_size(balance)
    if lot_size <= 0:
        print("❌ Invalid lot size calculated.")
        return False

    trade_action = mt5.ORDER_TYPE_BUY if trade_type.lower() == "buy" else mt5.ORDER_TYPE_SELL

    tick_info = mt5.symbol_info_tick(symbol)
    if tick_info is None:
        print(f"❌ Failed to get tick info for {symbol}. Error: {mt5.last_error()}")
        return False

    price = tick_info.ask if trade_type.lower() == "buy" else tick_info.bid
    #magic number 1
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": trade_action,
        "price": price,
        "deviation": 10,
        "magic": 10000,
        "comment": "Auto Trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    } 

    if tp:
        request["tp"] = tp
    if sl:
        request["sl"] = sl

    print(f"📤 Sending trade request: {request}")
    result = mt5.order_send(request)

    if result is None:
        print(f"❌ Order send failed. Error: {mt5.last_error()}")
    elif result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"✅ Trade placed: {trade_type.upper()} {lot_size} lots at {price}")
    else:
        print(f"❌ Trade failed, error code: {result.retcode}")

    return result


from datetime import datetime

def execute_automated_trade(symbol, prediction, sl=None, tp=None):
    """Execute trade based on prediction, with optional SL/TP."""
    if not prediction:
        print("❌ No valid prediction received, skipping trade.")
        return None

    prediction = prediction.lower()

    if prediction == "buy":
        print(f"{datetime.now()} - 📈 Prediction: BUY, placing trade on {symbol}...")
        return place_trade(symbol, "buy", sl=sl, tp=tp)

    if prediction == "sell":
        print(f"{datetime.now()} - 📉 Prediction: SELL, placing trade on {symbol}...")
        return place_trade(symbol, "sell", sl=sl, tp=tp)

    if prediction == "no trade":
        print(f"{datetime.now()} - 🛑 Prediction: NO TRADE, skipping execution.")
        return None

    print("❌ Invalid prediction received, skipping trade.")
    return None



import MetaTrader5 as mt5
import time

def get_m30_calibrated(symbol="EURUSD.e", offset=2):
    """
    Calculates seconds until the next 30m mark + an additional offset.
    Returns float (seconds.milliseconds).
    """
    # 1. Ensure symbol is selected so info isn't None
    mt5.symbol_select(symbol, True)
    
    info = mt5.symbol_info(symbol)
    if info is None:
        return None
    
    # 2. Calculate raw seconds remaining in the 1800s (30m) block
    # info.time % 1800 = seconds passed since last :00 or :30
    remaining = 1800 - (info.time % 1800)
    
    # 3. Add millisecond precision from local clock and the 2s addition
    # Subtracting (time.time() % 1) aligns us to the 000m s start of the second
    total_wait = remaining - (time.time() % 1) + offset
    
    return total_wait