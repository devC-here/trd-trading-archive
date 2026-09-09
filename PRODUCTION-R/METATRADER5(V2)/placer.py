import MetaTrader5 as mt5
from datetime import datetime
import math


# ------------------------------------------------------------
# MT5 Init
# ------------------------------------------------------------
def initialize_mt5():
    if not mt5.initialize():
        print("MT5 initialization failed!", mt5.last_error())
        return False
    return True


# ------------------------------------------------------------
# Account helpers
# ------------------------------------------------------------
def get_account_balance():
    account_info = mt5.account_info()
    if account_info is None:
        print("Failed to retrieve account info.", mt5.last_error())
        return None
    return account_info.balance


def calculate_lot_size(balance):
    if balance is None:
        return 0.01

    lot_units = math.floor(balance / 40)
    lot_size = max(lot_units * 0.01, 0.01)

    print(f"💰 Balance: {balance}, Lot Size: {lot_size}")
    return lot_size


# ------------------------------------------------------------
# Position safety
# ------------------------------------------------------------
def has_open_position(symbol, trade_type):
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return False

    for position in positions:
        if (trade_type == "buy" and position.type == mt5.ORDER_TYPE_SELL) or \
           (trade_type == "sell" and position.type == mt5.ORDER_TYPE_BUY):
            print(f"⚠️ Opposite position exists for {symbol}. Blocking trade.")
            return True

    return False


# ------------------------------------------------------------
# LIMIT order placement
# ------------------------------------------------------------
def place_trade(symbol, trade_type, sl=None, tp=None):
    if not initialize_mt5():
        return False

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None or not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            print(f"❌ Failed to select symbol {symbol}")
            return False

    if has_open_position(symbol, trade_type):
        print(f"❌ Trade blocked: Opposite position exists for {symbol}")
        return False

    balance = get_account_balance()
    lot_size = calculate_lot_size(balance)
    if lot_size <= 0:
        print("❌ Invalid lot size")
        return False

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"❌ Tick data unavailable for {symbol}")
        return False

    # --------------------------------------------------------
    # LIMIT order logic
    # --------------------------------------------------------
    if trade_type.lower() == "buy":
        order_type = mt5.ORDER_TYPE_BUY_LIMIT
        price = tick.bid          # buy lower
    else:
        order_type = mt5.ORDER_TYPE_SELL_LIMIT
        price = tick.ask          # sell higher

    request = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "magic": 10000,
        "comment": "Auto Limit Trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }

    if sl is not None:
        request["sl"] = sl
    if tp is not None:
        request["tp"] = tp

    print(f"📤 Sending LIMIT order: {request}")
    result = mt5.order_send(request)

    if result is None:
        print(f"❌ Order send failed: {mt5.last_error()}")
        return False

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"✅ LIMIT order placed: {trade_type.upper()} {lot_size} @ {price}")
        return True

    print(f"❌ Order failed, retcode: {result.retcode}")
    return False


# ------------------------------------------------------------
# Execution wrapper (unchanged interface)
# ------------------------------------------------------------
def execute_automated_trade(symbol, prediction, sl=None, tp=None):
    if not prediction:
        print("❌ No prediction, skipping trade.")
        return None

    prediction = prediction.lower()

    if prediction == "buy":
        print(f"{datetime.now()} 📈 BUY signal → LIMIT order")
        return place_trade(symbol, "buy", sl=sl, tp=tp)

    if prediction == "sell":
        print(f"{datetime.now()} 📉 SELL signal → LIMIT order")
        return place_trade(symbol, "sell", sl=sl, tp=tp)

    if prediction == "no trade":
        print(f"{datetime.now()} 🛑 NO TRADE")
        return None

    print("❌ Invalid prediction.")
    return None

import MetaTrader5 as mt5

import MetaTrader5 as mt5


import MetaTrader5 as mt5

def set_ticket_trailing_stop(ticket, trail_usd, symbol):
    """
    Calculates point distance from USD and sets an MT5 Global Variable 
    for the EA to read and execute.
    """
    # 1. Ensure MT5 is connected
    if not mt5.initialize():
        print("MT5 Initialize failed")
        return False

    # 2. Ensure symbol info is available
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Error: Could not find info for {symbol}")
        return False
        
    # 3. Calculate points based on the symbol's point size
    # Note: Use round() to ensure we send a clean number to MT5
    trail_points = round(float(trail_usd) / symbol_info.point)
    
    # 4. Set the Global Variable
    # Format: TSL_TicketNumber = DistanceInPoints
    var_name = f"TSL_{ticket}"
    
    success = mt5.global_variable_set(var_name, float(trail_points))
    
    if success:
        print(f"Success: Sent {trail_points} points trail for ticket {ticket}")
    else:
        print(f"Failed to set Global Variable for ticket {ticket}")
        
    return success

# Example usage:
# set_ticket_trailing_stop(85214796, 50.0, "NAS100")