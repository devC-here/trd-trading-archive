import MetaTrader5 as mt5
import math

def initialize_mt5():
    if not mt5.initialize():
        print("MT5 initialization failed!", mt5.last_error())
        return False
    return True

def get_free_margin():
    account_info = mt5.account_info()
    if account_info is None:
        print("Failed to retrieve account info.", mt5.last_error())
        return None
    return account_info.margin_free

def get_account_balance():
    account_info = mt5.account_info()
    if account_info is None:
        print("Failed to retrieve account info.", mt5.last_error())
        return None
    return account_info.balance

def get_lot_constraints(symbol):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"⚠️ Failed to retrieve symbol info for {symbol}.", mt5.last_error())
        return 0.01, 0.01, 1.0
    return symbol_info.volume_min, symbol_info.volume_step, symbol_info.volume_max

def get_open_positions():
    """Retrieve ongoing trades and return a dictionary of placed lots per symbol."""
    positions = mt5.positions_get()
    placed_lots = {}
    
    if positions is None:
        print("⚠️ Failed to retrieve open positions.", mt5.last_error())
        return placed_lots  # Return an empty dictionary if no positions are retrieved
    
    for position in positions:
        symbol = position.symbol
        volume = position.volume
        placed_lots[symbol] = placed_lots.get(symbol, 0) + volume  # Accumulate lots per symbol
    
    return placed_lots

def calculate_lot_size(free_margin, symbol, placed_lots, account_balance):
    min_lot, lot_step, max_lot = get_lot_constraints(symbol)
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        return min_lot
    
    margin_per_lot = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, symbol, 1.0, symbol_info.bid)
    if margin_per_lot is None or margin_per_lot <= 0:
        margin_per_lot = 150  # Fallback margin per lot
    
    raw_lot_size = free_margin / margin_per_lot
    lot_size = math.floor(raw_lot_size / lot_step) * lot_step  # Round down to nearest valid step
    lot_size = round(lot_size, 2)  # Ensure rounding to 2 decimal places
    
    max_allowed_lots = (account_balance / 10) * 0.01
    remaining_lots = max_allowed_lots - placed_lots
    lot_size = min(lot_size, remaining_lots, max_lot)
    
    return max(lot_size, min_lot) if remaining_lots > 0 else 0

def execute_automated_trade(symbol, prediction):
    if not prediction:
        return None
    prediction = prediction.lower()
    if prediction not in ["buy", "sell"]:
        return None
    
    free_margin = get_free_margin()
    account_balance = get_account_balance()
    if free_margin is None or account_balance is None:
        return False
    
    placed_lots = get_open_positions()  # Get ongoing trades dynamically
    lot_size = calculate_lot_size(free_margin, symbol, placed_lots.get(symbol, 0), account_balance)
    
    if lot_size <= 0:
        return False
    
    trade_action = mt5.ORDER_TYPE_BUY if prediction == "buy" else mt5.ORDER_TYPE_SELL
    tick_info = mt5.symbol_info_tick(symbol)
    price = tick_info.ask if prediction == "buy" else tick_info.bid
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
    
    result = mt5.order_send(request)
    
    return result if result and result.retcode == mt5.TRADE_RETCODE_DONE else False
