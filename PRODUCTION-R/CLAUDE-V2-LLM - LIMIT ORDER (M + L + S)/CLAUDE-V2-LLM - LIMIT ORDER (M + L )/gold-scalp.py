import MetaTrader5 as mt5
import time
from datetime import datetime

# ===== CONFIG =====
SYMBOL = "XAUUSD"
LOT = 0.01
SL_USD = 1.0
TP_USD = 1.5

mt5.initialize()

last_trade_minute = None


def usd_to_price(usd):
    return usd  # for gold (0.01 lot ≈ 1 oz)


def place_orders():
    tick = mt5.symbol_info_tick(SYMBOL)
    ask = tick.ask
    bid = tick.bid

    move_sl = usd_to_price(SL_USD)
    move_tp = usd_to_price(TP_USD)

    buy = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": LOT,
        "type": mt5.ORDER_TYPE_BUY,
        "price": ask,
        "sl": ask - move_sl,
        "tp": ask + move_tp,
        "deviation": 20,
        "magic": 111,
        "comment": "py_buy",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    sell = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": LOT,
        "type": mt5.ORDER_TYPE_SELL,
        "price": bid,
        "sl": bid + move_sl,
        "tp": bid - move_tp,
        "deviation": 20,
        "magic": 111,
        "comment": "py_sell",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    print("Placing trades...")
    print(mt5.order_send(buy))
    print(mt5.order_send(sell))


def get_m1_calibrated(symbol="XAUUSD", offset_sec=-1):
    mt5.symbol_select(symbol, True)
    tick = mt5.symbol_info_tick(symbol)

    if tick is None:
        return None

    server_time = tick.time
    sec_in_minute = server_time % 300
    remaining = 300 - sec_in_minute

    wait_time = remaining - (time.time() % 1) + offset_sec
    return wait_time


# ===== MAIN LOOP =====
while True:
    wait = get_m1_calibrated(SYMBOL, offset_sec=-1)

    if wait is None:
        time.sleep(1)
        continue

    if wait > 0:
        time.sleep(wait)

    # identify candle (by minute)
    now = datetime.now()
    current_minute = now.minute

    # prevent duplicate trades
    if current_minute != last_trade_minute:
        print("Trigger at:", now)

        place_orders()
        last_trade_minute = current_minute

    # small delay to avoid loop spam
    time.sleep(0.2)