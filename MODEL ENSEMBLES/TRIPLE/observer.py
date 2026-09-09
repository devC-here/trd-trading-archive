import MetaTrader5 as mt5
import threading
import time
import csv
import pandas as pd
import datetime

# Initialize MT5
if not mt5.initialize():
    print("❌ Failed to initialize MT5.")
    exit()

LOG_FILE = "trade_log.csv"

def log_trade(event, trade_id, profit):
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), event, trade_id, profit])

def calculate_sl_tp(symbol, order_type):
    now = datetime.datetime.now()
    minute = now.minute
    timeframe = mt5.TIMEFRAME_M30 if minute == 0 else mt5.TIMEFRAME_H1
    tf_name = "M30" if timeframe == mt5.TIMEFRAME_M30 else "H1"

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, 1)
    if rates is None or len(rates) == 0:
        return None, None

    rates_df = pd.DataFrame(rates)
    row = rates_df.iloc[0]

    open_price = row["open"]
    high_price = row["high"]
    low_price = row["low"]
    close_price = row["close"]

    body = abs(close_price - open_price)
    wick = high_price - max(open_price, close_price)
    tail = min(open_price, close_price) - low_price

    # Choose part
    if body < wick and body < tail:
        if order_type == mt5.ORDER_TYPE_BUY:
            chosen_part = "wick"
            size = wick
        else:
            chosen_part = "tail"
            size = tail
    else:
        chosen_part = "body"
        size = body

    sl_multiplier = 1.5
    tp_multiplier = 2.0

    if order_type == mt5.ORDER_TYPE_BUY:
        sl = close_price - (sl_multiplier * size)
        tp = close_price + (tp_multiplier * size)
    else:
        sl = close_price + (sl_multiplier * size)
        tp = close_price - (tp_multiplier * size)

    print(f"🕒 Using timeframe: {tf_name} for {symbol}")
    print(f"📊 Candle Data: Open={open_price}, High={high_price}, Low={low_price}, Close={close_price}")
    print(f"📎 Chosen part: {chosen_part} | Size used: {size:.5f}")
    print(f"✅ Final SL/TP: SL={sl:.5f}, TP={tp:.5f}")

    return round(sl, 5), round(tp, 5)

def update_sl_tp(position_id, symbol, sl, tp):
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": position_id,
        "symbol": symbol,
        "sl": sl,
        "tp": tp,
    }
    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"✅ SL/TP updated for trade {position_id}: SL={sl:.5f}, TP={tp:.5f}")
        return True
    else:
        print(f"❌ Failed to update SL/TP for trade {position_id} (Error Code: {result.retcode})")
        return False

def close_trade(position_id, symbol, volume, order_type, closing_profit):
    price = mt5.symbol_info_tick(symbol).bid if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "position": position_id,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_SELL if order_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
        "price": price,
        "deviation": 10,
        "magic": 0,
        "type_filling": mt5.ORDER_FILLING_IOC
    }
    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        log_trade("Closed", position_id, closing_profit)
        print(f"✅ Trade {position_id} closed successfully with profit {closing_profit:.2f} USD.")
    else:
        print(f"❌ Failed to close trade {position_id} (Error Code: {result.retcode})")

def monitor_trade(order, monitored_trades, lock):
    trade_id = order.ticket
    symbol = order.symbol
    volume = order.volume
    order_type = order.type

    sl, tp = calculate_sl_tp(symbol, order_type)
    if sl is None or tp is None:
        print(f"⚠️ Could not calculate SL/TP for trade {trade_id}. Skipping monitoring.")
        return

    updated = update_sl_tp(trade_id, symbol, sl, tp)
    if not updated:
        return

    print(f"📊 Monitoring trade {trade_id} (waiting for TP to hit)")
    trailing_active = False
    max_profit = -float('inf')

    while True:
        order_info = next((pos for pos in mt5.positions_get() if pos.ticket == trade_id), None)
        if not order_info:
            log_trade("Trade disappeared", trade_id, 0)
            print(f"⚠️ Trade {trade_id} no longer exists. Logging and stopping monitoring.")
            with lock:
                monitored_trades.discard(trade_id)
            break

        profit = order_info.profit
        max_profit = max(max_profit, profit)

        # Activate trailing only after TP hit
        if not trailing_active and profit >= tp:
            print(f"🎯 Trade {trade_id} hit TP. Trailing started.")
            trailing_active = True

        if trailing_active and profit <= max_profit * 0.66:
            log_trade("Trailing Exit", trade_id, profit)
            print(f"📉 Trade {trade_id} fell to 66% of max profit ({profit:.2f}). Closing...")
            close_trade(trade_id, symbol, volume, order_type, profit)
            break

        time.sleep(10)

def monitor_all_trades():
    print("🚀 Continuous trade monitoring started...")
    monitored_trades = set()
    lock = threading.Lock()

    while True:
        open_trades = mt5.positions_get()
        if open_trades:
            for trade in open_trades:
                with lock:
                    if trade.ticket not in monitored_trades:
                        print(f"🆕 New trade detected: {trade.ticket}. Starting monitoring...")
                        t = threading.Thread(target=monitor_trade, args=(trade, monitored_trades, lock), daemon=True)
                        t.start()
                        monitored_trades.add(trade.ticket)
        time.sleep(1)

monitor_all_trades()
