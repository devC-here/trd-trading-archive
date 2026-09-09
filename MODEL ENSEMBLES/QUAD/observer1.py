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

def calculate_sl_tp(symbol, order_type, entry_price):
    sl_distance = 5
    tp_distance = 7

    if order_type == mt5.ORDER_TYPE_BUY:
        sl = entry_price - sl_distance
        tp = entry_price + tp_distance
    else:
        sl = entry_price + sl_distance
        tp = entry_price - tp_distance

    print(f"📊 Using fixed SL/TP | SL={sl:.5f}, TP={tp:.5f}")
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

def monitor_trade(order, monitored_trades, lock):
    trade_id = order.ticket
    symbol = order.symbol
    order_type = order.type
    open_price = order.price_open

    sl, tp = calculate_sl_tp(symbol, order_type, open_price)
    if sl is None or tp is None:
        print(f"⚠️ Could not calculate SL/TP for trade {trade_id}. Skipping monitoring.")
        return

    updated = update_sl_tp(trade_id, symbol, sl, tp)
    if not updated:
        return

    print(f"📊 SL/TP set for trade {trade_id}. No further monitoring will be done.")
    with lock:
        monitored_trades.discard(trade_id)

def monitor_all_trades():
    print("🚀 Continuous trade SL/TP assignment started...")
    monitored_trades = set()
    lock = threading.Lock()

    while True:
        open_trades = mt5.positions_get()
        if open_trades:
            for trade in open_trades:
                with lock:
                    if trade.ticket not in monitored_trades:
                        print(f"🆕 New trade detected: {trade.ticket}. Setting SL/TP...")
                        t = threading.Thread(target=monitor_trade, args=(trade, monitored_trades, lock), daemon=True)
                        t.start()
                        monitored_trades.add(trade.ticket)
        time.sleep(0.2)

monitor_all_trades()
