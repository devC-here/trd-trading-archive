import MetaTrader5 as mt5
import threading
import time
import csv

# Initialize MT5
if not mt5.initialize():
    print("❌ Failed to initialize MT5.")
    exit()

# Trade log file
LOG_FILE = "trade_log.csv"

# Function to log trade events
def log_trade(event, trade_id, profit):
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), event, trade_id, profit])

def close_trade(position_id, symbol, volume, order_type):
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
        log_trade("Closed", position_id, 0)
        print(f"✅ Trade {position_id} closed successfully.")
    else:
        print(f"❌ Failed to close trade {position_id} (Error: {result.retcode})")

def monitor_trade(order, monitored_trades, lock):
    trade_id = order.ticket
    symbol = order.symbol
    volume = order.volume
    order_type = order.type

    # New SL/TP Calculation (2.5x original values)
    sl_profit = -0.35 * 2.5 * (volume / 0.01)
    tp_profit = 0.85 * 2.5 * (volume / 0.01)
    tsl_trigger = 0.5 * tp_profit  # Move SL to breakeven at 50% TP
    tsl_new_sl = 0.33 * tp_profit  # Move SL to 33% profit at 75% TP
    
    print(f"📊 Monitoring Trade {trade_id} - TP: {tp_profit:.2f} USD, SL: {sl_profit:.2f} USD")

    trailing_active = False

    while True:
        order_info = next((pos for pos in mt5.positions_get() if pos.ticket == trade_id), None)
        if not order_info:
            log_trade("Trade disappeared", trade_id, 0)
            print(f"⚠️ Trade {trade_id} no longer exists. Logging and stopping monitoring.")
            with lock:
                monitored_trades.discard(trade_id)
            break

        profit = order_info.profit

        # Activate Trailing SL at 50% TP
        if profit >= tsl_trigger and not trailing_active:
            sl_profit = 0  # Move SL to breakeven
            trailing_active = True
            print(f"🔄 Trade {trade_id} trailing SL activated. SL moved to breakeven.")

        # Adjust SL further at 75% TP
        if profit >= 0.75 * tp_profit:
            sl_profit = tsl_new_sl  # Move SL to 33% profit
            print(f"🔄 Trade {trade_id} SL moved to 33% profit at 75% TP.")

        # Close trade if it falls back after reaching at least 33% TP
        if profit >= 0.33 * tp_profit and profit <= 0.03:
            log_trade("Trailing SL hit", trade_id, profit)
            print(f"⚠️ Trade {trade_id} fell back to 0.03 profit. Closing trade to secure profit.")
            close_trade(trade_id, symbol, volume, order_type)
            break
        
        # Normal TP Hit
        if profit >= tp_profit:
            log_trade("Take Profit Hit", trade_id, profit)
            print(f"✅ Trade {trade_id} hit take profit ({profit:.2f} USD). Closing...")
            close_trade(trade_id, symbol, volume, order_type)
            break

        time.sleep(1)

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
