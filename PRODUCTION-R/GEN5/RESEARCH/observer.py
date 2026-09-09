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

# Function to close trade
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

# Function to monitor individual trades
def monitor_trade(order, monitored_trades, lock):
    trade_id = order.ticket
    symbol = order.symbol
    volume = order.volume
    order_type = order.type

    # **SL & TP Calculation (2.5x increase)**
    sl_profit = -0.875 * (volume / 0.01)  # SL (negative)
    tp_profit = 2.125 * (volume / 0.01)  # TP (positive)
    tp_33 = 0.33 * tp_profit  # **Trailing activation at 33% TP**
    
    profit_trailing_active = False  # Flag to track trailing

    print(f"📊 Monitoring Trade {trade_id} - SL: {sl_profit:.2f} USD, TP: {tp_profit:.2f} USD")

    while True:
        order_info = next((pos for pos in mt5.positions_get() if pos.ticket == trade_id), None)
        if not order_info:
            log_trade("Trade disappeared", trade_id, 0)
            print(f"⚠️ Trade {trade_id} no longer exists. Logging and stopping monitoring.")
            with lock:
                monitored_trades.discard(trade_id)
            break

        profit = order_info.profit

        # Stop Loss hit
        if profit <= sl_profit:
            log_trade("Stopped Out", trade_id, profit)
            print(f"⛔ Trade {trade_id} hit stop loss ({profit:.2f} USD). Closing...")
            close_trade(trade_id, symbol, volume, order_type, profit)
            break

        # Take Profit hit
        if profit >= tp_profit:
            log_trade("Take Profit Hit", trade_id, profit)
            print(f"✅ Trade {trade_id} hit take profit ({profit:.2f} USD). Closing...")
            close_trade(trade_id, symbol, volume, order_type, profit)
            break

        # **Enable Trailing Profit When TP 33% Reached**
        if profit >= tp_33:
            profit_trailing_active = True

        # **Safe Exit - If trade falls back after reaching TP 33% and drops to 0.03**
        if profit_trailing_active and profit <= 0.03:
            log_trade("Safe Exit", trade_id, profit)
            print(f"⚠️ Trade {trade_id} fell back to $0.03 after crossing 33% TP. Closing trade to secure capital.")
            close_trade(trade_id, symbol, volume, order_type, profit)
            break

        time.sleep(1)

# Function to continuously monitor trades
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

# Start monitoring
monitor_all_trades()
