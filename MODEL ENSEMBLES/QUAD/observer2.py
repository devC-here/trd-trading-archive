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

TP_STATIC = 1.5
SL_STATIC = 0.75

def log_trade(event, trade_id, profit):
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), event, trade_id, profit])


def choose_timeframe():
    now = datetime.datetime.now()
    minute = now.minute
    if abs(minute - 0) <= 5 or abs(minute - 60) <= 5:
        return mt5.TIMEFRAME_M30
    else:
        return mt5.TIMEFRAME_H1


def calculate_dynamic_sl_tp(symbol, order_type, entry_price):
    timeframe = choose_timeframe()

    N = 7  # Skip 0th candle, use pairs from (2,3) to (8,9)
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, N + 3)
    if rates is None or len(rates) < N + 3:
        print("⚠️ Insufficient candle data.")
        return None, None

    df = pd.DataFrame(rates)
    max_components = []
    min_components = []
    sizes = []

    for i in range(2, N + 2):
        c1 = df.iloc[i - 1]  # First candle in pair
        c2 = df.iloc[i]      # Second candle in pair

        # Anatomy from second candle
        body = abs(c2['close'] - c2['open'])
        wick = c2['high'] - max(c2['open'], c2['close'])
        tail = min(c2['open'], c2['close']) - c2['low']
        size = max(body, wick, tail)

        if size < 1e-5:
            continue

        up = (c1['high'] - c1['open']) / size
        down = (c1['open'] - c1['low']) / size

        max_components.append(max(up, down))
        min_components.append(min(up, down))
        sizes.append(size)

    if len(max_components) == 0 or len(min_components) == 0:
        print("⚠️ No valid component data.")
        return None, None

    tp_dynamic = pd.Series(max_components).quantile(0.75)
    sl_dynamic = pd.Series(min_components).quantile(0.75)

    tp = 0.6 * TP_STATIC + 0.4 * tp_dynamic
    sl = 0.6 * SL_STATIC + 0.4 * sl_dynamic

    tp = max(min(tp, TP_STATIC * 1.2), TP_STATIC * 0.8)
    sl = max(min(sl, SL_STATIC * 1.2), SL_STATIC * 0.8)

    latest_size = sizes[-1]

    if order_type == mt5.ORDER_TYPE_BUY:
        sl_price = entry_price - sl * latest_size
        tp_price = entry_price + tp * latest_size
    else:
        sl_price = entry_price + sl * latest_size
        tp_price = entry_price - tp * latest_size

    print(f"📊 Components — TP Mult: {tp:.3f}, SL Mult: {sl:.3f}, Size: {latest_size:.5f}")
    print(f"✅ SL/TP: SL={sl_price:.3f}, TP={tp_price:.3f}")

    return round(sl_price, 3), round(tp_price, 3)


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
        print(f"✅ SL/TP updated for trade {position_id}: SL={sl:.3f}, TP={tp:.3f}")
        return True
    else:
        print(f"❌ Failed to update SL/TP for trade {position_id} (Error Code: {result.retcode})")
        return False


def monitor_trade(order, monitored_trades, lock):
    trade_id = order.ticket
    symbol = order.symbol
    order_type = order.type
    open_price = order.price_open

    sl, tp = calculate_dynamic_sl_tp(symbol, order_type, open_price)
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
