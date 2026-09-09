import MetaTrader5 as mt5
import time

def close_all_trades():
    """Closes all open trades immediately."""
    orders = mt5.positions_get()
    if not orders:
        print("✅ No trades to close.")
        return

    for order in orders:
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": order.ticket,
            "symbol": order.symbol,
            "volume": order.volume,
            "type": mt5.ORDER_TYPE_SELL if order.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "deviation": 10,
            "magic": 0,
            "type_filling": mt5.ORDER_FILLING_IOC
        }
        result = mt5.order_send(request)

        if result is None:
            print(f"❌ Order failed: {mt5.last_error()}")
        elif result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"❌ Failed to close trade {order.ticket} (Error Code: {result.retcode})")
        else:
            print(f"✅ Trade {order.ticket} closed successfully.")

def get_total_lots_by_type(order_type):
    """Calculates total lots separately for buy and sell orders."""
    orders = mt5.positions_get()
    if not orders:
        return 0
    return sum(order.volume for order in orders if order.type == order_type)

def monitor_capital():
    """Monitors capital and closes trades if profit/loss breaches separate thresholds for buy and sell orders."""
    if not mt5.initialize():
        print("❌ Failed to initialize MT5.")
        return
    
    try:
        while True:
            account_info = mt5.account_info()
            if account_info is None:
                print("❌ Failed to retrieve account info.")
                break
            
            total_lots_buy = get_total_lots_by_type(mt5.ORDER_TYPE_BUY)
            total_lots_sell = get_total_lots_by_type(mt5.ORDER_TYPE_SELL)
            
            if total_lots_buy == 0 and total_lots_sell == 0:
                print("🔹 No trades open. Skipping threshold check.")
                time.sleep(0.1)
                continue

            if total_lots_buy > 0:
                threshold_buy = total_lots_buy * 400
                total_profit_buy = sum(order.profit for order in mt5.positions_get() if order.type == mt5.ORDER_TYPE_BUY)
                print(f"🔹 Buy P/L: {total_profit_buy} | Buy Threshold: {threshold_buy}")
                if abs(total_profit_buy) >= threshold_buy:
                    print("⚠️ Buy threshold breached! Closing all buy trades...")
                    close_all_trades()

            if total_lots_sell > 0:
                threshold_sell = total_lots_sell * 240
                total_profit_sell = sum(order.profit for order in mt5.positions_get() if order.type == mt5.ORDER_TYPE_SELL)
                print(f"🔹 Sell P/L: {total_profit_sell} | Sell Threshold: {threshold_sell}")
                if abs(total_profit_sell) >= threshold_sell:
                    print("⚠️ Sell threshold breached! Closing all sell trades...")
                    close_all_trades()
            
            time.sleep(0.1)
    
    finally:
        mt5.shutdown()
        print("🔻 MT5 connection closed.")

# **Example Usage**
monitor_capital()
