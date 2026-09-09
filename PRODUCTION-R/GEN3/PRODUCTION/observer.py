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
            "position": order.ticket,  # Closing specific trade
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

def get_total_lots():
    """Calculates total lots of open trades."""
    orders = mt5.positions_get()
    return sum(order.volume for order in orders) if orders else 0  # Total lot size

def monitor_capital():
    """Monitors capital and terminates trades if profit/loss breaches (lots * 110)."""
    if not mt5.initialize():
        print("❌ Failed to initialize MT5.")
        return
    
    try:
        while True:
            account_info = mt5.account_info()
            if account_info is None:
                print("❌ Failed to retrieve account info.")
                break
            
            total_lots = get_total_lots()

            # **Avoid threshold breach checks when no trades are open**
            if total_lots == 0:
                print("🔹 No trades open. Skipping threshold check.")
                time.sleep(1)
                continue

            threshold = total_lots * 160  # Profit/Loss threshold

            # Get floating profit/loss
            total_profit = sum(order.profit for order in mt5.positions_get())

            print(f"🔹 Current Floating P/L: {total_profit} | Threshold: {threshold}")

            # Check if profit/loss exceeds threshold
            if abs(total_profit) >= threshold:
                print("⚠️ Threshold breached! Closing all trades...")
                close_all_trades()
            
            time.sleep(1)  # **Strict 1-second interval**
    
    finally:
        mt5.shutdown()
        print("🔻 MT5 connection closed.")

# **Example Usage**
monitor_capital()
