import MetaTrader5 as mt5
import os

# --- SETTINGS ---
SYMBOL = "BTCUSD"
LOT_SIZE = 0.01
TSL_AMOUNT_USD = 100.0  # Trailing distance in USD
MAGIC_NUMBER = 123456

def run_combined_test():
    # 1. Initialize Connection to MT5
    if not mt5.initialize():
        print(f"❌ Failed to connect: {mt5.last_error()}")
        return

    print(f"--- Starting Test for {SYMBOL} ---")

    # 2. Prepare Symbol for Trading
    mt5.symbol_select(SYMBOL, True)
    tick = mt5.symbol_info_tick(SYMBOL)
    symbol_info = mt5.symbol_info(SYMBOL)
    
    if tick is None or symbol_info is None:
        print(f"❌ Could not get data for {SYMBOL}")
        mt5.shutdown()
        return

    # 3. Place the Buy Order
    print(f"Opening Buy Position at {tick.ask}...")
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": LOT_SIZE,
        "type": mt5.ORDER_TYPE_BUY,
        "price": tick.ask,
        "magic": MAGIC_NUMBER,
        "comment": "File-Bridge Test",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK, 
    }

    result = mt5.order_send(request)

    # 4. Handle Result and Create Instruction File
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        ticket = result.order
        print(f"✅ Trade Opened! Ticket: {ticket}")

        # Calculate point distance
        trail_points = round(float(TSL_AMOUNT_USD) / symbol_info.point)
        
        # FIND THE MT5 FILES FOLDER AUTOMATICALLY
        # terminal_info().data_path gives the root folder of your MT5 terminal
        terminal_data = mt5.terminal_info()
        if terminal_data is None:
            print("❌ Could not find terminal data path.")
            return

        # Path to MQL5/Files folder
        mql5_files_path = os.path.join(terminal_data.data_path, "MQL5", "Files")
        file_name = f"TSL_{ticket}.txt"
        full_file_path = os.path.join(mql5_files_path, file_name)

        # Write the trail points to the file
        try:
            with open(full_file_path, "w") as f:
                f.write(str(trail_points))
            print(f"🚀 Created trigger file: {file_name}")
            print(f"📍 Location: {full_file_path}")
            print(f"📉 Trail points: {trail_points}")
        except Exception as e:
            print(f"❌ Error writing file: {e}")
    
    else:
        print(f"❌ Trade Placement Failed: {result.comment}")

    mt5.shutdown()

if __name__ == "__main__":
    run_combined_test()