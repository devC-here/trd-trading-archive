import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta

# 1. Connect to MetaTrader 5
if not mt5.initialize():
    print("initialize() failed, error code =", mt5.last_error())
    quit()

# 2. Define the timeframe for history (e.g., last 30 days)
from_date = datetime.now() - timedelta(days=80)
to_date = datetime.now()

# 3. Pull all "deals" (actual executions)
deals = mt5.history_deals_get(from_date, to_date)

if deals is None or len(deals) == 0:
    print("No deals found in history.")
else:
    # Convert to a DataFrame
    df = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
    
    # Filter for only closed trades (Entries/Exits)
    # entry 0 = In, 1 = Out
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # CALCULATE COSTS
    total_commission = df['commission'].sum()
    total_swap = df['swap'].sum()
    total_profit = df['profit'].sum() # This is NET profit after spread
    
    # Display the results
    print("-" * 30)
    print(f"REPORT FOR THE LAST 30 DAYS")
    print("-" * 30)
    print(f"Total Trades: {len(df)}")
    print(f"Actual Commissions Paid: ${total_commission:.2f}")
    print(f"Actual Swaps Paid:       ${total_swap:.2f}")
    print(f"Net Profit (After All):  ${total_profit:.2f}")
    print("-" * 30)

    # Individual Deal Breakdown (First 5)
    print("Recent Deal Breakdown:")
    print(df[['time', 'symbol', 'volume', 'profit', 'commission']].tail())

# Always shut down the connection
mt5.shutdown()