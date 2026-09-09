import MetaTrader5 as mt5
import pandas as pd

# Assets to test
assets = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "XAUUSD", "US500", "USTEC", "GER40", "BTCUSD"]
timeframes = {
    "M5": mt5.TIMEFRAME_M5, "M10": mt5.TIMEFRAME_M10, "M15": mt5.TIMEFRAME_M15, 
    "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H2": mt5.TIMEFRAME_H2
}

def get_clean_rate(symbol, tf_name, tf_val, count=8000):
    rates = mt5.copy_rates_from_pos(symbol, tf_val, 0, count)
    if rates is None or len(rates) == 0: return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Restricted time logic
    def is_restricted(dt):
        m = dt.hour * 60 + dt.minute
        for s in [90, 330, 810, 1110]:
            if s <= m < s + 90: return True
        return False
        
    df = df[~df['time'].apply(is_restricted)].copy()
    if len(df) < 100: return None # Ensure enough data

    df['body'] = abs(df['open'] - df['close'])
    df['total_wick'] = (df['high'] - df['low']) - df['body']
    
    # Success: Body is at least 1.25x the total wick size
    rate = (df['body'] >= (df['total_wick'] * 1.25)).mean() * 100
    return {"Asset": symbol, "TF": tf_name, "Clean_Rate": round(rate, 2), "Count": len(df)}

if not mt5.initialize():
    quit()

results_list = []

for asset in assets:
    for name, val in timeframes.items():
        res = get_clean_rate(asset, name, val)
        if res:
            results_list.append(res)

# Convert to DataFrame and SORT by Clean_Rate descending
master_df = pd.DataFrame(results_list)
master_df = master_df.sort_values(by="Clean_Rate", ascending=False)

print("\n--- TOP CONTENDERS FOR YOUR INDICATOR (Sorted by Efficiency) ---")
print(master_df.to_string(index=False))

mt5.shutdown()