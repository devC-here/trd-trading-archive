import MetaTrader5 as mt5
import pandas as pd

# Initialize MT5
if not mt5.initialize():
    print("❌ Failed to initialize MT5")
    exit()

# Load 5001 candles to get 5000 adjacent pairs
symbol = "XAUUSDm"
timeframe = mt5.TIMEFRAME_M30
candles_needed = 5001

rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, candles_needed)

if rates is None or len(rates) < 2:
    print("❌ Not enough candle data retrieved.")
    mt5.shutdown()
    exit()

df = pd.DataFrame(rates)

results = []

for i in range(len(df) - 1):
    c1 = df.iloc[i]
    open_price = c1["open"]
    high_price = c1["high"]
    low_price = c1["low"]
    close_price = c1["close"]

    body = abs(close_price - open_price)
    wick = high_price - max(open_price, close_price)
    tail = min(open_price, close_price) - low_price

    size = max(body, wick, tail)

    if size < 1e-5:
        continue  # Skip near-zero candles

    up = (high_price - open_price) / size
    down = (open_price - low_price) / size

    max_component = max(up, down)
    min_component = min(up, down)

    results.append({
        "candle_index": i,
        "max_component": round(max_component, 5),
        "min_component": round(min_component, 5),
        "size": round(size, 5)
    })

results_df = pd.DataFrame(results)

# Save to CSV
csv_filename = "xauusd_candle_components.csv"
results_df.to_csv(csv_filename, index=False)

print(results_df.head())
print(f"\n✅ Data saved to {csv_filename} with {len(results_df)} records.")

mt5.shutdown()
