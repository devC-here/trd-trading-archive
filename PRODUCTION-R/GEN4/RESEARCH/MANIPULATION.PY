import pandas as pd

# Load dataset
df = pd.read_csv("trade_decisions1.csv")

# Separate Sell signals
sell_samples = df[df['trade_decision'] == 'Sell'].copy()

# Function to flip Sell candles into Buy
def flip_sell_candle(row):
    open_price = row['open']
    close_price = row['close']
    high = row['high']
    low = row['low']
    
    # Calculate original wick and tail sizes
    wick_size = high - max(open_price, close_price)
    tail_size = min(open_price, close_price) - low
    
    # Swap wick and tail sizes
    row['high'] = max(open_price, close_price) + tail_size
    row['low'] = min(open_price, close_price) - wick_size
    
    # Change decision to Buy
    row['trade_decision'] = 'Buy'
    return row

# Apply transformation to duplicate Sell samples
flipped_buy_samples = sell_samples.apply(flip_sell_candle, axis=1)

# Combine original dataset with newly generated Buy samples
final_df = pd.concat([df, flipped_buy_samples])

# Save the modified dataset
final_df.to_csv("flipped_trade_decisions.csv", index=False)

print("Dataset saved as 'flipped_trade_decisions.csv'")
