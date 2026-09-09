import MetaTrader5 as mt5
import pandas as pd
import numpy as np

# Connect to MT5
if not mt5.initialize():
    print("Failed to initialize MT5")
    quit()

# Parameters
symbol = "XAUUSDm"
timeframes = {"30min": mt5.TIMEFRAME_M30, "1H": mt5.TIMEFRAME_H1}
candles_count = 10000  # Number of candles to fetch

def fetch_ohlcv(symbol, timeframe, count):
    """Fetch OHLCV data from MT5."""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None:
        print(f"Failed to fetch data for {symbol} on timeframe {timeframe}")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def prepare_dataset(df):
    """Prepare dataset for training a time regression model or neural network."""
    df['Next_Open'] = df['open'].shift(-1)
    df['Next_Close'] = df['close'].shift(-1)
    df['Target'] = np.where(df['Next_Close'] > df['Next_Open'], 1, 0)  # 1 if next close > next open, else 0
    df.drop(columns=['Next_Open', 'Next_Close'], inplace=True)  # Remove unnecessary columns
    return df.dropna()

# Fetch and process data
ohlcv_data = {}
for tf_name, tf in timeframes.items():
    df = fetch_ohlcv(symbol, tf, candles_count)
    if df is not None:
        df = prepare_dataset(df)
        ohlcv_data[tf_name] = df

# Save to CSV
for tf_name, df in ohlcv_data.items():
    df.to_csv(f"XAUUSDm_{tf_name}_dataset.csv", index=False)
    print(f"Saved: XAUUSDm_{tf_name}_dataset.csv")

# Shutdown MT5
mt5.shutdown()
