import pandas as pd
import numpy as np
from tqdm import tqdm

def modify_ohlcv(datafile, outputfile, ohlc_variation=0.0001):
    # Load dataset
    df = pd.read_csv(datafile)
    
    # Normalize column names and convert to lowercase
    df.columns = df.columns.str.lower()
    
    # Ensure required OHLCV columns exist
    required_columns = ['open', 'high', 'low', 'close', 'volume', 'trade_decision']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("Dataset must contain OHLCV and trade_decision columns.")
    
    # Introduce OHLC variations for 'open' and 'close'
    for col in tqdm(['open', 'close'], desc="Modifying OHLC"):
        variation = df[col] * np.random.uniform(-ohlc_variation, ohlc_variation, size=len(df))
        df[col] = (df[col] + variation).round(4)  # Round to 4 decimal places
    
    # Ensure logical candlestick structure for high and low
    df['high'] = np.maximum(df[['open', 'close']].max(axis=1) + np.abs(df['high'] * 0.0002), df['high'])
    df['low'] = np.minimum(df[['open', 'close']].min(axis=1) - np.abs(df['low'] * 0.0002), df['low'])
    
    # Safe division function
    def safe_divide(a, b, eps=1e-6):
        return np.round(np.where(np.abs(b) > eps, a / b, 0), 3)  # Round to 3 decimal places

    # Compute ratio features
    df['high_low_high_close_ratio'] = safe_divide(df['high'] - df['low'], df['high'] - df['close'])  # HLHC
    df['high_close_low_close_ratio'] = safe_divide(df['high'] - df['close'], df['low'] - df['close'])  # HCLC
    df['open_low_high_close_ratio'] = safe_divide(df['open'] - df['low'], df['high'] - df['close'])  # OLHC
    df['high_low_low_close_ratio'] = safe_divide(df['high'] - df['low'], df['low'] - df['close'])  # HLLC
    df['open_close_low_close_ratio'] = safe_divide(df['open'] - df['close'], df['low'] - df['close'])  # OCLC
    
    # Compute additional candlestick shape features
    df['wick_to_length_ratio'] = safe_divide(df['high'] - df[['open', 'close']].max(axis=1), df['high'] - df['low'])
    df['body_to_length_ratio'] = safe_divide(np.abs(df['close'] - df['open']), df['high'] - df['low'])
    df['tail_to_length_ratio'] = safe_divide(df[['open', 'close']].min(axis=1) - df['low'], df['high'] - df['low'])
    
    # Ensure volume is kept (redundant assignment but for clarity)
    df['volume'] = df['volume']
    
    # Keep only the selected features (order matters for later processing)
    df = df[['open', 'close', 
             'high_low_high_close_ratio', 'high_close_low_close_ratio', 'open_low_high_close_ratio',
             'high_low_low_close_ratio', 'open_close_low_close_ratio', 
             'wick_to_length_ratio', 'body_to_length_ratio', 'tail_to_length_ratio', 'volume',
             'trade_decision']]
    
    # Shuffle the dataset to introduce randomness
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Save the modified (and shuffled) dataset
    df.to_csv(outputfile, index=False)
    print(f"Modified and shuffled dataset saved as {outputfile}")

# Run the function
modify_ohlcv("trade_decisions1.csv", "trade_decisions2.csv")
