import pandas as pd
import numpy as np
import random

N_SAMPLES = 10000
SEQ_LEN = 5
OUTPUT_PATH = "synthetic_candle_sequence_dataset.csv"

def generate_candle(base_price, direction="sideways", volatility=0.5):
    """Simulate a single OHLCV candle"""
    body_size = np.random.uniform(0.05, 0.3) * volatility
    wick_size = np.random.uniform(0.05, 0.2) * volatility

    if direction == "bull":
        open_price = base_price - body_size / 2
        close_price = base_price + body_size / 2
    elif direction == "bear":
        open_price = base_price + body_size / 2
        close_price = base_price - body_size / 2
    else:
        # Random order for doji / weak candle
        if np.random.rand() > 0.5:
            open_price = base_price - body_size / 2
            close_price = base_price + body_size / 2
        else:
            open_price = base_price + body_size / 2
            close_price = base_price - body_size / 2

    high = max(open_price, close_price) + wick_size
    low = min(open_price, close_price) - wick_size
    volume = np.random.randint(1000, 10000)

    return {
        "open": open_price,
        "high": high,
        "low": low,
        "close": close_price,
        "volume": volume
    }

def label_sequence(seq_df):
    """Label pattern type, dominance, strength"""
    o = seq_df["open"].values
    c = seq_df["close"].values
    bodies = np.abs(c - o)

    net_move = c[-1] - o[0]
    direction = np.sign(net_move)

    avg_body = np.mean(bodies)
    volatility = np.std(seq_df["high"] - seq_df["low"])

    # Reversal detection
    if direction > 0 and c[0] > o[0] and c[-1] > c[0]:
        seq_type = 1  # Bullish reversal
    elif direction < 0 and c[0] < o[0] and c[-1] < c[0]:
        seq_type = 2  # Bearish reversal
    elif direction > 0:
        seq_type = 3  # Bullish continuation
    elif direction < 0:
        seq_type = 4  # Bearish continuation
    else:
        seq_type = 0  # Neutral

    dominance = 2 if direction > 0 else 0 if direction < 0 else 1

    if avg_body > 0.2 and volatility > 0.3:
        strength = 2
    elif avg_body > 0.1:
        strength = 1
    else:
        strength = 0

    return seq_type, dominance, strength

def generate_sequence(base_price=100.0):
    """Generate a sequence of 5 candles and label it"""
    direction = random.choices(
        ["bull", "bear", "sideways"], weights=[0.35, 0.35, 0.3]
    )[0]

    sequence = []
    prices = [base_price]
    for _ in range(SEQ_LEN):
        next_candle = generate_candle(prices[-1], direction)
        sequence.append(next_candle)
        prices.append(next_candle["close"])

    seq_df = pd.DataFrame(sequence)
    seq_type, dominance, strength = label_sequence(seq_df)

    # Feature engineering
    seq_df["range"] = seq_df["high"] - seq_df["low"]
    seq_df["body"] = abs(seq_df["close"] - seq_df["open"])
    seq_df["wick"] = seq_df["high"] - seq_df[["open", "close"]].max(axis=1)
    seq_df["tail"] = seq_df[["open", "close"]].min(axis=1) - seq_df["low"]
    seq_df["body_dir"] = np.sign(seq_df["close"] - seq_df["open"])
    seq_df["body_ratio"] = seq_df["body"] / (seq_df["range"] + 1e-6)
    seq_df["tail_ratio"] = seq_df["tail"] / (seq_df["range"] + 1e-6)
    seq_df["wick_ratio"] = seq_df["wick"] / (seq_df["range"] + 1e-6)

    # Flatten sequence to one row
    flat_row = {}
    for i in range(SEQ_LEN):
        for col in ["body", "wick", "tail", "body_ratio", "tail_ratio", "wick_ratio", "body_dir"]:
            flat_row[f"{col}_t-{i}"] = seq_df.iloc[i][col]

    flat_row["sequence_type"] = seq_type
    flat_row["candle_dominance"] = dominance
    flat_row["pattern_strength"] = strength

    return flat_row

# --- Main Generation ---
if __name__ == "__main__":
    print(f"Generating {N_SAMPLES} candlestick sequences...")
    data = [generate_sequence() for _ in range(N_SAMPLES)]
    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH} — shape: {df.shape}")
    print(df.head())
