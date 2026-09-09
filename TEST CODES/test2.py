import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

# ---------- CONFIG ----------
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M15   # 5-minute candles
N_CANDLES = 100                # last N candles
RANGE_THRESHOLD = 0.00030       # example: 5 pips for EURUSD
# ----------------------------

def get_ohlcv_and_compare(symbol, timeframe, n_candles, range_threshold):
    # Initialize MT5
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    # Ensure symbol is available
    if not mt5.symbol_select(symbol, True):
        mt5.shutdown()
        raise RuntimeError(f"Failed to select symbol {symbol}")

    # Fetch last N candles
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_candles)

    mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise RuntimeError("No OHLCV data retrieved")

    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # Candle length = High - Low
    df['range'] = df['high'] - df['low']

    # Count candles where range > threshold
    count_greater = (df['range'] > range_threshold).sum()

    return count_greater, df


if __name__ == "__main__":
    count, candles_df = get_ohlcv_and_compare(
        SYMBOL,
        TIMEFRAME,
        N_CANDLES,
        RANGE_THRESHOLD
    )

    print(f"Symbol        : {SYMBOL}")
    print(f"Timeframe     : {TIMEFRAME}")
    print(f"Total candles : {N_CANDLES}")
    print(f"Threshold     : {RANGE_THRESHOLD}")
    print(f"Candles > threshold : {count}")
