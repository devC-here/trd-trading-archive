# sltp.py
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

# ==============================
# CONFIG
# ==============================
REF_CANDLES = 6
EPS = 1e-8

# ==============================
# RESTRICTED TIME WINDOWS
# ==============================
def is_restricted_time(ts):
    h, m = ts.hour, ts.minute

    if 0 <= h <= 1:
        return True
    if h == 12 and m >= 30:
        return True
    if h == 13:
        return True
    if 18 <= h <= 19:
        return True

    return False

# ==============================
# CORE UTILS
# ==============================
def candle_size(o, c):
    return abs(o - c)

def variation_weighted_avg(sizes):
    mean = np.mean(sizes)
    weights = [1.0 / (abs(s - mean) + EPS) for s in sizes]
    return float(np.sum(np.array(sizes) * np.array(weights)) / np.sum(weights))

# ==============================
# PUBLIC API
# ==============================
def predict_size(symbol: str, timeframe, lookback: int = 100) -> float:
    """
    Returns:
        float -> variation-weighted average candle size
    """

    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")

    if not mt5.symbol_select(symbol, True):
        mt5.shutdown()
        raise RuntimeError(f"Symbol not found: {symbol}")

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, lookback)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise RuntimeError("No market data")

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # ---- collect last valid candle sizes ----
    sizes = []

    for i in range(len(df) - 1, -1, -1):
        row = df.iloc[i]
        if is_restricted_time(row.time):
            continue

        sizes.append(candle_size(row.open, row.close))

        if len(sizes) == REF_CANDLES:
            break

    if len(sizes) < REF_CANDLES:
        raise RuntimeError("Not enough valid candles")

    return variation_weighted_avg(sizes)
