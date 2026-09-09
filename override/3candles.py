import MetaTrader5 as mt5
from datetime import datetime, timedelta

def get_next_3_candles(symbol, timeframe, start_time):
    """
    Returns next 3 consecutive candles starting from start_time (inclusive).
    start_time MUST be a candle open time.
    """

    # Calculate safe forward range
    tf_minutes = {
        mt5.TIMEFRAME_M1: 1,
        mt5.TIMEFRAME_M5: 5,
        mt5.TIMEFRAME_M15: 15,
        mt5.TIMEFRAME_M30: 30,
        mt5.TIMEFRAME_H1: 60,
    }[timeframe]

    end_time = start_time + timedelta(minutes=tf_minutes * 3)

    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")

    rates = mt5.copy_rates_range(
        symbol,
        timeframe,
        start_time,
        end_time
    )

    mt5.shutdown()

    if rates is None or len(rates) < 3:
        raise RuntimeError("Not enough candles returned")

    # Keep exactly 3 candles
    return [
        {
            "time": datetime.fromtimestamp(r["time"]),
            "open": r["open"],
            "high": r["high"],
            "low": r["low"],
            "close": r["close"],
            "volume": r["tick_volume"]
        }
        for r in rates[:3]
    ]


# ------------------ USAGE ------------------
candles = get_next_3_candles(
    symbol="EURUSD",
    timeframe=mt5.TIMEFRAME_M15,
    start_time=datetime(2026, 1, 21, 8, 30)
)

for c in candles:
    print(c)
