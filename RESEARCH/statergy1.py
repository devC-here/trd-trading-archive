import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime

SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M5
BARS = 5000


# -------------------------
# CONNECT MT5
# -------------------------

if not mt5.initialize():
    print("MT5 initialization failed")
    quit()

print("MT5 connected")


# -------------------------
# GET DATA
# -------------------------

rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, BARS)

df = pd.DataFrame(rates)

df['time'] = pd.to_datetime(df['time'], unit='s')


# -------------------------
# INDICATORS
# -------------------------

df['ma9'] = df['close'].rolling(9).mean()
df['ma15'] = df['close'].rolling(15).mean()


# -------------------------
# CANDLE FEATURES
# -------------------------

df['body'] = abs(df['close'] - df['open'])
df['range'] = df['high'] - df['low']

df['upper_wick'] = df['high'] - df[['close','open']].max(axis=1)
df['lower_wick'] = df[['close','open']].min(axis=1) - df['low']


# -------------------------
# PATTERN DETECTION
# -------------------------

hammer = (df['lower_wick'] > df['body'] * 2) & (df['upper_wick'] < df['body'])

bull_marubozu = (
    (df['body'] / df['range'] > 0.8) &
    (df['close'] > df['open'])
)

bear_marubozu = (
    (df['body'] / df['range'] > 0.8) &
    (df['close'] < df['open'])
)

big_bull = (
    (df['close'] > df['open']) &
    (df['body'] > df['range'] * 0.6)
)

big_bear = (
    (df['close'] < df['open']) &
    (df['body'] > df['range'] * 0.6)
)

df['bull_signal'] = hammer | bull_marubozu | big_bull
df['bear_signal'] = bear_marubozu | big_bear


# -------------------------
# BACKTEST LOOP
# -------------------------

trades = []

for i in range(20, len(df)):

    row = df.iloc[i]

    # BUY CONDITION
    if row.ma9 > row.ma15 and row.bull_signal:

        entry = row.high
        sl = row.low

        trades.append({
            "time": row.time,
            "type": "BUY",
            "entry": entry,
            "sl": sl,
            "ma9": row.ma9,
            "ma15": row.ma15
        })

        print(f"BUY signal {row.time} entry={entry} SL={sl}")


    # SELL CONDITION
    elif row.ma9 < row.ma15 and row.bear_signal:

        entry = row.low
        sl = row.high

        trades.append({
            "time": row.time,
            "type": "SELL",
            "entry": entry,
            "sl": sl,
            "ma9": row.ma9,
            "ma15": row.ma15
        })

        print(f"SELL signal {row.time} entry={entry} SL={sl}")


# -------------------------
# RESULTS
# -------------------------

trades_df = pd.DataFrame(trades)

print("\nTotal signals:", len(trades_df))

print(trades_df.head())

trades_df.to_csv("strategy_integrity_test.csv", index=False)


mt5.shutdown()