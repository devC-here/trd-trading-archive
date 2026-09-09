import MetaTrader5 as mt5
import pandas as pd
import numpy as np

# ==============================
# CONFIG
# ==============================

SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M15
LOOKBACK = 10000

INERTIA = 0.6
RESPONSE = 0.4

ATR_LEN = 14
Z_THRESHOLD = 5.0

# ==============================
# INIT MT5
# ==============================

if not mt5.initialize():
    raise RuntimeError("MT5 init failed")

rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, LOOKBACK)

mt5.shutdown()

df = pd.DataFrame(rates)
df["time"] = pd.to_datetime(df["time"], unit="s")


# ==============================
# BARYCENTER
# ==============================

df["B"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4


# ==============================
# MEAN WAVE
# ==============================

mean_wave = np.zeros(len(df))
mean_wave[0] = df["B"].iloc[0]

for i in range(1, len(df)):
    mean_wave[i] = (
        mean_wave[i-1] * INERTIA +
        df["B"].iloc[i] * RESPONSE
    )

df["mean"] = mean_wave


# ==============================
# ATR
# ==============================

df["tr"] = np.maximum(
    df["high"] - df["low"],
    np.maximum(
        abs(df["high"] - df["close"].shift()),
        abs(df["low"] - df["close"].shift())
    )
)

df["atr"] = df["tr"].rolling(ATR_LEN).mean()


# ==============================
# Z SCORE
# ==============================

df["z"] = (df["close"] - df["mean"]) / df["atr"]


# ==============================
# TEST REVERSAL
# ==============================

reversals = []
tests = 0

for i in range(ATR_LEN, len(df) - 2):

    z = df["z"].iloc[i]

    if abs(z) > Z_THRESHOLD:

        tests += 1

        future_move = df["close"].iloc[i+2] - df["close"].iloc[i]

        # check snapback
        if z > 0 and future_move < 0:
            reversals.append(1)

        elif z < 0 and future_move > 0:
            reversals.append(1)

        else:
            reversals.append(0)


# ==============================
# RESULTS
# ==============================

if tests > 0:
    rate = sum(reversals) / tests
else:
    rate = 0

print("\n===== Z-SCORE SNAPBACK TEST =====")
print("Tests:", tests)
print("Reversal rate:", rate)