import MetaTrader5 as mt5
import pandas as pd
import numpy as np

# ==============================
# CONFIG
# ==============================

SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M15
LOOKBACK = 20000

INERTIA = 0.6
RESPONSE = 0.4


# ==============================
# LOAD DATA
# ==============================

if not mt5.initialize():
    raise RuntimeError("MT5 init failed")

mt5.symbol_select(SYMBOL, True)

rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, LOOKBACK)

mt5.shutdown()

df = pd.DataFrame(rates)
df["time"] = pd.to_datetime(df["time"], unit="s")


# ==============================
# EXTRACT PRICES
# ==============================

close = df["close"].values
open_  = df["open"].values
high   = df["high"].values
low    = df["low"].values


# ==============================
# BARYCENTER
# ==============================

B = (open_ + high + low + close) / 4


# ==============================
# MEAN WAVE
# ==============================

mean_wave = np.zeros(len(B))
mean_wave[0] = B[0]

for i in range(1,len(B)):
    mean_wave[i] = mean_wave[i-1]*INERTIA + B[i]*RESPONSE


# ==============================
# FEATURES
# ==============================

deviation = close - mean_wave

velocity = np.diff(close, prepend=close[0])

stretch = np.abs(deviation) * np.abs(velocity)


# ==============================
# NEXT RETURN
# ==============================

next_return = np.diff(close, append=close[-1])


# ==============================
# REVERSAL TEST
# ==============================

reversal = np.sign(velocity) != np.sign(next_return)

data = pd.DataFrame({
    "stretch": stretch,
    "reversal": reversal
})


# ==============================
# QUANTILE ANALYSIS
# ==============================

data["quantile"] = pd.qcut(data["stretch"],5,labels=False)

results = data.groupby("quantile")["reversal"].mean()

print("\n===== RUBBER BAND TEST =====\n")

print("Reversal probability by stretch level:\n")
print(results)