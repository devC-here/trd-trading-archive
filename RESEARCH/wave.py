import MetaTrader5 as mt5
import pandas as pd
import numpy as np

# ==============================
# CONFIG
# ==============================

SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M15
LOOKBACK = 2000

INERTIA = 0.6
RESPONSE = 0.4


# ==============================
# RESTRICTED PERIODS
# ==============================

def is_restricted_time(ts):

    minutes = ts.hour*60 + ts.minute

    asian_open = 0*60
    london_open = 8*60
    ny_open = 13*60

    if asian_open <= minutes < asian_open+60:
        return True

    if london_open <= minutes < london_open+60:
        return True

    if ny_open <= minutes < ny_open+60:
        return True

    return False


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
# FIND ONE CYCLE
# ==============================

segment = []
started = False

for i in range(len(df)):

    row = df.iloc[i]

    if is_restricted_time(row.time):

        if started:
            break
        else:
            continue

    started = True
    segment.append(row)

seg_df = pd.DataFrame(segment)


# ==============================
# EXTRACT WAVES
# ==============================

open_wave  = seg_df["open"].values
high_wave  = seg_df["high"].values
low_wave   = seg_df["low"].values
close_wave = seg_df["close"].values


# ==============================
# BARYCENTER (CENTER OF GRAVITY)
# ==============================

B = (open_wave + high_wave + low_wave + close_wave) / 4


# ==============================
# MEAN WAVE (INERTIA FILTER)
# ==============================

mean_wave = np.zeros(len(B))
mean_wave[0] = B[0]

for i in range(1, len(B)):
    mean_wave[i] = (
        mean_wave[i-1] * INERTIA +
        B[i] * RESPONSE
    )


# ==============================
# OUTPUT DATAFRAME
# ==============================

wave_df = pd.DataFrame({
    "open_wave": open_wave,
    "high_wave": high_wave,
    "low_wave": low_wave,
    "close_wave": close_wave,
    "mean_wave": mean_wave
})


print("\nSEGMENT START:", seg_df.iloc[0]["time"])
print("SEGMENT END:", seg_df.iloc[-1]["time"])

print("\n===== FIVE WAVES =====\n")
print(wave_df)

print("\nNumber of candles in cycle:", len(wave_df))