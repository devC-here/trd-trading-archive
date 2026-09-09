from datetime import datetime, timedelta, UTC
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

# ==============================
# CONFIG
# ==============================

SYMBOL = "EURUSD"
HOURS_BACK = 4
FFT_WINDOW = 4096

# ==============================
# INIT MT5
# ==============================

if not mt5.initialize():
    raise RuntimeError("MT5 initialization failed")

if not mt5.symbol_select(SYMBOL, True):
    raise RuntimeError("Symbol not found")

# ==============================
# GET LAST TICK (server time)
# ==============================

latest = mt5.symbol_info_tick(SYMBOL)

if latest is None:
    raise RuntimeError("Cannot get latest tick")

server_time = datetime.fromtimestamp(latest.time, tz=UTC)

to_time = server_time
from_time = server_time - timedelta(hours=HOURS_BACK)

# ==============================
# DOWNLOAD TICKS
# ==============================

ticks = mt5.copy_ticks_range(
    SYMBOL,
    from_time,
    to_time,
    mt5.COPY_TICKS_ALL
)

mt5.shutdown()

if ticks is None or len(ticks) == 0:
    raise RuntimeError("No tick data received")

print("Ticks received:", len(ticks))

# ==============================
# DATAFRAME
# ==============================

df = pd.DataFrame(ticks)
df["time"] = pd.to_datetime(df["time"], unit="s")

# ==============================
# BUILD WAVES
# ==============================

df["mid"] = (df["bid"] + df["ask"]) / 2
df["spread"] = df["ask"] - df["bid"]

waves = {
    "bid_wave": df["bid"].values,
    "ask_wave": df["ask"].values,
    "mid_wave": df["mid"].values,
    "spread_wave": df["spread"].values
}

# ==============================
# FFT ANALYSIS
# ==============================

def analyze_wave(name, series):

    wave = np.diff(series)

    if len(wave) < FFT_WINDOW:
        return

    wave = wave - np.mean(wave)
    wave = wave[-FFT_WINDOW:]

    fft_vals = np.fft.fft(wave)
    freqs = np.fft.fftfreq(len(wave))
    magnitude = np.abs(fft_vals)

    idx = np.argsort(magnitude)[-6:]

    print("\n===== ", name, " =====")

    for i in idx:
        print("freq:", freqs[i], "mag:", magnitude[i])


# ==============================
# RUN ANALYSIS
# ==============================

for name, wave in waves.items():
    analyze_wave(name, wave)