import MetaTrader5 as mt5
import pandas as pd
import numpy as np

# ==============================
# CONFIG
# ==============================

SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M15
LOOKBACK = 3000

INERTIA = 0.6
RESPONSE = 0.4
MA_LEN = 10


# ==============================
# RESTRICTED TIMES
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

mt5.initialize()
mt5.symbol_select(SYMBOL, True)

rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, LOOKBACK)

mt5.shutdown()

df = pd.DataFrame(rates)
df["time"] = pd.to_datetime(df["time"], unit="s")


# ==============================
# EXTRACT ONE CYCLE
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
# WAVES
# ==============================

open_wave  = seg_df["open"].values
high_wave  = seg_df["high"].values
low_wave   = seg_df["low"].values
close_wave = seg_df["close"].values


# ==============================
# BARYCENTER
# ==============================

B = (open_wave + high_wave + low_wave + close_wave) / 4


# ==============================
# MEAN WAVE (YOUR MODEL)
# ==============================

mean_wave = np.zeros(len(B))
mean_wave[0] = B[0]

for i in range(1, len(B)):
    mean_wave[i] = mean_wave[i-1]*INERTIA + B[i]*RESPONSE


# ==============================
# SMA
# ==============================

sma = pd.Series(close_wave).rolling(MA_LEN).mean().values


# ==============================
# EMA
# ==============================

ema = pd.Series(close_wave).ewm(span=MA_LEN).mean().values


# ==============================
# MID PRICE
# ==============================

mid = (high_wave + low_wave)/2


# ==============================
# PARKINSON VOLATILITY
# ==============================

parkinson = np.sqrt(
    (1/(4*np.log(2))) *
    np.mean(np.log(high_wave/low_wave)**2)
)


# ==============================
# METRIC FUNCTION
# ==============================

def compute_metrics(signal, price):

    valid = ~np.isnan(signal)

    signal = signal[valid]
    price  = price[valid]

    noise = price - signal

    signal_var = np.var(signal)
    noise_var  = np.var(noise)

    snr = signal_var / noise_var if noise_var > 0 else np.nan

    corr = np.corrcoef(signal, price)[0,1]

    return signal_var, noise_var, snr, corr


# ==============================
# RUN TESTS
# ==============================

tests = {
    "MeanWave": mean_wave,
    "SMA10": sma,
    "EMA10": ema,
    "MidPrice": mid
}

results = []

for name, signal in tests.items():

    s_var, n_var, snr, corr = compute_metrics(signal, close_wave)

    results.append({
        "model": name,
        "signal_var": s_var,
        "noise_var": n_var,
        "snr": snr,
        "correlation": corr
    })


results_df = pd.DataFrame(results)


# ==============================
# OUTPUT
# ==============================

print("\nSEGMENT START:", seg_df.iloc[0]["time"])
print("SEGMENT END:", seg_df.iloc[-1]["time"])

print("\nParkinson Volatility:", parkinson)

print("\n===== COMPARATIVE RESULTS =====\n")

print(results_df.sort_values("snr", ascending=False))