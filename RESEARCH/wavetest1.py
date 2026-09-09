import MetaTrader5 as mt5
import pandas as pd
import numpy as np

# ==============================
# CONFIG
# ==============================

SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M15
LOOKBACK = 15000

INERTIA = 0.6
RESPONSE = 0.4
MA_LEN = 5


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

if not mt5.initialize():
    raise RuntimeError("MT5 init failed")

mt5.symbol_select(SYMBOL, True)

rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, LOOKBACK)

mt5.shutdown()

df = pd.DataFrame(rates)
df["time"] = pd.to_datetime(df["time"], unit="s")


# ==============================
# DETECT SEGMENTS
# ==============================

segments = []
current = []

for i in range(len(df)):

    row = df.iloc[i]

    if is_restricted_time(row.time):

        if len(current) > 5:
            segments.append(pd.DataFrame(current))

        current = []
        continue

    current.append(row)


# ==============================
# METRIC FUNCTIONS
# ==============================

def compute_snr(signal, price):

    valid = ~np.isnan(signal)

    signal = signal[valid]
    price  = price[valid]

    noise = price - signal

    signal_var = np.var(signal)
    noise_var  = np.var(noise)

    if noise_var == 0:
        return np.nan

    return signal_var / noise_var


def gravity_test(signal, price):

    valid = ~np.isnan(signal)

    signal = signal[valid]
    price  = price[valid]

    deviation = price - signal

    returns = np.diff(price)

    dev = deviation[:-1]

    corr = np.corrcoef(dev, returns)[0,1]

    abs_dev = np.abs(deviation)

    pullbacks = 0
    total = len(abs_dev)-1

    for i in range(total):

        if abs_dev[i+1] < abs_dev[i]:
            pullbacks += 1

    pull_rate = pullbacks / total if total>0 else np.nan

    return corr, pull_rate


# ==============================
# TEST ALL SEGMENTS
# ==============================

results = []

for seg in segments:

    open_wave  = seg["open"].values
    high_wave  = seg["high"].values
    low_wave   = seg["low"].values
    close_wave = seg["close"].values

    if len(close_wave) < MA_LEN+2:
        continue

    # barycenter
    B = (open_wave + high_wave + low_wave + close_wave)/4

    # MeanWave
    mean_wave = np.zeros(len(B))
    mean_wave[0] = B[0]

    for i in range(1,len(B)):
        mean_wave[i] = mean_wave[i-1]*INERTIA + B[i]*RESPONSE

    # SMA / EMA
    sma = pd.Series(close_wave).rolling(MA_LEN).mean().values
    ema = pd.Series(close_wave).ewm(span=MA_LEN).mean().values

    # MidPrice
    mid = (high_wave + low_wave)/2

    models = {
        "MeanWave": mean_wave,
        "MidPrice": mid,
        "EMA": ema,
        "SMA": sma
    }

    for name,signal in models.items():

        snr = compute_snr(signal,close_wave)

        corr,pull = gravity_test(signal,close_wave)

        results.append({
            "model":name,
            "snr":snr,
            "gravity_corr":corr,
            "pullback_rate":pull
        })


results_df = pd.DataFrame(results)

summary = results_df.groupby("model").mean()

print("\n===== FINAL STATISTICAL REPORT =====\n")
print(summary.sort_values("snr",ascending=False))