import MetaTrader5 as mt5
import pandas as pd
import numpy as np

# ==============================
# CONFIG
# ==============================
SYMBOL = "EURUSD.e"
TIMEFRAME = mt5.TIMEFRAME_M30
LOOKBACK = 650        # slightly higher for M30

LOW_TH = 0.00013
HIGH_TH = 0.00030
EPS = 1e-8

# ==============================
# TIME GATES (NO COMPARE / NO WAR)
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
# UTILS
# ==============================
def candle_size(o, c):
    return abs(o - c)

def variation_weighted_avg(sizes):
    mean = np.mean(sizes)
    weights = [1.0 / (abs(s - mean) + EPS) for s in sizes]
    return np.sum(np.array(sizes) * np.array(weights)) / np.sum(weights)

# ==============================
# MT5 INIT
# ==============================
if not mt5.initialize():
    raise RuntimeError("MT5 init failed")

if not mt5.symbol_select(SYMBOL, True):
    mt5.shutdown()
    raise RuntimeError(f"Symbol not found: {SYMBOL}")

rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, LOOKBACK)
mt5.shutdown()

if rates is None or len(rates) < 30:
    raise RuntimeError("Not enough data")

df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')

# ==============================
# COLLECTIVE ANALYTICS
# ==============================
greater = 0
smaller = 0
skipped = 0

# stop early so C+2 exists
for i in range(12, len(df) - 2):
    ts = df.iloc[i - 1]['time']

    # ---- collect previous valid candles for WAR ----
    sizes = []
    j = i - 1

    while j >= 0 and len(sizes) < 6:
        row = df.iloc[j]
        if not is_restricted_time(row.time):
            sizes.append(candle_size(row.open, row.close))
        j -= 1

    if len(sizes) < 6:
        continue

    war = variation_weighted_avg(sizes)

    # ---- HARD ZONES ----
    if war > HIGH_TH:
        greater += 1
        continue

    if war < LOW_TH:
        smaller += 1
        continue

    # ---- MID ZONE ----
    if is_restricted_time(ts):
        skipped += 1
        continue

    c0 = df.iloc[i - 1]   # C
    c1 = df.iloc[i]       # C+1
    c2 = df.iloc[i + 1]   # C+2

    s0 = candle_size(c0.open, c0.close)
    s1 = candle_size(c0.open, c1.close)
    s2 = candle_size(c0.open, c2.close)

    if max(s0, s1, s2) > war:
        greater += 1
    else:
        smaller += 1

# ==============================
# RESULTS
# ==============================
total = greater + smaller

print("===== EURUSD.e M30 VARIATION-WEIGHTED REGIME =====")
print(f"Lookback candles     : {LOOKBACK}")
print(f"Valid decisions      : {total}")
print(f"Skipped (time gate)  : {skipped}")
print(f"GREATER              : {greater} ({(greater/total)*100:.2f}%)")
print(f"SMALLER              : {smaller} ({(smaller/total)*100:.2f}%)")
