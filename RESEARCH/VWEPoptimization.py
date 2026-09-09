import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================
LOOKBACK          = 8000
ATR_PERIOD        = 14
PARK_PERIOD       = 14
EPS               = 1e-8
REF_CANDLES_RANGE = range(1, 101)

symbols = [
    "EURUSD","USDJPY","XAUUSD","DXYUSD",
    "BA.N","US SP 500","XAGUSD","BTCUSD",
    "GBPUSD","US Oil","ETHUSD","NGAS"
]

timeframes = {
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1
}

def is_restricted_time(ts):
    minutes = ts.hour * 60 + ts.minute
    if 0   <= minutes < 60:  return True
    if 480 <= minutes < 540: return True
    if 780 <= minutes < 840: return True
    return False

def compute_atr(df, period):
    h, l, c = df['high'], df['low'], df['close']
    tr = pd.concat([h - l, abs(h - c.shift()), abs(l - c.shift())], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_parkinson(df, period):
    hl = np.log(df["high"] / df["low"]) ** 2
    return np.sqrt((1 / (4 * np.log(2))) * hl.rolling(period).mean())

def variation_weighted_avg(arr):
    mean    = arr.mean()
    weights = 1.0 / (np.abs(arr - mean) + EPS)
    return (arr * weights).sum() / weights.sum()

def sweep(df):
    n       = len(df)
    times   = df["time"].values
    highs   = df["high"].values
    lows    = df["low"].values
    opens   = df["open"].values
    closes  = df["close"].values
    atrs    = df["atr"].values
    parks   = df["parkinson"].values
    restr   = np.array([is_restricted_time(pd.Timestamp(t)) for t in times])

    # We now track Hit1, Hit2, and Hit3 for more robust fit analysis
    hit1 = np.zeros(101, dtype=np.int64)
    hit2 = np.zeros(101, dtype=np.int64)
    hit3 = np.zeros(101, dtype=np.int64)
    total = np.zeros(101, dtype=np.int64)

    for i in range(10, n - 4):
        if restr[i] or np.isnan(atrs[i]) or np.isnan(parks[i]):
            continue

        op = opens[i]
        
        # Checking high/low of next 3 candles for the multi-hit fit
        c1_h, c1_l = highs[i], lows[i]
        c2_h, c2_l = highs[i+1], lows[i+1]
        c3_h, c3_l = highs[i+2], lows[i+2]

        sizes = []
        j = i - 1
        # Collect historical sizes for lookback
        while j >= 0 and len(sizes) < 100:
            if not restr[j]:
                sizes.append(abs(opens[j] - closes[j]))
            j -= 1

        arr = np.array(sizes, dtype=np.float64)
        for rc in REF_CANDLES_RANGE:
            if rc > len(arr): break
            
            war   = variation_weighted_avg(arr[:rc])
            upper = op + war
            lower = op - war
            
            total[rc] += 1
            # Hit 1
            if c1_h >= upper or c1_l <= lower:
                hit1[rc] += 1
            # Hit 2 (Cumulative)
            if max(c1_h, c2_h) >= upper or min(c1_l, c2_l) <= lower:
                hit2[rc] += 1
            # Hit 3 (Cumulative)
            if max(c1_h, c2_h, c3_h) >= upper or min(c1_l, c2_l, c3_l) <= lower:
                hit3[rc] += 1

    return hit1, hit2, hit3, total

# ==============================
# EXECUTION
# ==============================
if not mt5.initialize():
    raise RuntimeError("MT5 init failed")

# Global accumulators
global_hit1 = np.zeros(101)
global_hit3 = np.zeros(101)
global_total = np.zeros(101)

total_jobs = len(symbols) * len(timeframes)

with tqdm(total=total_jobs, desc="Analyzing Best Fit", unit="job") as pbar:
    for symbol in symbols:
        mt5.symbol_select(symbol, True)
        for tf_name, tf in timeframes.items():
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, LOOKBACK)
            if rates is not None:
                df = pd.DataFrame(rates)
                df["time"] = pd.to_datetime(df["time"], unit="s")
                df["atr"] = compute_atr(df, ATR_PERIOD)
                df["parkinson"] = compute_parkinson(df, PARK_PERIOD)

                h1, h2, h3, tot = sweep(df)
                
                global_hit1 += h1
                global_hit3 += h3
                global_total += tot
            
            pbar.update(1)

mt5.shutdown()

# ==============================
# RESULTS
# ==============================
# Calculate Global Averages
avg_h1 = np.divide(global_hit1, global_total, out=np.zeros_like(global_hit1), where=global_total!=0)
avg_h3 = np.divide(global_hit3, global_total, out=np.zeros_like(global_hit3), where=global_total!=0)

# We want an RC that balances Hit1 accuracy with Hit3 stability
# Using a score: (Hit1 * 0.4) + (Hit3 * 0.6)
combined_score = (avg_h1 * 0.4) + (avg_h3 * 0.6)
best_rc = int(np.argmax(combined_score))

print("\n" + "="*50)
print(f"COLLECTIVE BEST FIT LOOKBACK: {best_rc}")
print(f"Avg Hit1 across all symbols: {avg_h1[best_rc]:.4f}")
print(f"Avg Hit3 across all symbols: {avg_h3[best_rc]:.4f}")
print("="*50)

print("\nTop 5 Lookbacks by Collective Stability:")
top_5 = np.argsort(combined_score)[::-1][:5]
for rank, rc in enumerate(top_5, 1):
    print(f"{rank}. RC={rc:2d} | Score: {combined_score[rc]:.4f} (H1: {avg_h1[rc]:.4f}, H3: {avg_h3[rc]:.4f})")