import MetaTrader5 as mt5
import pandas as pd
import numpy as np

# ==============================
# CONFIG
# ==============================
SYMBOL = "EURUSD"
LOOKBACK = 5000  
REF_CANDLES = 6
MAX_WAIT = 5  # Small window to catch the immediate volatility
EXT_MULT = 1.3  # 1.3x the estimate
EPS = 1e-8

timeframes = {"M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1}

def variation_weighted_avg(sizes):
    if len(sizes) == 0: return 0
    mean = np.mean(sizes)
    weights = [1/(abs(s-mean)+EPS) for s in sizes]
    return np.sum(np.array(sizes)*np.array(weights)) / np.sum(weights)

if not mt5.initialize():
    raise RuntimeError("MT5 init failed")

final_results = []

for tf_name, tf in timeframes.items():
    rates = mt5.copy_rates_from_pos(SYMBOL, tf, 0, LOOKBACK)
    if rates is None: continue
    df = pd.DataFrame(rates)

    total_triggers = 0
    both_hit_with_ext = 0

    for i in range(10, len(df) - MAX_WAIT):
        # Calculate WAR for current candle
        ref_sizes = [abs(df.iloc[i-k].open - df.iloc[i-k].close) for k in range(1, REF_CANDLES+1)]
        war = variation_weighted_avg(ref_sizes)

        curr_open = df.iloc[i].open
        
        # Defining the levels
        band_upper = curr_open + war
        band_lower = curr_open - war
        ext_upper = curr_open + (war * EXT_MULT)
        ext_lower = curr_open - (war * EXT_MULT)

        # Track if both sides are hit within the window
        upper_touched = False
        lower_touched = False
        extension_reached = False

        for k in range(0, MAX_WAIT):
            candle = df.iloc[i + k]
            
            # Check Upper
            if candle.high >= band_upper:
                upper_touched = True
            if candle.high >= ext_upper:
                extension_reached = True
                
            # Check Lower
            if candle.low <= band_lower:
                lower_touched = True
            if candle.low <= ext_lower:
                extension_reached = True

            # Condition: Both bands hit AND at least one went to 1.3x
            if upper_touched and lower_touched and extension_reached:
                both_hit_with_ext += 1
                break
        
        total_triggers += 1

    final_results.append({
        "TF": tf_name,
        "Total Candles Tested": total_triggers,
        "Both Hit + 1.3x Ext": both_hit_with_ext,
        "Frequency %": round((both_hit_with_ext / total_triggers * 100), 2)
    })

mt5.shutdown()

print(f"\n===== VOLATILITY SYMMETRY TEST (BOTH BANDS + 1.3x) FOR {SYMBOL} =====")
print(pd.DataFrame(final_results))