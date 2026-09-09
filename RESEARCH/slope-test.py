import MetaTrader5 as mt5
import pandas as pd
import numpy as np

# ==============================
# CONFIG
# ==============================
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M30
LOOKBACK = 200
POINT_VALUE = 0.0001 

# ==============================
# INIT MT5
# ==============================
if not mt5.initialize():
    raise RuntimeError("MT5 initialization failed")

mt5.symbol_select(SYMBOL, True)
rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, LOOKBACK)
mt5.shutdown()

# ==============================
# DATAFRAME
# ==============================
df = pd.DataFrame(rates)
df["time"] = pd.to_datetime(df["time"], unit="s")

# 1. Slope Calculation (Using OPEN Price)
# Difference between this candle's Open and the last candle's Open
df["open_diff"] = df["open"].diff()
df["slope_pips"] = df["open_diff"] / POINT_VALUE
df["angle_deg"] = np.degrees(np.arctan(df["slope_pips"]))

# 2. Candle Body Size Calculation (For the Filter)
df["body_size_pips"] = abs(df["close"] - df["open"]) / POINT_VALUE

# ==============================
# VOLATILITY FILTER LOGIC
# ==============================
def check_volatility_flag(row_index, data):
    # Need at least 6 previous candles for comparison
    if row_index < 7: return False
    
    current_slope = abs(data.loc[row_index, "slope_pips"])
    
    # Get the body sizes of the LAST 6 candles (not including current)
    last_6_bodies = data.loc[row_index-6 : row_index-1, "body_size_pips"].values
    sorted_bodies = sorted(last_6_bodies, reverse=True)
    
    # CONDITION 1: Slope is larger than 3 out of the last 6 bodies
    # (Checking if current slope > the 3rd largest body)
    cond_a = current_slope > sorted_bodies[2] 
    
    # CONDITION 2: Slope is larger than any two bodies combined
    # (Checking against the sum of the two largest)
    sum_top_2 = sorted_bodies[0] + sorted_bodies[1]
    cond_b = current_slope > sum_top_2
    
    return cond_a or cond_b

# Apply filter
df["is_flagged"] = [check_volatility_flag(i, df) for i in range(len(df))]

# ==============================
# RESULTS
# ==============================
# Show the results where the slope is significant
flagged_results = df[df["is_flagged"] == True][["time", "open", "slope_pips", "angle_deg"]].tail(20)

if flagged_results.empty:
    print("\nNo flags found. The market might be in a low-volatility phase.")
    print("Latest Slope:", df["slope_pips"].iloc[-1], "Pips")
else:
    print("\n===== OPEN-TO-OPEN FLAGGED SLOPES =====")
    print(flagged_results.to_string(index=False))