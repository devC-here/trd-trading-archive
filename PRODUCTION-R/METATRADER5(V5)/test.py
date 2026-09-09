import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

# ==========================
# SETTINGS
# ==========================
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M30
CANDLES = 14

# ==========================
# CONNECT TO MT5
# ==========================
if not mt5.initialize():
    print("Initialize() failed")
    mt5.shutdown()
    quit()

# Ensure symbol is available
if not mt5.symbol_select(SYMBOL, True):
    print(f"Failed to select {SYMBOL}")
    mt5.shutdown()
    quit()

# ==========================
# GET LAST 14 CLOSED CANDLES
# ==========================
rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 1, CANDLES)

if rates is None:
    print("Failed to get rates")
    mt5.shutdown()
    quit()

# Convert to DataFrame
df = pd.DataFrame(rates)

# Convert time to readable format
df['time'] = pd.to_datetime(df['time'], unit='s')

# ==========================
# PRINT TICK VOLUMES
# ==========================
print("\nLast 14 Closed M30 Candle Tick Volumes:\n")
print(df[['time', 'tick_volume']])

mt5.shutdown()
