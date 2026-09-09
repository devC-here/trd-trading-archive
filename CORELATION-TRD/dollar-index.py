import MetaTrader5 as mt5
import pandas as pd
import time

# 1. The 6 core pairs needed to calculate the "True" Synthetic DXY
CORE_PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD', 'USDCAD', 'USDSEK', 'USDCHF']

def find_all_index_symbols():
    """Scans all symbols in Market Watch for potential Dollar Index names."""
    all_symbols = mt5.symbols_get()
    # Looking for symbols that contain 'DXY', 'USDX', or 'DOLLAR'
    matches = [s.name for s in all_symbols if any(x in s.name.upper() for x in ["DXY", "USDX", "DOLLAR"])]
    # Filter out the core pairs themselves to avoid circular logic
    return [m for m in matches if m not in CORE_PAIRS]

def get_live_quotes(symbols_list):
    """Fetches the latest Bid price for a list of symbols."""
    quotes = {}
    for sym in symbols_list:
        tick = mt5.symbol_info_tick(sym)
        if tick:
            quotes[sym] = tick.bid
    return quotes

def calculate_synthetic(p):
    """Mathematical DXY formula based on ICE standards."""
    try:
        # Weighted geometric mean formula
        return 50.14348112 * (
            (p['EURUSD'] ** -0.576) *
            (p['USDJPY'] ** 0.136) *
            (p['GBPUSD'] ** -0.119) *
            (p['USDCAD'] ** 0.091) *
            (p['USDSEK'] ** 0.042) *
            (p['USDCHF'] ** 0.036)
        )
    except Exception:
        return None

# --- Main Logic ---
if not mt5.initialize():
    print("MT5 Init Failed")
    quit()

# Auto-detect every relevant symbol your broker has
target_indices = find_all_index_symbols()
print(f"Monitoring Opportunities in: {target_indices}")

try:
    print(f"{'Time':<10} | {'Synthetic':<10} | {'Symbol':<15} | {'Broker':<10} | {'Gap (Opp)'}")
    print("-" * 65)

    while True:
        # Get prices for BOTH core pairs and target symbols
        all_prices = get_live_quotes(CORE_PAIRS + target_indices)
        synth_dxy = calculate_synthetic(all_prices)

        if synth_dxy:
            for index_sym in target_indices:
                broker_price = all_prices.get(index_sym)
                if broker_price:
                    gap = synth_dxy - broker_price
                    # Opportunity threshold: usually > 0.005 is worth watching
                    alert = "★ ALERT ★" if abs(gap) > 0.01 else ""
                    
                    print(f"{time.strftime('%H:%M:%S'):<10} | {synth_dxy:<10.3f} | {index_sym:<15} | {broker_price:<10.3f} | {gap:+.5f} {alert}")
        
        time.sleep(1) # Scan every second

except KeyboardInterrupt:
    print("Scanner stopped.")
finally:
    mt5.shutdown()