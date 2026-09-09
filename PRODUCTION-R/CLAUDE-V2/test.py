import MetaTrader5 as mt5
import time

# 1. Configuration - ADJUST SYMBOL NAMES TO YOUR MARKET WATCH
SYMBOLS = {
    "DXY": "DXYUSD",      
    "EURUSD": "EURUSD",
    "USDJPY": "USDJPY",
    "GBPUSD": "GBPUSD",
    "USDCAD": "USDCAD",
    "USDSEK": "USDSEK",
    "USDCHF": "USDCHF"
}

LOT_SIZE = 0.1  # Base lot for DXY; others will be weighted proportionally

# 2. Connect to MT5
if not mt5.initialize():
    print("Initialize failed")
    quit()

def get_prices(symbol):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        mt5.symbol_select(symbol, True)
        tick = mt5.symbol_info_tick(symbol)
    return {"bid": tick.bid, "ask": tick.ask, "mid": (tick.ask + tick.bid) / 2}

def execute_basket(direction):
    """
    Direction: 'SHORT_DXY' means Sell DXY and Buy the components (Long USD).
    """
    print(f"!!! EXECUTING {direction} BASKET !!!")
    
    # This is a simplified execution. In production, you'd use mt5.order_send() 
    # for each pair in a loop with calculated lot sizes.
    for label, sym in SYMBOLS.items():
        # Example logic for DXY leg:
        order_type = mt5.ORDER_TYPE_SELL if direction == 'SHORT_DXY' and label == 'DXY' else mt5.ORDER_TYPE_BUY
        # ... mt5.order_send logic goes here ...
        print(f"Order Sent: {label} ({sym})")

# --- MAIN LOOP ---
print("Monitoring Live Arbitrage Opportunity (Spread-Adjusted)...")

try:
    while True:
        p = {k: get_prices(v) for k, v in SYMBOLS.items()}
        
        # 1. Calculate Synthetic MID
        syn_mid = (50.14348112 * (p["EURUSD"]["mid"]**-0.576) * (p["USDJPY"]["mid"]**0.136) * (p["GBPUSD"]["mid"]**-0.119) * (p["USDCAD"]["mid"]**0.091) * (p["USDSEK"]["mid"]**0.042) * (p["USDCHF"]["mid"]**0.036))
        
        # 2. Calculate Execution Cost (Crossing the spread)
        # To "Sell" the overvalued DXY, we hit the BID.
        # To "Buy" the undervalued Synthetic, we hit the ASKS of the components.
        syn_buy_cost = (50.14348112 * (p["EURUSD"]["bid"]**-0.576) * (p["USDJPY"]["ask"]**0.136) * (p["GBPUSD"]["bid"]**-0.119) * (p["USDCAD"]["ask"]**0.091) * (p["USDSEK"]["ask"]**0.042) * (p["USDCHF"]["ask"]**0.036))

        market_dxy_bid = p["DXY"]["bid"]
        net_profit = market_dxy_bid - syn_buy_cost
        
        # Output status
        print(f"Mkt Bid: {market_dxy_bid:.3f} | Syn Ask: {syn_buy_cost:.3f} | NET: {net_profit:.4f}")
        
        # 3. Trigger Logic
        if net_profit > 0.015: # 1.5 index points threshold
            execute_basket('SHORT_DXY')
            break # Stop loop after execution to avoid multiple fills

        time.sleep(0.2) # High-frequency polling

except Exception as e:
    print(f"Error: {e}")
finally:
    mt5.shutdown()