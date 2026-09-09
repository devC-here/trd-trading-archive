import MetaTrader5 as mt5
import time
from datetime import datetime

# ================= CONFIG =================
SYMBOLS = ["EURUSD", "DXYUSD"]  # adjust DXY symbol if needed
# ==========================================

def connect():
    if not mt5.initialize():
        print("MT5 init failed:", mt5.last_error())
        quit()
    print("Connected to MT5")

def setup_symbols():
    for sym in SYMBOLS:
        if not mt5.symbol_select(sym, True):
            print(f"Failed to select {sym}")
            quit()

def tick_detector():
    last_ticks = {sym: None for sym in SYMBOLS}
    last_time_msc = {sym: 0 for sym in SYMBOLS}

    while True:
        updated = {}

        for sym in SYMBOLS:
            tick = mt5.symbol_info_tick(sym)
            if tick is None:
                continue

            # 🔥 Only react if NEW tick arrived
            if tick.time_msc != last_time_msc[sym]:
                updated[sym] = tick
                last_time_msc[sym] = tick.time_msc

        # If no new ticks → skip
        if not updated:
            continue

        print(f"\n[{datetime.now()}] --- NEW TICK EVENT ---")

        for sym, tick in updated.items():
            prev = last_ticks[sym]

            if prev:
                delta = tick.bid - prev.bid
                direction = "UP" if delta > 0 else "DOWN" if delta < 0 else "FLAT"
                speed = abs(delta) / max((tick.time_msc - prev.time_msc)/1000, 0.001)

                print(f"{sym}: {tick.bid:.5f} | {direction} | Δ={delta:.5f} | speed={speed:.5f}")
            else:
                print(f"{sym}: INIT @ {tick.bid:.5f}")

            last_ticks[sym] = tick

        # 🔥 Cross-symbol logic (EURUSD vs DXY)
        if all(last_ticks.values()):
            e = last_ticks[SYMBOLS[0]]
            d = last_ticks[SYMBOLS[1]]

            prev_e = last_ticks[SYMBOLS[0]]
            prev_d = last_ticks[SYMBOLS[1]]

            # simple directional relation
            if e.bid > prev_e.bid and d.bid < prev_d.bid:
                print("✔ EURUSD ↑ & DXY ↓ (classic inverse)")
            elif e.bid < prev_e.bid and d.bid > prev_d.bid:
                print("✔ EURUSD ↓ & DXY ↑ (classic inverse)")
            elif e.bid > prev_e.bid and d.bid > prev_d.bid:
                print("⚠ Both Rising (momentum phase)")
            elif e.bid < prev_e.bid and d.bid < prev_d.bid:
                print("⚠ Both Falling (risk-off / USD flow?)")

# ================= RUN =================
connect()
setup_symbols()
tick_detector() 