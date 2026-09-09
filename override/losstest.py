import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# =========================================================
# =============== USER INPUT (EDIT THIS) ==================
# =========================================================

SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M15

TRADE_OPEN_TIME = "2026-01-21 08:30"   # YYYY-MM-DD HH:MM
DIRECTION = "BUY"                     # BUY or SELL
ENTRY_PRICE = 1.17231

TP_MULTIPLIER = 1.0                   # keep 1.0 for diagnostics

# =========================================================
# ================== INTERNAL CONFIG ======================
# =========================================================

REF_CANDLES = 6
FORWARD_CANDLES = 3
LOOKBACK_MINUTES = 300

LOW_TH = 0.00013
HIGH_TH = 0.00030
EPS = 1e-8

# =========================================================
# ================== TIME RESTRICTIONS ====================
# =========================================================

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

# =========================================================
# ================== CORE UTILS ===========================
# =========================================================

def body_size(o, c):
    return abs(o - c)

def variation_weighted_avg(sizes):
    mean = np.mean(sizes)
    weights = [1.0 / (abs(s - mean) + EPS) for s in sizes]
    return float(np.sum(np.array(sizes) * np.array(weights)) / np.sum(weights))

def clamp_size(v):
    return max(LOW_TH, min(HIGH_TH, v))

# =========================================================
# ================== MAIN ANALYSIS ========================
# =========================================================

def run_trade_comparison():
    trade_time = datetime.strptime(TRADE_OPEN_TIME, "%Y-%m-%d %H:%M")

    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")

    if not mt5.symbol_select(SYMBOL, True):
        mt5.shutdown()
        raise RuntimeError(f"Symbol not found: {SYMBOL}")

    rates = mt5.copy_rates_from(
        SYMBOL,
        TIMEFRAME,
        trade_time - timedelta(minutes=LOOKBACK_MINUTES),
        300
    )

    mt5.shutdown()

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    before = df[df['time'] < trade_time]
    after = df[df['time'] >= trade_time]

    # ------------------ Reference candles -----------------
    ref_sizes = []
    ref_details = []

    for _, row in before.iloc[::-1].iterrows():
        if is_restricted_time(row.time):
            continue

        size = body_size(row.open, row.close)
        ref_sizes.append(size)
        ref_details.append(
            f"{row.time} | O:{row.open} C:{row.close} | size:{size:.5f}"
        )

        if len(ref_sizes) == REF_CANDLES:
            break

    if len(ref_sizes) < REF_CANDLES:
        raise RuntimeError("Not enough valid reference candles")

    raw_war = variation_weighted_avg(ref_sizes)
    war = clamp_size(raw_war)

    tp_distance = war * TP_MULTIPLIER
    tp_price = (
        ENTRY_PRICE + tp_distance
        if DIRECTION == "BUY"
        else ENTRY_PRICE - tp_distance
    )

    # ------------------ Next 3 candles --------------------
    max_price = ENTRY_PRICE
    max_distance = 0.0
    hit = False
    hit_candle = None

    for i, row in enumerate(after.iloc[:FORWARD_CANDLES].itertuples(), start=1):
        if DIRECTION == "BUY":
            max_price = max(max_price, row.high)
            max_distance = max(max_distance, max_price - ENTRY_PRICE)
            if row.high >= tp_price:
                hit = True
                hit_candle = i
                break
        else:
            max_price = min(max_price, row.low)
            max_distance = max(max_distance, ENTRY_PRICE - max_price)
            if row.low <= tp_price:
                hit = True
                hit_candle = i
                break

    # ------------------ OUTPUT -----------------------------
    print("\n===== TRADE TP COMPARISON =====")
    print(f"Symbol        : {SYMBOL}")
    print(f"Timeframe     : {TIMEFRAME}")
    print(f"Trade time    : {TRADE_OPEN_TIME}")
    print(f"Direction     : {DIRECTION}")
    print(f"Entry price   : {ENTRY_PRICE}")
    print("\n--- Reference candle sizes (used) ---")
    for r in ref_details:
        print(r)

    print("\n--- WAR & TP ---")
    print(f"Raw WAR       : {raw_war:.6f}")
    print(f"Capped WAR    : {war:.6f}")
    print(f"TP distance   : {tp_distance:.6f}")
    print(f"TP price      : {tp_price:.6f}")

    print("\n--- Outcome (next 3 candles) ---")
    print(f"TP hit        : {hit}")
    print(f"Hit candle #  : {hit_candle}")
    print(f"Max price     : {max_price:.6f}")
    print(f"Max distance  : {max_distance:.6f}")

# =========================================================
# ================== RUN ==================================
# =========================================================

if __name__ == "__main__":
    run_trade_comparison()
