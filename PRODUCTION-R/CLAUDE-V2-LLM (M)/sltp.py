import MetaTrader5 as mt5
import pandas as pd

# ==============================
# INTERNAL CONFIG
# ==============================
LOOKBACK_BARS = 650     # how much history to scan
LOOKBACK_WAR  = 6       # how many valid candles to collect
EPS = 1e-8

# ==============================
# TIME RESTRICTION (UNCHANGED)
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
# STABLE AVERAGING FUNCTION (KEPT)
# ==============================
def variation_weighted_avg(values):
    mean = sum(values) / len(values)
    weights = [1.0 / (abs(v - mean) + EPS) for v in values]
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)

# ==============================
# MAIN PUBLIC API
# ==============================
def compute_sltp(symbol, timeframe):
    """
    Returns:
        estimate (float): Typical recent M30 body expansion in price units.
                          This is the ONLY output now.
    """

    # --- MT5 Init ---
    if not mt5.initialize():
        return None

    if not mt5.symbol_select(symbol, True):
        mt5.shutdown()
        return None

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, LOOKBACK_BARS)
    mt5.shutdown()

    if rates is None or len(rates) < 50:
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    # Start from last CLOSED candle
    start_idx = len(df) - 2

    body_vals = []

    j = start_idx
    while j >= 0 and len(body_vals) < LOOKBACK_WAR:
        row = df.iloc[j]

        # Respect your session filter
        if not is_restricted_time(row.time):

            # ONLY BODY SIZE MATTERS NOW
            body = abs(row.open - row.close)

            if body > EPS:   # ignore doji / zero moves
                body_vals.append(body)

        j -= 1

    if len(body_vals) < LOOKBACK_WAR:
        return None

    # ------------------------------------------
    # FINAL RESULT → SINGLE MARKET ESTIMATE
    # ------------------------------------------
    estimate = variation_weighted_avg(body_vals)

    return estimate


# ============================================================
# SELF TEST (RUN FILE DIRECTLY TO VERIFY)
# ============================================================
 
if __name__ == "__main__":

    SYMBOL = "EURUSD"

    print("\n--- MULTI TF EXPANSION TEST ---")

    # --- estimates ---
    est_m15 = compute_sltp(SYMBOL, mt5.TIMEFRAME_M15)
    est_m30 = compute_sltp(SYMBOL, mt5.TIMEFRAME_M30)
    est_h1  = compute_sltp(SYMBOL, mt5.TIMEFRAME_H1)

    if None in (est_m15, est_m30, est_h1):
        print("Estimate calculation failed.")
        exit()

    # --- get latest M15 candle ---
    if not mt5.initialize():
        print("MT5 init failed")
        exit()

    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M15, 0, 2)

    if rates is None or len(rates) < 2:
        print("Failed to fetch M15 data")
        mt5.shutdown()
        exit()

    df = pd.DataFrame(rates)

    last_closed = df.iloc[-2]

    current_move = abs(last_closed["open"] - last_closed["close"])

    mt5.shutdown()

    # --- coverage ---
    cov_m30 = (current_move / est_m30) * 100
    cov_h1  = (current_move / est_h1) * 100

    # --- output ---
    print(f"M15 Estimate : {est_m15:.6f}")
    print(f"M30 Estimate : {est_m30:.6f}")
    print(f"H1  Estimate : {est_h1:.6f}")

    print("\nCurrent M15 Move :", round(current_move, 6))

    print("\n--- COVERAGE ---")
    print(f"M30 Covered : {cov_m30:.2f}%")
    print(f"H1  Covered : {cov_h1:.2f}%")

    print("\n--- TEST END ---\n")