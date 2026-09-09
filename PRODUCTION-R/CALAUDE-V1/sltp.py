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

    SYMBOL = "ETHUSD"
    TF = mt5.TIMEFRAME_M5

    print("\n--- ESTIMATE TEST START ---")

    estimate = compute_sltp(SYMBOL, TF)

    if estimate is None:
        print("Estimate calculation failed.")
    else:
        print(f"Estimated M30 Expansion : {estimate:.6f}")

    print("--- ESTIMATE TEST END ---\n")
