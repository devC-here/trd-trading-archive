import MetaTrader5 as mt5
import pandas as pd

# ==============================
# INTERNAL CONFIG
# ==============================
LOOKBACK_BARS = 650
LOOKBACK_WAR  = 6
EPS = 1e-8


# ==============================
# TIME RESTRICTION
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
# STABLE VARIATION WEIGHTED AVG
# ==============================
def variation_weighted_avg(values):

    mean = sum(values) / len(values)

    weights = [
        1.0 / (abs(v - mean) + EPS)
        for v in values
    ]

    return sum(v * w for v, w in zip(values, weights)) / sum(weights)


# ==============================
# ESTIMATE CALCULATOR
# ==============================
def compute_estimate(symbol, timeframe):

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, LOOKBACK_BARS)

    if rates is None or len(rates) < 50:
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    start_idx = len(df) - 2

    body_vals = []

    j = start_idx

    while j >= 0 and len(body_vals) < LOOKBACK_WAR:

        row = df.iloc[j]

        if not is_restricted_time(row.time):

            body = abs(row.open - row.close)

            if body > EPS:
                body_vals.append(body)

        j -= 1

    if len(body_vals) < LOOKBACK_WAR:
        return None

    estimate = variation_weighted_avg(body_vals)

    return estimate


# ==============================
# MAIN FUNCTION
# ==============================
def compute_all_estimates(symbol):
    """
    Returns:
        M15 estimate
        M30 estimate
        H1  estimate
    """

    if not mt5.initialize():
        print("MT5 initialization failed")
        return None

    if not mt5.symbol_select(symbol, True):
        print("Symbol selection failed")
        mt5.shutdown()
        return None

    m15_est = compute_estimate(symbol, mt5.TIMEFRAME_M15)
    m30_est = compute_estimate(symbol, mt5.TIMEFRAME_M30)
    h1_est  = compute_estimate(symbol, mt5.TIMEFRAME_H1)

    mt5.shutdown()

    return m15_est, m30_est, h1_est


# ============================================================
# TEST RUN
# ============================================================
if __name__ == "__main__":

    SYMBOL = "EURUSD"

    print("\n--- EXPANSION ESTIMATE TEST ---\n")

    result = compute_all_estimates(SYMBOL)

    if result is None:
        print("Estimation failed.")
        exit()

    m15, m30, h1 = result

    print(f"M15 Estimate : {m15:.6f}")
    print(f"M30 Estimate : {m30:.6f}")
    print(f"H1  Estimate : {h1:.6f}")

    print("\n--- TEST COMPLETE ---\n")