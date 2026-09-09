import MetaTrader5 as mt5
import pandas as pd

# ==============================
# CONFIG
# ==============================
LOOKBACK_BARS = 650   # Total candles to fetch
LOOKBACK_WAR  = 5     # Number of valid candles to sample
EPS           = 1e-8  # Guard against zero-division / doji filter


# ==============================
# SESSION FILTER  —  uses BROKER clock, not local machine
# ==============================
def get_broker_time() -> pd.Timestamp | None:
    """
    Pulls current broker server time from the latest tick.
    Returns None if MT5 is not initialised or symbol unavailable.
    """
    tick = mt5.symbol_info_tick("EURUSD")   # lightweight, always available
    if tick is None:
        return None
    return pd.Timestamp(tick.time, unit="s")


def is_restricted_time(ts: pd.Timestamp) -> bool:
    """
    All comparisons are against broker server timestamps —
    no dependency on the local machine clock or timezone.
    """
    h, m = ts.hour, ts.minute
    return (
        0  <= h <= 1           or   # Asian dead-zone / rollover
        (h == 12 and m >= 30)  or   # Pre-NY lunch drain
        h == 13                or   # NY open chop
        18 <= h <= 19               # End-of-day thin liquidity
    )


# ==============================
# DUAL-WEIGHTED AVERAGE
# ==============================
def dual_weighted_avg(values: list[float]) -> float:
    """
    Combines two independent weight signals multiplicatively:

      recency_weight[i]   = 1 / (rank + 1)   → w1 > w2 > ... > w5
                            rank 0 = newest candle

      deviation_weight[i] = 1 / (|v - mean| + EPS)
                            → candles near the cluster mean outweigh spikes

      final_weight[i]     = recency_weight[i] × deviation_weight[i]

    This means a recent AND typical candle dominates;
    an old OR extreme candle is progressively down-weighted.
    """
    n    = len(values)
    mean = sum(values) / n

    weights = [
        (1.0 / (rank + 1)) * (1.0 / (abs(v - mean) + EPS))
        for rank, v in enumerate(values)   # rank 0 = most recent
    ]

    total_w = sum(weights)
    return sum(v * w for v, w in zip(values, weights)) / total_w


# ==============================
# MAIN API
# ==============================
def compute_sltp(symbol: str, timeframe: int) -> float | None:
    """
    Scans recent candles on BROKER time (not local machine time),
    collects the last LOOKBACK_WAR non-doji, non-restricted body sizes,
    and returns a dual-weighted average body expansion in price units.

    Weight priority:
        newer candle  >  older candle       (recency layer)
        typical body  >  spike/doji body    (deviation layer)

    Args:
        symbol:    MT5 symbol string, e.g. "ETHUSD"
        timeframe: MT5 timeframe constant,  e.g. mt5.TIMEFRAME_M5

    Returns:
        Weighted average body size (float), or None on failure.
    """
    if not mt5.initialize():
        return None

    try:
        if not mt5.symbol_select(symbol, True):
            return None

        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, LOOKBACK_BARS)
        if rates is None or len(rates) < 50:
            return None

    finally:
        mt5.shutdown()

    df         = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    # ↑ These are already broker-server timestamps as stored by MT5.
    #   No local timezone conversion applied — broker time preserved as-is.

    body_vals = []

    for row in df.iloc[-2::-1].itertuples():   # newest closed candle first
        if len(body_vals) >= LOOKBACK_WAR:
            break

        if is_restricted_time(row.time):       # broker timestamp checked
            continue

        body = abs(row.open - row.close)
        if body > EPS:
            body_vals.append(body)             # index 0 = most recent

    if len(body_vals) < LOOKBACK_WAR:
        return None

    return dual_weighted_avg(body_vals)        # rank 0 carries highest recency weight


# ==============================
# SELF-TEST
# ==============================
if __name__ == "__main__":
    SYMBOL = "EURUSD"
    TF     = mt5.TIMEFRAME_H1

    print("\n--- ESTIMATE TEST START ---")

    estimate = compute_sltp(SYMBOL, TF)

    if estimate is None:
        print("Estimate calculation failed.")
    else:
        print(f"Estimated body expansion ({SYMBOL}, TF={TF}): {estimate:.6f}")

    print("--- ESTIMATE TEST END ---\n")