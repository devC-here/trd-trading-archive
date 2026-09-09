"""
========================================================================================
MT5 SIGNAL SCANNER: THE TRADING SCHOOL FIBONACCI GOLDEN ZONE STRATEGY
========================================================================================
Source: Short Video epI1Clb4r54 | 1-Year Backtest: +43.2% Return, 60% Win Rate, 16.8% DD
Creator Attribution: The Trading School
Market: Forex (EURUSD, GBPUSD) / Universal MT5
Timeframe: M15 Execution (Swing Swings on M15/H1)
Indicators: Fibonacci Retracement Levels (0.50, 0.618, 0.786)
Logic:
  - Concept:
      90% of retail traders enter too early at 0.382 or 0.50 and get stopped out.
      Institutions enter at the "Golden Phantom Zone" between 0.618 and 0.786.
  - Buy Setup:
      1. Major Swing: Price made a clear Swing Low to Swing High move.
      2. Deep Pullback: Price retraces deeply into the 0.618 - 0.786 Golden Zone.
      3. Confirmation Reclaim: Price rejects the golden zone and closes a candle back ABOVE 0.50 level.
      4. Trigger: Confirmed candle close above 0.50 Fibonacci level.
      5. Stop Loss: Placed below the 0.786 Fibonacci level (structural invalidation).
      6. Take Profit: 1:2 Risk-to-Reward Ratio.
  - Sell Setup (Inverted):
      1. Major Swing: Price made a clear Swing High to Swing Low move.
      2. Deep Pullback: Price retraces deeply up into 0.618 - 0.786 Golden Zone.
      3. Confirmation Reclaim: Price rejects the golden zone and closes a candle back BELOW 0.50 level.
      4. Trigger: Confirmed candle close below 0.50 Fibonacci level.
      5. Stop Loss: Placed above the 0.786 Fibonacci level.
      6. Take Profit: 1:2 Risk-to-Reward Ratio.
========================================================================================
NOTE: SIGNAL ONLY - No trade execution or live orders are placed.
========================================================================================
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import time
import datetime
import pandas as pd
import numpy as np
import MetaTrader5 as mt5

# ==========================================
# 1. CONFIGURATION & STRATEGY SETTINGS
# ==========================================
DEFAULT_SYMBOLS = ["EURUSD.0", "EURUSD", "EURUSDm", "EURUSD.pro", "GBPUSD", "XAUUSD"]
TIMEFRAME = mt5.TIMEFRAME_M15    # 15-Minute Chart
SWING_BARS = 7                   # Pivot detection lookback
RISK_REWARD_RATIO = 2.0
POLL_INTERVAL = 5                # Seconds between poll checks
LOOKBACK_BARS = 150

ACTIVE_SYMBOL = None
PIP_SIZE = 0.0001


def resolve_symbol():
    """Auto-detects active MT5 symbol and pip scaling."""
    global ACTIVE_SYMBOL, PIP_SIZE
    for s in DEFAULT_SYMBOLS:
        info = mt5.symbol_info(s)
        if info is not None:
            if not info.visible:
                mt5.symbol_select(s, True)
            ACTIVE_SYMBOL = s
            PIP_SIZE = 0.0001 if info.digits in (4, 5) else (0.01 if info.digits in (2, 3) else 1.0)
            return ACTIVE_SYMBOL
    ACTIVE_SYMBOL = "EURUSD"
    return ACTIVE_SYMBOL


def initialize_mt5():
    """Connects to MetaTrader 5 terminal."""
    if not mt5.initialize():
        print(f"[-] MT5 Connection failed: {mt5.last_error()}")
        return False
    symbol = resolve_symbol()
    print(f"[+] Connected to MT5. Monitoring symbol: {symbol}")
    return True


def fetch_rates(count=LOOKBACK_BARS):
    """Fetches M15 candle rates."""
    rates = mt5.copy_rates_from_pos(ACTIVE_SYMBOL, TIMEFRAME, 0, count)
    if rates is None or len(rates) < 50:
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


def find_recent_swings(df: pd.DataFrame, n=SWING_BARS):
    """Finds the most recent prominent Swing High and Swing Low."""
    highs = df['high'].values
    lows = df['low'].values
    length = len(df)

    last_high_idx = None
    last_low_idx = None

    for i in range(length - n - 2, n, -1):
        if last_high_idx is None:
            if highs[i] == max(highs[i-n:i+n+1]):
                last_high_idx = i

        if last_low_idx is None:
            if lows[i] == min(lows[i-n:i+n+1]):
                last_low_idx = i

        if last_high_idx is not None and last_low_idx is not None:
            break

    return last_high_idx, last_low_idx


def emit_signal(sig_type: str, candle_time: datetime.datetime, entry_price: float, sl_price: float, tp_price: float, fib_618: float, fib_786: float):
    """Prints a structured ASCII signal card to console."""
    risk_pips = abs(entry_price - sl_price) / PIP_SIZE
    reward_pips = abs(tp_price - entry_price) / PIP_SIZE

    print("\n" + "=" * 75)
    print(f" [ALERT] MT5 SIGNAL ALERT: FIBONACCI GOLDEN ZONE ({sig_type.upper()})")
    print("=" * 75)
    print(f" Symbol:        {ACTIVE_SYMBOL} | Timeframe: M15")
    print(f" Bar Time:      {candle_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Signal Type:   {sig_type.upper()} (0.618 - 0.786 Golden Retracement Reclaim)")
    print(f" Golden Zone:   {fib_618:.5f} - {fib_786:.5f}")
    print(f" Entry Price:   {entry_price:.5f}")
    print(f" Stop Loss:     {sl_price:.5f}  ({risk_pips:.1f} pips)")
    print(f" Take Profit:   {tp_price:.5f}  ({reward_pips:.1f} pips)")
    print(f" Risk/Reward:   1:{RISK_REWARD_RATIO:.1f}")
    print(f" Status:        SIGNAL ONLY (Ready for manual or external execution)")
    print("=" * 75 + "\n")


def run_scanner():
    """Main live market scanner loop."""
    if not initialize_mt5():
        return

    print("\n" + "=" * 75)
    print(f"   FIBONACCI GOLDEN PHANTOM ZONE SCANNER ({ACTIVE_SYMBOL} M15)   ")
    print("=" * 75)
    print(" Strategy: 0.618 - 0.786 Institutional Retracement + 0.50 Reclaim")
    print(" Status: Scanning live bars... Press Ctrl+C to stop.\n")

    last_processed_candle_time = None

    try:
        while True:
            df = fetch_rates()
            if df is None:
                time.sleep(POLL_INTERVAL)
                continue

            current_candle_time = df['time'].iloc[-1]
            closed_bar = df.iloc[-2]
            prev_bar = df.iloc[-3]

            if last_processed_candle_time != current_candle_time:
                last_processed_candle_time = current_candle_time

                high_idx, low_idx = find_recent_swings(df)
                if high_idx is None or low_idx is None:
                    time.sleep(POLL_INTERVAL)
                    continue

                swing_high = df['high'].iloc[high_idx]
                swing_low = df['low'].iloc[low_idx]
                swing_range = swing_high - swing_low

                if swing_range < (10 * PIP_SIZE):
                    time.sleep(POLL_INTERVAL)
                    continue

                # ----------------------------------------------------
                # BUY LOGIC: Low happened before High (Bullish impulse)
                # ----------------------------------------------------
                if low_idx < high_idx:
                    fib_500 = swing_high - (0.500 * swing_range)
                    fib_618 = swing_high - (0.618 * swing_range)
                    fib_786 = swing_high - (0.786 * swing_range)

                    # Check if pullback reached the golden zone (between 0.618 and 0.786)
                    recent_low_after_high = df['low'].iloc[high_idx:].min()
                    in_golden_zone = (recent_low_after_high <= fib_618) and (recent_low_after_high >= fib_786 - (3 * PIP_SIZE))

                    # Reclaim trigger: closed_bar crosses back above 0.50 level
                    reclaimed_50 = (prev_bar['close'] <= fib_500) and (closed_bar['close'] > fib_500)

                    if in_golden_zone and reclaimed_50:
                        entry = closed_bar['close']
                        sl = fib_786 - (2.0 * PIP_SIZE)
                        risk = entry - sl
                        if risk > 0:
                            tp = entry + (risk * RISK_REWARD_RATIO)
                            emit_signal("BUY", current_candle_time, entry, sl, tp, fib_618, fib_786)

                # ----------------------------------------------------
                # SELL LOGIC: High happened before Low (Bearish impulse)
                # ----------------------------------------------------
                elif high_idx < low_idx:
                    fib_500 = swing_low + (0.500 * swing_range)
                    fib_618 = swing_low + (0.618 * swing_range)
                    fib_786 = swing_low + (0.786 * swing_range)

                    recent_high_after_low = df['high'].iloc[low_idx:].max()
                    in_golden_zone = (recent_high_after_low >= fib_618) and (recent_high_after_low <= fib_786 + (3 * PIP_SIZE))

                    # Reclaim trigger: closed_bar crosses back below 0.50 level
                    reclaimed_50 = (prev_bar['close'] >= fib_500) and (closed_bar['close'] < fib_500)

                    if in_golden_zone and reclaimed_50:
                        entry = closed_bar['close']
                        sl = fib_786 + (2.0 * PIP_SIZE)
                        risk = sl - entry
                        if risk > 0:
                            tp = entry - (risk * RISK_REWARD_RATIO)
                            emit_signal("SELL", current_candle_time, entry, sl, tp, fib_618, fib_786)

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n[STOP] Scanner stopped by user.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    run_scanner()
