"""
Pure Price Action Double Breakout Retest Forex Signal Scanner (MetaTrader 5)
============================================================================
Source: YouTube Shorts @UnfilteredFinanceWithSahil (Video: VhNXhA6cmDE)
Creator: Sahil Pure Price Action Strategy
Backtest Metrics: +69.0% Return in 1 Year, 48% Win Rate, 11.5% Max Drawdown
Timeframe: M15
Markets: EURUSD, GBPUSD, XAUUSD, USDJPY

Strategy Rules:
1. Support & Resistance Identification:
   - Mark Key Resistance (local swing highs) and Key Support (local swing lows).
2. The Trap Avoidance Filter:
   - When price breaks through Resistance, DO NOT ENTER immediately (prevents false breakout trap).
3. The Retest Phase:
   - Price pulls back to the broken Resistance level (polarity flip: Resistance turns into Support).
4. The Second Breakout (Trigger):
   - Price bounces from the retested level and forms a Second Breakout candle that closes
     ABOVE the high of the initial breakout candle.
   - Entry: At the close of the Second Breakout candle.
   - Stop Loss: Slightly below the retest swing low.
   - Take Profit: 1:2 to 1:3 Risk-to-Reward.
5. Reverse completely for Bearish Support Breakdowns.

SIGNAL ONLY - NO TRADE EXECUTION
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import time
import argparse
from datetime import datetime
import pandas as pd
import numpy as np

try:
    import MetaTrader5 as mt5
except ImportError:
    print("[ERROR] MetaTrader5 package not installed. Run: pip install MetaTrader5")
    sys.exit(1)


def resolve_symbol(requested_symbol: str) -> str:
    if not mt5.initialize():
        return requested_symbol
    symbols = mt5.symbols_get()
    if not symbols:
        return requested_symbol
    all_names = [s.name for s in symbols]
    if requested_symbol in all_names:
        return requested_symbol
    for s in all_names:
        if s.startswith(requested_symbol) or requested_symbol.startswith(s.split(".")[0]):
            return s
    return requested_symbol


def find_pivots(df: pd.DataFrame, window: int = 5):
    highs = df["high"]
    lows = df["low"]
    pivot_highs = []
    pivot_lows = []

    for i in range(window, len(df) - window):
        if highs.iloc[i] == highs.iloc[i - window:i + window + 1].max():
            pivot_highs.append({"idx": i, "price": float(highs.iloc[i]), "time": df["time"].iloc[i]})
        if lows.iloc[i] == lows.iloc[i - window:i + window + 1].min():
            pivot_lows.append({"idx": i, "price": float(lows.iloc[i]), "time": df["time"].iloc[i]})

    return pivot_highs, pivot_lows


def scan_symbol(symbol: str, timeframe=mt5.TIMEFRAME_M15) -> dict:
    resolved = resolve_symbol(symbol)
    if not mt5.symbol_select(resolved, True):
        return {"action": "HOLD", "reason": f"Symbol {resolved} not found"}

    rates = mt5.copy_rates_from_pos(resolved, timeframe, 0, 150)
    if rates is None or len(rates) < 60:
        return {"action": "HOLD", "reason": "Insufficient bars retrieved"}

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    pivot_highs, pivot_lows = find_pivots(df, window=5)

    bar = df.iloc[-2]       # Last closed candle
    prev_bar = df.iloc[-3]  # Candle before
    close = float(bar["close"])
    high = float(bar["high"])
    low = float(bar["low"])

    # Check for Bullish Double Breakout Retest
    for ph in reversed(pivot_highs[-6:]):
        res_level = ph["price"]
        # Find bars after the pivot high
        bars_after = df.iloc[ph["idx"] + 1:-1]
        if len(bars_after) < 6:
            continue

        # Check if an initial breakout occurred
        initial_breakouts = bars_after[bars_after["close"] > res_level]
        if len(initial_breakouts) == 0:
            continue

        first_bo_idx = initial_breakouts.index[0]
        first_bo_high = float(bars_after.loc[first_bo_idx, "high"])

        # Check for retest after first breakout
        bars_between = bars_after.loc[first_bo_idx + 1:]
        if len(bars_between) < 2:
            continue

        retested = bars_between["low"].min() <= res_level * 1.001 and bars_between["close"].min() >= res_level * 0.998
        retest_low = float(bars_between["low"].min())

        # Check if current bar is the Second Breakout closing above first_bo_high
        if retested and close > first_bo_high and prev_bar["close"] <= first_bo_high:
            sl = retest_low - (first_bo_high - retest_low) * 0.15
            risk = close - sl
            if risk <= 0:
                risk = close * 0.002
                sl = close - risk
            tp = close + 2.5 * risk
            rr = round((tp - close) / risk, 2)
            return {
                "action": "BUY",
                "symbol": resolved,
                "entry": round(close, 5),
                "sl": round(sl, 5),
                "tp": round(tp, 5),
                "rr": rr,
                "resistance_level": round(res_level, 5),
                "first_breakout_high": round(first_bo_high, 5),
                "retest_low": round(retest_low, 5),
                "reason": f"Double Breakout Retest above Resistance {round(res_level,5)} (Peak: {round(first_bo_high,5)})"
            }

    # Check for Bearish Double Breakdown Retest
    for pl in reversed(pivot_lows[-6:]):
        sup_level = pl["price"]
        bars_after = df.iloc[pl["idx"] + 1:-1]
        if len(bars_after) < 6:
            continue

        initial_breakdowns = bars_after[bars_after["close"] < sup_level]
        if len(initial_breakdowns) == 0:
            continue

        first_bd_idx = initial_breakdowns.index[0]
        first_bd_low = float(bars_after.loc[first_bd_idx, "low"])

        bars_between = bars_after.loc[first_bd_idx + 1:]
        if len(bars_between) < 2:
            continue

        retested = bars_between["high"].max() >= sup_level * 0.999 and bars_between["close"].max() <= sup_level * 1.002
        retest_high = float(bars_between["high"].max())

        if retested and close < first_bd_low and prev_bar["close"] >= first_bd_low:
            sl = retest_high + (retest_high - first_bd_low) * 0.15
            risk = sl - close
            if risk <= 0:
                risk = close * 0.002
                sl = close + risk
            tp = close - 2.5 * risk
            rr = round((close - tp) / risk, 2)
            return {
                "action": "SELL",
                "symbol": resolved,
                "entry": round(close, 5),
                "sl": round(sl, 5),
                "tp": round(tp, 5),
                "rr": rr,
                "support_level": round(sup_level, 5),
                "first_breakdown_low": round(first_bd_low, 5),
                "retest_high": round(retest_high, 5),
                "reason": f"Double Breakdown Retest below Support {round(sup_level,5)} (Valley: {round(first_bd_low,5)})"
            }

    return {
        "action": "HOLD",
        "symbol": resolved,
        "entry": round(float(df.iloc[-1]["close"]), 5),
        "pivot_highs": len(pivot_highs),
        "pivot_lows": len(pivot_lows),
        "reason": "Waiting for secondary breakout confirmation following support/resistance retest"
    }


def main():
    parser = argparse.ArgumentParser(description="Double Breakout Retest Signal Scanner")
    parser.add_argument("symbol", nargs="?", default="EURUSD.0", help="Forex Symbol")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    args = parser.parse_args()

    if not mt5.initialize():
        print(f"[ERROR] MT5 init failed: {mt5.last_error()}")
        sys.exit(1)

    print("=" * 75)
    print("  PURE PRICE ACTION DOUBLE BREAKOUT RETEST FOREX SCANNER")
    print(f"  Target Symbol: {args.symbol} | Timeframe: M15")
    print(f"  Source: Sahil Pure Price Action (VhNXhA6cmDE) | +69.0% Return, 11.5% DD")
    print("=" * 75)

    try:
        while True:
            res = scan_symbol(args.symbol)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if res["action"] in ["BUY", "SELL"]:
                print(f"\n>>> [{ts}] [{res['action']}_SIGNAL] {res['symbol']}")
                print(f"    Entry: {res['entry']} | SL: {res['sl']} | TP: {res['tp']} | R:R: 1:{res['rr']}")
                print(f"    Reason: {res['reason']}")
            else:
                print(f"[{ts}] {res.get('symbol', args.symbol)}: {res['action']} - {res['reason']}")

            if not args.loop:
                break
            time.sleep(30)
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
