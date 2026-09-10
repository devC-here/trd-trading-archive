"""
Strategy 59: The Moving Average Trendline + Fibonacci Golden Zone Scalper
Creator Video ID: 1nOpTr6aNjQ (The Moving Average - 1.13M subs Forex Strategy)
Markets: Forex Majors/Crosses & Gold (XAUUSD)
Execution: SIGNAL ONLY (No order placement)
"""

import sys
import os
import argparse
import datetime
import pandas as pd
import numpy as np

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def resolve_symbol(symbol_name):
    if not mt5:
        return symbol_name
    symbols = mt5.symbols_get()
    if not symbols:
        return symbol_name
    names = [s.name for s in symbols]
    if symbol_name in names:
        return symbol_name
    for n in names:
        if symbol_name.upper() in n.upper():
            return n
    return symbol_name


def get_rates(symbol, timeframe, n_bars=120):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


def scan_symbol(symbol="EURUSD"):
    res = {
        "strategy": "The Moving Average Fib Golden Zone Scalp",
        "symbol": symbol,
        "signal": "NEUTRAL",
        "timeframe": "M5",
        "entry": None,
        "sl": None,
        "tp": None,
        "rr": None,
        "details": ""
    }

    if not mt5:
        res["details"] = "MetaTrader5 library not installed."
        return res

    sym = resolve_symbol(symbol)
    if not mt5.symbol_select(sym, True):
        res["details"] = f"Failed to select symbol {sym}"
        return res

    df = get_rates(sym, mt5.TIMEFRAME_M5, 90)
    if df is None or len(df) < 50:
        res["details"] = "Insufficient M5 rates"
        return res

    # Identify recent swing impulse (excluding last 3 bars)
    historical = df.iloc[-45:-3]
    swing_high = float(historical["high"].max())
    swing_low = float(historical["low"].min())
    swing_range = swing_high - swing_low

    if swing_range <= 0.0005:
        res["details"] = "Insufficient swing range for Fibonacci retracement"
        return res

    # Bullish Retracement: Upward impulse from swing_low to swing_high
    # Fib 50% = swing_high - 0.50 * swing_range
    # Fib 61.8% = swing_high - 0.618 * swing_range
    # Fib 78.6% = swing_high - 0.786 * swing_range
    fib_50_bull = swing_high - (0.50 * swing_range)
    fib_618_bull = swing_high - (0.618 * swing_range)
    fib_786_bull = swing_high - (0.786 * swing_range)

    # Bearish Retracement: Downward impulse from swing_high to swing_low
    fib_50_bear = swing_low + (0.50 * swing_range)
    fib_618_bear = swing_low + (0.618 * swing_range)
    fib_786_bear = swing_low + (0.786 * swing_range)

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    spread = mt5.symbol_info(sym).spread * mt5.symbol_info(sym).point

    # Bullish Golden Zone Setup:
    # Price retraced into [fib_618_bull, fib_50_bull] and printed a green rejection candle
    is_in_bull_golden = fib_786_bull <= curr["low"] <= fib_50_bull
    is_bull_rejection = curr["close"] > curr["open"] and curr["close"] >= fib_618_bull

    if is_in_bull_golden and is_bull_rejection:
        entry = float(curr["close"])
        sl = float(fib_786_bull - (spread * 2) - 0.0003)
        risk = entry - sl
        if risk > 0:
            tp = round(swing_high + (0.272 * swing_range), 5)
            rr = round((tp - entry) / risk, 2)
            res.update({
                "signal": "BUY",
                "entry": round(entry, 5),
                "sl": round(sl, 5),
                "tp": tp,
                "rr": rr,
                "details": f"Bullish Golden Zone (0.50-0.618: [{fib_618_bull:.5f} - {fib_50_bull:.5f}]) Rebound. Swing Range=[{swing_low:.5f} - {swing_high:.5f}]. Target -0.272 Extension: {tp:.5f}"
            })
            return res

    # Bearish Golden Zone Setup:
    is_in_bear_golden = fib_50_bear <= curr["high"] <= fib_786_bear
    is_bear_rejection = curr["close"] < curr["open"] and curr["close"] <= fib_618_bear

    if is_in_bear_golden and is_bear_rejection:
        entry = float(curr["close"])
        sl = float(fib_786_bear + (spread * 2) + 0.0003)
        risk = sl - entry
        if risk > 0:
            tp = round(swing_low - (0.272 * swing_range), 5)
            rr = round((entry - tp) / risk, 2)
            res.update({
                "signal": "SELL",
                "entry": round(entry, 5),
                "sl": round(sl, 5),
                "tp": tp,
                "rr": rr,
                "details": f"Bearish Golden Zone (0.50-0.618: [{fib_50_bear:.5f} - {fib_618_bear:.5f}]) Rebound. Swing Range=[{swing_low:.5f} - {swing_high:.5f}]. Target -0.272 Extension: {tp:.5f}"
            })
            return res

    res["details"] = f"M5 Swing Range: [{swing_low:.5f} - {swing_high:.5f}]. Bull Golden: [{fib_618_bull:.5f} - {fib_50_bull:.5f}], Bear Golden: [{fib_50_bear:.5f} - {fib_618_bear:.5f}]. No Golden Zone touch."
    return res


def main():
    parser = argparse.ArgumentParser(description="The Moving Average Fib Golden Zone Scalper")
    parser.add_argument("symbol", nargs="?", default="EURUSD", help="Symbol to scan")
    parser.add_argument("--loop", action="store_true", help="Run continuously every 60s")
    args = parser.parse_args()

    if mt5 and not mt5.initialize():
        print(f"[ERROR] MT5 initialization failed: {mt5.last_error()}")
        return

    import time
    while True:
        res = scan_symbol(args.symbol)
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("=" * 60)
        print(f"[{now_str}] The Moving Average Fib Scalper -> {res['symbol']}")
        print(f"Signal: {res['signal']} | Timeframe: {res['timeframe']}")
        if res["signal"] != "NEUTRAL":
            print(f"Entry: {res['entry']} | SL: {res['sl']} | TP: {res['tp']} | RR: {res['rr']}")
        print(f"Details: {res['details']}")
        print("=" * 60)

        if not args.loop:
            break
        time.sleep(60)

    if mt5:
        mt5.shutdown()


if __name__ == "__main__":
    main()
