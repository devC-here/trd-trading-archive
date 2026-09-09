"""
Strategy 56: Bill Williams Alligator Momentum & Teeth Cross Scanner
Creator Video ID: tW3pNDWhKHE (+28.4% Annual Return, 54.0% Win Rate, 16.8% Max DD)
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


def get_rates(symbol, timeframe, n_bars=150):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


def calculate_alligator(df):
    median_price = (df["high"] + df["low"]) / 2.0
    # Standard Bill Williams Smoothed MA (SMMA) or Wilder's Smoothing
    jaw = median_price.ewm(alpha=1.0 / 13, min_periods=13, adjust=False).mean()
    teeth = median_price.ewm(alpha=1.0 / 8, min_periods=8, adjust=False).mean()
    lips = median_price.ewm(alpha=1.0 / 5, min_periods=5, adjust=False).mean()
    return jaw, teeth, lips


def scan_symbol(symbol="EURUSD"):
    res = {
        "strategy": "Bill Williams Alligator Teeth Cross",
        "symbol": symbol,
        "signal": "NEUTRAL",
        "timeframe": "H1",
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

    df = get_rates(sym, mt5.TIMEFRAME_H1, 100)
    if df is None or len(df) < 50:
        res["details"] = "Insufficient H1 rates"
        return res

    jaw, teeth, lips = calculate_alligator(df)
    df["jaw"] = jaw      # Blue
    df["teeth"] = teeth  # Red
    df["lips"] = lips    # Green

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    spread = mt5.symbol_info(sym).spread * mt5.symbol_info(sym).point

    # Bullish Alignment: Lips > Teeth > Jaw
    # Trigger: Price dips to or crosses above Teeth
    alligator_awake_bull = curr["lips"] > curr["teeth"] and curr["teeth"] > curr["jaw"]
    bull_cross = (prev["close"] <= prev["teeth"] and curr["close"] > curr["teeth"]) or \
                 (prev2["close"] <= prev2["teeth"] and prev["close"] > prev["teeth"] and curr["close"] > curr["teeth"])

    if alligator_awake_bull and bull_cross and curr["close"] > curr["open"]:
        entry = float(curr["close"])
        sl = float(curr["jaw"] - (spread * 2) - 0.0003)
        risk = entry - sl
        if risk > 0:
            tp = round(entry + (risk * 2.0), 5)
            rr = round((tp - entry) / risk, 2)
            res.update({
                "signal": "BUY",
                "entry": round(entry, 5),
                "sl": round(sl, 5),
                "tp": tp,
                "rr": rr,
                "details": f"Alligator Bullish: Lips ({curr['lips']:.5f}) > Teeth ({curr['teeth']:.5f}) > Jaw ({curr['jaw']:.5f}). Candle reclaimed above Teeth. SL at Jaw."
            })
            return res

    # Bearish Alignment: Lips < Teeth < Jaw
    alligator_awake_bear = curr["lips"] < curr["teeth"] and curr["teeth"] < curr["jaw"]
    bear_cross = (prev["close"] >= prev["teeth"] and curr["close"] < curr["teeth"]) or \
                 (prev2["close"] >= prev2["teeth"] and prev["close"] < prev["teeth"] and curr["close"] < curr["teeth"])

    if alligator_awake_bear and bear_cross and curr["close"] < curr["open"]:
        entry = float(curr["close"])
        sl = float(curr["jaw"] + (spread * 2) + 0.0003)
        risk = sl - entry
        if risk > 0:
            tp = round(entry - (risk * 2.0), 5)
            rr = round((entry - tp) / risk, 2)
            res.update({
                "signal": "SELL",
                "entry": round(entry, 5),
                "sl": round(sl, 5),
                "tp": tp,
                "rr": rr,
                "details": f"Alligator Bearish: Lips ({curr['lips']:.5f}) < Teeth ({curr['teeth']:.5f}) < Jaw ({curr['jaw']:.5f}). Candle broke below Teeth. SL at Jaw."
            })
            return res

    status = "Awake Bullish" if alligator_awake_bull else ("Awake Bearish" if alligator_awake_bear else "Sleeping (Consolidation)")
    res["details"] = f"Alligator Status: {status}. Lips={curr['lips']:.5f}, Teeth={curr['teeth']:.5f}, Jaw={curr['jaw']:.5f}. Waiting for Teeth cross."
    return res


def main():
    parser = argparse.ArgumentParser(description="Bill Williams Alligator Scanner")
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
        print(f"[{now_str}] Bill Williams Alligator Scanner -> {res['symbol']}")
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
