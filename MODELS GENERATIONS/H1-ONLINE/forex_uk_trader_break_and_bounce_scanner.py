"""
Strategy 69: UK 20-Year Pro Trader Break and Bounce Scanner
Creator Video ID: ulkbJJzPHzE (71.0% Win Rate, $10,000/Month Proven Model)
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
        "strategy": "UK Trader Break and Bounce",
        "symbol": symbol,
        "signal": "NEUTRAL",
        "timeframe": "M15",
        "entry": None,
        "sl": None,
        "tp": None,
        "rr": 2.0,
        "details": ""
    }

    if not mt5:
        res["details"] = "MetaTrader5 library not installed."
        return res

    sym = resolve_symbol(symbol)
    if not mt5.symbol_select(sym, True):
        res["details"] = f"Failed to select symbol {sym}"
        return res

    df = get_rates(sym, mt5.TIMEFRAME_M15, 90)
    if df is None or len(df) < 50:
        res["details"] = "Insufficient M15 rates"
        return res

    # Establish structural resistance & support (excluding last 6 bars)
    ref_bars = df.iloc[-45:-6]
    resistance = float(ref_bars["high"].max())
    support = float(ref_bars["low"].min())

    curr = df.iloc[-1]
    prev1 = df.iloc[-2]
    prev2 = df.iloc[-3]
    prev3 = df.iloc[-4]

    spread = mt5.symbol_info(sym).spread * mt5.symbol_info(sym).point

    # Bullish Break and Bounce:
    # 1. One of the previous 3 candles broke cleanly above Resistance
    # 2. Current or prev candle dipped back to test Resistance (within 0.15%)
    # 3. Current candle bounced and closed green above Resistance
    breakout_above = (prev3["close"] > resistance or prev2["close"] > resistance or prev1["close"] > resistance)
    retested_resistance = curr["low"] <= resistance * 1.0015
    bounced_green = curr["close"] > resistance and curr["close"] > curr["open"]

    if breakout_above and retested_resistance and bounced_green:
        entry = float(curr["close"])
        sl = float(min(curr["low"], prev1["low"], resistance) - (spread * 2) - 0.0003)
        risk = entry - sl
        if risk > 0:
            tp = round(entry + (risk * 2.2), 5)
            rr = round((tp - entry) / risk, 2)
            res.update({
                "signal": "BUY",
                "entry": round(entry, 5),
                "sl": round(sl, 5),
                "tp": tp,
                "rr": rr,
                "details": f"Break and Bounce Bullish: Resistance {resistance:.5f} broken & cleanly bounced. Retest Low: {curr['low']:.5f}. Target: {tp:.5f}"
            })
            return res

    # Bearish Break and Bounce:
    breakout_below = (prev3["close"] < support or prev2["close"] < support or prev1["close"] < support)
    retested_support = curr["high"] >= support * 0.9985
    bounced_red = curr["close"] < support and curr["close"] < curr["open"]

    if breakout_below and retested_support and bounced_red:
        entry = float(curr["close"])
        sl = float(max(curr["high"], prev1["high"], support) + (spread * 2) + 0.0003)
        risk = sl - entry
        if risk > 0:
            tp = round(entry - (risk * 2.2), 5)
            rr = round((entry - tp) / risk, 2)
            res.update({
                "signal": "SELL",
                "entry": round(entry, 5),
                "sl": round(sl, 5),
                "tp": tp,
                "rr": rr,
                "details": f"Break and Bounce Bearish: Support {support:.5f} broken & cleanly bounced downward. Retest High: {curr['high']:.5f}. Target: {tp:.5f}"
            })
            return res

    res["details"] = f"M15 S/R: Support={support:.5f}, Resistance={resistance:.5f}. Current={curr['close']:.5f}. Awaiting clean break and bounce."
    return res


def main():
    parser = argparse.ArgumentParser(description="UK Trader Break and Bounce Scanner")
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
        print(f"[{now_str}] UK Trader Break and Bounce Scanner -> {res['symbol']}")
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
