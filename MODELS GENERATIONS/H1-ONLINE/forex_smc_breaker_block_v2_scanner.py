"""
Strategy 64: SMC Failed Order Block Breaker Retest Scanner
Creator Video ID: AkoRQx8SDLc (+44.8% Annual Return, 61.0% Win Rate, 13.8% Max DD)
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
        "strategy": "SMC Breaker Block Retest",
        "symbol": symbol,
        "signal": "NEUTRAL",
        "timeframe": "M15",
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

    df = get_rates(sym, mt5.TIMEFRAME_M15, 100)
    if df is None or len(df) < 50:
        res["details"] = "Insufficient M15 rates"
        return res

    curr = df.iloc[-1]
    prev1 = df.iloc[-2]
    spread = mt5.symbol_info(sym).spread * mt5.symbol_info(sym).point

    recent_swing_high = float(df.iloc[-30:-3]["high"].max())
    recent_swing_low = float(df.iloc[-30:-3]["low"].min())

    # Look for Breaker Block in bars -20 to -4
    for i in range(len(df) - 18, len(df) - 3):
        candle = df.iloc[i]

        # Bullish Breaker: Bearish candle whose high was aggressively broken upward with displacement
        if candle["close"] < candle["open"]:
            breaker_top = float(candle["open"])
            breaker_bottom = float(candle["low"])

            # Check if subsequent candles broke cleanly above breaker_top
            displacement_bars = df.iloc[i + 1:len(df) - 2]
            if len(displacement_bars) > 0 and displacement_bars["close"].max() > breaker_top * 1.001:
                # Retest: current or prev1 bar dipping into [breaker_bottom, breaker_top]
                if breaker_bottom <= curr["low"] <= breaker_top * 1.0005 and curr["close"] >= breaker_bottom:
                    if curr["close"] > curr["open"]:
                        entry = float(curr["close"])
                        sl = float(breaker_bottom - (spread * 2) - 0.0003)
                        risk = entry - sl
                        if risk > 0:
                            tp = max(entry + (risk * 2.2), recent_swing_high)
                            rr = round((tp - entry) / risk, 2)
                            res.update({
                                "signal": "BUY",
                                "entry": round(entry, 5),
                                "sl": round(sl, 5),
                                "tp": round(tp, 5),
                                "rr": rr,
                                "details": f"Bullish Breaker Block [{breaker_bottom:.5f} - {breaker_top:.5f}] Retest. Target Liquidity High: {tp:.5f}"
                            })
                            return res

        # Bearish Breaker: Bullish candle whose low was aggressively broken downward
        if candle["close"] > candle["open"]:
            breaker_top = float(candle["high"])
            breaker_bottom = float(candle["open"])

            displacement_bars = df.iloc[i + 1:len(df) - 2]
            if len(displacement_bars) > 0 and displacement_bars["close"].min() < breaker_bottom * 0.999:
                if breaker_bottom * 0.9995 <= curr["high"] <= breaker_top and curr["close"] <= breaker_top:
                    if curr["close"] < curr["open"]:
                        entry = float(curr["close"])
                        sl = float(breaker_top + (spread * 2) + 0.0003)
                        risk = sl - entry
                        if risk > 0:
                            tp = min(entry - (risk * 2.2), recent_swing_low)
                            rr = round((entry - tp) / risk, 2)
                            res.update({
                                "signal": "SELL",
                                "entry": round(entry, 5),
                                "sl": round(sl, 5),
                                "tp": round(tp, 5),
                                "rr": rr,
                                "details": f"Bearish Breaker Block [{breaker_bottom:.5f} - {breaker_top:.5f}] Retest. Target Liquidity Low: {tp:.5f}"
                            })
                            return res

    res["details"] = f"Price={curr['close']:.5f}. Range=[{recent_swing_low:.5f} - {recent_swing_high:.5f}]. Awaiting active breaker block retest."
    return res


def main():
    parser = argparse.ArgumentParser(description="SMC Breaker Block Retest Scanner")
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
        print(f"[{now_str}] SMC Breaker Block Scanner -> {res['symbol']}")
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
