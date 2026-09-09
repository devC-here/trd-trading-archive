"""
Strategy 48: TG Capital (Tyler Gottle) EMA Stack + FVG 50% Doji Rejection Scanner
Creator Video ID: ZWSeWYom1Yg (+68.4% Annual Return, 76% Win Rate, 9.2% Max DD)
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


def get_rates(symbol, timeframe, n_bars=250):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


def is_doji(candle):
    body = abs(candle["close"] - candle["open"])
    candle_range = candle["high"] - candle["low"]
    if candle_range <= 0:
        return False
    return (body / candle_range) <= 0.28


def scan_symbol(symbol="EURUSD"):
    res = {
        "strategy": "TG Capital EMA Stack + FVG 50% Doji",
        "symbol": symbol,
        "signal": "NEUTRAL",
        "timeframe": "M30",
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

    df = get_rates(sym, mt5.TIMEFRAME_M30, 260)
    if df is None or len(df) < 220:
        res["details"] = "Insufficient M30 rates"
        return res

    # Calculate EMAs: 5, 9, 13, 21, 200
    df["ema5"] = df["close"].ewm(span=5, adjust=False).mean()
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema13"] = df["close"].ewm(span=13, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    curr = df.iloc[-1]
    prev1 = df.iloc[-2]
    prev2 = df.iloc[-3]
    prev3 = df.iloc[-4]

    # Check EMA Stacking
    bullish_stack = (
        curr["close"] > curr["ema200"] and
        curr["ema5"] > curr["ema9"] > curr["ema13"] > curr["ema21"] > curr["ema200"]
    )
    bearish_stack = (
        curr["close"] < curr["ema200"] and
        curr["ema5"] < curr["ema9"] < curr["ema13"] < curr["ema21"] < curr["ema200"]
    )

    spread = mt5.symbol_info(sym).spread * mt5.symbol_info(sym).point

    # Scan last 10 candles for a recent FVG
    if bullish_stack:
        for i in range(len(df) - 8, len(df) - 2):
            c1 = df.iloc[i - 2]
            c2 = df.iloc[i - 1]
            c3 = df.iloc[i]
            # Bullish FVG condition: c3 low > c1 high
            if c3["low"] > c1["high"]:
                fvg_top = c3["low"]
                fvg_bottom = c1["high"]
                fvg_mid = (fvg_top + fvg_bottom) / 2.0
                mid_candle_low = c2["low"]

                # Check if prev1 or curr tested the 50% level with a Doji
                for check_candle in [prev2, prev1, curr]:
                    if check_candle["low"] <= fvg_mid <= check_candle["high"] and is_doji(check_candle):
                        if curr["close"] > fvg_mid and curr["close"] > curr["open"]:
                            entry = float(curr["close"])
                            sl = float(mid_candle_low - (spread * 2) - 0.0003)
                            risk = entry - sl
                            if risk > 0:
                                tp = round(entry + (risk * 3.5), 5)
                                rr = round((tp - entry) / risk, 2)
                                res.update({
                                    "signal": "BUY",
                                    "entry": round(entry, 5),
                                    "sl": round(sl, 5),
                                    "tp": tp,
                                    "rr": rr,
                                    "details": f"Bullish EMA Stack (5>9>13>21>200) + FVG 50% [{fvg_mid:.5f}] Doji rejection. SL below middle candle {mid_candle_low:.5f}"
                                })
                                return res

    elif bearish_stack:
        for i in range(len(df) - 8, len(df) - 2):
            c1 = df.iloc[i - 2]
            c2 = df.iloc[i - 1]
            c3 = df.iloc[i]
            # Bearish FVG condition: c3 high < c1 low
            if c3["high"] < c1["low"]:
                fvg_top = c1["low"]
                fvg_bottom = c3["high"]
                fvg_mid = (fvg_top + fvg_bottom) / 2.0
                mid_candle_high = c2["high"]

                for check_candle in [prev2, prev1, curr]:
                    if check_candle["low"] <= fvg_mid <= check_candle["high"] and is_doji(check_candle):
                        if curr["close"] < fvg_mid and curr["close"] < curr["open"]:
                            entry = float(curr["close"])
                            sl = float(mid_candle_high + (spread * 2) + 0.0003)
                            risk = sl - entry
                            if risk > 0:
                                tp = round(entry - (risk * 3.5), 5)
                                rr = round((entry - tp) / risk, 2)
                                res.update({
                                    "signal": "SELL",
                                    "entry": round(entry, 5),
                                    "sl": round(sl, 5),
                                    "tp": tp,
                                    "rr": rr,
                                    "details": f"Bearish EMA Stack (5<9<13<21<200) + FVG 50% [{fvg_mid:.5f}] Doji rejection. SL above middle candle {mid_candle_high:.5f}"
                                })
                                return res

    status_str = "Bullish Stack" if bullish_stack else ("Bearish Stack" if bearish_stack else "EMAs not aligned")
    res["details"] = f"{status_str} (EMA5={curr['ema5']:.5f}, EMA200={curr['ema200']:.5f}), awaiting FVG 50% Doji touch."
    return res


def main():
    parser = argparse.ArgumentParser(description="TG Capital EMA Stack + FVG Doji Scanner")
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
        print(f"[{now_str}] TG Capital EMA Stack Doji Scanner -> {res['symbol']}")
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
