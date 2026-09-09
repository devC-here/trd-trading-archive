"""
Strategy 65: Steven Hart 4H Market Structure + H1 Pullback Breakout Scanner
Creator Video ID: Em-Ao3orPbU (66.4% Win Rate, Steven Hart - 2.87M subs Strategy)
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


def get_rates(symbol, timeframe, n_bars=100):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


def scan_symbol(symbol="EURUSD"):
    res = {
        "strategy": "Steven Hart 4H Trend Pullback",
        "symbol": symbol,
        "signal": "NEUTRAL",
        "timeframe": "H4 (Macro Zone) + H1 (Pullback Trigger)",
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

    # 1. Fetch H4 Data
    df_h4 = get_rates(sym, mt5.TIMEFRAME_H4, 60)
    if df_h4 is None or len(df_h4) < 30:
        res["details"] = "Insufficient H4 rates"
        return res

    # Detect H4 Trend
    h4_now = df_h4.iloc[-1]
    h4_swing_high = float(df_h4.iloc[-25:-5]["high"].max())
    h4_swing_low = float(df_h4.iloc[-25:-5]["low"].min())
    h4_prev_break_high = float(df_h4.iloc[-15:-3]["high"].max())
    h4_prev_break_low = float(df_h4.iloc[-15:-3]["low"].min())

    # 2. Fetch H1 Data
    df_h1 = get_rates(sym, mt5.TIMEFRAME_H1, 70)
    if df_h1 is None or len(df_h1) < 35:
        res["details"] = "Insufficient H1 rates"
        return res

    h1_now = df_h1.iloc[-1]
    h1_prev = df_h1.iloc[-2]
    spread = mt5.symbol_info(sym).spread * mt5.symbol_info(sym).point

    # Bullish Trend: H4 making higher highs, price tested previous broken swing high zone
    h4_bull = h4_now["close"] > h4_swing_low
    tested_support_zone = abs(h1_now["low"] - h4_swing_high) / h4_swing_high < 0.0035 or h1_now["low"] <= h4_swing_high

    if h4_bull and tested_support_zone and h1_now["close"] > h1_now["open"] and h1_now["close"] >= h4_swing_high:
        entry = float(h1_now["close"])
        sl = float(min(h1_now["low"], h1_prev["low"]) - (spread * 2) - 0.0003)
        risk = entry - sl
        if risk > 0:
            tp = max(entry + (risk * 2.2), float(df_h4.iloc[-15:]["high"].max()))
            rr = round((tp - entry) / risk, 2)
            res.update({
                "signal": "BUY",
                "entry": round(entry, 5),
                "sl": round(sl, 5),
                "tp": round(tp, 5),
                "rr": rr,
                "details": f"Steven Hart 4H Bullish Trend: Support Zone {h4_swing_high:.5f} tested on H1 with green rejection candle. Target H4 High: {tp:.5f}"
            })
            return res

    # Bearish Trend: H4 making lower lows, price tested previous broken swing low zone
    h4_bear = h4_now["close"] < h4_swing_high
    tested_resist_zone = abs(h1_now["high"] - h4_swing_low) / h4_swing_low < 0.0035 or h1_now["high"] >= h4_swing_low

    if h4_bear and tested_resist_zone and h1_now["close"] < h1_now["open"] and h1_now["close"] <= h4_swing_low:
        entry = float(h1_now["close"])
        sl = float(max(h1_now["high"], h1_prev["high"]) + (spread * 2) + 0.0003)
        risk = sl - entry
        if risk > 0:
            tp = min(entry - (risk * 2.2), float(df_h4.iloc[-15:]["low"].min()))
            rr = round((entry - tp) / risk, 2)
            res.update({
                "signal": "SELL",
                "entry": round(entry, 5),
                "sl": round(sl, 5),
                "tp": round(tp, 5),
                "rr": rr,
                "details": f"Steven Hart 4H Bearish Trend: Resistance Zone {h4_swing_low:.5f} tested on H1 with red rejection candle. Target H4 Low: {tp:.5f}"
            })
            return res

    res["details"] = f"H4 Levels: High={h4_swing_high:.5f}, Low={h4_swing_low:.5f}. Current H1 Price={h1_now['close']:.5f}. Waiting for zone touch & rejection."
    return res


def main():
    parser = argparse.ArgumentParser(description="Steven Hart 4H Trend Pullback Scanner")
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
        print(f"[{now_str}] Steven Hart 4H Trend Pullback Scanner -> {res['symbol']}")
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
