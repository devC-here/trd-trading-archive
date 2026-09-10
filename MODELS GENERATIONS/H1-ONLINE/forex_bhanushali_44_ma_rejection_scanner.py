"""
Strategy 66: Siddharth Bhanushali 44 Moving Average Dynamic Swing Scanner
Creator Video ID: EIF1PHduEbo (51.0% Win Rate, 44 SMA Rejection Strategy)
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
        "strategy": "Siddharth Bhanushali 44 MA Rejection",
        "symbol": symbol,
        "signal": "NEUTRAL",
        "timeframe": "D1 (Swing)",
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

    df = get_rates(sym, mt5.TIMEFRAME_D1, 100)
    if df is None or len(df) < 55:
        res["details"] = "Insufficient D1 rates"
        return res

    df["sma44"] = df["close"].rolling(44).mean()

    curr = df.iloc[-1]
    prev1 = df.iloc[-2]
    prev2 = df.iloc[-3]

    sma44 = float(curr["sma44"])
    sma44_prev = float(prev1["sma44"])
    spread = mt5.symbol_info(sym).spread * mt5.symbol_info(sym).point

    # Bullish: 44 SMA sloping up + price tested 44 SMA + green candle breaks previous high
    sma_slope_up = sma44 > sma44_prev
    tested_sma_bull = curr["low"] <= sma44 * 1.003 or prev1["low"] <= sma44 * 1.003
    bull_trigger = curr["close"] > curr["open"] and curr["close"] > prev1["high"]

    if sma_slope_up and tested_sma_bull and bull_trigger:
        entry = float(curr["close"])
        sl = float(min(curr["low"], prev1["low"], sma44) - (spread * 2) - 0.0005)
        risk = entry - sl
        if risk > 0:
            tp = round(entry + (risk * 2.0), 5)
            res.update({
                "signal": "BUY",
                "entry": round(entry, 5),
                "sl": round(sl, 5),
                "tp": tp,
                "rr": 2.0,
                "details": f"Bhanushali 44 SMA Bullish Rejection: Rising 44 SMA ({sma44:.5f}) tested with green trigger breaking previous high ({prev1['high']:.5f}). Target 1:2 RR."
            })
            return res

    # Bearish: 44 SMA sloping down + price tested 44 SMA + red candle breaks previous low
    sma_slope_down = sma44 < sma44_prev
    tested_sma_bear = curr["high"] >= sma44 * 0.997 or prev1["high"] >= sma44 * 0.997
    bear_trigger = curr["close"] < curr["open"] and curr["close"] < prev1["low"]

    if sma_slope_down and tested_sma_bear and bear_trigger:
        entry = float(curr["close"])
        sl = float(max(curr["high"], prev1["high"], sma44) + (spread * 2) + 0.0005)
        risk = sl - entry
        if risk > 0:
            tp = round(entry - (risk * 2.0), 5)
            res.update({
                "signal": "SELL",
                "entry": round(entry, 5),
                "sl": round(sl, 5),
                "tp": tp,
                "rr": 2.0,
                "details": f"Bhanushali 44 SMA Bearish Rejection: Falling 44 SMA ({sma44:.5f}) tested with red trigger breaking previous low ({prev1['low']:.5f}). Target 1:2 RR."
            })
            return res

    slope_str = "Rising" if sma_slope_up else "Falling"
    res["details"] = f"44 SMA={sma44:.5f} ({slope_str}). Current Price={curr['close']:.5f}. Distance={abs(curr['close'] - sma44):.5f}."
    return res


def main():
    parser = argparse.ArgumentParser(description="Siddharth Bhanushali 44 MA Scanner")
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
        print(f"[{now_str}] Bhanushali 44 MA Scanner -> {res['symbol']}")
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
