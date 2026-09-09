"""
Strategy 68: LuxAlgo Confirmation Trend & Baseline Momentum Scanner
Creator Video ID: x-7CInJes_s (55.0% Win Rate, LuxAlgo Proprietary System)
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


def calculate_lux_oscillator(df, length=14):
    diff = df["close"] - df["close"].shift(length)
    vol = df["high"] - df["low"]
    norm = diff / (vol.rolling(length).mean() + 1e-10)
    return norm.ewm(span=5, adjust=False).mean()


def scan_symbol(symbol="EURUSD"):
    res = {
        "strategy": "LuxAlgo Confirmation Trend",
        "symbol": symbol,
        "signal": "NEUTRAL",
        "timeframe": "H1",
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

    df = get_rates(sym, mt5.TIMEFRAME_H1, 120)
    if df is None or len(df) < 60:
        res["details"] = "Insufficient H1 rates"
        return res

    df["baseline"] = df["close"].ewm(span=50, adjust=False).mean()
    df["osc"] = calculate_lux_oscillator(df, 14)

    curr = df.iloc[-1]
    prev1 = df.iloc[-2]
    prev2 = df.iloc[-3]

    spread = mt5.symbol_info(sym).spread * mt5.symbol_info(sym).point
    recent_swing_low = float(df.iloc[-15:-1]["low"].min())
    recent_swing_high = float(df.iloc[-15:-1]["high"].max())

    # Bullish: Price > Baseline (50 EMA) + Oscillator crossed above 0 + Green candle
    bull_baseline = curr["close"] > curr["baseline"]
    osc_cross_up = (prev1["osc"] <= 0 and curr["osc"] > 0) or (prev2["osc"] <= 0 and prev1["osc"] > 0)

    if bull_baseline and osc_cross_up and curr["close"] > curr["open"]:
        entry = float(curr["close"])
        sl = float(min(recent_swing_low, curr["baseline"]) - (spread * 2) - 0.0003)
        risk = entry - sl
        if risk > 0:
            tp = round(entry + (risk * 2.0), 5)
            res.update({
                "signal": "BUY",
                "entry": round(entry, 5),
                "sl": round(sl, 5),
                "tp": tp,
                "rr": 2.0,
                "details": f"LuxAlgo Bullish Confirmation: Price > Baseline ({curr['baseline']:.5f}), Confirmation Oscillator flipped positive ({curr['osc']:.2f}). Target 1:2 RR: {tp:.5f}"
            })
            return res

    # Bearish: Price < Baseline + Oscillator crossed below 0 + Red candle
    bear_baseline = curr["close"] < curr["baseline"]
    osc_cross_down = (prev1["osc"] >= 0 and curr["osc"] < 0) or (prev2["osc"] >= 0 and prev1["osc"] < 0)

    if bear_baseline and osc_cross_down and curr["close"] < curr["open"]:
        entry = float(curr["close"])
        sl = float(max(recent_swing_high, curr["baseline"]) + (spread * 2) + 0.0003)
        risk = sl - entry
        if risk > 0:
            tp = round(entry - (risk * 2.0), 5)
            res.update({
                "signal": "SELL",
                "entry": round(entry, 5),
                "sl": round(sl, 5),
                "tp": tp,
                "rr": 2.0,
                "details": f"LuxAlgo Bearish Confirmation: Price < Baseline ({curr['baseline']:.5f}), Confirmation Oscillator flipped negative ({curr['osc']:.2f}). Target 1:2 RR: {tp:.5f}"
            })
            return res

    res["details"] = f"Baseline={curr['baseline']:.5f}, Osc={curr['osc']:.2f}. Price={curr['close']:.5f}. Waiting for synchronized baseline & oscillator cross."
    return res


def main():
    parser = argparse.ArgumentParser(description="LuxAlgo Confirmation Trend Scanner")
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
        print(f"[{now_str}] LuxAlgo Confirmation Scanner -> {res['symbol']}")
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
