"""
Strategy 55: Umar Punjabi Gold London Session 1st Hour Killzone Scanner
Creator Video ID: uXG8eHYhG9E (+44.6% Annual Return, 59.0% Win Rate, 11.8% Max DD)
Markets: Gold (XAUUSD) & Forex Majors
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


def scan_symbol(symbol="XAUUSD"):
    res = {
        "strategy": "Umar Punjabi Gold London Killzone",
        "symbol": symbol,
        "signal": "NEUTRAL",
        "timeframe": "M15 (Asian Range) + M5 (BOS Trigger)",
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

    # 1. M15 Session Rates
    df_m15 = get_rates(sym, mt5.TIMEFRAME_M15, 60)
    if df_m15 is None or len(df_m15) < 30:
        res["details"] = "Insufficient M15 rates"
        return res

    # Identify Asian consolidation (lookback 8 to 28 bars)
    asian_df = df_m15.iloc[-28:-6]
    asian_high = float(asian_df["high"].max())
    asian_low = float(asian_df["low"].min())

    # Pre-London / London test bars (recent 6 M15 bars)
    london_test = df_m15.iloc[-6:]
    high_swept = bool(london_test["high"].max() > asian_high)
    low_swept = bool(london_test["low"].min() < asian_low)

    # 2. M5 Trigger Rates
    df_m5 = get_rates(sym, mt5.TIMEFRAME_M5, 60)
    if df_m5 is None or len(df_m5) < 30:
        res["details"] = "Insufficient M5 rates"
        return res

    curr_m5 = df_m5.iloc[-1]
    prev_m5 = df_m5.iloc[-2]
    recent_m5_low = float(df_m5.iloc[-15:-3]["low"].min())
    recent_m5_high = float(df_m5.iloc[-15:-3]["high"].max())

    spread = mt5.symbol_info(sym).spread * mt5.symbol_info(sym).point

    # Bearish Setup: Asian High swept -> M5 Bearish Break of Structure (BOS)
    if high_swept:
        sweep_peak = float(london_test["high"].max())
        is_bos_bearish = (curr_m5["close"] < recent_m5_low) or (prev_m5["close"] < recent_m5_low)
        if is_bos_bearish and curr_m5["close"] < curr_m5["open"]:
            entry = float(curr_m5["close"])
            sl = float(sweep_peak + (spread * 2) + 0.0005)
            risk = sl - entry
            if risk > 0:
                tp = round(entry - (risk * 2.0), 5)
                res.update({
                    "signal": "SELL",
                    "entry": round(entry, 5),
                    "sl": round(sl, 5),
                    "tp": tp,
                    "rr": 2.0,
                    "details": f"Asian High ({asian_high:.5f}) swept (Peak: {sweep_peak:.5f}). M5 Bearish BOS broken below {recent_m5_low:.5f}. SL at peak, Target 1:2 RR."
                })
                return res

    # Bullish Setup: Asian Low swept -> M5 Bullish BOS
    if low_swept:
        sweep_trough = float(london_test["low"].min())
        is_bos_bullish = (curr_m5["close"] > recent_m5_high) or (prev_m5["close"] > recent_m5_high)
        if is_bos_bullish and curr_m5["close"] > curr_m5["open"]:
            entry = float(curr_m5["close"])
            sl = float(sweep_trough - (spread * 2) - 0.0005)
            risk = entry - sl
            if risk > 0:
                tp = round(entry + (risk * 2.0), 5)
                res.update({
                    "signal": "BUY",
                    "entry": round(entry, 5),
                    "sl": round(sl, 5),
                    "tp": tp,
                    "rr": 2.0,
                    "details": f"Asian Low ({asian_low:.5f}) swept (Trough: {sweep_trough:.5f}). M5 Bullish BOS broken above {recent_m5_high:.5f}. SL at trough, Target 1:2 RR."
                })
                return res

    status_str = f"Asian Range: [{asian_low:.5f} - {asian_high:.5f}]. High Swept: {high_swept}, Low Swept: {low_swept}. M5 BOS pending."
    res["details"] = status_str
    return res


def main():
    parser = argparse.ArgumentParser(description="Umar Punjabi Gold London Killzone Scanner")
    parser.add_argument("symbol", nargs="?", default="XAUUSD", help="Symbol to scan")
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
        print(f"[{now_str}] Umar Punjabi Gold London Scanner -> {res['symbol']}")
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
