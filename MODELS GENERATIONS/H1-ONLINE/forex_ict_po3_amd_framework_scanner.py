"""
Strategy 57: ICT PO3 (Accumulation Manipulation Distribution) / CRT Scanner
Creator Video ID: ujdzVPO97bU (+56.8% Annual Return, 54.0% Win Rate, 15.4% Max DD)
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
        "strategy": "ICT PO3 AMD Framework",
        "symbol": symbol,
        "signal": "NEUTRAL",
        "timeframe": "M15 (CRT Range) + M5 (Distribution Trigger)",
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

    # 1. Fetch M15 Data
    df_m15 = get_rates(sym, mt5.TIMEFRAME_M15, 60)
    if df_m15 is None or len(df_m15) < 30:
        res["details"] = "Insufficient M15 rates"
        return res

    # CRT Candle range (accumulation zone from 4 to 12 bars ago)
    crt_df = df_m15.iloc[-14:-4]
    crt_high = float(crt_df["high"].max())
    crt_low = float(crt_df["low"].min())

    recent_m15 = df_m15.iloc[-4:]
    manipulation_low_swept = bool(recent_m15["low"].min() < crt_low)
    manipulation_high_swept = bool(recent_m15["high"].max() > crt_high)

    # 2. Fetch M5 Data
    df_m5 = get_rates(sym, mt5.TIMEFRAME_M5, 60)
    if df_m5 is None or len(df_m5) < 30:
        res["details"] = "Insufficient M5 rates"
        return res

    curr_m5 = df_m5.iloc[-1]
    prev_m5 = df_m5.iloc[-2]
    spread = mt5.symbol_info(sym).spread * mt5.symbol_info(sym).point

    # Bullish PO3: CRT Low was swept (Manipulation), M5 reclaims back above crt_low
    if manipulation_low_swept:
        sweep_extreme = float(recent_m15["low"].min())
        # M5 confirmation: price closed back above crt_low with bullish candle
        if curr_m5["close"] > crt_low and curr_m5["close"] > curr_m5["open"]:
            entry = float(curr_m5["close"])
            sl = float(sweep_extreme - (spread * 2) - 0.0003)
            risk = entry - sl
            if risk > 0:
                tp = float(crt_high)
                rr = round((tp - entry) / risk, 2)
                res.update({
                    "signal": "BUY",
                    "entry": round(entry, 5),
                    "sl": round(sl, 5),
                    "tp": round(tp, 5),
                    "rr": rr,
                    "details": f"Bullish PO3 / AMD: CRT Low ({crt_low:.5f}) swept down to {sweep_extreme:.5f} (Manipulation). M5 reclaimed into range. Target CRT High: {crt_high:.5f}"
                })
                return res

    # Bearish PO3: CRT High was swept (Manipulation), M5 reclaims back below crt_high
    if manipulation_high_swept:
        sweep_extreme = float(recent_m15["high"].max())
        if curr_m5["close"] < crt_high and curr_m5["close"] < curr_m5["open"]:
            entry = float(curr_m5["close"])
            sl = float(sweep_extreme + (spread * 2) + 0.0003)
            risk = sl - entry
            if risk > 0:
                tp = float(crt_low)
                rr = round((entry - tp) / risk, 2)
                res.update({
                    "signal": "SELL",
                    "entry": round(entry, 5),
                    "sl": round(sl, 5),
                    "tp": round(tp, 5),
                    "rr": rr,
                    "details": f"Bearish PO3 / AMD: CRT High ({crt_high:.5f}) swept up to {sweep_extreme:.5f} (Manipulation). M5 rejected below range. Target CRT Low: {crt_low:.5f}"
                })
                return res

    res["details"] = f"M15 CRT Accumulation: [{crt_low:.5f} - {crt_high:.5f}]. Low Swept: {manipulation_low_swept}, High Swept: {manipulation_high_swept}."
    return res


def main():
    parser = argparse.ArgumentParser(description="ICT PO3 AMD Framework Scanner")
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
        print(f"[{now_str}] ICT PO3 AMD Scanner -> {res['symbol']}")
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
