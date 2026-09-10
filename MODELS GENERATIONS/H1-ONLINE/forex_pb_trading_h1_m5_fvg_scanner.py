"""
Strategy 70: PB Trading Multi-Timeframe H1 + M5 FVG Confluence Scanner
Creator Video ID: N3o-py_LA3A (70.0% Win Rate, Multi-TF Institutional FVG Model)
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
        "strategy": "PB Trading Multi-TF FVG Confluence",
        "symbol": symbol,
        "signal": "NEUTRAL",
        "timeframe": "H1 (Zone) + M5 (FVG Trigger)",
        "entry": None,
        "sl": None,
        "tp": None,
        "rr": 3.0,
        "details": ""
    }

    if not mt5:
        res["details"] = "MetaTrader5 library not installed."
        return res

    sym = resolve_symbol(symbol)
    if not mt5.symbol_select(sym, True):
        res["details"] = f"Failed to select symbol {sym}"
        return res

    # 1. H1 Rates: Find active FVG
    df_h1 = get_rates(sym, mt5.TIMEFRAME_H1, 50)
    if df_h1 is None or len(df_h1) < 25:
        res["details"] = "Insufficient H1 rates"
        return res

    # 2. M5 Rates: Trigger
    df_m5 = get_rates(sym, mt5.TIMEFRAME_M5, 60)
    if df_m5 is None or len(df_m5) < 30:
        res["details"] = "Insufficient M5 rates"
        return res

    curr_price = float(df_m5.iloc[-1]["close"])
    spread = mt5.symbol_info(sym).spread * mt5.symbol_info(sym).point

    # Find Bullish H1 FVG in recent 15 bars
    for i in range(len(df_h1) - 12, len(df_h1) - 1):
        c1 = df_h1.iloc[i - 2]
        c2 = df_h1.iloc[i - 1]
        c3 = df_h1.iloc[i]

        # Bullish H1 FVG: c3 low > c1 high
        if c3["low"] > c1["high"]:
            h1_fvg_top = float(c3["low"])
            h1_fvg_bot = float(c1["high"])

            # Is price currently inside or tapping H1 FVG?
            if h1_fvg_bot * 0.999 <= curr_price <= h1_fvg_top * 1.001:
                # Check for fresh M5 bullish FVG in recent 5 bars
                for m in range(len(df_m5) - 6, len(df_m5) - 1):
                    mc1 = df_m5.iloc[m - 2]
                    mc2 = df_m5.iloc[m - 1]
                    mc3 = df_m5.iloc[m]
                    if mc3["low"] > mc1["high"]:
                        m5_fvg_top = float(mc3["low"])
                        m5_fvg_bot = float(mc1["high"])
                        # Retest check on M5
                        curr_m5 = df_m5.iloc[-1]
                        if curr_m5["low"] <= m5_fvg_top and curr_m5["close"] >= m5_fvg_bot and curr_m5["close"] > curr_m5["open"]:
                            entry = float(curr_m5["close"])
                            sl = float(mc2["low"] - (spread * 2) - 0.0003)
                            risk = entry - sl
                            if risk > 0:
                                tp = round(entry + (risk * 3.0), 5)
                                res.update({
                                    "signal": "BUY",
                                    "entry": round(entry, 5),
                                    "sl": round(sl, 5),
                                    "tp": tp,
                                    "rr": 3.0,
                                    "details": f"PB Trading: Price in H1 FVG [{h1_fvg_bot:.5f} - {h1_fvg_top:.5f}], fresh M5 FVG [{m5_fvg_bot:.5f} - {m5_fvg_top:.5f}] retested with bullish rejection. Target 1:3 RR."
                                })
                                return res

        # Bearish H1 FVG: c3 high < c1 low
        if c3["high"] < c1["low"]:
            h1_fvg_top = float(c1["low"])
            h1_fvg_bot = float(c3["high"])

            if h1_fvg_bot * 0.999 <= curr_price <= h1_fvg_top * 1.001:
                for m in range(len(df_m5) - 6, len(df_m5) - 1):
                    mc1 = df_m5.iloc[m - 2]
                    mc2 = df_m5.iloc[m - 1]
                    mc3 = df_m5.iloc[m]
                    if mc3["high"] < mc1["low"]:
                        m5_fvg_top = float(mc1["low"])
                        m5_fvg_bot = float(mc3["high"])
                        curr_m5 = df_m5.iloc[-1]
                        if curr_m5["high"] >= m5_fvg_bot and curr_m5["close"] <= m5_fvg_top and curr_m5["close"] < curr_m5["open"]:
                            entry = float(curr_m5["close"])
                            sl = float(mc2["high"] + (spread * 2) + 0.0003)
                            risk = sl - entry
                            if risk > 0:
                                tp = round(entry - (risk * 3.0), 5)
                                res.update({
                                    "signal": "SELL",
                                    "entry": round(entry, 5),
                                    "sl": round(sl, 5),
                                    "tp": tp,
                                    "rr": 3.0,
                                    "details": f"PB Trading: Price in H1 FVG [{h1_fvg_bot:.5f} - {h1_fvg_top:.5f}], fresh M5 FVG [{m5_fvg_bot:.5f} - {m5_fvg_top:.5f}] retested with bearish rejection. Target 1:3 RR."
                                })
                                return res

    res["details"] = f"Price={curr_price:.5f}. Awaiting active H1 FVG tap and M5 displacement FVG retest."
    return res


def main():
    parser = argparse.ArgumentParser(description="PB Trading Multi-TF FVG Confluence Scanner")
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
        print(f"[{now_str}] PB Trading Multi-TF FVG Scanner -> {res['symbol']}")
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
