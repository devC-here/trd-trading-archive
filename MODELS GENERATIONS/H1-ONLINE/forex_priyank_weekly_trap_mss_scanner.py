"""
Strategy 52: Priyank Sharma MTF Weekly Trap + M15 MSS Scanner
Creator Video ID: fipzUyTxUxw (+52.4% Annual Return, 63.0% Win Rate, 12.6% Max DD)
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
        "strategy": "Priyank Sharma Weekly Trap + M15 MSS",
        "symbol": symbol,
        "signal": "NEUTRAL",
        "timeframe": "W1 (Levels) + D1 (Trap) + M15 (MSS Trigger)",
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

    # 1. Weekly Rates
    df_w1 = get_rates(sym, mt5.TIMEFRAME_W1, 10)
    if df_w1 is None or len(df_w1) < 3:
        res["details"] = "Insufficient W1 rates"
        return res

    # Completed previous week is index -2
    prev_week = df_w1.iloc[-2]
    pwh = float(prev_week["high"])
    pwl = float(prev_week["low"])

    # 2. Daily Rates (Check if recent Daily candles formed a trap)
    df_d1 = get_rates(sym, mt5.TIMEFRAME_D1, 20)
    if df_d1 is None or len(df_d1) < 10:
        res["details"] = "Insufficient D1 rates"
        return res

    recent_d1 = df_d1.iloc[-6:]
    pwl_trapped = False
    pwh_trapped = False
    trap_extreme_low = pwl
    trap_extreme_high = pwh

    for _, d_row in recent_d1.iterrows():
        # PWL Trap: low dipped below PWL, but close is above PWL
        if d_row["low"] < pwl and d_row["close"] >= pwl:
            pwl_trapped = True
            trap_extreme_low = min(trap_extreme_low, d_row["low"])
        # PWH Trap: high pierced above PWH, but close is below PWH
        if d_row["high"] > pwh and d_row["close"] <= pwh:
            pwh_trapped = True
            trap_extreme_high = max(trap_extreme_high, d_row["high"])

    # 3. M15 Rates (Check for Market Structure Shift)
    df_m15 = get_rates(sym, mt5.TIMEFRAME_M15, 60)
    if df_m15 is None or len(df_m15) < 30:
        res["details"] = "Insufficient M15 rates"
        return res

    curr = df_m15.iloc[-1]
    prev1 = df_m15.iloc[-2]
    prev2 = df_m15.iloc[-3]

    recent_m15_high = df_m15.iloc[-20:-3]["high"].max()
    recent_m15_low = df_m15.iloc[-20:-3]["low"].min()

    spread = mt5.symbol_info(sym).spread * mt5.symbol_info(sym).point

    if pwl_trapped:
        # Bullish MSS: M15 closes above recent swing high
        mss_bullish = curr["close"] > recent_m15_high or prev1["close"] > recent_m15_high
        if mss_bullish and curr["close"] > curr["open"]:
            entry = float(curr["close"])
            sl = float(trap_extreme_low - (spread * 2) - 0.0003)
            risk = entry - sl
            if risk > 0:
                tp = max(entry + (risk * 2.0), pwh)
                rr = round((tp - entry) / risk, 2)
                res.update({
                    "signal": "BUY",
                    "entry": round(entry, 5),
                    "sl": round(sl, 5),
                    "tp": round(tp, 5),
                    "rr": rr,
                    "details": f"Weekly Low Trap: PWL={pwl:.5f} swept (Extreme Low: {trap_extreme_low:.5f}), M15 MSS broken above {recent_m15_high:.5f}. Target PWH: {pwh:.5f}"
                })
                return res

    if pwh_trapped:
        # Bearish MSS: M15 closes below recent swing low
        mss_bearish = curr["close"] < recent_m15_low or prev1["close"] < recent_m15_low
        if mss_bearish and curr["close"] < curr["open"]:
            entry = float(curr["close"])
            sl = float(trap_extreme_high + (spread * 2) + 0.0003)
            risk = sl - entry
            if risk > 0:
                tp = min(entry - (risk * 2.0), pwl)
                rr = round((entry - tp) / risk, 2)
                res.update({
                    "signal": "SELL",
                    "entry": round(entry, 5),
                    "sl": round(sl, 5),
                    "tp": round(tp, 5),
                    "rr": rr,
                    "details": f"Weekly High Trap: PWH={pwh:.5f} swept (Extreme High: {trap_extreme_high:.5f}), M15 MSS broken below {recent_m15_low:.5f}. Target PWL: {pwl:.5f}"
                })
                return res

    res["details"] = f"PWH: {pwh:.5f}, PWL: {pwl:.5f}. PWL Trapped: {pwl_trapped}, PWH Trapped: {pwh_trapped}. M15 MSS pending."
    return res


def main():
    parser = argparse.ArgumentParser(description="Priyank Sharma Weekly Trap + M15 MSS Scanner")
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
        print(f"[{now_str}] Priyank Sharma Weekly Trap Scanner -> {res['symbol']}")
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
