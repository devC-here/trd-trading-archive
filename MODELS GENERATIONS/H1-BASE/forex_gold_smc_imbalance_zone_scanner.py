"""
Strategy 62: Gold (XAUUSD) SMC Imbalance Zone Institutional Scanner
Creator Video ID: qQgjJIBE2hk (58.5% Win Rate, Institutional Imbalance Strategy)
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


def get_rates(symbol, timeframe, n_bars=120):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


def scan_symbol(symbol="XAUUSD"):
    res = {
        "strategy": "Gold SMC Imbalance Zone",
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

    df = get_rates(sym, mt5.TIMEFRAME_M30, 90)
    if df is None or len(df) < 40:
        res["details"] = "Insufficient M30 rates"
        return res

    curr = df.iloc[-1]
    prev1 = df.iloc[-2]
    spread = mt5.symbol_info(sym).spread * mt5.symbol_info(sym).point

    recent_swing_high = float(df.iloc[-25:-2]["high"].max())
    recent_swing_low = float(df.iloc[-25:-2]["low"].min())

    # Scan for recent 3-candle imbalance (FVG)
    for i in range(len(df) - 8, len(df) - 2):
        c1 = df.iloc[i - 2]
        c2 = df.iloc[i - 1]
        c3 = df.iloc[i]

        # Bullish Imbalance: c3 low > c1 high
        if c3["low"] > c1["high"]:
            imb_top = float(c3["low"])
            imb_bot = float(c1["high"])
            imb_mid = (imb_top + imb_bot) / 2.0

            # Retest into imbalance 50%
            if curr["low"] <= imb_mid <= curr["high"] or (prev1["low"] <= imb_mid and curr["close"] > imb_mid):
                if curr["close"] > curr["open"]:
                    entry = float(curr["close"])
                    sl = float(c2["low"] - (spread * 2) - 0.0005)
                    risk = entry - sl
                    if risk > 0:
                        tp = max(entry + (risk * 2.5), recent_swing_high)
                        rr = round((tp - entry) / risk, 2)
                        res.update({
                            "signal": "BUY",
                            "entry": round(entry, 5),
                            "sl": round(sl, 5),
                            "tp": round(tp, 5),
                            "rr": rr,
                            "details": f"Gold Bullish Imbalance Zone [{imb_bot:.5f} - {imb_top:.5f}], Midpoint={imb_mid:.5f} tested with bullish rejection. Target Swing High: {tp:.5f}"
                        })
                        return res

        # Bearish Imbalance: c3 high < c1 low
        if c3["high"] < c1["low"]:
            imb_top = float(c1["low"])
            imb_bot = float(c3["high"])
            imb_mid = (imb_top + imb_bot) / 2.0

            if curr["low"] <= imb_mid <= curr["high"] or (prev1["high"] >= imb_mid and curr["close"] < imb_mid):
                if curr["close"] < curr["open"]:
                    entry = float(curr["close"])
                    sl = float(c2["high"] + (spread * 2) + 0.0005)
                    risk = sl - entry
                    if risk > 0:
                        tp = min(entry - (risk * 2.5), recent_swing_low)
                        rr = round((entry - tp) / risk, 2)
                        res.update({
                            "signal": "SELL",
                            "entry": round(entry, 5),
                            "sl": round(sl, 5),
                            "tp": round(tp, 5),
                            "rr": rr,
                            "details": f"Gold Bearish Imbalance Zone [{imb_bot:.5f} - {imb_top:.5f}], Midpoint={imb_mid:.5f} tested with bearish rejection. Target Swing Low: {tp:.5f}"
                        })
                        return res

    res["details"] = f"Price={curr['close']:.5f}. Range=[{recent_swing_low:.5f} - {recent_swing_high:.5f}]. Awaiting institutional imbalance retest."
    return res


def main():
    parser = argparse.ArgumentParser(description="Gold SMC Imbalance Zone Scanner")
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
        print(f"[{now_str}] Gold SMC Imbalance Zone Scanner -> {res['symbol']}")
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
