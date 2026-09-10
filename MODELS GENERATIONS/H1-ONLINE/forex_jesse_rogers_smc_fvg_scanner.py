"""
Strategy 60: Jesse Rogers SMC FVG Sniper Scalping Scanner
Creator Video ID: 0O2w9OYLuTs (83.0% Claimed Win Rate, SMC FVG Strategy)
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
        "strategy": "Jesse Rogers SMC FVG Sniper",
        "symbol": symbol,
        "signal": "NEUTRAL",
        "timeframe": "M5",
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

    df = get_rates(sym, mt5.TIMEFRAME_M5, 90)
    if df is None or len(df) < 40:
        res["details"] = "Insufficient M5 rates"
        return res

    # Establish reference session range (bars -25 to -10)
    ref_window = df.iloc[-30:-10]
    opening_high = float(ref_window["high"].max())
    opening_low = float(ref_window["low"].min())

    # Scan recent 8 candles for FVG creation after breakout
    curr = df.iloc[-1]
    prev1 = df.iloc[-2]
    spread = mt5.symbol_info(sym).spread * mt5.symbol_info(sym).point

    # Bullish FVG: Breakout above opening_high leaving an FVG (C3 Low > C1 High)
    for i in range(len(df) - 6, len(df) - 1):
        c1 = df.iloc[i - 2]
        c2 = df.iloc[i - 1]
        c3 = df.iloc[i]

        if c2["close"] > opening_high and c3["low"] > c1["high"]:
            fvg_top = float(c3["low"])
            fvg_bottom = float(c1["high"])
            fvg_mid = (fvg_top + fvg_bottom) / 2.0

            # Retest check: Did current or prev candle touch into the FVG zone?
            if curr["low"] <= fvg_top and curr["close"] >= fvg_bottom and curr["close"] > curr["open"]:
                entry = float(curr["close"])
                sl = float(c2["low"] - (spread * 2) - 0.0003)
                risk = entry - sl
                if risk > 0:
                    tp = round(entry + (risk * 2.5), 5)
                    rr = round((tp - entry) / risk, 2)
                    res.update({
                        "signal": "BUY",
                        "entry": round(entry, 5),
                        "sl": round(sl, 5),
                        "tp": tp,
                        "rr": rr,
                        "details": f"Jesse Rogers Bullish FVG [{fvg_bottom:.5f} - {fvg_top:.5f}] Retest. Range High={opening_high:.5f}. SL below displacement candle {c2['low']:.5f}"
                    })
                    return res

        # Bearish FVG: Breakdown below opening_low leaving an FVG (C3 High < C1 Low)
        if c2["close"] < opening_low and c3["high"] < c1["low"]:
            fvg_top = float(c1["low"])
            fvg_bottom = float(c3["high"])
            fvg_mid = (fvg_top + fvg_bottom) / 2.0

            if curr["high"] >= fvg_bottom and curr["close"] <= fvg_top and curr["close"] < curr["open"]:
                entry = float(curr["close"])
                sl = float(c2["high"] + (spread * 2) + 0.0003)
                risk = sl - entry
                if risk > 0:
                    tp = round(entry - (risk * 2.5), 5)
                    rr = round((entry - tp) / risk, 2)
                    res.update({
                        "signal": "SELL",
                        "entry": round(entry, 5),
                        "sl": round(sl, 5),
                        "tp": tp,
                        "rr": rr,
                        "details": f"Jesse Rogers Bearish FVG [{fvg_bottom:.5f} - {fvg_top:.5f}] Retest. Range Low={opening_low:.5f}. SL above displacement candle {c2['high']:.5f}"
                    })
                    return res

    res["details"] = f"M5 Range: [{opening_low:.5f} - {opening_high:.5f}]. Current={curr['close']:.5f}. Awaiting breakout FVG creation & retest."
    return res


def main():
    parser = argparse.ArgumentParser(description="Jesse Rogers SMC FVG Sniper Scanner")
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
        print(f"[{now_str}] Jesse Rogers SMC FVG Scanner -> {res['symbol']}")
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
