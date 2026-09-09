"""
Strategy 58: Oliver Velez Bull/Bear 180 Elephant Candle Scanner
Creator Video ID: Lzzi6ylRObY (82.0% Claimed Win Rate, 39-Yr Prop Firm Strategy)
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


def scan_symbol(symbol="EURUSD"):
    res = {
        "strategy": "Oliver Velez Bull/Bear 180",
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

    df = get_rates(sym, mt5.TIMEFRAME_M5, 250)
    if df is None or len(df) < 210:
        res["details"] = "Insufficient M5 rates"
        return res

    # SMAs: 8, 20, 200
    df["sma8"] = df["close"].rolling(8).mean()
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma200"] = df["close"].rolling(200).mean()

    curr = df.iloc[-1]
    prev = df.iloc[-2]

    # Average body over last 20 candles
    avg_body = abs(df.iloc[-22:-2]["close"] - df.iloc[-22:-2]["open"]).mean()

    curr_body = abs(curr["close"] - curr["open"])
    prev_body = abs(prev["close"] - prev["open"])
    spread = mt5.symbol_info(sym).spread * mt5.symbol_info(sym).point

    # Bull 180 Setup:
    # 1. Trend: 20 SMA > 200 SMA (or price near 20 SMA)
    # 2. Previous candle is Red
    # 3. Current candle is a Large Green Elephant Candle (curr_body >= 1.4 * avg_body and curr_body >= prev_body)
    # 4. Current candle closes above previous candle high (180 degree momentum reversal)
    bull_trend = curr["sma20"] > curr["sma200"] or curr["close"] >= curr["sma20"]
    is_bull_180 = (
        prev["close"] < prev["open"] and
        curr["close"] > curr["open"] and
        curr_body >= avg_body * 1.35 and
        curr["close"] >= prev["open"] and
        curr["open"] <= prev["close"]
    )

    if bull_trend and is_bull_180:
        entry = float(curr["close"])
        sl = float(min(curr["low"], prev["low"]) - (spread * 2) - 0.0003)
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
                "details": f"Oliver Velez Bull 180: Large Green Elephant Bar (Body={curr_body:.5f} vs Avg={avg_body:.5f}) engulfing red bar. 20 SMA={curr['sma20']:.5f} > 200 SMA={curr['sma200']:.5f}."
            })
            return res

    # Bear 180 Setup:
    bear_trend = curr["sma20"] < curr["sma200"] or curr["close"] <= curr["sma20"]
    is_bear_180 = (
        prev["close"] > prev["open"] and
        curr["close"] < curr["open"] and
        curr_body >= avg_body * 1.35 and
        curr["close"] <= prev["open"] and
        curr["open"] >= prev["close"]
    )

    if bear_trend and is_bear_180:
        entry = float(curr["close"])
        sl = float(max(curr["high"], prev["high"]) + (spread * 2) + 0.0003)
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
                "details": f"Oliver Velez Bear 180: Large Red Elephant Bar (Body={curr_body:.5f} vs Avg={avg_body:.5f}) erasing green bar. 20 SMA={curr['sma20']:.5f} < 200 SMA={curr['sma200']:.5f}."
            })
            return res

    trend_str = "Bullish (SMA20 > SMA200)" if curr["sma20"] > curr["sma200"] else "Bearish (SMA20 < SMA200)"
    res["details"] = f"{trend_str}: SMA20={curr['sma20']:.5f}, SMA200={curr['sma200']:.5f}. Avg Body={avg_body:.5f}. Waiting for 180 Elephant reversal bar."
    return res


def main():
    parser = argparse.ArgumentParser(description="Oliver Velez Bull/Bear 180 Scanner")
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
        print(f"[{now_str}] Oliver Velez Bull/Bear 180 Scanner -> {res['symbol']}")
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
