"""
Strategy 49: Rayner Teo 'Catch The Wave' 50 EMA Pullback Swing Scanner
Creator Video ID: KGKC9Tkiii0 (+48.2% Annual Return, 64.0% Win Rate)
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


def is_bullish_trigger(curr, prev):
    body = abs(curr["close"] - curr["open"])
    c_range = curr["high"] - curr["low"]
    lower_wick = min(curr["open"], curr["close"]) - curr["low"]
    upper_wick = curr["high"] - max(curr["open"], curr["close"])

    if c_range <= 0:
        return False, ""

    # Hammer / Pinbar
    if lower_wick >= 2 * body and curr["close"] > curr["open"] and upper_wick <= 0.3 * c_range:
        return True, "Hammer Rejection"

    # Bullish Engulfing
    if curr["close"] > curr["open"] and prev["close"] < prev["open"]:
        if curr["close"] >= prev["open"] and curr["open"] <= prev["close"]:
            return True, "Bullish Engulfing"

    # Strong Expansion Rejection Green Candle
    if curr["close"] > curr["open"] and (body / c_range) >= 0.65:
        return True, "Strong Expansion Green Candle"

    return False, ""


def is_bearish_trigger(curr, prev):
    body = abs(curr["close"] - curr["open"])
    c_range = curr["high"] - curr["low"]
    lower_wick = min(curr["open"], curr["close"]) - curr["low"]
    upper_wick = curr["high"] - max(curr["open"], curr["close"])

    if c_range <= 0:
        return False, ""

    # Shooting Star / Inverted Pinbar
    if upper_wick >= 2 * body and curr["close"] < curr["open"] and lower_wick <= 0.3 * c_range:
        return True, "Shooting Star Rejection"

    # Bearish Engulfing
    if curr["close"] < curr["open"] and prev["close"] > prev["open"]:
        if curr["close"] <= prev["open"] and curr["open"] >= prev["close"]:
            return True, "Bearish Engulfing"

    # Strong Expansion Red Candle
    if curr["close"] < curr["open"] and (body / c_range) >= 0.65:
        return True, "Strong Expansion Red Candle"

    return False, ""


def scan_symbol(symbol="EURUSD"):
    res = {
        "strategy": "Rayner Teo 50 EMA Pullback Swing",
        "symbol": symbol,
        "signal": "NEUTRAL",
        "timeframe": "D1 (Swing)",
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

    df = get_rates(sym, mt5.TIMEFRAME_D1, 100)
    if df is None or len(df) < 60:
        res["details"] = "Insufficient D1 rates"
        return res

    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    ema50 = curr["ema50"]
    spread = mt5.symbol_info(sym).spread * mt5.symbol_info(sym).point

    recent_swing_high = df.iloc[-25:-2]["high"].max()
    recent_swing_low = df.iloc[-25:-2]["low"].min()

    # Bullish Setup:
    # 1. Trend: close above 50 EMA
    # 2. Pullback: low touched or tested near 50 EMA (within 0.5% or low <= ema50)
    # 3. Candlestick trigger
    is_near_ema_bull = curr["low"] <= ema50 * 1.004 or prev["low"] <= ema50 * 1.004
    is_bull, pattern_bull = is_bullish_trigger(curr, prev)

    if curr["close"] > ema50 and is_near_ema_bull and is_bull:
        entry = float(curr["close"])
        sl = float(min(curr["low"], prev["low"]) - (spread * 2) - 0.0005)
        risk = entry - sl
        if risk > 0:
            tp = max(entry + (risk * 2.0), float(recent_swing_high * 0.998))
            rr = round((tp - entry) / risk, 2)
            res.update({
                "signal": "BUY",
                "entry": round(entry, 5),
                "sl": round(sl, 5),
                "tp": round(tp, 5),
                "rr": rr,
                "details": f"Bullish 50 EMA Pullback [{pattern_bull}]. Price={entry:.5f}, EMA50={ema50:.5f}, Target Swing High={recent_swing_high:.5f}"
            })
            return res

    # Bearish Setup:
    is_near_ema_bear = curr["high"] >= ema50 * 0.996 or prev["high"] >= ema50 * 0.996
    is_bear, pattern_bear = is_bearish_trigger(curr, prev)

    if curr["close"] < ema50 and is_near_ema_bear and is_bear:
        entry = float(curr["close"])
        sl = float(max(curr["high"], prev["high"]) + (spread * 2) + 0.0005)
        risk = sl - entry
        if risk > 0:
            tp = min(entry - (risk * 2.0), float(recent_swing_low * 1.002))
            rr = round((entry - tp) / risk, 2)
            res.update({
                "signal": "SELL",
                "entry": round(entry, 5),
                "sl": round(sl, 5),
                "tp": round(tp, 5),
                "rr": rr,
                "details": f"Bearish 50 EMA Pullback [{pattern_bear}]. Price={entry:.5f}, EMA50={ema50:.5f}, Target Swing Low={recent_swing_low:.5f}"
            })
            return res

    trend_str = "Bullish (>50 EMA)" if curr["close"] > ema50 else "Bearish (<50 EMA)"
    res["details"] = f"{trend_str}: Price={curr['close']:.5f}, EMA50={ema50:.5f}. Distance to EMA: {abs(curr['close'] - ema50):.5f}"
    return res


def main():
    parser = argparse.ArgumentParser(description="Rayner Teo 50 EMA Pullback Swing Scanner")
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
        print(f"[{now_str}] Rayner Teo 50 EMA Swing Scanner -> {res['symbol']}")
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
