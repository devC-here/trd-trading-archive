"""
Strategy 61: Theta Gainers 50/200 EMA + RSI 40-55 Value Pullback Scanner
Creator Video ID: vGHK12UgsmE (58.0% Win Rate, High RR Pullback Strategy)
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


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100.0 - (100.0 / (1.0 + rs))


def scan_symbol(symbol="EURUSD"):
    res = {
        "strategy": "Theta Gainers EMA + RSI Value Pullback",
        "symbol": symbol,
        "signal": "NEUTRAL",
        "timeframe": "M15",
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

    df = get_rates(sym, mt5.TIMEFRAME_M15, 260)
    if df is None or len(df) < 210:
        res["details"] = "Insufficient M15 rates"
        return res

    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
    df["rsi"] = calculate_rsi(df["close"], 14)

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    ema50 = float(curr["ema50"])
    ema200 = float(curr["ema200"])
    rsi = float(curr["rsi"])
    spread = mt5.symbol_info(sym).spread * mt5.symbol_info(sym).point

    recent_swing_low = float(df.iloc[-15:-1]["low"].min())
    recent_swing_high = float(df.iloc[-15:-1]["high"].max())

    # Bullish Conditions:
    # 1. Trend: EMA 50 > EMA 200
    # 2. Pullback: Low touched or tested within 0.15% of EMA 50
    # 3. RSI is in healthy pullback range (40 to 55)
    # 4. Confirmation: Green bullish candle rejecting EMA 50
    bull_trend = ema50 > ema200
    tested_ema_bull = curr["low"] <= ema50 * 1.002 or prev["low"] <= ema50 * 1.002
    rsi_in_bull_zone = 38.0 <= rsi <= 56.0
    bull_rejection = curr["close"] > curr["open"] and curr["close"] >= ema50

    if bull_trend and tested_ema_bull and rsi_in_bull_zone and bull_rejection:
        entry = float(curr["close"])
        sl = float(min(recent_swing_low, ema50) - (spread * 2) - 0.0003)
        risk = entry - sl
        if risk > 0:
            tp = round(entry + (risk * 2.2), 5)
            rr = round((tp - entry) / risk, 2)
            res.update({
                "signal": "BUY",
                "entry": round(entry, 5),
                "sl": round(sl, 5),
                "tp": tp,
                "rr": rr,
                "details": f"Theta Gainers Bullish Pullback: EMA50 ({ema50:.5f}) > EMA200 ({ema200:.5f}), RSI={rsi:.1f} in 40-55 value zone. Green rejection candle. Target: {tp:.5f}"
            })
            return res

    # Bearish Conditions:
    bear_trend = ema50 < ema200
    tested_ema_bear = curr["high"] >= ema50 * 0.998 or prev["high"] >= ema50 * 0.998
    rsi_in_bear_zone = 44.0 <= rsi <= 62.0
    bear_rejection = curr["close"] < curr["open"] and curr["close"] <= ema50

    if bear_trend and tested_ema_bear and rsi_in_bear_zone and bear_rejection:
        entry = float(curr["close"])
        sl = float(max(recent_swing_high, ema50) + (spread * 2) + 0.0003)
        risk = sl - entry
        if risk > 0:
            tp = round(entry - (risk * 2.2), 5)
            rr = round((entry - tp) / risk, 2)
            res.update({
                "signal": "SELL",
                "entry": round(entry, 5),
                "sl": round(sl, 5),
                "tp": tp,
                "rr": rr,
                "details": f"Theta Gainers Bearish Pullback: EMA50 ({ema50:.5f}) < EMA200 ({ema200:.5f}), RSI={rsi:.1f} in 45-60 value zone. Red rejection candle. Target: {tp:.5f}"
            })
            return res

    trend_str = "Bullish (EMA50 > EMA200)" if bull_trend else "Bearish (EMA50 < EMA200)"
    res["details"] = f"{trend_str}: EMA50={ema50:.5f}, RSI={rsi:.1f}. Distance to EMA50={abs(curr['close'] - ema50):.5f}."
    return res


def main():
    parser = argparse.ArgumentParser(description="Theta Gainers EMA + RSI Pullback Scanner")
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
        print(f"[{now_str}] Theta Gainers EMA RSI Scanner -> {res['symbol']}")
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
