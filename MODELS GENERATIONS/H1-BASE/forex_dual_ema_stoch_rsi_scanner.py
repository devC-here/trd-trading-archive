"""
Strategy 63: 50/200 Dual EMA + Stochastic RSI Momentum Scalper
Creator Video ID: FvHJKuGaoqg (+72.0% Annual Return, 1L -> 1.72L, 53% Win Rate)
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


def calculate_stoch_rsi(series, period=14, smooth_k=3, smooth_d=3):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    rsi_low = rsi.rolling(period).min()
    rsi_high = rsi.rolling(period).max()
    stoch = 100.0 * (rsi - rsi_low) / (rsi_high - rsi_low + 1e-10)
    k = stoch.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()
    return k, d


def scan_symbol(symbol="EURUSD"):
    res = {
        "strategy": "Dual EMA + Stoch RSI Scalper",
        "symbol": symbol,
        "signal": "NEUTRAL",
        "timeframe": "M5",
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

    df = get_rates(sym, mt5.TIMEFRAME_M5, 260)
    if df is None or len(df) < 210:
        res["details"] = "Insufficient M5 rates"
        return res

    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
    k, d = calculate_stoch_rsi(df["close"], 14, 3, 3)
    df["k"] = k
    df["d"] = d

    curr = df.iloc[-1]
    prev1 = df.iloc[-2]
    prev2 = df.iloc[-3]

    spread = mt5.symbol_info(sym).spread * mt5.symbol_info(sym).point
    recent_swing_low = float(df.iloc[-15:-1]["low"].min())
    recent_swing_high = float(df.iloc[-15:-1]["high"].max())

    # Bullish: 50 EMA > 200 EMA + Stoch RSI cross up from < 25
    bull_trend = curr["ema50"] > curr["ema200"]
    was_oversold = df.iloc[-5:-1]["k"].min() <= 25.0
    k_cross_up = (prev1["k"] <= prev1["d"] and curr["k"] > curr["d"]) or (prev2["k"] <= prev2["d"] and prev1["k"] > prev1["d"])

    if bull_trend and was_oversold and k_cross_up and curr["close"] > curr["open"]:
        entry = float(curr["close"])
        sl = float(recent_swing_low - (spread * 2) - 0.0003)
        risk = entry - sl
        if risk > 0:
            tp = round(entry + (risk * 2.0), 5)
            res.update({
                "signal": "BUY",
                "entry": round(entry, 5),
                "sl": round(sl, 5),
                "tp": tp,
                "rr": 2.0,
                "details": f"Bullish Trend (EMA50 > EMA200). Stoch RSI %K ({curr['k']:.1f}) crossed above %D ({curr['d']:.1f}) from oversold. Target 1:2 RR."
            })
            return res

    # Bearish: 50 EMA < 200 EMA + Stoch RSI cross down from > 75
    bear_trend = curr["ema50"] < curr["ema200"]
    was_overbought = df.iloc[-5:-1]["k"].max() >= 75.0
    k_cross_down = (prev1["k"] >= prev1["d"] and curr["k"] < curr["d"]) or (prev2["k"] >= prev2["d"] and prev1["k"] < prev1["d"])

    if bear_trend and was_overbought and k_cross_down and curr["close"] < curr["open"]:
        entry = float(curr["close"])
        sl = float(recent_swing_high + (spread * 2) + 0.0003)
        risk = sl - entry
        if risk > 0:
            tp = round(entry - (risk * 2.0), 5)
            res.update({
                "signal": "SELL",
                "entry": round(entry, 5),
                "sl": round(sl, 5),
                "tp": tp,
                "rr": 2.0,
                "details": f"Bearish Trend (EMA50 < EMA200). Stoch RSI %K ({curr['k']:.1f}) crossed below %D ({curr['d']:.1f}) from overbought. Target 1:2 RR."
            })
            return res

    trend_str = "Bullish (EMA50 > EMA200)" if bull_trend else "Bearish (EMA50 < EMA200)"
    res["details"] = f"{trend_str}: EMA50={curr['ema50']:.5f}, EMA200={curr['ema200']:.5f}. Stoch RSI: %K={curr['k']:.1f}, %D={curr['d']:.1f}."
    return res


def main():
    parser = argparse.ArgumentParser(description="Dual EMA + Stoch RSI Scalper")
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
        print(f"[{now_str}] Dual EMA Stoch RSI Scanner -> {res['symbol']}")
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
