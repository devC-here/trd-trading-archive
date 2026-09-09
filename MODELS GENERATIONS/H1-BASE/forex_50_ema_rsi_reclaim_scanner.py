"""
Strategy 50: 50 EMA Trend + RSI 30/70 Reclaim Scanner
Creator Video ID: ryoDybQXEs4 (+47.6% Annual Return, 60.7% Win Rate, 18.2% Max DD)
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
        "strategy": "50 EMA + RSI 30/70 Reclaim",
        "symbol": symbol,
        "signal": "NEUTRAL",
        "timeframe": "H1",
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

    df = get_rates(sym, mt5.TIMEFRAME_H1, 120)
    if df is None or len(df) < 70:
        res["details"] = "Insufficient H1 rates"
        return res

    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["rsi"] = calculate_rsi(df["close"], 14)

    curr = df.iloc[-1]
    prev1 = df.iloc[-2]
    prev2 = df.iloc[-3]
    prev3 = df.iloc[-4]

    spread = mt5.symbol_info(sym).spread * mt5.symbol_info(sym).point
    recent_swing_high = df.iloc[-25:-2]["high"].max()
    recent_swing_low = df.iloc[-25:-2]["low"].min()

    # Look back 5 candles for RSI oversold/overbought condition
    recent_rsi_min = df.iloc[-7:-1]["rsi"].min()
    recent_rsi_max = df.iloc[-7:-1]["rsi"].max()

    # Bullish Setup:
    # 1. Price above 50 EMA
    # 2. RSI dipped below 30 in recent bars
    # 3. RSI crossed back above 30 on prev1 or curr
    # 4. Confirmation bullish candle (close > open)
    is_bullish_trend = curr["close"] > curr["ema50"]
    was_oversold = recent_rsi_min <= 31.0
    rsi_reclaimed_30 = (prev1["rsi"] < 30.0 and curr["rsi"] >= 30.0) or (prev2["rsi"] < 30.0 and prev1["rsi"] >= 30.0 and curr["rsi"] > 30.0)
    is_bull_candle = curr["close"] > curr["open"]

    if is_bullish_trend and was_oversold and rsi_reclaimed_30 and is_bull_candle:
        entry = float(curr["close"])
        pullback_low = float(df.iloc[-7:]["low"].min())
        sl = float(pullback_low - (spread * 2) - 0.0003)
        risk = entry - sl
        if risk > 0:
            tp = max(entry + (risk * 2.2), float(recent_swing_high))
            rr = round((tp - entry) / risk, 2)
            res.update({
                "signal": "BUY",
                "entry": round(entry, 5),
                "sl": round(sl, 5),
                "tp": round(tp, 5),
                "rr": rr,
                "details": f"Bullish Reclaim: Price above EMA50 ({curr['ema50']:.5f}), RSI dipped to {recent_rsi_min:.1f} and reclaimed {curr['rsi']:.1f} with green candle. Target: {tp:.5f}"
            })
            return res

    # Bearish Setup:
    # 1. Price below 50 EMA
    # 2. RSI peaked above 70 in recent bars
    # 3. RSI crossed back below 70
    # 4. Confirmation bearish candle (close < open)
    is_bearish_trend = curr["close"] < curr["ema50"]
    was_overbought = recent_rsi_max >= 69.0
    rsi_reclaimed_70 = (prev1["rsi"] > 70.0 and curr["rsi"] <= 70.0) or (prev2["rsi"] > 70.0 and prev1["rsi"] <= 70.0 and curr["rsi"] < 70.0)
    is_bear_candle = curr["close"] < curr["open"]

    if is_bearish_trend and was_overbought and rsi_reclaimed_70 and is_bear_candle:
        entry = float(curr["close"])
        pullback_high = float(df.iloc[-7:]["high"].max())
        sl = float(pullback_high + (spread * 2) + 0.0003)
        risk = sl - entry
        if risk > 0:
            tp = min(entry - (risk * 2.2), float(recent_swing_low))
            rr = round((entry - tp) / risk, 2)
            res.update({
                "signal": "SELL",
                "entry": round(entry, 5),
                "sl": round(sl, 5),
                "tp": round(tp, 5),
                "rr": rr,
                "details": f"Bearish Reclaim: Price below EMA50 ({curr['ema50']:.5f}), RSI peaked at {recent_rsi_max:.1f} and dropped to {curr['rsi']:.1f} with red candle. Target: {tp:.5f}"
            })
            return res

    trend_str = "Above EMA50" if is_bullish_trend else "Below EMA50"
    res["details"] = f"{trend_str} (Price={curr['close']:.5f}, EMA50={curr['ema50']:.5f}). RSI={curr['rsi']:.1f} (Recent Range: {recent_rsi_min:.1f} - {recent_rsi_max:.1f})."
    return res


def main():
    parser = argparse.ArgumentParser(description="50 EMA + RSI 30/70 Reclaim Scanner")
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
        print(f"[{now_str}] 50 EMA RSI Reclaim Scanner -> {res['symbol']}")
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
