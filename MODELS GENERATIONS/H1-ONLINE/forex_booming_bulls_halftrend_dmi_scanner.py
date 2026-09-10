"""
Strategy 51: Booming Bulls (Anish Singh Thakur) HalfTrend + DMI Intraday Scanner
Creator Video ID: pFs09yBjQU4 (+63.28% Annual Return, 53.45% Win Rate, 1:3 RR)
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


def get_rates(symbol, timeframe, n_bars=150):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


def calculate_dmi(df, period=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    plus_di = 100.0 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    minus_di = 100.0 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean() / (atr + 1e-10))

    dx = 100.0 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10))
    adx = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    return plus_di, minus_di, adx


def calculate_half_trend(df, amplitude=2, channel_dev=2):
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(df)

    ht = np.zeros(n)
    trend = np.zeros(n, dtype=int)

    tr1 = df["high"] - df["low"]
    tr2 = abs(df["high"] - df["close"].shift(1))
    tr3 = abs(df["low"] - df["close"].shift(1))
    atr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(amplitude * 5).mean().bfill().values

    trend[0] = 1
    ht[0] = close[0]

    for i in range(1, n):
        max_high = np.max(high[max(0, i - amplitude):i + 1])
        min_low = np.min(low[max(0, i - amplitude):i + 1])

        dev = channel_dev * atr[i]

        if trend[i - 1] == 1:
            curr_ht = max(ht[i - 1], min_low)
            if close[i] < curr_ht - dev * 0.5:
                trend[i] = -1
                ht[i] = max_high
            else:
                trend[i] = 1
                ht[i] = curr_ht
        else:
            curr_ht = min(ht[i - 1], max_high)
            if close[i] > curr_ht + dev * 0.5:
                trend[i] = 1
                ht[i] = min_low
            else:
                trend[i] = -1
                ht[i] = curr_ht

    return pd.Series(ht, index=df.index), pd.Series(trend, index=df.index)


def scan_symbol(symbol="EURUSD"):
    res = {
        "strategy": "Booming Bulls HalfTrend + DMI Intraday",
        "symbol": symbol,
        "signal": "NEUTRAL",
        "timeframe": "M15",
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

    df = get_rates(sym, mt5.TIMEFRAME_M15, 120)
    if df is None or len(df) < 60:
        res["details"] = "Insufficient M15 rates"
        return res

    plus_di, minus_di, adx = calculate_dmi(df, 14)
    ht_line, ht_trend = calculate_half_trend(df, amplitude=2, channel_dev=2)

    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    df["adx"] = adx
    df["ht_line"] = ht_line
    df["ht_trend"] = ht_trend

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    spread = mt5.symbol_info(sym).spread * mt5.symbol_info(sym).point

    # Bullish Conditions:
    # 1. HalfTrend is Bullish (trend == 1)
    # 2. DMI is Bullish (plus_di > minus_di)
    # 3. New crossover trigger: HalfTrend flipped bullish or plus_di crossed minus_di in last 2 bars
    ht_bull = curr["ht_trend"] == 1
    dmi_bull = curr["plus_di"] > curr["minus_di"]
    new_bull_trigger = (prev["ht_trend"] == -1 and curr["ht_trend"] == 1) or \
                       (prev["plus_di"] <= prev["minus_di"] and curr["plus_di"] > curr["minus_di"]) or \
                       (prev2["ht_trend"] == -1 and prev["ht_trend"] == 1 and curr["ht_trend"] == 1)

    if ht_bull and dmi_bull and new_bull_trigger:
        entry = float(curr["close"])
        sl = float(curr["ht_line"] - (spread * 2) - 0.0003)
        risk = entry - sl
        if risk > 0:
            tp = round(entry + (risk * 3.0), 5)
            res.update({
                "signal": "BUY",
                "entry": round(entry, 5),
                "sl": round(sl, 5),
                "tp": tp,
                "rr": 3.0,
                "details": f"Bullish HalfTrend (+1) & DMI (+DI={curr['plus_di']:.1f} > -DI={curr['minus_di']:.1f}, ADX={curr['adx']:.1f}). HalfTrend SL={curr['ht_line']:.5f} (1:3 RR)"
            })
            return res

    # Bearish Conditions:
    ht_bear = curr["ht_trend"] == -1
    dmi_bear = curr["minus_di"] > curr["plus_di"]
    new_bear_trigger = (prev["ht_trend"] == 1 and curr["ht_trend"] == -1) or \
                       (prev["minus_di"] <= prev["plus_di"] and curr["minus_di"] > curr["plus_di"]) or \
                       (prev2["ht_trend"] == 1 and prev["ht_trend"] == -1 and curr["ht_trend"] == -1)

    if ht_bear and dmi_bear and new_bear_trigger:
        entry = float(curr["close"])
        sl = float(curr["ht_line"] + (spread * 2) + 0.0003)
        risk = sl - entry
        if risk > 0:
            tp = round(entry - (risk * 3.0), 5)
            res.update({
                "signal": "SELL",
                "entry": round(entry, 5),
                "sl": round(sl, 5),
                "tp": tp,
                "rr": 3.0,
                "details": f"Bearish HalfTrend (-1) & DMI (-DI={curr['minus_di']:.1f} > +DI={curr['plus_di']:.1f}, ADX={curr['adx']:.1f}). HalfTrend SL={curr['ht_line']:.5f} (1:3 RR)"
            })
            return res

    ht_status = "BULLISH" if ht_bull else "BEARISH"
    dmi_status = "BULLISH (+DI > -DI)" if dmi_bull else "BEARISH (-DI > +DI)"
    res["details"] = f"HalfTrend: {ht_status} (HT Line={curr['ht_line']:.5f}), DMI: {dmi_status} (+DI={curr['plus_di']:.1f}, -DI={curr['minus_di']:.1f}, ADX={curr['adx']:.1f}). Waiting for synchronized entry flip."
    return res


def main():
    parser = argparse.ArgumentParser(description="Booming Bulls HalfTrend + DMI Scanner")
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
        print(f"[{now_str}] Booming Bulls HalfTrend DMI Scanner -> {res['symbol']}")
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
