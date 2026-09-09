"""
Strategy 53: Traders Paradise 'Red Box' S/R Fakeout Reversal Scanner
Creator Video ID: KJmLvxyzXSg (+43.0% Annual Return, 53.0% Win Rate, 10.0% Max DD)
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
        "strategy": "Traders Paradise Red Box Fakeout Reversal",
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

    df = get_rates(sym, mt5.TIMEFRAME_M15, 100)
    if df is None or len(df) < 50:
        res["details"] = "Insufficient M15 rates"
        return res

    # Establish S/R from historical bars (excluding last 5 bars)
    historical = df.iloc[-50:-6]
    resistance = float(historical["high"].max())
    support = float(historical["low"].min())

    # Average candle body over last 30 bars
    avg_body = abs(df.iloc[-30:-1]["close"] - df.iloc[-30:-1]["open"]).mean()

    recent = df.iloc[-6:]
    curr = df.iloc[-1]
    prev1 = df.iloc[-2]
    prev2 = df.iloc[-3]

    curr_body = abs(curr["close"] - curr["open"])
    spread = mt5.symbol_info(sym).spread * mt5.symbol_info(sym).point

    # 1. Bearish "Red Box" Setup:
    # - In last 5 candles, price breached Resistance (high > resistance)
    # - Current or prev1 candle is a large red candle (curr_body >= avg_body * 1.25 and close < open)
    # - Close is now back at or below resistance
    resistance_breached = recent["high"].max() > resistance
    is_large_red = (curr["close"] < curr["open"]) and (curr_body >= avg_body * 1.2)
    fakeout_high = recent["high"].max()

    if resistance_breached and is_large_red and curr["close"] <= resistance * 1.0005:
        entry = float(curr["close"])
        sl = float(max(fakeout_high, curr["high"]) + (spread * 2) + 0.0003)
        risk = sl - entry
        if risk > 0:
            tp = min(entry - (risk * 2.0), support)
            rr = round((entry - tp) / risk, 2)
            res.update({
                "signal": "SELL",
                "entry": round(entry, 5),
                "sl": round(sl, 5),
                "tp": round(tp, 5),
                "rr": rr,
                "details": f"Red Box Bearish Reversal: Fakeout above Resistance {resistance:.5f} (Peak {fakeout_high:.5f}), followed by large Red Box (Body={curr_body:.5f} vs Avg={avg_body:.5f}). Target Support: {support:.5f}"
            })
            return res

    # 2. Bullish "Green Box" Setup:
    support_breached = recent["low"].min() < support
    is_large_green = (curr["close"] > curr["open"]) and (curr_body >= avg_body * 1.2)
    fakeout_low = recent["low"].min()

    if support_breached and is_large_green and curr["close"] >= support * 0.9995:
        entry = float(curr["close"])
        sl = float(min(fakeout_low, curr["low"]) - (spread * 2) - 0.0003)
        risk = entry - sl
        if risk > 0:
            tp = max(entry + (risk * 2.0), resistance)
            rr = round((tp - entry) / risk, 2)
            res.update({
                "signal": "BUY",
                "entry": round(entry, 5),
                "sl": round(sl, 5),
                "tp": round(tp, 5),
                "rr": rr,
                "details": f"Green Box Bullish Reversal: Fakeout below Support {support:.5f} (Trough {fakeout_low:.5f}), followed by large Green Box (Body={curr_body:.5f} vs Avg={avg_body:.5f}). Target Resistance: {resistance:.5f}"
            })
            return res

    res["details"] = f"M15 S/R: Support={support:.5f}, Resistance={resistance:.5f}, Current Price={curr['close']:.5f}. Avg Body={avg_body:.5f}. No fakeout box trigger."
    return res


def main():
    parser = argparse.ArgumentParser(description="Traders Paradise Red Box Scanner")
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
        print(f"[{now_str}] Traders Paradise Red Box Scanner -> {res['symbol']}")
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
