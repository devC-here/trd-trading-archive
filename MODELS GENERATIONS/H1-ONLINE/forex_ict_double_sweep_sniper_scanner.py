"""
Strategy 46: ICT Double Sweep Sniper Entry Scanner (Inter Equity Trading Model)
Creator Video ID: bffQ8_1ShO8 (+58.2% Annual Return, 63.5% Win Rate, 12.4% Max DD)
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


def scan_symbol(symbol="EURUSD"):
    res = {
        "strategy": "ICT Double Sweep Sniper",
        "symbol": symbol,
        "signal": "NEUTRAL",
        "timeframe": "H1 (HTF Bias) + M15 (Trigger)",
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

    # 1. Higher Timeframe (H1) Bias Check
    df_htf = get_rates(sym, mt5.TIMEFRAME_H1, 80)
    if df_htf is None or len(df_htf) < 40:
        res["details"] = "Insufficient H1 rates"
        return res

    # Check recent swing & momentum on H1
    recent_h1 = df_htf.iloc[-25:-1]
    h1_high = recent_h1["high"].max()
    h1_low = recent_h1["low"].min()
    curr_h1_close = df_htf.iloc[-1]["close"]
    
    # 2. Lower Timeframe (M15) Rates
    df_ltf = get_rates(sym, mt5.TIMEFRAME_M15, 60)
    if df_ltf is None or len(df_ltf) < 30:
        res["details"] = "Insufficient M15 rates"
        return res

    c_now = df_ltf.iloc[-1]
    c_prev = df_ltf.iloc[-2]
    c_prev2 = df_ltf.iloc[-3]
    c_prev3 = df_ltf.iloc[-4]

    # Look back 5 to 20 candles for swing reference
    lookback = df_ltf.iloc[-25:-5]
    recent_swing_low = lookback["low"].min()
    recent_swing_high = lookback["high"].max()

    # Bullish Double Sweep:
    # 1. Candle 1 sweeps recent swing low: c_prev2 or c_prev3 dipped below recent_swing_low
    # 2. Candle 2 establishes a low
    # 3. Current or previous candle sweeps Candle 2's low (secondary sweep) and closes back above it
    bullish_sweep = False
    bearish_sweep = False

    # Scenario A: c_prev2 swept swing low, c_prev set interim low, c_now swept c_prev low and closed above it
    if c_prev2["low"] < recent_swing_low:
        candle2_low = c_prev["low"]
        if c_now["low"] < candle2_low and c_now["close"] > candle2_low and c_now["close"] > c_now["open"]:
            bullish_sweep = True
            sweep1_low = min(c_prev2["low"], c_prev["low"], c_now["low"])
    # Scenario B: c_prev3 swept swing low, c_prev2 set low, c_prev swept and c_now confirms
    elif c_prev3["low"] < recent_swing_low:
        candle2_low = c_prev2["low"]
        if (c_prev["low"] < candle2_low and c_prev["close"] > candle2_low) or \
           (c_now["low"] < candle2_low and c_now["close"] > candle2_low):
            bullish_sweep = True
            sweep1_low = min(c_prev3["low"], c_prev2["low"], c_prev["low"])

    # Bearish Double Sweep:
    if c_prev2["high"] > recent_swing_high:
        candle2_high = c_prev["high"]
        if c_now["high"] > candle2_high and c_now["close"] < candle2_high and c_now["close"] < c_now["open"]:
            bearish_sweep = True
            sweep1_high = max(c_prev2["high"], c_prev["high"], c_now["high"])
    elif c_prev3["high"] > recent_swing_high:
        candle2_high = c_prev2["high"]
        if (c_prev["high"] > candle2_high and c_prev["close"] < candle2_high) or \
           (c_now["high"] > candle2_high and c_now["close"] < candle2_high):
            bearish_sweep = True
            sweep1_high = max(c_prev3["high"], c_prev2["high"], c_prev["high"])

    spread = mt5.symbol_info(sym).spread * mt5.symbol_info(sym).point

    if bullish_sweep:
        entry = float(c_now["close"])
        sl = float(sweep1_low - (spread * 2) - 0.0003)
        risk = entry - sl
        if risk > 0:
            tp = max(entry + (risk * 2.5), float(h1_high))
            rr = round((tp - entry) / risk, 2)
            res.update({
                "signal": "BUY",
                "entry": round(entry, 5),
                "sl": round(sl, 5),
                "tp": round(tp, 5),
                "rr": rr,
                "details": f"Bullish Double Sweep Sniper: Swing low {recent_swing_low:.5f} swept, secondary low swept with bullish reclaim. HTF High: {h1_high:.5f}"
            })
    elif bearish_sweep:
        entry = float(c_now["close"])
        sl = float(sweep1_high + (spread * 2) + 0.0003)
        risk = sl - entry
        if risk > 0:
            tp = min(entry - (risk * 2.5), float(h1_low))
            rr = round((entry - tp) / risk, 2)
            res.update({
                "signal": "SELL",
                "entry": round(entry, 5),
                "sl": round(sl, 5),
                "tp": round(tp, 5),
                "rr": rr,
                "details": f"Bearish Double Sweep Sniper: Swing high {recent_swing_high:.5f} swept, secondary high swept with bearish reclaim. HTF Low: {h1_low:.5f}"
            })
    else:
        res["details"] = f"No Double Sweep trigger. M15 Range: {c_now['low']:.5f} - {c_now['high']:.5f}, HTF Range: {h1_low:.5f} - {h1_high:.5f}"

    return res


def main():
    parser = argparse.ArgumentParser(description="ICT Double Sweep Sniper Scanner")
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
        print(f"[{now_str}] ICT Double Sweep Sniper Scanner -> {res['symbol']}")
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
