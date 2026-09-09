"""
Strategy 67: Mukul Chowdhary 10 EMA + Session VWAP 1:4 RR Scanner
Creator Video ID: I7i8MnUsqJ4 (70.0% Win Rate, 1:4 Risk-to-Reward System)
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


def calculate_vwap(df):
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["tick_volume"].replace(0, 1)
    cum_vol = vol.cumsum()
    cum_pv = (typical_price * vol).cumsum()
    return cum_pv / (cum_vol + 1e-10)


def scan_symbol(symbol="EURUSD"):
    res = {
        "strategy": "Mukul Chowdhary 10 EMA + VWAP",
        "symbol": symbol,
        "signal": "NEUTRAL",
        "timeframe": "M15",
        "entry": None,
        "sl": None,
        "tp": None,
        "rr": 4.0,
        "details": ""
    }

    if not mt5:
        res["details"] = "MetaTrader5 library not installed."
        return res

    sym = resolve_symbol(symbol)
    if not mt5.symbol_select(sym, True):
        res["details"] = f"Failed to select symbol {sym}"
        return res

    df = get_rates(sym, mt5.TIMEFRAME_M15, 90)
    if df is None or len(df) < 40:
        res["details"] = "Insufficient M15 rates"
        return res

    df["ema10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["vwap"] = calculate_vwap(df)

    curr = df.iloc[-1]
    prev1 = df.iloc[-2]

    ema10 = float(curr["ema10"])
    vwap = float(curr["vwap"])
    spread = mt5.symbol_info(sym).spread * mt5.symbol_info(sym).point

    # Bullish: Price above VWAP + Pullback touch 10 EMA + Green rejection candle
    bull_bias = curr["close"] > vwap
    tested_ema_bull = curr["low"] <= ema10 * 1.001 or prev1["low"] <= ema10 * 1.001
    bull_rejection = curr["close"] > curr["open"] and curr["close"] >= ema10

    if bull_bias and tested_ema_bull and bull_rejection:
        entry = float(curr["close"])
        sl = float(min(curr["low"], prev1["low"], ema10) - (spread * 2) - 0.0003)
        risk = entry - sl
        if risk > 0:
            tp = round(entry + (risk * 4.0), 5)
            res.update({
                "signal": "BUY",
                "entry": round(entry, 5),
                "sl": round(sl, 5),
                "tp": tp,
                "rr": 4.0,
                "details": f"Mukul Chowdhary Bullish Setup: Price > VWAP ({vwap:.5f}), 10 EMA ({ema10:.5f}) pullback bounced with green expansion. Fixed 1:4 RR Target: {tp:.5f}"
            })
            return res

    # Bearish: Price below VWAP + Pullback touch 10 EMA + Red rejection candle
    bear_bias = curr["close"] < vwap
    tested_ema_bear = curr["high"] >= ema10 * 0.999 or prev1["high"] >= ema10 * 0.999
    bear_rejection = curr["close"] < curr["open"] and curr["close"] <= ema10

    if bear_bias and tested_ema_bear and bear_rejection:
        entry = float(curr["close"])
        sl = float(max(curr["high"], prev1["high"], ema10) + (spread * 2) + 0.0003)
        risk = sl - entry
        if risk > 0:
            tp = round(entry - (risk * 4.0), 5)
            res.update({
                "signal": "SELL",
                "entry": round(entry, 5),
                "sl": round(sl, 5),
                "tp": tp,
                "rr": 4.0,
                "details": f"Mukul Chowdhary Bearish Setup: Price < VWAP ({vwap:.5f}), 10 EMA ({ema10:.5f}) pullback rejected with red expansion. Fixed 1:4 RR Target: {tp:.5f}"
            })
            return res

    res["details"] = f"Price={curr['close']:.5f}, 10 EMA={ema10:.5f}, VWAP={vwap:.5f}. Bias: {'BULLISH' if bull_bias else 'BEARISH'}. Awaiting 10 EMA touch."
    return res


def main():
    parser = argparse.ArgumentParser(description="Mukul Chowdhary 10 EMA + VWAP Scanner")
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
        print(f"[{now_str}] Mukul Chowdhary 10 EMA VWAP Scanner -> {res['symbol']}")
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
