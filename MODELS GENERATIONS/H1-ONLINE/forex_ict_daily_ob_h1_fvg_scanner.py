"""
Strategy 47: MS Trading ICT Daily Order Block + H1 FVG Retest Scanner
Creator Video ID: 4W3SUtxl0u8 (+74.1% Annual Return, 53.2% Win Rate, 14.3% Max DD)
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


def get_rates(symbol, timeframe, n_bars=100):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


def find_daily_order_blocks(df_d1):
    # Scan recent 30 daily candles for clear order blocks
    bullish_obs = []
    bearish_obs = []

    for i in range(2, len(df_d1) - 2):
        c_prev = df_d1.iloc[i]
        c_next1 = df_d1.iloc[i + 1]
        c_next2 = df_d1.iloc[i + 2]

        # Bullish OB: Red candle followed by strong bullish displacement
        if c_prev["close"] < c_prev["open"]:
            displacement = (c_next1["close"] - c_next1["open"]) + (c_next2["close"] - c_next2["open"])
            avg_body = (df_d1["high"] - df_d1["low"]).mean()
            if displacement > avg_body * 1.2:
                ob_high = max(c_prev["open"], c_prev["close"])
                ob_low = c_prev["low"]
                ob_mid = (ob_high + ob_low) / 2.0
                bullish_obs.append({
                    "time": c_prev["time"],
                    "high": ob_high,
                    "low": ob_low,
                    "mid": ob_mid,
                    "target": df_d1.iloc[i:]["high"].max()
                })

        # Bearish OB: Green candle followed by strong bearish displacement
        if c_prev["close"] > c_prev["open"]:
            displacement = (c_next1["open"] - c_next1["close"]) + (c_next2["open"] - c_next2["close"])
            avg_body = (df_d1["high"] - df_d1["low"]).mean()
            if displacement > avg_body * 1.2:
                ob_high = c_prev["high"]
                ob_low = min(c_prev["open"], c_prev["close"])
                ob_mid = (ob_high + ob_low) / 2.0
                bearish_obs.append({
                    "time": c_prev["time"],
                    "high": ob_high,
                    "low": ob_low,
                    "mid": ob_mid,
                    "target": df_d1.iloc[i:]["low"].min()
                })

    return bullish_obs, bearish_obs


def scan_symbol(symbol="EURUSD"):
    res = {
        "strategy": "ICT Daily OB + H1 FVG Retest",
        "symbol": symbol,
        "signal": "NEUTRAL",
        "timeframe": "D1 (OB Zone) + H1 (MSS & FVG Retest)",
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

    # 1. Fetch Daily Data
    df_d1 = get_rates(sym, mt5.TIMEFRAME_D1, 60)
    if df_d1 is None or len(df_d1) < 25:
        res["details"] = "Insufficient D1 rates"
        return res

    bullish_obs, bearish_obs = find_daily_order_blocks(df_d1)
    curr_d1 = df_d1.iloc[-1]
    curr_price = curr_d1["close"]

    # 2. Fetch H1 Data
    df_h1 = get_rates(sym, mt5.TIMEFRAME_H1, 70)
    if df_h1 is None or len(df_h1) < 30:
        res["details"] = "Insufficient H1 rates"
        return res

    recent_h1 = df_h1.iloc[-15:]
    h1_now = df_h1.iloc[-1]
    h1_prev = df_h1.iloc[-2]
    h1_prev2 = df_h1.iloc[-3]

    # Check Bullish Setup: Price in Bullish Daily OB zone and H1 MSS + FVG
    signal = "NEUTRAL"
    spread = mt5.symbol_info(sym).spread * mt5.symbol_info(sym).point

    for ob in reversed(bullish_obs[-5:]):
        # Is price tapping or inside OB zone?
        if ob["low"] <= curr_price <= ob["high"] * 1.0025 or ob["low"] <= h1_now["low"] <= ob["high"]:
            # Check H1 Market Structure Shift (breaking swing high) and FVG
            swing_high_h1 = df_h1.iloc[-25:-3]["high"].max()
            # FVG between h1_prev2 and h1_now
            has_fvg = h1_now["low"] > h1_prev2["high"] or h1_prev["low"] > df_h1.iloc[-4]["high"]
            is_mss = h1_prev["close"] > swing_high_h1 or h1_now["close"] > swing_high_h1

            if is_mss or (has_fvg and h1_now["close"] > h1_now["open"] and curr_price >= ob["mid"]):
                entry = float(h1_now["close"])
                sl = float(min(ob["mid"], h1_now["low"]) - (spread * 2) - 0.0003)
                risk = entry - sl
                if risk > 0:
                    target_liq = float(max(ob["target"], df_d1.iloc[-15:]["high"].max()))
                    tp = max(entry + (risk * 2.0), target_liq)
                    rr = round((tp - entry) / risk, 2)
                    res.update({
                        "signal": "BUY",
                        "entry": round(entry, 5),
                        "sl": round(sl, 5),
                        "tp": round(tp, 5),
                        "rr": rr,
                        "details": f"Price in Daily Bullish OB [{ob['low']:.5f} - {ob['high']:.5f}], Mid: {ob['mid']:.5f}. H1 MSS/FVG Confirmed. Target Daily Liq: {tp:.5f}"
                    })
                    return res

    for ob in reversed(bearish_obs[-5:]):
        if ob["low"] * 0.9975 <= curr_price <= ob["high"] or ob["low"] <= h1_now["high"] <= ob["high"]:
            swing_low_h1 = df_h1.iloc[-25:-3]["low"].min()
            has_fvg = h1_now["high"] < h1_prev2["low"] or h1_prev["high"] < df_h1.iloc[-4]["low"]
            is_mss = h1_prev["close"] < swing_low_h1 or h1_now["close"] < swing_low_h1

            if is_mss or (has_fvg and h1_now["close"] < h1_now["open"] and curr_price <= ob["mid"]):
                entry = float(h1_now["close"])
                sl = float(max(ob["mid"], h1_now["high"]) + (spread * 2) + 0.0003)
                risk = sl - entry
                if risk > 0:
                    target_liq = float(min(ob["target"], df_d1.iloc[-15:]["low"].min()))
                    tp = min(entry - (risk * 2.0), target_liq)
                    rr = round((entry - tp) / risk, 2)
                    res.update({
                        "signal": "SELL",
                        "entry": round(entry, 5),
                        "sl": round(sl, 5),
                        "tp": round(tp, 5),
                        "rr": rr,
                        "details": f"Price in Daily Bearish OB [{ob['low']:.5f} - {ob['high']:.5f}], Mid: {ob['mid']:.5f}. H1 MSS/FVG Confirmed. Target Daily Liq: {tp:.5f}"
                    })
                    return res

    res["details"] = f"Price {curr_price:.5f} outside active Daily OB threshold or waiting for H1 MSS/FVG alignment."
    return res


def main():
    parser = argparse.ArgumentParser(description="ICT Daily OB + H1 FVG Retest Scanner")
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
        print(f"[{now_str}] ICT Daily OB + H1 FVG Scanner -> {res['symbol']}")
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
