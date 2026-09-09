"""
The Trading School Secret MBP Session Setup - Forex/Gold Signal Scanner (MetaTrader 5)
======================================================================================
Source: YouTube Shorts @UnfilteredFinanceWithSahil (Video: 9Jxhj7mVb1w)
Creator: The Trading School
Backtest Metrics: +84.6% Return in 1 Year, 46% Win Rate, 18.2% Drawdown, Fixed 1:3 RR
Timeframe: H1 (Session Opening Range) -> M5 (FVG Retest & Execution)
Markets: Gold (XAUUSD), EURUSD, GBPUSD, USDJPY

Strategy Rules:
1. High Timeframe (H1):
   - Mark the First H1 Candle High and Low of the active daily session.
2. Low Timeframe (M5):
   - Wait for price to break out above H1 High (Bullish) or below H1 Low (Bearish).
   - Identify the Fair Value Gap (FVG) created during or immediately following the breakout:
       Bullish FVG: Bar1 High < Bar3 Low (Imbalance zone [Bar1 High, Bar3 Low])
       Bearish FVG: Bar1 Low > Bar3 High (Imbalance zone [Bar3 High, Bar1 Low])
3. Entry Trigger:
   - Price retraces back into the FVG zone.
   - A confirmation candle closes in the breakout direction (Bullish candle for Buy).
   - Stop Loss: Slightly below the FVG zone low.
   - Take Profit: Fixed 1:3 Risk-to-Reward.

SIGNAL ONLY - NO TRADE EXECUTION
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import time
import argparse
from datetime import datetime
import pandas as pd
import numpy as np

try:
    import MetaTrader5 as mt5
except ImportError:
    print("[ERROR] MetaTrader5 package not installed. Run: pip install MetaTrader5")
    sys.exit(1)


def resolve_symbol(requested_symbol: str) -> str:
    if not mt5.initialize():
        return requested_symbol
    symbols = mt5.symbols_get()
    if not symbols:
        return requested_symbol
    all_names = [s.name for s in symbols]
    if requested_symbol in all_names:
        return requested_symbol
    for s in all_names:
        if s.startswith(requested_symbol) or requested_symbol.startswith(s.split(".")[0]):
            return s
    return requested_symbol


def get_h1_session_range(symbol: str):
    """
    Fetches H1 bars to determine today's opening H1 candle range.
    """
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 24)
    if rates is None or len(rates) < 2:
        return None, None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    
    # Filter bars belonging to the current day
    today = df["time"].iloc[-1].date()
    today_bars = df[df["time"].dt.date == today]
    if len(today_bars) > 0:
        first_h1 = today_bars.iloc[0]
    else:
        first_h1 = df.iloc[-2]
        
    return first_h1["high"], first_h1["low"]


def find_m5_fvgs(df: pd.DataFrame):
    """
    Finds bullish and bearish FVGs on M5.
    """
    bull_fvgs = []
    bear_fvgs = []
    for i in range(len(df) - 25, len(df) - 1):
        c1 = df.iloc[i - 2]
        c2 = df.iloc[i - 1]
        c3 = df.iloc[i]
        # Bullish FVG
        if c1["high"] < c3["low"] and c2["close"] > c2["open"]:
            bull_fvgs.append({
                "bar_idx": i,
                "low": c1["high"],
                "high": c3["low"],
                "time": c2["time"]
            })
        # Bearish FVG
        if c1["low"] > c3["high"] and c2["close"] < c2["open"]:
            bear_fvgs.append({
                "bar_idx": i,
                "low": c3["high"],
                "high": c1["low"],
                "time": c2["time"]
            })
    return bull_fvgs, bear_fvgs


def scan_symbol(symbol: str) -> dict:
    resolved = resolve_symbol(symbol)
    if not mt5.symbol_select(resolved, True):
        return {"action": "HOLD", "reason": f"Symbol {resolved} not found"}

    h1_high, h1_low = get_h1_session_range(resolved)
    if h1_high is None:
        return {"action": "HOLD", "reason": "Failed to retrieve H1 session range"}

    rates_m5 = mt5.copy_rates_from_pos(resolved, mt5.TIMEFRAME_M5, 0, 120)
    if rates_m5 is None or len(rates_m5) < 30:
        return {"action": "HOLD", "reason": "Insufficient M5 bars"}

    df_m5 = pd.DataFrame(rates_m5)
    df_m5["time"] = pd.to_datetime(df_m5["time"], unit="s")

    bull_fvgs, bear_fvgs = find_m5_fvgs(df_m5)

    bar = df_m5.iloc[-2]  # last closed M5 candle
    close = bar["close"]
    low = bar["low"]
    high = bar["high"]
    is_bull = bar["close"] > bar["open"]
    is_bear = bar["close"] < bar["open"]

    # Bullish scenario: Price broke above H1 High, then pulled back to Bullish FVG
    if close > h1_high:
        for fvg in reversed(bull_fvgs):
            if low <= fvg["high"] and close >= fvg["low"] and is_bull:
                sl = fvg["low"] - (fvg["high"] - fvg["low"]) * 0.2
                risk = close - sl
                if risk <= 0:
                    risk = close * 0.002
                    sl = close - risk
                tp = close + 3.0 * risk  # Fixed 1:3 RR
                rr = round((tp - close) / risk, 2)
                return {
                    "action": "BUY",
                    "symbol": resolved,
                    "entry": round(close, 5),
                    "sl": round(sl, 5),
                    "tp": round(tp, 5),
                    "rr": rr,
                    "h1_high": round(float(h1_high), 5),
                    "fvg": [round(float(fvg["low"]), 5), round(float(fvg["high"]), 5)],
                    "reason": f"MBP H1 Breakout above {round(float(h1_high),5)} + M5 FVG Retest"
                }

    # Bearish scenario: Price broke below H1 Low, then pulled back to Bearish FVG
    if close < h1_low:
        for fvg in reversed(bear_fvgs):
            if high >= fvg["low"] and close <= fvg["high"] and is_bear:
                sl = fvg["high"] + (fvg["high"] - fvg["low"]) * 0.2
                risk = sl - close
                if risk <= 0:
                    risk = close * 0.002
                    sl = close + risk
                tp = close - 3.0 * risk  # Fixed 1:3 RR
                rr = round((close - tp) / risk, 2)
                return {
                    "action": "SELL",
                    "symbol": resolved,
                    "entry": round(close, 5),
                    "sl": round(sl, 5),
                    "tp": round(tp, 5),
                    "rr": rr,
                    "h1_low": round(float(h1_low), 5),
                    "fvg": [round(float(fvg["low"]), 5), round(float(fvg["high"]), 5)],
                    "reason": f"MBP H1 Breakdown below {round(float(h1_low),5)} + M5 Bearish FVG Retest"
                }

    return {
        "action": "HOLD",
        "symbol": resolved,
        "entry": round(float(df_m5.iloc[-1]["close"]), 5),
        "h1_range": [round(float(h1_low), 5), round(float(h1_high), 5)],
        "reason": f"Price {round(float(close),5)} inside or waiting for FVG retest of H1 range [{round(float(h1_low),5)}, {round(float(h1_high),5)}]"
    }


def main():
    parser = argparse.ArgumentParser(description="The Trading School Secret MBP Session Setup")
    parser.add_argument("symbol", nargs="?", default="EURUSD.0", help="Forex / Gold Symbol")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    args = parser.parse_args()

    if not mt5.initialize():
        print(f"[ERROR] MT5 init failed: {mt5.last_error()}")
        sys.exit(1)

    print("=" * 75)
    print("  THE TRADING SCHOOL SECRET MBP SESSION SETUP SCANNER")
    print(f"  Target Symbol: {args.symbol} | Multi-TF: H1 Session Range -> M5 FVG")
    print(f"  Source: The Trading School (9Jxhj7mVb1w) | +84.6% Return, 1:3 Fixed RR")
    print("=" * 75)

    try:
        while True:
            res = scan_symbol(args.symbol)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if res["action"] in ["BUY", "SELL"]:
                print(f"\n>>> [{ts}] [{res['action']}_SIGNAL] {res['symbol']}")
                print(f"    Entry: {res['entry']} | SL: {res['sl']} | TP: {res['tp']} | R:R: 1:{res['rr']}")
                print(f"    FVG Zone: {res['fvg']}")
                print(f"    Reason: {res['reason']}")
            else:
                print(f"[{ts}] {res.get('symbol', args.symbol)}: {res['action']} - {res['reason']}")

            if not args.loop:
                break
            time.sleep(30)
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
