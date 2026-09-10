"""
Hold With Priyank 21 EMA Pullback Breakout Signal Scanner (MetaTrader 5)
========================================================================
Source: YouTube Shorts @UnfilteredFinanceWithSahil (Video: 0pbzM_4QkOs)
Creator: Hold with Priyank (SEBI Registered Analyst)
Backtest Metrics: +44.80% Return in 1 Year, 103/212 Wins, 14.6% Max Drawdown
Timeframe: M15 (or M5)
Markets: Universal Forex (EURUSD, GBPUSD, USDJPY), Gold (XAUUSD), Indices

Strategy Rules:
1. 21 EMA Breakout:
   - A distinct Breakout candle crosses over the 21 EMA (from below to above for Longs).
2. The 3-Candle Retest Window:
   - Within the next 1 to 3 candles following the breakout, at least ONE candle must be a
     Bearish/Pullback candle (Red candle).
   - The close of this pullback candle must remain strictly ABOVE the 21 EMA (showing support).
3. The High-Breakout Trigger:
   - When any subsequent candle breaks above the High of this pullback candle:
   - Entry: On break or close above pullback high.
   - Stop Loss: Slightly below the low of the pullback candle.
   - Take Profit: 1:1.5 to 1:2.0 Risk-to-Reward.
4. Reverse completely for Bearish Short Setups (Break below 21 EMA, Green pullback bar holding below EMA, break below low).

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


def scan_symbol(symbol: str, timeframe=mt5.TIMEFRAME_M15) -> dict:
    resolved = resolve_symbol(symbol)
    if not mt5.symbol_select(resolved, True):
        return {"action": "HOLD", "reason": f"Symbol {resolved} not found"}

    rates = mt5.copy_rates_from_pos(resolved, timeframe, 0, 100)
    if rates is None or len(rates) < 40:
        return {"action": "HOLD", "reason": "Insufficient bars retrieved"}

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()

    bar = df.iloc[-2]       # Last closed candle
    close = float(bar["close"])
    low = float(bar["low"])
    high = float(bar["high"])
    ema21 = float(bar["ema21"])

    # Search past 8 bars for a 21 EMA breakout followed by a pullback candle
    for i in range(len(df) - 8, len(df) - 2):
        c_prev = df.iloc[i - 1]
        c_bo = df.iloc[i]

        # Bullish Breakout across 21 EMA
        if c_prev["close"] <= c_prev["ema21"] and c_bo["close"] > c_bo["ema21"]:
            # Look at subsequent candles (up to 3) for a red pullback candle
            post_bars = df.iloc[i + 1:len(df) - 1]
            pullback_reds = post_bars[(post_bars["close"] < post_bars["open"]) & (post_bars["close"] >= post_bars["ema21"])]

            if len(pullback_reds) > 0:
                pb_bar = pullback_reds.iloc[0]
                pb_high = float(pb_bar["high"])
                pb_low = float(pb_bar["low"])

                # Trigger: Did current bar break above pb_high?
                if close > pb_high and float(df.iloc[-3]["close"]) <= pb_high:
                    sl = pb_low - (high - low) * 0.15
                    risk = close - sl
                    if risk <= 0:
                        risk = close * 0.002
                        sl = close - risk
                    tp = close + 1.5 * risk
                    rr = round((tp - close) / risk, 2)
                    return {
                        "action": "BUY",
                        "symbol": resolved,
                        "entry": round(close, 5),
                        "sl": round(sl, 5),
                        "tp": round(tp, 5),
                        "rr": rr,
                        "ema21": round(ema21, 5),
                        "pullback_high": round(pb_high, 5),
                        "pullback_low": round(pb_low, 5),
                        "reason": f"Hold With Priyank 21 EMA Pullback Breakout above {round(pb_high,5)} (EMA: {round(ema21,5)})"
                    }

        # Bearish Breakdown across 21 EMA
        if c_prev["close"] >= c_prev["ema21"] and c_bo["close"] < c_bo["ema21"]:
            post_bars = df.iloc[i + 1:len(df) - 1]
            pullback_greens = post_bars[(post_bars["close"] > post_bars["open"]) & (post_bars["close"] <= post_bars["ema21"])]

            if len(pullback_greens) > 0:
                pb_bar = pullback_greens.iloc[0]
                pb_high = float(pb_bar["high"])
                pb_low = float(pb_bar["low"])

                # Trigger: Did current bar break below pb_low?
                if close < pb_low and float(df.iloc[-3]["close"]) >= pb_low:
                    sl = pb_high + (high - low) * 0.15
                    risk = sl - close
                    if risk <= 0:
                        risk = close * 0.002
                        sl = close + risk
                    tp = close - 1.5 * risk
                    rr = round((close - tp) / risk, 2)
                    return {
                        "action": "SELL",
                        "symbol": resolved,
                        "entry": round(close, 5),
                        "sl": round(sl, 5),
                        "tp": round(tp, 5),
                        "rr": rr,
                        "ema21": round(ema21, 5),
                        "pullback_high": round(pb_high, 5),
                        "pullback_low": round(pb_low, 5),
                        "reason": f"Hold With Priyank 21 EMA Pullback Breakdown below {round(pb_low,5)} (EMA: {round(ema21,5)})"
                    }

    return {
        "action": "HOLD",
        "symbol": resolved,
        "entry": round(float(df.iloc[-1]["close"]), 5),
        "ema21": round(ema21, 5),
        "reason": f"No active 21 EMA pullback breakout trigger (Price: {round(close,5)}, 21 EMA: {round(ema21,5)})"
    }


def main():
    parser = argparse.ArgumentParser(description="Hold With Priyank 21 EMA Pullback Scanner")
    parser.add_argument("symbol", nargs="?", default="EURUSD.0", help="Forex Symbol")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    args = parser.parse_args()

    if not mt5.initialize():
        print(f"[ERROR] MT5 init failed: {mt5.last_error()}")
        sys.exit(1)

    print("=" * 75)
    print("  HOLD WITH PRIYANK 21 EMA PULLBACK BREAKOUT FOREX SCANNER")
    print(f"  Target Symbol: {args.symbol} | Timeframe: M15")
    print(f"  Source: Hold with Priyank (0pbzM_4QkOs) | +44.80% Return, 1:1.5 RR")
    print("=" * 75)

    try:
        while True:
            res = scan_symbol(args.symbol)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if res["action"] in ["BUY", "SELL"]:
                print(f"\n>>> [{ts}] [{res['action']}_SIGNAL] {res['symbol']}")
                print(f"    Entry: {res['entry']} | SL: {res['sl']} | TP: {res['tp']} | R:R: 1:{res['rr']}")
                print(f"    21 EMA: {res['ema21']} | Pullback Range: [{res['pullback_low']}, {res['pullback_high']}]")
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
