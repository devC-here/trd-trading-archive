"""
Trade Like Berlin 'Final Boss' 7 EMA Forex Signal Scanner (MetaTrader 5)
========================================================================
Source: YouTube Shorts @UnfilteredFinanceWithSahil (Video: 2NZW106gUfU)
Creator: Trade Like Berlin (250k Subscribers, "Final Boss of All EMA Strategies")
Backtest Metrics: +52.3% Return in 1 Year, 48% Win Rate, 12.8% Max Drawdown
Timeframe: H1 (Macro Direction Filter) -> M15 (7 EMA Momentum Execution)
Markets: EURUSD, GBPUSD, XAUUSD, USDJPY

Strategy Rules:
1. Pure 7 EMA System:
   - Only ONE indicator used across multiple timeframes: 7 EMA.
2. Higher Timeframe Filter (H1):
   - If H1 Close > 7 EMA: Regime is Bullish -> ONLY BUY setups allowed.
   - If H1 Close < 7 EMA: Regime is Bearish -> ONLY SELL setups allowed.
3. Lower Timeframe Execution (M15):
   - Price pulls back to touch the 7 EMA.
   - A confirmation candle closes in the direction of the H1 trend (above 7 EMA for Longs).
   - Entry: At the close of the confirmation candle.
   - Stop Loss: Slightly below the entry candle low.
   - Take Profit: Dynamic Risk-to-Reward:
       If SL risk is tight (< 20 pips), Target 1:3 RR.
       If SL risk is wide (>= 20 pips), Target 1:2 RR.
4. Reverse completely for Sell setups when H1 is Bearish.

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


def get_h1_trend(symbol: str):
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 30)
    if rates is None or len(rates) < 15:
        return None, None
    df = pd.DataFrame(rates)
    df["ema7"] = df["close"].ewm(span=7, adjust=False).mean()
    last_close = float(df["close"].iloc[-2])
    last_ema = float(df["ema7"].iloc[-2])
    trend = "BULLISH" if last_close > last_ema else "BEARISH"
    return trend, last_ema


def scan_symbol(symbol: str) -> dict:
    resolved = resolve_symbol(symbol)
    if not mt5.symbol_select(resolved, True):
        return {"action": "HOLD", "reason": f"Symbol {resolved} not found"}

    h1_trend, h1_ema = get_h1_trend(resolved)
    if h1_trend is None:
        return {"action": "HOLD", "reason": "Failed to get H1 trend"}

    rates_m15 = mt5.copy_rates_from_pos(resolved, mt5.TIMEFRAME_M15, 0, 60)
    if rates_m15 is None or len(rates_m15) < 20:
        return {"action": "HOLD", "reason": "Insufficient M15 bars"}

    df_m15 = pd.DataFrame(rates_m15)
    df_m15["time"] = pd.to_datetime(df_m15["time"], unit="s")
    df_m15["ema7"] = df_m15["close"].ewm(span=7, adjust=False).mean()

    bar = df_m15.iloc[-2]       # Last closed candle
    prev_bar = df_m15.iloc[-3]  # Previous candle
    close = float(bar["close"])
    low = float(bar["low"])
    high = float(bar["high"])
    ema7 = float(bar["ema7"])
    prev_ema7 = float(prev_bar["ema7"])

    # Bullish condition:
    # 1. H1 is Bullish
    # 2. Previous bar or current bar low touched/penetrated 7 EMA
    # 3. Current bar closes cleanly above 7 EMA
    if h1_trend == "BULLISH":
        touched_ema = (prev_bar["low"] <= prev_ema7 * 1.0005) or (low <= ema7 * 1.0005)
        closed_above = (close > ema7) and (prev_bar["close"] <= prev_ema7 or bar["close"] > bar["open"])
        if touched_ema and closed_above:
            sl = low - (high - low) * 0.15
            risk = close - sl
            if risk <= 0:
                risk = close * 0.002
                sl = close - risk

            # Dynamic RR: tight SL -> 1:3, larger SL -> 1:2
            point_risk = risk / (close * 0.0001)  # approx pips
            mult = 3.0 if point_risk < 20.0 else 2.0
            tp = close + mult * risk
            rr = round((tp - close) / risk, 2)
            return {
                "action": "BUY",
                "symbol": resolved,
                "entry": round(close, 5),
                "sl": round(sl, 5),
                "tp": round(tp, 5),
                "rr": rr,
                "h1_trend": h1_trend,
                "m15_7_ema": round(ema7, 5),
                "reason": f"Berlin 7 EMA Rebound in H1 Bullish Trend (Dynamic RR 1:{mult})"
            }

    # Bearish condition:
    if h1_trend == "BEARISH":
        touched_ema = (prev_bar["high"] >= prev_ema7 * 0.9995) or (high >= ema7 * 0.9995)
        closed_below = (close < ema7) and (prev_bar["close"] >= prev_ema7 or bar["close"] < bar["open"])
        if touched_ema and closed_below:
            sl = high + (high - low) * 0.15
            risk = sl - close
            if risk <= 0:
                risk = close * 0.002
                sl = close + risk

            point_risk = risk / (close * 0.0001)
            mult = 3.0 if point_risk < 20.0 else 2.0
            tp = close - mult * risk
            rr = round((close - tp) / risk, 2)
            return {
                "action": "SELL",
                "symbol": resolved,
                "entry": round(close, 5),
                "sl": round(sl, 5),
                "tp": round(tp, 5),
                "rr": rr,
                "h1_trend": h1_trend,
                "m15_7_ema": round(ema7, 5),
                "reason": f"Berlin 7 EMA Rejection in H1 Bearish Trend (Dynamic RR 1:{mult})"
            }

    return {
        "action": "HOLD",
        "symbol": resolved,
        "entry": round(float(df_m15.iloc[-1]["close"]), 5),
        "h1_trend": h1_trend,
        "m15_7_ema": round(ema7, 5),
        "reason": f"H1 Trend: {h1_trend} | Close: {round(close,5)} vs M15 7 EMA: {round(ema7,5)}"
    }


def main():
    parser = argparse.ArgumentParser(description="Trade Like Berlin 7 EMA Scanner")
    parser.add_argument("symbol", nargs="?", default="EURUSD.0", help="Forex Symbol")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    args = parser.parse_args()

    if not mt5.initialize():
        print(f"[ERROR] MT5 init failed: {mt5.last_error()}")
        sys.exit(1)

    print("=" * 75)
    print("  TRADE LIKE BERLIN 'FINAL BOSS' 7 EMA FOREX SCANNER")
    print(f"  Target Symbol: {args.symbol} | Multi-TF: H1 Trend Filter -> M15 Execution")
    print(f"  Source: Trade Like Berlin (2NZW106gUfU) | +52.3% Return, Dynamic 1:2/1:3 RR")
    print("=" * 75)

    try:
        while True:
            res = scan_symbol(args.symbol)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if res["action"] in ["BUY", "SELL"]:
                print(f"\n>>> [{ts}] [{res['action']}_SIGNAL] {res['symbol']}")
                print(f"    Entry: {res['entry']} | SL: {res['sl']} | TP: {res['tp']} | R:R: 1:{res['rr']}")
                print(f"    H1 Trend: {res['h1_trend']} | M15 7 EMA: {res['m15_7_ema']}")
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
