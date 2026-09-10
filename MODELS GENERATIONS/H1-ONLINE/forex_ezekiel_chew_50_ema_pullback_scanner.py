"""
Ezekiel Chew 50 EMA Pullback Engulfing Signal Scanner (MetaTrader 5)
====================================================================
Source: YouTube Shorts @UnfilteredFinanceWithSahil (Video: AKuM1cTgPks)
Creator: Ezekiel Chew (Trainer to Banks, Fund Houses & Prop Firms)
Markets: Universal Forex (EURUSD, GBPUSD, USDJPY, XAUUSD), Crypto & Indices
Timeframe: H1 (or M15)

Strategy Rules:
1. 50 EMA Breakout & Structure:
   - Price breaks above 50 EMA and prints a clean Higher High (HH).
2. The 2-Candle Minimum Pullback:
   - At least 2 consecutive Red (Bearish) pullback candles retreat towards the 50 EMA line.
3. The Engulfing Trigger:
   - A strong Bullish Engulfing candle forms directly off the 50 EMA (touches near 50 EMA
     and engulfs the body of the preceding pullback candle).
   - Entry: On candle close of the Bullish Engulfing bar.
   - Stop Loss: Slightly below the low of the Engulfing candle.
   - Take Profit: Previous swing high or minimum 1:2 Risk-to-Reward.
4. Reverse completely for Sell Trades on Bearish 50 EMA breakdowns.

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


def scan_symbol(symbol: str, timeframe=mt5.TIMEFRAME_H1) -> dict:
    resolved = resolve_symbol(symbol)
    if not mt5.symbol_select(resolved, True):
        return {"action": "HOLD", "reason": f"Symbol {resolved} not found"}

    rates = mt5.copy_rates_from_pos(resolved, timeframe, 0, 150)
    if rates is None or len(rates) < 60:
        return {"action": "HOLD", "reason": "Insufficient bars retrieved"}

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    bar = df.iloc[-2]         # Current signal candle (Engulfing trigger)
    pb1 = df.iloc[-3]         # Pullback candle 1
    pb2 = df.iloc[-4]         # Pullback candle 2

    close = float(bar["close"])
    open_p = float(bar["open"])
    low = float(bar["low"])
    high = float(bar["high"])
    ema50 = float(bar["ema50"])

    # Pullback candles
    pb1_red = pb1["close"] < pb1["open"]
    pb2_red = pb2["close"] < pb2["open"]
    pb1_green = pb1["close"] > pb1["open"]
    pb2_green = pb2["close"] > pb2["open"]

    # Bullish Setup:
    # 1. Price above 50 EMA
    # 2. At least 2 consecutive red candles pulling back towards 50 EMA
    # 3. Bar low tests near 50 EMA (within 0.15% distance)
    # 4. Bar is a Bullish Engulfing candle (green body covering pb1 body)
    is_bullish_engulfing = (close > open_p) and (close > pb1["open"]) and (open_p <= pb1["close"] * 1.0005)
    near_ema_bull = (low <= ema50 * 1.0015) and (close >= ema50)

    if pb1_red and pb2_red and near_ema_bull and is_bullish_engulfing:
        swing_high = float(df["high"].iloc[-15:-3].max())
        sl = low - (high - low) * 0.15
        risk = close - sl
        if risk <= 0:
            risk = close * 0.002
            sl = close - risk
        tp = max(swing_high, close + 2.0 * risk)
        rr = round((tp - close) / risk, 2)
        return {
            "action": "BUY",
            "symbol": resolved,
            "entry": round(close, 5),
            "sl": round(sl, 5),
            "tp": round(tp, 5),
            "rr": rr,
            "ema50": round(ema50, 5),
            "reason": f"Ezekiel Chew 50 EMA Bullish Engulfing after 2 Red Pullback Candles (EMA: {round(ema50,5)})"
        }

    # Bearish Setup:
    is_bearish_engulfing = (close < open_p) and (close < pb1["open"]) and (open_p >= pb1["close"] * 0.9995)
    near_ema_bear = (high >= ema50 * 0.9985) and (close <= ema50)

    if pb1_green and pb2_green and near_ema_bear and is_bearish_engulfing:
        swing_low = float(df["low"].iloc[-15:-3].min())
        sl = high + (high - low) * 0.15
        risk = sl - close
        if risk <= 0:
            risk = close * 0.002
            sl = close + risk
        tp = min(swing_low, close - 2.0 * risk)
        rr = round((close - tp) / risk, 2)
        return {
            "action": "SELL",
            "symbol": resolved,
            "entry": round(close, 5),
            "sl": round(sl, 5),
            "tp": round(tp, 5),
            "rr": rr,
            "ema50": round(ema50, 5),
            "reason": f"Ezekiel Chew 50 EMA Bearish Engulfing after 2 Green Pullback Candles (EMA: {round(ema50,5)})"
        }

    return {
        "action": "HOLD",
        "symbol": resolved,
        "entry": round(float(df.iloc[-1]["close"]), 5),
        "ema50": round(ema50, 5),
        "reason": f"Awaiting 2-candle pullback + 50 EMA Engulfing trigger (Price: {round(close,5)}, 50 EMA: {round(ema50,5)})"
    }


def main():
    parser = argparse.ArgumentParser(description="Ezekiel Chew 50 EMA Pullback Engulfing Scanner")
    parser.add_argument("symbol", nargs="?", default="EURUSD.0", help="Forex Symbol")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    args = parser.parse_args()

    if not mt5.initialize():
        print(f"[ERROR] MT5 init failed: {mt5.last_error()}")
        sys.exit(1)

    print("=" * 75)
    print("  EZEKIEL CHEW (PROP FIRM TRAINER) 50 EMA PULLBACK SCANNER")
    print(f"  Target Symbol: {args.symbol} | Timeframe: H1")
    print(f"  Source: Ezekiel Chew (AKuM1cTgPks) | 2 Pullback Candles + Engulfing Trigger")
    print("=" * 75)

    try:
        while True:
            res = scan_symbol(args.symbol)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if res["action"] in ["BUY", "SELL"]:
                print(f"\n>>> [{ts}] [{res['action']}_SIGNAL] {res['symbol']}")
                print(f"    Entry: {res['entry']} | SL: {res['sl']} | TP: {res['tp']} | R:R: 1:{res['rr']}")
                print(f"    50 EMA: {res['ema50']}")
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
