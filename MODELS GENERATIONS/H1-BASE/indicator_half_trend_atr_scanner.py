"""
Half Trend + ATR Algorithmic Forex Signal Scanner (MetaTrader 5)
================================================================
Source: YouTube Shorts @UnfilteredFinanceWithSahil (Video: WRczqfxjR4g)
Creator: ICT / TradingView Half Trend Algo Setup
Backtest Metrics: +67.4% Return in 1 Year, 55% Win Rate, 16.8% Max Drawdown
Timeframe: M15
Markets: Forex (EURUSD, GBPUSD, USDJPY), Gold (XAUUSD), Crypto

Strategy Rules:
1. Half Trend Indicator:
   - Evaluates price action relative to dynamic ATR-based trailing bands.
   - Amplitude: 2, Channel Deviation: 2.0
2. Long Signal Trigger:
   - Half Trend direction flips from Bearish (-1) to Bullish (+1).
   - Entry: At the close of the trigger candle.
   - Stop Loss: Placed on the Half Trend dynamic support line.
   - Take Profit: 1:2.0 Risk-to-Reward.
3. Short Signal Trigger:
   - Half Trend direction flips from Bullish (+1) to Bearish (-1).
   - Entry: At the close of the trigger candle.
   - Stop Loss: Placed on the Half Trend dynamic resistance line.
   - Take Profit: 1:2.0 Risk-to-Reward.

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


def compute_half_trend(df: pd.DataFrame, amplitude: int = 2, channel_dev: float = 2.0):
    """
    Computes Half Trend indicator direction and dynamic trailing lines.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # True Range & ATR
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=100).mean().fillna(tr.rolling(window=14).mean())
    dev = channel_dev * atr / 2.0

    trend = pd.Series(index=df.index, dtype=int)
    ht_line = pd.Series(index=df.index, dtype=float)

    curr_trend = 1
    curr_ht = float(close.iloc[0])

    for i in range(amplitude, len(df)):
        max_high = high.iloc[i - amplitude:i + 1].max()
        min_low = low.iloc[i - amplitude:i + 1].min()
        c = close.iloc[i]

        if curr_trend == 1:
            if c < curr_ht:
                curr_trend = -1
                curr_ht = max_high
            else:
                curr_ht = max(curr_ht, min_low)
        else:
            if c > curr_ht:
                curr_trend = 1
                curr_ht = min_low
            else:
                curr_ht = min(curr_ht, max_high)

        trend.iloc[i] = curr_trend
        ht_line.iloc[i] = curr_ht

    return trend, ht_line, dev


def scan_symbol(symbol: str, timeframe=mt5.TIMEFRAME_M15) -> dict:
    resolved = resolve_symbol(symbol)
    if not mt5.symbol_select(resolved, True):
        return {"action": "HOLD", "reason": f"Symbol {resolved} not found"}

    rates = mt5.copy_rates_from_pos(resolved, timeframe, 0, 150)
    if rates is None or len(rates) < 60:
        return {"action": "HOLD", "reason": "Insufficient bars retrieved"}

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    trend, ht_line, dev = compute_half_trend(df, amplitude=2, channel_dev=2.0)
    df["trend"] = trend
    df["ht_line"] = ht_line

    bar = df.iloc[-2]       # Last closed bar
    prev_bar = df.iloc[-3]  # Previous bar
    close = float(bar["close"])
    low = float(bar["low"])
    high = float(bar["high"])

    curr_t = int(bar["trend"])
    prev_t = int(prev_bar["trend"])
    ht_val = float(bar["ht_line"])

    # Bullish Flip: Trend flipped from -1 to 1
    if curr_t == 1 and prev_t == -1:
        sl = ht_val - (high - low) * 0.15
        risk = close - sl
        if risk <= 0:
            risk = close * 0.002
            sl = close - risk
        tp = close + 2.0 * risk
        rr = round((tp - close) / risk, 2)
        reason_str = f"Half Trend Bullish Flip (Support Line: {round(ht_val,5)})"
        return {
            "strategy": "HalfTrend ATR Dynamic Channel",
            "timeframe": "M15",
            "signal": "BUY",
            "action": "BUY",
            "symbol": resolved,
            "entry": round(close, 5),
            "sl": round(sl, 5),
            "tp": round(tp, 5),
            "rr": rr,
            "ht_line": round(ht_val, 5),
            "reason": reason_str,
            "details": reason_str
        }

    # Bearish Flip: Trend flipped from 1 to -1
    if curr_t == -1 and prev_t == 1:
        sl = ht_val + (high - low) * 0.15
        risk = sl - close
        if risk <= 0:
            risk = close * 0.002
            sl = close + risk
        tp = close - 2.0 * risk
        rr = round((close - tp) / risk, 2)
        reason_str = f"Half Trend Bearish Flip (Resistance Line: {round(ht_val,5)})"
        return {
            "strategy": "HalfTrend ATR Dynamic Channel",
            "timeframe": "M15",
            "signal": "SELL",
            "action": "SELL",
            "symbol": resolved,
            "entry": round(close, 5),
            "sl": round(sl, 5),
            "tp": round(tp, 5),
            "rr": rr,
            "ht_line": round(ht_val, 5),
            "reason": reason_str,
            "details": reason_str
        }

    reason_str = f"No flip. Trend is {'BULLISH' if curr_t==1 else 'BEARISH'} (Price: {round(close,5)}, HT Line: {round(ht_val,5)})"
    return {
        "strategy": "HalfTrend ATR Dynamic Channel",
        "timeframe": "M15",
        "signal": "NEUTRAL",
        "action": "HOLD",
        "symbol": resolved,
        "entry": round(float(df.iloc[-1]["close"]), 5),
        "ht_line": round(ht_val, 5),
        "trend": "BULLISH" if curr_t == 1 else "BEARISH",
        "reason": reason_str,
        "details": reason_str
    }


def main():
    parser = argparse.ArgumentParser(description="Half Trend + ATR Forex Signal Scanner")
    parser.add_argument("symbol", nargs="?", default="EURUSD.0", help="Forex Symbol")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    args = parser.parse_args()

    if not mt5.initialize():
        print(f"[ERROR] MT5 init failed: {mt5.last_error()}")
        sys.exit(1)

    print("=" * 75)
    print("  HALF TREND + ATR ALGORITHMIC FOREX SIGNAL SCANNER")
    print(f"  Target Symbol: {args.symbol} | Timeframe: M15")
    print(f"  Source: Half Trend Strategy (WRczqfxjR4g) | +67.4% Return, 55% Win Rate")
    print("=" * 75)

    try:
        while True:
            res = scan_symbol(args.symbol)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if res["action"] in ["BUY", "SELL"]:
                print(f"\n>>> [{ts}] [{res['action']}_SIGNAL] {res['symbol']}")
                print(f"    Entry: {res['entry']} | SL: {res['sl']} | TP: {res['tp']} | R:R: 1:{res['rr']}")
                print(f"    HT Line: {res['ht_line']}")
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
