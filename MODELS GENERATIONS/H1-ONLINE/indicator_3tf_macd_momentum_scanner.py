"""
3-Timeframe MACD Momentum Algorithmic Forex Signal Scanner (MetaTrader 5)
========================================================================
Source: YouTube Shorts @UnfilteredFinanceWithSahil (Video: aUKmS6ZDeq4)
Creator: Quant Institutional Setup ("62% Win Rate with Multi-Timeframe MACD")
Backtest Metrics: +61.8% Return in 1 Year, 62% Win Rate, Average 1:3.5 Risk-to-Reward
Timeframes: Daily (Zero-Line Regime) -> H4 (Cross Confirmation) -> H1 (Histogram Trigger)
Markets: Forex (EURUSD, GBPUSD, USDJPY), Gold (XAUUSD)

Strategy Rules:
1. Daily MACD Baseline (12, 26, 9):
   - MACD Line > 0: Macro Bullish Bias -> ONLY BUY setups allowed.
   - MACD Line < 0: Macro Bearish Bias -> ONLY SELL setups allowed.
2. H4 Trend Alignment:
   - Bullish setup: H4 MACD Line is above H4 Signal Line (Bullish Cross active).
   - Bearish setup: H4 MACD Line is below H4 Signal Line (Bearish Cross active).
3. H1 Execution Trigger:
   - H1 MACD Histogram flips from negative/red to positive/green (or shows momentum expansion).
   - Entry: Close of the first green candle on H1.
   - Stop Loss: Slightly below recent H1 swing low.
   - Take Profit: 1:2.5 to 1:3.5 Risk-to-Reward.

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


def compute_macd(df: pd.DataFrame, fast=12, slow=26, signal=9):
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def scan_symbol(symbol: str) -> dict:
    resolved = resolve_symbol(symbol)
    if not mt5.symbol_select(resolved, True):
        return {"action": "HOLD", "reason": f"Symbol {resolved} not found"}

    # 1. Daily Bias
    rates_d1 = mt5.copy_rates_from_pos(resolved, mt5.TIMEFRAME_D1, 0, 45)
    if rates_d1 is None or len(rates_d1) < 30:
        return {"action": "HOLD", "reason": "Insufficient D1 bars for MACD"}
    df_d1 = pd.DataFrame(rates_d1)
    d1_macd, d1_sig, _ = compute_macd(df_d1)
    d1_val = float(d1_macd.iloc[-2])
    d1_bias = "BULLISH" if d1_val > 0 else "BEARISH"

    # 2. H4 Alignment
    rates_h4 = mt5.copy_rates_from_pos(resolved, mt5.TIMEFRAME_H4, 0, 50)
    if rates_h4 is None or len(rates_h4) < 30:
        return {"action": "HOLD", "reason": "Insufficient H4 bars for MACD"}
    df_h4 = pd.DataFrame(rates_h4)
    h4_macd, h4_sig, _ = compute_macd(df_h4)
    h4_bull = float(h4_macd.iloc[-2]) > float(h4_sig.iloc[-2])
    h4_bear = float(h4_macd.iloc[-2]) < float(h4_sig.iloc[-2])

    # 3. H1 Trigger
    rates_h1 = mt5.copy_rates_from_pos(resolved, mt5.TIMEFRAME_H1, 0, 60)
    if rates_h1 is None or len(rates_h1) < 35:
        return {"action": "HOLD", "reason": "Insufficient H1 bars for MACD"}
    df_h1 = pd.DataFrame(rates_h1)
    h1_macd, h1_sig, h1_hist = compute_macd(df_h1)

    bar = df_h1.iloc[-2]       # Last closed H1 bar
    prev_bar = df_h1.iloc[-3]  # Previous H1 bar
    close = float(bar["close"])
    low = float(bar["low"])
    high = float(bar["high"])

    curr_hist = float(h1_hist.iloc[-2])
    prev_hist = float(h1_hist.iloc[-3])

    # Bullish condition:
    # 1. Daily MACD > 0
    # 2. H4 MACD > Signal
    # 3. H1 Histogram flipped positive or expanding green candle
    hist_flip_green = (prev_hist <= 0 and curr_hist > 0) or (curr_hist > prev_hist and curr_hist > 0)
    is_green_candle = bar["close"] > bar["open"]

    if d1_bias == "BULLISH" and h4_bull and hist_flip_green and is_green_candle:
        recent_low = float(df_h1["low"].iloc[-10:-2].min())
        sl = min(recent_low, low) - (high - low) * 0.15
        risk = close - sl
        if risk <= 0:
            risk = close * 0.002
            sl = close - risk
        tp = close + 3.0 * risk  # 1:3 RR
        rr = round((tp - close) / risk, 2)
        reason_str = f"3-TF MACD Triple Confluence Bullish (D1 > 0, H4 Cross, H1 Hist: {round(curr_hist,6)})"
        return {
            "strategy": "3-Timeframe MACD Momentum",
            "timeframe": "D1 (Trend) + H4 (Wave) + H1 (Entry)",
            "signal": "BUY",
            "action": "BUY",
            "symbol": resolved,
            "entry": round(close, 5),
            "sl": round(sl, 5),
            "tp": round(tp, 5),
            "rr": rr,
            "d1_macd": round(d1_val, 5),
            "h1_hist": round(curr_hist, 6),
            "reason": reason_str,
            "details": reason_str
        }

    # Bearish condition:
    hist_flip_red = (prev_hist >= 0 and curr_hist < 0) or (curr_hist < prev_hist and curr_hist < 0)
    is_red_candle = bar["close"] < bar["open"]

    if d1_bias == "BEARISH" and h4_bear and hist_flip_red and is_red_candle:
        recent_high = float(df_h1["high"].iloc[-10:-2].max())
        sl = max(recent_high, high) + (high - low) * 0.15
        risk = sl - close
        if risk <= 0:
            risk = close * 0.002
            sl = close + risk
        tp = close - 3.0 * risk
        rr = round((close - tp) / risk, 2)
        reason_str = f"3-TF MACD Triple Confluence Bearish (D1 < 0, H4 Cross, H1 Hist: {round(curr_hist,6)})"
        return {
            "strategy": "3-Timeframe MACD Momentum",
            "timeframe": "D1 (Trend) + H4 (Wave) + H1 (Entry)",
            "signal": "SELL",
            "action": "SELL",
            "symbol": resolved,
            "entry": round(close, 5),
            "sl": round(sl, 5),
            "tp": round(tp, 5),
            "rr": rr,
            "d1_macd": round(d1_val, 5),
            "h1_hist": round(curr_hist, 6),
            "reason": reason_str,
            "details": reason_str
        }

    reason_str = f"D1 Bias: {d1_bias}, H4 Aligned: {h4_bull if d1_bias=='BULLISH' else h4_bear}, H1 Hist: {round(curr_hist,6)}"
    return {
        "strategy": "3-Timeframe MACD Momentum",
        "timeframe": "D1 (Trend) + H4 (Wave) + H1 (Entry)",
        "signal": "NEUTRAL",
        "action": "HOLD",
        "symbol": resolved,
        "entry": round(float(df_h1.iloc[-1]["close"]), 5),
        "d1_bias": d1_bias,
        "h4_aligned": h4_bull if d1_bias == "BULLISH" else h4_bear,
        "h1_hist": round(curr_hist, 6),
        "reason": reason_str,
        "details": reason_str
    }


def main():
    parser = argparse.ArgumentParser(description="3-Timeframe MACD Momentum Scanner")
    parser.add_argument("symbol", nargs="?", default="EURUSD.0", help="Forex Symbol")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    args = parser.parse_args()

    if not mt5.initialize():
        print(f"[ERROR] MT5 init failed: {mt5.last_error()}")
        sys.exit(1)

    print("=" * 75)
    print("  3-TIMEFRAME MACD MOMENTUM ALGORITHMIC FOREX SCANNER")
    print(f"  Target Symbol: {args.symbol} | Multi-TF: D1 Baseline -> H4 Cross -> H1 Hist")
    print(f"  Source: Multi-TF MACD (aUKmS6ZDeq4) | +61.8% Return, 62% Win Rate")
    print("=" * 75)

    try:
        while True:
            res = scan_symbol(args.symbol)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if res["action"] in ["BUY", "SELL"]:
                print(f"\n>>> [{ts}] [{res['action']}_SIGNAL] {res['symbol']}")
                print(f"    Entry: {res['entry']} | SL: {res['sl']} | TP: {res['tp']} | R:R: 1:{res['rr']}")
                print(f"    D1 MACD: {res['d1_macd']} | H1 Hist: {res['h1_hist']}")
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
