"""
Bharat Jhunjhunwala RSI 60/40 Momentum + Bollinger Band Forex Signal Scanner (MetaTrader 5)
===========================================================================================
Source: YouTube Shorts @UnfilteredFinanceWithSahil (Video: 4gvWmhCzE30)
Creator: Bharat Jhunjhunwala (CMT, MFTA Technical Analyst)
Backtest Metrics: +58.4% Return in 1 Year, 63% Win Rate, 12.6% Max Drawdown
Timeframe: H1 (or Daily)
Markets: Forex (EURUSD, GBPUSD, USDJPY), Gold (XAUUSD)

Strategy Concept:
- The RSI 60/40 Range Shift:
  RSI > 60 is NOT overbought; it represents true institutional momentum!
  In a bull market, RSI shifts into the 40-80 zone (40 acts as floor/support).
  In a bear market, RSI shifts into the 20-60 zone (60 acts as ceiling/resistance).
- Setup Rules:
  1. Bullish Momentum: RSI(14) recently crossed above 60.
  2. Pullback: Price pulls back to the 20 SMA (Bollinger Middle Band) or Lower Band.
  3. Strict Rule: During the pullback, RSI must hold strictly ABOVE 40!
  4. Trigger: Price forms a reversal candle breaking above the pullback high with RSI curling up.
  5. Stop Loss: Below recent swing low.
  6. Take Profit: Upper Bollinger Band or 1:2 Risk-to-Reward.
- Reverse completely for Bearish Short setups (RSI < 40 momentum, pullback holds below 60, breaks low).

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


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return sma, upper, lower


def scan_symbol(symbol: str, timeframe=mt5.TIMEFRAME_H1) -> dict:
    resolved = resolve_symbol(symbol)
    if not mt5.symbol_select(resolved, True):
        return {"action": "HOLD", "reason": f"Symbol {resolved} not found"}

    rates = mt5.copy_rates_from_pos(resolved, timeframe, 0, 150)
    if rates is None or len(rates) < 60:
        return {"action": "HOLD", "reason": "Insufficient bars retrieved"}

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    df["rsi"] = compute_rsi(df["close"], period=14)
    mid_bb, upper_bb, lower_bb = compute_bollinger_bands(df["close"], period=20, num_std=2.0)
    df["mid_bb"] = mid_bb
    df["upper_bb"] = upper_bb
    df["lower_bb"] = lower_bb

    bar = df.iloc[-2]       # Last closed candle
    prev_bar = df.iloc[-3]  # Previous candle
    close = float(bar["close"])
    low = float(bar["low"])
    high = float(bar["high"])
    rsi = float(bar["rsi"])
    mid = float(bar["mid_bb"])
    upper = float(bar["upper_bb"])
    lower = float(bar["lower_bb"])

    # Look back 15 bars for momentum regime
    recent_bars = df.iloc[-15:-2]
    had_bull_momentum = (recent_bars["rsi"] >= 60).any()
    had_bear_momentum = (recent_bars["rsi"] <= 40).any()

    # Pullback check
    pullback_bars = df.iloc[-6:-2]
    min_rsi_pullback = float(pullback_bars["rsi"].min())
    max_rsi_pullback = float(pullback_bars["rsi"].max())

    # Bullish condition:
    # 1. Had RSI >= 60 momentum
    # 2. Pulled back near or below 20 SMA (mid_bb)
    # 3. RSI stayed strictly >= 40 throughout the pullback
    # 4. Current bar closes green above prev_bar high
    if had_bull_momentum and min_rsi_pullback >= 40:
        touched_bb = (low <= mid * 1.002)
        if touched_bb and close > prev_bar["high"] and rsi > float(prev_bar["rsi"]):
            recent_low = float(df["low"].iloc[-6:-2].min())
            sl = min(recent_low, low) - (high - low) * 0.15
            risk = close - sl
            if risk <= 0:
                risk = close * 0.002
                sl = close - risk
            tp = max(upper, close + 2.0 * risk)
            rr = round((tp - close) / risk, 2)
            return {
                "action": "BUY",
                "symbol": resolved,
                "entry": round(close, 5),
                "sl": round(sl, 5),
                "tp": round(tp, 5),
                "rr": rr,
                "rsi": round(rsi, 2),
                "mid_bb": round(mid, 5),
                "upper_bb": round(upper, 5),
                "reason": f"Bharat Jhunjhunwala RSI 60/40 Bullish Rebound (RSI: {round(rsi,1)}, Min PB: {round(min_rsi_pullback,1)})"
            }

    # Bearish condition:
    if had_bear_momentum and max_rsi_pullback <= 60:
        touched_bb = (high >= mid * 0.998)
        if touched_bb and close < prev_bar["low"] and rsi < float(prev_bar["rsi"]):
            recent_high = float(df["high"].iloc[-6:-2].max())
            sl = max(recent_high, high) + (high - low) * 0.15
            risk = sl - close
            if risk <= 0:
                risk = close * 0.002
                sl = close + risk
            tp = min(lower, close - 2.0 * risk)
            rr = round((close - tp) / risk, 2)
            return {
                "action": "SELL",
                "symbol": resolved,
                "entry": round(close, 5),
                "sl": round(sl, 5),
                "tp": round(tp, 5),
                "rr": rr,
                "rsi": round(rsi, 2),
                "mid_bb": round(mid, 5),
                "lower_bb": round(lower, 5),
                "reason": f"Bharat Jhunjhunwala RSI 60/40 Bearish Rejection (RSI: {round(rsi,1)}, Max PB: {round(max_rsi_pullback,1)})"
            }

    return {
        "action": "HOLD",
        "symbol": resolved,
        "entry": round(float(df.iloc[-1]["close"]), 5),
        "rsi": round(rsi, 2),
        "mid_bb": round(mid, 5),
        "reason": f"RSI at {round(rsi,2)} | Awaiting 60/40 range shift & BB retest"
    }


def main():
    parser = argparse.ArgumentParser(description="Bharat Jhunjhunwala RSI 60/40 BB Scanner")
    parser.add_argument("symbol", nargs="?", default="EURUSD.0", help="Forex Symbol")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    args = parser.parse_args()

    if not mt5.initialize():
        print(f"[ERROR] MT5 init failed: {mt5.last_error()}")
        sys.exit(1)

    print("=" * 75)
    print("  BHARAT JHUNJHUNWALA RSI 60/40 + BOLLINGER BAND FOREX SCANNER")
    print(f"  Target Symbol: {args.symbol} | Timeframe: H1")
    print(f"  Source: Bharat Jhunjhunwala (4gvWmhCzE30) | +58.4% Return, 63% Win Rate")
    print("=" * 75)

    try:
        while True:
            res = scan_symbol(args.symbol)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if res["action"] in ["BUY", "SELL"]:
                print(f"\n>>> [{ts}] [{res['action']}_SIGNAL] {res['symbol']}")
                print(f"    Entry: {res['entry']} | SL: {res['sl']} | TP: {res['tp']} | R:R: 1:{res['rr']}")
                print(f"    RSI: {res['rsi']} | Mid BB: {res['mid_bb']}")
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
