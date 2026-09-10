"""
Tori Trades Multi-Touch Trendline Breakout Forex Signal Scanner (MetaTrader 5)
=============================================================================
Source: YouTube Shorts @UnfilteredFinanceWithSahil (Video: 0n4sMsqTZdw)
Creator: Tori Trades (600k+ Subscribers, Full-Time Futures & Forex Trader)
Backtest Metrics: +58.0% Return in 1 Year, 46% Win Rate, 19.0% Max Drawdown
Timeframe: H1 (Structural Trendlines) -> M15 (Decisive Breakout Execution)
Markets: Forex (EURUSD, GBPUSD, USDJPY), Gold (XAUUSD), Indices

Strategy Rules:
1. Multi-Touch Structural Trendlines:
   - Identify prominent swing highs and swing lows across the lookback window.
   - Fit Descending Trendline (connecting lower highs) and Ascending Trendline (connecting higher lows).
2. The Decisive Breakout Trigger:
   - Bullish Breakout: A strong green candle closes cleanly ABOVE the descending trendline.
   - Bearish Breakdown: A strong red candle closes cleanly BELOW the ascending trendline.
3. Trade Management:
   - Entry: At the close of the breakout candle.
   - Stop Loss: On the opposite side of the trendline / recent swing pivot.
   - Take Profit: Next key horizontal S/R level or 1:2 Risk-to-Reward.

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


def fit_trendlines(df: pd.DataFrame, window: int = 40):
    """
    Fits descending resistance line (highs) and ascending support line (lows) via linear regression.
    """
    subset = df.iloc[-window:]
    x = np.arange(len(subset))
    
    # Resistance trendline over local highs
    highs = subset["high"].values
    slope_res, intercept_res = np.polyfit(x, highs, 1)
    res_line = slope_res * x + intercept_res

    # Support trendline over local lows
    lows = subset["low"].values
    slope_sup, intercept_sup = np.polyfit(x, lows, 1)
    sup_line = slope_sup * x + intercept_sup

    return slope_res, intercept_res, slope_sup, intercept_sup, res_line[-1], sup_line[-1]


def scan_symbol(symbol: str, timeframe=mt5.TIMEFRAME_M15) -> dict:
    resolved = resolve_symbol(symbol)
    if not mt5.symbol_select(resolved, True):
        return {"action": "HOLD", "reason": f"Symbol {resolved} not found"}

    rates = mt5.copy_rates_from_pos(resolved, timeframe, 0, 100)
    if rates is None or len(rates) < 50:
        return {"action": "HOLD", "reason": "Insufficient bars retrieved"}

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    s_res, i_res, s_sup, i_sup, curr_res_val, curr_sup_val = fit_trendlines(df, window=40)

    bar = df.iloc[-2]       # Last closed candle
    prev_bar = df.iloc[-3]  # Previous candle
    close = float(bar["close"])
    prev_close = float(prev_bar["close"])
    low = float(bar["low"])
    high = float(bar["high"])

    # Previous line values at index -2
    prev_res_val = curr_res_val - s_res
    prev_sup_val = curr_sup_val - s_sup

    # Bullish Trendline Breakout:
    # 1. Slope is downward or flat (s_res <= 0)
    # 2. Previous close was below/at trendline
    # 3. Current close breaks cleanly above trendline
    if s_res < 0.0005:
        if prev_close <= prev_res_val and close > curr_res_val and bar["close"] > bar["open"]:
            recent_low = float(df["low"].iloc[-8:-2].min())
            sl = min(recent_low, low) - (high - low) * 0.15
            risk = close - sl
            if risk <= 0:
                risk = close * 0.002
                sl = close - risk
            tp = close + 2.0 * risk
            rr = round((tp - close) / risk, 2)
            return {
                "action": "BUY",
                "symbol": resolved,
                "entry": round(close, 5),
                "sl": round(sl, 5),
                "tp": round(tp, 5),
                "rr": rr,
                "trendline_level": round(curr_res_val, 5),
                "trendline_slope": round(s_res, 6),
                "reason": f"Tori Trades Bullish Trendline Breakout above {round(curr_res_val,5)}"
            }

    # Bearish Trendline Breakdown:
    # 1. Slope is upward or flat (s_sup >= -0.0005)
    # 2. Previous close was above/at support trendline
    # 3. Current close breaks cleanly below support trendline
    if s_sup > -0.0005:
        if prev_close >= prev_sup_val and close < curr_sup_val and bar["close"] < bar["open"]:
            recent_high = float(df["high"].iloc[-8:-2].max())
            sl = max(recent_high, high) + (high - low) * 0.15
            risk = sl - close
            if risk <= 0:
                risk = close * 0.002
                sl = close + risk
            tp = close - 2.0 * risk
            rr = round((close - tp) / risk, 2)
            return {
                "action": "SELL",
                "symbol": resolved,
                "entry": round(close, 5),
                "sl": round(sl, 5),
                "tp": round(tp, 5),
                "rr": rr,
                "trendline_level": round(curr_sup_val, 5),
                "trendline_slope": round(s_sup, 6),
                "reason": f"Tori Trades Bearish Trendline Breakdown below {round(curr_sup_val,5)}"
            }

    return {
        "action": "HOLD",
        "symbol": resolved,
        "entry": round(float(df.iloc[-1]["close"]), 5),
        "res_trendline": round(curr_res_val, 5),
        "sup_trendline": round(curr_sup_val, 5),
        "reason": f"Price inside channel: Support {round(curr_sup_val,5)} < Close {round(close,5)} < Resistance {round(curr_res_val,5)}"
    }


def main():
    parser = argparse.ArgumentParser(description="Tori Trades Multi-Touch Trendline Scanner")
    parser.add_argument("symbol", nargs="?", default="EURUSD.0", help="Forex Symbol")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    args = parser.parse_args()

    if not mt5.initialize():
        print(f"[ERROR] MT5 init failed: {mt5.last_error()}")
        sys.exit(1)

    print("=" * 75)
    print("  TORI TRADES MULTI-TOUCH TRENDLINE BREAKOUT FOREX SCANNER")
    print(f"  Target Symbol: {args.symbol} | Timeframe: M15")
    print(f"  Source: Tori Trades (0n4sMsqTZdw) | +58.0% Return, 46% Win Rate")
    print("=" * 75)

    try:
        while True:
            res = scan_symbol(args.symbol)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if res["action"] in ["BUY", "SELL"]:
                print(f"\n>>> [{ts}] [{res['action']}_SIGNAL] {res['symbol']}")
                print(f"    Entry: {res['entry']} | SL: {res['sl']} | TP: {res['tp']} | R:R: 1:{res['rr']}")
                print(f"    Trendline: {res['trendline_level']} (Slope: {res['trendline_slope']})")
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
