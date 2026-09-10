"""
Jenfer 'One-Candle' Impulse Scalping Signal Scanner (MetaTrader 5)
=================================================================
Source: YouTube Shorts @UnfilteredFinanceWithSahil (Video: AD36fIiJWVs)
Creator: Jenfer (Prop Firm Payout Trader)
Backtest Metrics: +36.4% Return in 1 Year, 56% Win Rate, 15.8% Max Drawdown
Timeframe: M5
Markets: Forex (EURUSD, GBPUSD, USDJPY), Gold (XAUUSD), Indices

Strategy Rules:
1. The 15-Minute Range Baseline:
   - Identify recent range (high & low of the prior 3-candle 15-minute cluster).
2. The Trend-Setting Impulse Candle:
   - A single prominent big-body candle (body > 60% of candle range) closes cleanly outside the range.
   - Volume confirmation: Tick volume higher than the preceding 2 candles.
3. The 50% - 75% Equilibrium Retracement Entry:
   - DO NOT chase the breakout close!
   - Wait for subsequent candles to retrace into the 50% - 75% discount zone of this impulse candle:
       Bullish Zone: [Impulse_Low + 0.25 * Impulse_Range, Impulse_Low + 0.50 * Impulse_Range]
       (representing a 50% to 75% retracement from the impulse peak)
   - Entry: When price touches/tests this zone and forms a rejection confirmation.
   - Stop Loss: Slightly below the low of the Impulse Candle.
   - Take Profit: Fixed 1:2.0 Risk-to-Reward.
4. Reverse completely for Bearish Impulse breakdown candles.

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


def scan_symbol(symbol: str, timeframe=mt5.TIMEFRAME_M5) -> dict:
    resolved = resolve_symbol(symbol)
    if not mt5.symbol_select(resolved, True):
        return {"action": "HOLD", "reason": f"Symbol {resolved} not found"}

    rates = mt5.copy_rates_from_pos(resolved, timeframe, 0, 100)
    if rates is None or len(rates) < 40:
        return {"action": "HOLD", "reason": "Insufficient bars retrieved"}

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    bar = df.iloc[-2]       # Last closed candle
    close = float(bar["close"])
    low = float(bar["low"])
    high = float(bar["high"])

    # Search for an impulse candle within the last 3 to 10 bars
    for i in range(len(df) - 12, len(df) - 3):
        impulse_c = df.iloc[i]
        c_open = float(impulse_c["open"])
        c_close = float(impulse_c["close"])
        c_high = float(impulse_c["high"])
        c_low = float(impulse_c["low"])
        c_range = c_high - c_low
        c_vol = float(impulse_c["tick_volume"])

        # Prior 3 candles range baseline
        prior_3 = df.iloc[i - 3:i]
        p_high = float(prior_3["high"].max())
        p_low = float(prior_3["low"].min())
        p_vol_max = float(prior_3["tick_volume"].max())

        # Check for Bullish Impulse Candle
        is_big_body = (c_close - c_open) > 0.60 * c_range and c_range > 0.0002
        is_breakout = c_close > p_high
        is_high_vol = c_vol >= p_vol_max

        if is_big_body and is_breakout and is_high_vol:
            # 50% to 75% retracement zone of the impulse candle
            retrace_50 = c_high - 0.50 * c_range
            retrace_75 = c_high - 0.75 * c_range
            zone_low = min(retrace_50, retrace_75)
            zone_high = max(retrace_50, retrace_75)

            # Did subsequent candles (between impulse and current) pull back into zone?
            post_bars = df.iloc[i + 1:len(df) - 1]
            touched_zone = (post_bars["low"].min() <= zone_high) and (post_bars["close"].min() >= c_low)

            if touched_zone and bar["close"] > bar["open"] and low <= zone_high:
                sl = c_low - c_range * 0.10
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
                    "impulse_candle_range": [round(c_low, 5), round(c_high, 5)],
                    "retracement_zone": [round(zone_low, 5), round(zone_high, 5)],
                    "reason": f"Jenfer Impulse Candle 50-75% Retracement Test [{round(zone_low,5)} - {round(zone_high,5)}]"
                }

        # Check for Bearish Impulse Candle
        is_big_body_bear = (c_open - c_close) > 0.60 * c_range and c_range > 0.0002
        is_breakdown = c_close < p_low

        if is_big_body_bear and is_breakdown and is_high_vol:
            retrace_50 = c_low + 0.50 * c_range
            retrace_75 = c_low + 0.75 * c_range
            zone_low = min(retrace_50, retrace_75)
            zone_high = max(retrace_50, retrace_75)

            post_bars = df.iloc[i + 1:len(df) - 1]
            touched_zone = (post_bars["high"].max() >= zone_low) and (post_bars["close"].max() <= c_high)

            if touched_zone and bar["close"] < bar["open"] and high >= zone_low:
                sl = c_high + c_range * 0.10
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
                    "impulse_candle_range": [round(c_low, 5), round(c_high, 5)],
                    "retracement_zone": [round(zone_low, 5), round(zone_high, 5)],
                    "reason": f"Jenfer Bearish Impulse Candle 50-75% Retracement Test [{round(zone_low,5)} - {round(zone_high,5)}]"
                }

    return {
        "action": "HOLD",
        "symbol": resolved,
        "entry": round(float(df.iloc[-1]["close"]), 5),
        "reason": "Awaiting qualified trend-setting impulse candle + 50-75% equilibrium retracement"
    }


def main():
    parser = argparse.ArgumentParser(description="Jenfer One-Candle Impulse Scalping Scanner")
    parser.add_argument("symbol", nargs="?", default="EURUSD.0", help="Forex Symbol")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    args = parser.parse_args()

    if not mt5.initialize():
        print(f"[ERROR] MT5 init failed: {mt5.last_error()}")
        sys.exit(1)

    print("=" * 75)
    print("  JENFER 'ONE-CANDLE' IMPULSE RETRACEMENT SCALPING SCANNER")
    print(f"  Target Symbol: {args.symbol} | Timeframe: M5")
    print(f"  Source: Jenfer (AD36fIiJWVs) | +36.4% Return, 56% Win Rate, Fixed 1:2 RR")
    print("=" * 75)

    try:
        while True:
            res = scan_symbol(args.symbol)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if res["action"] in ["BUY", "SELL"]:
                print(f"\n>>> [{ts}] [{res['action']}_SIGNAL] {res['symbol']}")
                print(f"    Entry: {res['entry']} | SL: {res['sl']} | TP: {res['tp']} | R:R: 1:{res['rr']}")
                print(f"    Retracement Zone: {res['retracement_zone']}")
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
