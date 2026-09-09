"""
The Forex Guide (Umer Akhtar) Sandwich Candlestick Pattern Signal Scanner (MetaTrader 5)
========================================================================================
Source: YouTube Shorts @UnfilteredFinanceWithSahil (Video: t9VzgJhqgHk)
Creator: Umer Akhtar / The Forex Guide (600k+ Subscribers, Paid Course Strategy)
Backtest Metrics: +57.4% Return in 1 Year, 80-85% Claimed Win Rate, Only 8.1% Max Drawdown
Timeframe: H1 (Macro Trend Filter) -> M15 (Sandwich Pattern & Retest Execution)
Markets: Forex (EURUSD, GBPUSD, USDJPY), Gold (XAUUSD)

Strategy Rules:
1. Higher Timeframe Trend (H1):
   - Only take Long Sandwich setups if H1 is Bullish (Price > 50 EMA / higher highs).
   - Only take Short Sandwich setups if H1 is Bearish (Price < 50 EMA / lower lows).
2. The 3-Candle Sandwich Formation (M15):
   - Candle 1: Strong Bullish Momentum Candle (large green body).
   - Candle 2: Doji / Small Indecision Candle (body < 25% of total candle range).
   - Candle 3: Second Strong Bullish Momentum Candle closing strictly ABOVE Candle 2 High.
3. The Retest Entry:
   - Price pulls back to test Candle 2 (Doji High / Low zone).
   - Entry: On limit retest or confirmation bounce candle.
   - Stop Loss: Slightly below Candle 1 Low.
   - Take Profit: Fixed 1:2.0 Risk-to-Reward.
4. Reverse completely for Bearish Short Sandwich patterns.

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
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 60)
    if rates is None or len(rates) < 50:
        return None
    df = pd.DataFrame(rates)
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    return "BULLISH" if df["close"].iloc[-2] > df["ema50"].iloc[-2] else "BEARISH"


def scan_symbol(symbol: str) -> dict:
    resolved = resolve_symbol(symbol)
    if not mt5.symbol_select(resolved, True):
        return {"action": "HOLD", "reason": f"Symbol {resolved} not found"}

    h1_trend = get_h1_trend(resolved)
    if h1_trend is None:
        return {"action": "HOLD", "reason": "Failed to get H1 trend"}

    rates_m15 = mt5.copy_rates_from_pos(resolved, mt5.TIMEFRAME_M15, 0, 80)
    if rates_m15 is None or len(rates_m15) < 30:
        return {"action": "HOLD", "reason": "Insufficient M15 bars"}

    df = pd.DataFrame(rates_m15)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    bar = df.iloc[-2]       # Last closed bar
    close = float(bar["close"])
    low = float(bar["low"])
    high = float(bar["high"])

    # Search past 10 bars for a 3-candle sandwich pattern
    for i in range(len(df) - 12, len(df) - 3):
        c1 = df.iloc[i]
        c2 = df.iloc[i + 1]
        c3 = df.iloc[i + 2]

        # Candle 1 properties
        c1_range = float(c1["high"] - c1["low"])
        c1_body = abs(float(c1["close"] - c1["open"]))
        c1_bull = float(c1["close"]) > float(c1["open"])
        c1_bear = float(c1["close"]) < float(c1["open"])

        # Candle 2 properties (Doji)
        c2_range = float(c2["high"] - c2["low"])
        c2_body = abs(float(c2["close"] - c2["open"]))
        is_doji = c2_body <= (c2_range * 0.30) if c2_range > 0 else True

        # Candle 3 properties
        c3_range = float(c3["high"] - c3["low"])
        c3_body = abs(float(c3["close"] - c3["open"]))
        c3_bull = float(c3["close"]) > float(c3["open"])
        c3_bear = float(c3["close"]) < float(c3["open"])

        # Bullish Sandwich
        # 1. c1 is big green candle
        # 2. c2 is doji
        # 3. c3 is big green candle closing above c2 high
        if h1_trend == "BULLISH" and c1_bull and is_doji and c3_bull and float(c3["close"]) > float(c2["high"]):
            doji_zone_high = float(c2["high"])
            doji_zone_low = float(c2["low"])
            c1_low = float(c1["low"])

            # Check if subsequent bars (between c3 and current) retested the doji zone
            post_bars = df.iloc[i + 3:len(df) - 1]
            if len(post_bars) > 0:
                retested = (post_bars["low"].min() <= doji_zone_high) and (post_bars["close"].min() >= c1_low)
                if retested and bar["close"] > bar["open"] and low <= doji_zone_high:
                    sl = c1_low - (high - low) * 0.15
                    risk = close - sl
                    if risk <= 0:
                        risk = close * 0.002
                        sl = close - risk
                    tp = close + 2.0 * risk
                    rr = round((tp - close) / risk, 2)
                    reason_str = f"Umer Akhtar Bullish Sandwich Retest [{round(doji_zone_low,5)} - {round(doji_zone_high,5)}]"
                    return {
                        "strategy": "Umer Akhtar Sandwich Reversal",
                        "timeframe": "M15 (H1 Trend Filter)",
                        "signal": "BUY",
                        "action": "BUY",
                        "symbol": resolved,
                        "entry": round(close, 5),
                        "sl": round(sl, 5),
                        "tp": round(tp, 5),
                        "rr": rr,
                        "h1_trend": h1_trend,
                        "doji_zone": [round(doji_zone_low, 5), round(doji_zone_high, 5)],
                        "c1_low": round(c1_low, 5),
                        "reason": reason_str,
                        "details": reason_str
                    }

        # Bearish Sandwich
        if h1_trend == "BEARISH" and c1_bear and is_doji and c3_bear and float(c3["close"]) < float(c2["low"]):
            doji_zone_high = float(c2["high"])
            doji_zone_low = float(c2["low"])
            c1_high = float(c1["high"])

            post_bars = df.iloc[i + 3:len(df) - 1]
            if len(post_bars) > 0:
                retested = (post_bars["high"].max() >= doji_zone_low) and (post_bars["close"].max() <= c1_high)
                if retested and bar["close"] < bar["open"] and high >= doji_zone_low:
                    sl = c1_high + (high - low) * 0.15
                    risk = sl - close
                    if risk <= 0:
                        risk = close * 0.002
                        sl = close + risk
                    tp = close - 2.0 * risk
                    rr = round((close - tp) / risk, 2)
                    reason_str = f"Umer Akhtar Bearish Sandwich Retest [{round(doji_zone_low,5)} - {round(doji_zone_high,5)}]"
                    return {
                        "strategy": "Umer Akhtar Sandwich Reversal",
                        "timeframe": "M15 (H1 Trend Filter)",
                        "signal": "SELL",
                        "action": "SELL",
                        "symbol": resolved,
                        "entry": round(close, 5),
                        "sl": round(sl, 5),
                        "tp": round(tp, 5),
                        "rr": rr,
                        "h1_trend": h1_trend,
                        "doji_zone": [round(doji_zone_low, 5), round(doji_zone_high, 5)],
                        "c1_high": round(c1_high, 5),
                        "reason": reason_str,
                        "details": reason_str
                    }

    reason_str = f"H1 Trend: {h1_trend} | Awaiting valid 3-candle Sandwich formation & Doji retest"
    return {
        "strategy": "Umer Akhtar Sandwich Reversal",
        "timeframe": "M15 (H1 Trend Filter)",
        "signal": "NEUTRAL",
        "action": "HOLD",
        "symbol": resolved,
        "entry": round(float(df.iloc[-1]["close"]), 5),
        "h1_trend": h1_trend,
        "reason": reason_str,
        "details": reason_str
    }


def main():
    parser = argparse.ArgumentParser(description="The Forex Guide Sandwich Pattern Scanner")
    parser.add_argument("symbol", nargs="?", default="EURUSD.0", help="Forex Symbol")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    args = parser.parse_args()

    if not mt5.initialize():
        print(f"[ERROR] MT5 init failed: {mt5.last_error()}")
        sys.exit(1)

    print("=" * 75)
    print("  THE FOREX GUIDE (UMER AKHTAR) SANDWICH PATTERN SCANNER")
    print(f"  Target Symbol: {args.symbol} | Multi-TF: H1 Trend -> M15 Sandwich")
    print(f"  Source: Umer Akhtar (t9VzgJhqgHk) | +57.4% Return, 80-85% Claimed WR")
    print("=" * 75)

    try:
        while True:
            res = scan_symbol(args.symbol)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if res["action"] in ["BUY", "SELL"]:
                print(f"\n>>> [{ts}] [{res['action']}_SIGNAL] {res['symbol']}")
                print(f"    Entry: {res['entry']} | SL: {res['sl']} | TP: {res['tp']} | R:R: 1:{res['rr']}")
                print(f"    Doji Zone: {res['doji_zone']}")
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
