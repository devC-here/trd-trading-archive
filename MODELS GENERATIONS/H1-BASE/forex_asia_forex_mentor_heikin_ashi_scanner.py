"""
Asia Forex Mentor (Ezekiel Chew) Smoothed Heikin Ashi Signal Scanner (MetaTrader 5)
==================================================================================
Source: YouTube Shorts @UnfilteredFinanceWithSahil (Video: wVSM7VuZbT8)
Creator: Ezekiel Chew (Asia Forex Mentor - Bank & Prop Firm Trainer)
Backtest Metrics: +47.0% Return in 1 Year, 59% Win Rate, Only 8.4% Max Drawdown
Tested Markets: Dedicated Forex Pair EURUSD (Universal on GBPUSD, USDJPY, XAUUSD)
Timeframe: H1 (or M30 / H4)

Strategy Rules:
1. Smoothed Heikin Ashi Indicator:
   - Double-smoothed EMA calculation to eliminate false chop while preserving structural trends.
2. Trend Flip Detection:
   - Identify when Smoothed Heikin Ashi flips from Red (Bearish) to Green (Bullish).
3. The Retest / Value Pullback Rule:
   - DO NOT chase the first green candle (avoids entering at swing exhaustion).
   - Wait for price to pull back and touch or retest the Smoothed Heikin Ashi line/band.
4. Confirmation Trigger:
   - A strong Bullish Confirmation candle (Bullish Engulfing or strong green body) closes rejecting
     the Smoothed HA line.
   - Entry: On candle close.
   - Stop Loss: Below recent swing low.
   - Take Profit: Next resistance or 1:2 to 1:3 Risk-to-Reward.
5. Reverse completely for Sell Trades on Bearish flips.

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


def compute_smoothed_heikin_ashi(df: pd.DataFrame, period1: int = 10, period2: int = 10):
    """
    Computes Smoothed Heikin Ashi:
    1. Smooth raw OHLC with EMA(period1)
    2. Compute Heikin Ashi values from smoothed series
    3. Smooth HA values with EMA(period2)
    """
    # 1. First EMA smoothing of raw candles
    o_s = df["open"].ewm(span=period1, adjust=False).mean()
    h_s = df["high"].ewm(span=period1, adjust=False).mean()
    l_s = df["low"].ewm(span=period1, adjust=False).mean()
    c_s = df["close"].ewm(span=period1, adjust=False).mean()

    # 2. Compute HA
    ha_close = (o_s + h_s + l_s + c_s) / 4.0
    ha_open = pd.Series(index=df.index, dtype=float)
    ha_open.iloc[0] = (o_s.iloc[0] + c_s.iloc[0]) / 2.0

    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2.0

    ha_high = pd.concat([h_s, ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([l_s, ha_open, ha_close], axis=1).min(axis=1)

    # 3. Second EMA smoothing
    sha_open = ha_open.ewm(span=period2, adjust=False).mean()
    sha_close = ha_close.ewm(span=period2, adjust=False).mean()
    sha_high = ha_high.ewm(span=period2, adjust=False).mean()
    sha_low = ha_low.ewm(span=period2, adjust=False).mean()

    color = np.where(sha_close >= sha_open, 1, -1)  # 1 = Green (Bull), -1 = Red (Bear)
    return sha_open, sha_close, sha_high, sha_low, pd.Series(color, index=df.index)


def scan_symbol(symbol: str, timeframe=mt5.TIMEFRAME_H1) -> dict:
    resolved = resolve_symbol(symbol)
    if not mt5.symbol_select(resolved, True):
        return {"action": "HOLD", "reason": f"Symbol {resolved} not found"}

    rates = mt5.copy_rates_from_pos(resolved, timeframe, 0, 150)
    if rates is None or len(rates) < 60:
        return {"action": "HOLD", "reason": "Insufficient bars retrieved"}

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    sha_o, sha_c, sha_h, sha_l, sha_color = compute_smoothed_heikin_ashi(df, 10, 10)
    df["sha_color"] = sha_color
    df["sha_high"] = sha_h
    df["sha_low"] = sha_l

    bar = df.iloc[-2]       # Last closed candle
    prev_bar = df.iloc[-3]  # Previous candle
    curr_color = sha_color.iloc[-2]

    close = float(bar["close"])
    open_p = float(bar["open"])
    low = float(bar["low"])
    high = float(bar["high"])

    # Recent trend history: check if color flipped to green within last 12 bars
    recent_colors = sha_color.iloc[-15:-2]
    flips_to_green = (recent_colors.shift(1) == -1) & (recent_colors == 1)
    flips_to_red = (recent_colors.shift(1) == 1) & (recent_colors == -1)

    has_recent_green_flip = flips_to_green.any()
    has_recent_red_flip = flips_to_red.any()

    # Pullback check: Did price pull back and touch the Smoothed HA zone?
    sha_level_low = float(sha_l.iloc[-2])
    sha_level_high = float(sha_h.iloc[-2])

    is_bullish_engulfing = (close > open_p) and (prev_bar["close"] < prev_bar["open"]) and (close > prev_bar["open"])
    is_strong_green = (close > open_p) and ((close - open_p) > (high - low) * 0.5)

    is_bearish_engulfing = (close < open_p) and (prev_bar["close"] > prev_bar["open"]) and (close < prev_bar["open"])
    is_strong_red = (close < open_p) and ((open_p - close) > (high - low) * 0.5)

    # Bullish condition
    if curr_color == 1 and has_recent_green_flip:
        # Pullback touched SHA line
        touched_sha = (low <= sha_level_high) and (close >= sha_level_low)
        if touched_sha and (is_bullish_engulfing or is_strong_green):
            recent_swing_low = float(df["low"].iloc[-10:-2].min())
            sl = min(recent_swing_low, low) - (high - low) * 0.2
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
                "sha_color": "GREEN",
                "sha_zone": [round(sha_level_low, 5), round(sha_level_high, 5)],
                "reason": f"Asia Forex Mentor Smoothed HA Pullback Rejection [{round(sha_level_low,5)} - {round(sha_level_high,5)}]"
            }

    # Bearish condition
    if curr_color == -1 and has_recent_red_flip:
        touched_sha = (high >= sha_level_low) and (close <= sha_level_high)
        if touched_sha and (is_bearish_engulfing or is_strong_red):
            recent_swing_high = float(df["high"].iloc[-10:-2].max())
            sl = max(recent_swing_high, high) + (high - low) * 0.2
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
                "sha_color": "RED",
                "sha_zone": [round(sha_level_low, 5), round(sha_level_high, 5)],
                "reason": f"Asia Forex Mentor Smoothed HA Pullback Rejection [{round(sha_level_low,5)} - {round(sha_level_high,5)}]"
            }

    return {
        "action": "HOLD",
        "symbol": resolved,
        "entry": round(float(df.iloc[-1]["close"]), 5),
        "sha_color": "GREEN" if curr_color == 1 else "RED",
        "sha_zone": [round(sha_level_low, 5), round(sha_level_high, 5)],
        "reason": f"No trigger. Smoothed HA is {'GREEN' if curr_color==1 else 'RED'}, awaiting pullback touch & confirmation candle"
    }


def main():
    parser = argparse.ArgumentParser(description="Asia Forex Mentor Smoothed HA Scanner")
    parser.add_argument("symbol", nargs="?", default="EURUSD.0", help="Forex Symbol")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    args = parser.parse_args()

    if not mt5.initialize():
        print(f"[ERROR] MT5 init failed: {mt5.last_error()}")
        sys.exit(1)

    print("=" * 75)
    print("  ASIA FOREX MENTOR (EZEKIEL CHEW) SMOOTHED HEIKIN ASHI SCANNER")
    print(f"  Target Symbol: {args.symbol} | Timeframe: H1")
    print(f"  Source: Asia Forex Mentor (wVSM7VuZbT8) | +47.0% Return, 59% Win Rate, 8.4% DD")
    print("=" * 75)

    try:
        while True:
            res = scan_symbol(args.symbol)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if res["action"] in ["BUY", "SELL"]:
                print(f"\n>>> [{ts}] [{res['action']}_SIGNAL] {res['symbol']}")
                print(f"    Entry: {res['entry']} | SL: {res['sl']} | TP: {res['tp']} | R:R: 1:{res['rr']}")
                print(f"    SHA Color: {res['sha_color']} | Zone: {res['sha_zone']}")
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
