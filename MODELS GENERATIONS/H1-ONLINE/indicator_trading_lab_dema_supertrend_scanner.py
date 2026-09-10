"""
Trading Lab 200 DEMA + Supertrend Algorithmic Signal Scanner (MetaTrader 5)
==========================================================================
Source: YouTube Shorts @UnfilteredFinanceWithSahil (Video: WFMCsknKHD4)
Creator: Trading Lab (2M+ Subscribers, "130% Returns in 100 Trades")
Backtest Metrics: +60.0% to +130% Return in 1 Year, 42% Win Rate, 28% Drawdown
Timeframe: M15
Markets: Forex (EURUSD, GBPUSD, USDJPY), Gold (XAUUSD), Crypto, Indices

Strategy Rules:
1. 200 Double Exponential Moving Average (DEMA):
   - EMA1 = EMA(Close, 200)
   - EMA2 = EMA(EMA1, 200)
   - DEMA = 2 * EMA1 - EMA2
   - Acts as institutional trend filter (Longs only above DEMA, Shorts only below DEMA).
2. Supertrend Indicator (Period=10, Multiplier=3.0):
   - ATR-based dynamic trailing channel.
3. Long Signal Trigger:
   - Price trades firmly ABOVE 200 DEMA.
   - Supertrend flips from Bearish to Bullish (BUY Alert).
   - Entry: On candle close of trigger bar.
   - Stop Loss: Placed on the Supertrend support line (or recent swing low).
   - Take Profit: 1:2.0 Risk-to-Reward.
4. Short Signal Trigger:
   - Price trades firmly BELOW 200 DEMA.
   - Supertrend flips from Bullish to Bearish (SELL Alert).
   - Entry: On candle close of trigger bar.
   - Stop Loss: Placed on the Supertrend resistance line.
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


def compute_dema(series: pd.Series, period: int = 200) -> pd.Series:
    ema1 = series.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    return 2 * ema1 - ema2


def compute_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    hl2 = (df["high"] + df["low"]) / 2.0
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    final_upper = pd.Series(index=df.index, dtype=float)
    final_lower = pd.Series(index=df.index, dtype=float)
    trend = pd.Series(index=df.index, dtype=int)

    final_upper.iloc[0] = basic_upper.iloc[0]
    final_lower.iloc[0] = basic_lower.iloc[0]
    trend.iloc[0] = 1

    for i in range(1, len(df)):
        c = df["close"].iloc[i]
        c_prev = df["close"].iloc[i - 1]

        # Lower band
        if basic_lower.iloc[i] > final_lower.iloc[i - 1] or c_prev < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

        # Upper band
        if basic_upper.iloc[i] < final_upper.iloc[i - 1] or c_prev > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        # Trend direction
        if c > final_upper.iloc[i - 1]:
            trend.iloc[i] = 1
        elif c < final_lower.iloc[i - 1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i - 1]

    st_line = pd.Series(np.where(trend == 1, final_lower, final_upper), index=df.index)
    return st_line, trend


def scan_symbol(symbol: str, timeframe=mt5.TIMEFRAME_M15) -> dict:
    resolved = resolve_symbol(symbol)
    if not mt5.symbol_select(resolved, True):
        return {"action": "HOLD", "reason": f"Symbol {resolved} not found"}

    # Fetch 300 bars for 200 DEMA calculation
    rates = mt5.copy_rates_from_pos(resolved, timeframe, 0, 320)
    if rates is None or len(rates) < 220:
        return {"action": "HOLD", "reason": "Insufficient bars retrieved for 200 DEMA"}

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    df["dema200"] = compute_dema(df["close"], period=200)
    st_line, trend = compute_supertrend(df, period=10, multiplier=3.0)
    df["supertrend"] = st_line
    df["trend"] = trend

    bar = df.iloc[-2]       # Last closed candle
    prev_bar = df.iloc[-3]  # Candle before
    close = float(bar["close"])
    dema = float(bar["dema200"])
    curr_trend = int(bar["trend"])
    prev_trend = int(prev_bar["trend"])
    st_val = float(bar["supertrend"])

    # Bullish Flip above 200 DEMA
    bullish_flip = (curr_trend == 1 and prev_trend == -1)
    bullish_dema_bounce = (curr_trend == 1 and close > dema and prev_bar["close"] <= float(prev_bar["dema200"]))

    if close > dema and (bullish_flip or bullish_dema_bounce):
        sl = min(st_val, float(bar["low"])) - (float(bar["high"]) - float(bar["low"])) * 0.15
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
            "dema200": round(dema, 5),
            "supertrend": round(st_val, 5),
            "reason": f"Trading Lab Supertrend Bullish Flip Above 200 DEMA ({round(dema,5)})"
        }

    # Bearish Flip below 200 DEMA
    bearish_flip = (curr_trend == -1 and prev_trend == 1)
    bearish_dema_bounce = (curr_trend == -1 and close < dema and prev_bar["close"] >= float(prev_bar["dema200"]))

    if close < dema and (bearish_flip or bearish_dema_bounce):
        sl = max(st_val, float(bar["high"])) + (float(bar["high"]) - float(bar["low"])) * 0.15
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
            "dema200": round(dema, 5),
            "supertrend": round(st_val, 5),
            "reason": f"Trading Lab Supertrend Bearish Flip Below 200 DEMA ({round(dema,5)})"
        }

    return {
        "action": "HOLD",
        "symbol": resolved,
        "entry": round(float(df.iloc[-1]["close"]), 5),
        "dema200": round(dema, 5),
        "supertrend": round(st_val, 5),
        "trend": "BULLISH" if curr_trend == 1 else "BEARISH",
        "reason": f"Trend: {'BULL' if curr_trend==1 else 'BEAR'} | Close: {round(close,5)} vs 200 DEMA: {round(dema,5)}"
    }


def main():
    parser = argparse.ArgumentParser(description="Trading Lab 200 DEMA + Supertrend Scanner")
    parser.add_argument("symbol", nargs="?", default="EURUSD.0", help="Forex Symbol")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    args = parser.parse_args()

    if not mt5.initialize():
        print(f"[ERROR] MT5 init failed: {mt5.last_error()}")
        sys.exit(1)

    print("=" * 75)
    print("  TRADING LAB 200 DEMA + SUPERTREND ALGORITHMIC FOREX SCANNER")
    print(f"  Target Symbol: {args.symbol} | Timeframe: M15")
    print(f"  Source: Trading Lab (WFMCsknKHD4) | +60% to +130% Return in 100 Trades")
    print("=" * 75)

    try:
        while True:
            res = scan_symbol(args.symbol)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if res["action"] in ["BUY", "SELL"]:
                print(f"\n>>> [{ts}] [{res['action']}_SIGNAL] {res['symbol']}")
                print(f"    Entry: {res['entry']} | SL: {res['sl']} | TP: {res['tp']} | R:R: 1:{res['rr']}")
                print(f"    200 DEMA: {res['dema200']} | Supertrend: {res['supertrend']}")
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
