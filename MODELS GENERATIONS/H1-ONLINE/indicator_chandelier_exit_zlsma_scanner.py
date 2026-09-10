"""
Chandelier Exit + ZLSMA Algorithmic Forex Signal Scanner (MetaTrader 5)
=======================================================================
Source: YouTube Shorts @UnfilteredFinanceWithSahil (Video: -0oWb3Uv9FM)
Strategy: 300% Profit Waali Chandelier Exit + ZLSMA System
Backtest Metrics: +89.5% Return in 1 Year, 55% Win Rate, 17.0% Max Drawdown
Timeframe: H1 (or H4)
Markets: Forex, Gold, Crypto (24-Hour Continuous)

Rules:
- Zero Lag SMA (ZLSMA, Period=32):
    Lag = (Period - 1) / 2
    ZL_Data = Close + (Close - Close[Lag])
    ZLSMA = SMA(ZL_Data, Period)
- Chandelier Exit (ATR Period=22, Multiplier=3.0):
    Long Stop = Highest(High, 22) - 3.0 * ATR(22)
    Short Stop = Lowest(Low, 22) + 3.0 * ATR(22)
- BUY SIGNAL:
    Close > ZLSMA and Chandelier Exit flips/is Bullish (Close > Short Stop)
    Entry at Close, Stop Loss at Chandelier Long Stop (or recent swing low), TP at 1:2 RR.
- SELL SIGNAL:
    Close < ZLSMA and Chandelier Exit flips/is Bearish (Close < Long Stop)
    Entry at Close, Stop Loss at Chandelier Short Stop (or recent swing high), TP at 1:2 RR.

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


def compute_atr(df: pd.DataFrame, period: int = 22) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def compute_zlsma(series: pd.Series, period: int = 32) -> pd.Series:
    lag = int((period - 1) / 2)
    zl_data = series + (series - series.shift(lag))
    return zl_data.rolling(window=period).mean()


def compute_chandelier_exit(df: pd.DataFrame, period: int = 22, mult: float = 3.0):
    atr = compute_atr(df, period)
    highest_high = df["high"].rolling(window=period).max()
    lowest_low = df["low"].rolling(window=period).min()

    long_stop = highest_high - mult * atr
    short_stop = lowest_low + mult * atr

    direction = pd.Series(index=df.index, dtype=int)
    current_dir = 1
    
    for i in range(period, len(df)):
        close_val = df["close"].iloc[i]
        
        if close_val > short_stop.iloc[i - 1]:
            current_dir = 1
        elif close_val < long_stop.iloc[i - 1]:
            current_dir = -1
            
        direction.iloc[i] = current_dir

    return long_stop, short_stop, direction


def scan_symbol(symbol: str, timeframe=mt5.TIMEFRAME_H1) -> dict:
    resolved = resolve_symbol(symbol)
    if not mt5.symbol_select(resolved, True):
        return {"action": "HOLD", "reason": f"Symbol {resolved} not found"}

    rates = mt5.copy_rates_from_pos(resolved, timeframe, 0, 150)
    if rates is None or len(rates) < 60:
        return {"action": "HOLD", "reason": "Insufficient bars retrieved"}

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    
    # Compute indicators
    df["zlsma"] = compute_zlsma(df["close"], period=32)
    long_stop, short_stop, direction = compute_chandelier_exit(df, period=22, mult=3.0)
    df["long_stop"] = long_stop
    df["short_stop"] = short_stop
    df["direction"] = direction

    # Analyze last closed bar (index -2) and latest bar (-1)
    bar = df.iloc[-2]
    prev_bar = df.iloc[-3]
    latest = df.iloc[-1]

    close = bar["close"]
    zlsma = bar["zlsma"]
    dir_curr = bar["direction"]
    dir_prev = prev_bar["direction"]

    # Bullish condition: Close > ZLSMA and direction flipped to Bullish (or bar just closed above ZLSMA while Chandelier is Bullish)
    bullish_flip = (dir_curr == 1 and dir_prev == -1 and close > zlsma)
    bullish_zlsma_reclaim = (dir_curr == 1 and close > zlsma and prev_bar["close"] <= prev_bar["zlsma"])

    # Bearish condition: Close < ZLSMA and direction flipped to Bearish (or bar just closed below ZLSMA while Chandelier is Bearish)
    bearish_flip = (dir_curr == -1 and dir_prev == 1 and close < zlsma)
    bearish_zlsma_reclaim = (dir_curr == -1 and close < zlsma and prev_bar["close"] >= prev_bar["zlsma"])

    if bullish_flip or bullish_zlsma_reclaim:
        sl = min(bar["low"], bar["long_stop"])
        risk = close - sl
        if risk <= 0:
            risk = bar["close"] * 0.002
            sl = close - risk
        tp = close + 2.0 * risk
        rr = round((tp - close) / (close - sl), 2)
        return {
            "action": "BUY",
            "symbol": resolved,
            "entry": round(close, 5),
            "sl": round(sl, 5),
            "tp": round(tp, 5),
            "rr": rr,
            "zlsma": round(zlsma, 5),
            "chandelier_stop": round(bar["long_stop"], 5),
            "reason": "Chandelier Exit Bullish & Candle confirmed above ZLSMA 32"
        }

    elif bearish_flip or bearish_zlsma_reclaim:
        sl = max(bar["high"], bar["short_stop"])
        risk = sl - close
        if risk <= 0:
            risk = bar["close"] * 0.002
            sl = close + risk
        tp = close - 2.0 * risk
        rr = round((close - tp) / (sl - close), 2)
        return {
            "action": "SELL",
            "symbol": resolved,
            "entry": round(close, 5),
            "sl": round(sl, 5),
            "tp": round(tp, 5),
            "rr": rr,
            "zlsma": round(zlsma, 5),
            "chandelier_stop": round(bar["short_stop"], 5),
            "reason": "Chandelier Exit Bearish & Candle confirmed below ZLSMA 32"
        }

    return {
        "action": "HOLD",
        "symbol": resolved,
        "entry": round(latest["close"], 5),
        "zlsma": round(bar["zlsma"], 5),
        "direction": "BULLISH" if dir_curr == 1 else "BEARISH",
        "reason": f"No trigger. Direction: {'BULL' if dir_curr==1 else 'BEAR'}, Close: {round(close,5)} vs ZLSMA: {round(zlsma,5)}"
    }


def main():
    parser = argparse.ArgumentParser(description="Chandelier Exit + ZLSMA Signal Scanner")
    parser.add_argument("symbol", nargs="?", default="EURUSD.0", help="Forex Symbol")
    parser.add_argument("--loop", action="store_true", help="Run in continuous loop")
    args = parser.parse_args()

    if not mt5.initialize():
        print(f"[ERROR] MT5 init failed: {mt5.last_error()}")
        sys.exit(1)

    print("=" * 70)
    print("  CHANDELIER EXIT + ZLSMA FOREX SIGNAL SCANNER")
    print(f"  Target Symbol: {args.symbol} | Timeframe: H1")
    print(f"  Source: Sahil Algo Setup (-0oWb3Uv9FM) | +89.5% Backtest Return")
    print("=" * 70)

    try:
        while True:
            res = scan_symbol(args.symbol)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if res["action"] in ["BUY", "SELL"]:
                print(f"\n>>> [{ts}] [{res['action']}_SIGNAL] {res['symbol']}")
                print(f"    Entry: {res['entry']} | SL: {res['sl']} | TP: {res['tp']} | R:R: 1:{res['rr']}")
                print(f"    ZLSMA: {res['zlsma']} | Chandelier Stop: {res['chandelier_stop']}")
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
