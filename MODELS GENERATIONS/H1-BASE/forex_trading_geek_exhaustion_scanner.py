"""
The Trading Geek (Brad Goh) S/R Exhaustion Price Action Signal Scanner (MetaTrader 5)
=====================================================================================
Source: YouTube Shorts @UnfilteredFinanceWithSahil (Video: aPEzamtaYw4)
Creator: Brad Goh / The Trading Geek (2M+ Followers, Made First $1M at Age 21)
Backtest Metrics: +34.8% Return in 1 Year, 57% Win Rate, 14.6% Max Drawdown
Tested Markets: Dedicated Gold (XAUUSD) & Forex (EURUSD, GBPUSD)
Timeframe: H4 / Daily (Key Zones) -> H1 / M15 (Exhaustion & Reversal Execution)

Strategy Rules:
1. Key Support & Resistance Zones:
   - Identify major swing highs and lows from H4/Daily timeframe.
2. The Seller Exhaustion Pattern:
   - Price approaches the Key Support level.
   - Bearish candles show decreasing body sizes (loss of selling momentum).
   - Candles display prominent lower wicks (sellers being absorbed by institutional limit orders).
3. The Expansion Reversal Trigger:
   - A large, solid Bullish candle closes decisively higher, rejecting Support.
   - Entry: On candle close of this expansion candle.
   - Stop Loss: Below the lowest wick of the exhaustion cluster.
   - Take Profit: Next major Resistance level (min 1:2 Risk-to-Reward).
4. Reverse completely for Bearish Exhaustion at Key Resistance.

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


def get_h4_levels(symbol: str):
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, 50)
    if rates is None or len(rates) < 20:
        return [], []
    df = pd.DataFrame(rates)
    res_levels = [float(df["high"].iloc[i]) for i in range(5, len(df) - 5)
                  if df["high"].iloc[i] == df["high"].iloc[i-5:i+6].max()]
    sup_levels = [float(df["low"].iloc[i]) for i in range(5, len(df) - 5)
                  if df["low"].iloc[i] == df["low"].iloc[i-5:i+6].min()]
    return res_levels, sup_levels


def scan_symbol(symbol: str, timeframe=mt5.TIMEFRAME_H1) -> dict:
    resolved = resolve_symbol(symbol)
    if not mt5.symbol_select(resolved, True):
        return {"action": "HOLD", "reason": f"Symbol {resolved} not found"}

    res_levels, sup_levels = get_h4_levels(resolved)

    rates = mt5.copy_rates_from_pos(resolved, timeframe, 0, 100)
    if rates is None or len(rates) < 30:
        return {"action": "HOLD", "reason": "Insufficient bars retrieved"}

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    bar = df.iloc[-2]       # Reversal expansion candle
    c1 = df.iloc[-3]        # Preceding candle 1
    c2 = df.iloc[-4]        # Preceding candle 2

    close = float(bar["close"])
    open_p = float(bar["open"])
    low = float(bar["low"])
    high = float(bar["high"])

    body = abs(close - open_p)
    b1 = abs(float(c1["close"]) - float(c1["open"]))
    b2 = abs(float(c2["close"]) - float(c2["open"]))

    # Check Support Exhaustion (Bullish setup)
    for sup in reversed(sup_levels[-4:]):
        # Price approached support within 0.25%
        near_sup = abs(low - sup) / sup < 0.0035 or abs(float(c1["low"]) - sup) / sup < 0.0035
        # Preceding candles had shrinking bodies or long lower wicks
        lower_wick1 = min(float(c1["open"]), float(c1["close"])) - float(c1["low"])
        lower_wick2 = min(float(c2["open"]), float(c2["close"])) - float(c2["low"])
        had_exhaustion_wicks = (lower_wick1 > b1 * 0.4) or (lower_wick2 > b2 * 0.4) or (b1 < b2)

        # Trigger candle is strong green expansion
        is_bull_expansion = (close > open_p) and (body > b1 * 1.2) and (close > float(c1["high"]))

        if near_sup and had_exhaustion_wicks and is_bull_expansion:
            exhaustion_low = min(low, float(c1["low"]), float(c2["low"]))
            sl = exhaustion_low - (high - low) * 0.15
            risk = close - sl
            if risk <= 0:
                risk = close * 0.002
                sl = close - risk

            target_res = min([r for r in res_levels if r > close], default=close + 2.5 * risk)
            tp = max(target_res, close + 2.0 * risk)
            rr = round((tp - close) / risk, 2)
            return {
                "action": "BUY",
                "symbol": resolved,
                "entry": round(close, 5),
                "sl": round(sl, 5),
                "tp": round(tp, 5),
                "rr": rr,
                "support_level": round(sup, 5),
                "reason": f"The Trading Geek Support Exhaustion Reversal at {round(sup,5)}"
            }

    # Check Resistance Exhaustion (Bearish setup)
    for res in reversed(res_levels[-4:]):
        near_res = abs(high - res) / res < 0.0035 or abs(float(c1["high"]) - res) / res < 0.0035
        upper_wick1 = float(c1["high"]) - max(float(c1["open"]), float(c1["close"]))
        upper_wick2 = float(c2["high"]) - max(float(c2["open"]), float(c2["close"]))
        had_exhaustion_wicks = (upper_wick1 > b1 * 0.4) or (upper_wick2 > b2 * 0.4) or (b1 < b2)

        is_bear_expansion = (close < open_p) and (body > b1 * 1.2) and (close < float(c1["low"]))

        if near_res and had_exhaustion_wicks and is_bear_expansion:
            exhaustion_high = max(high, float(c1["high"]), float(c2["high"]))
            sl = exhaustion_high + (high - low) * 0.15
            risk = sl - close
            if risk <= 0:
                risk = close * 0.002
                sl = close + risk

            target_sup = max([s for s in sup_levels if s < close], default=close - 2.5 * risk)
            tp = min(target_sup, close - 2.0 * risk)
            rr = round((close - tp) / risk, 2)
            return {
                "action": "SELL",
                "symbol": resolved,
                "entry": round(close, 5),
                "sl": round(sl, 5),
                "tp": round(tp, 5),
                "rr": rr,
                "resistance_level": round(res, 5),
                "reason": f"The Trading Geek Resistance Exhaustion Reversal at {round(res,5)}"
            }

    return {
        "action": "HOLD",
        "symbol": resolved,
        "entry": round(float(df.iloc[-1]["close"]), 5),
        "h4_supports": len(sup_levels),
        "h4_resistances": len(res_levels),
        "reason": "Awaiting candle exhaustion pattern at key H4 Support/Resistance levels"
    }


def main():
    parser = argparse.ArgumentParser(description="The Trading Geek S/R Exhaustion Scanner")
    parser.add_argument("symbol", nargs="?", default="EURUSD.0", help="Forex / Gold Symbol")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    args = parser.parse_args()

    if not mt5.initialize():
        print(f"[ERROR] MT5 init failed: {mt5.last_error()}")
        sys.exit(1)

    print("=" * 75)
    print("  THE TRADING GEEK (BRAD GOH) S/R EXHAUSTION FOREX SCANNER")
    print(f"  Target Symbol: {args.symbol} | Multi-TF: H4 Key Levels -> H1 Execution")
    print(f"  Source: Brad Goh (aPEzamtaYw4) | +34.8% Return, 57% Win Rate")
    print("=" * 75)

    try:
        while True:
            res = scan_symbol(args.symbol)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if res["action"] in ["BUY", "SELL"]:
                print(f"\n>>> [{ts}] [{res['action']}_SIGNAL] {res['symbol']}")
                print(f"    Entry: {res['entry']} | SL: {res['sl']} | TP: {res['tp']} | R:R: 1:{res['rr']}")
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
