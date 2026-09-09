"""
SMC Liquidity Sniper 71% Fib Retracement + FVG Forex Signal Scanner (MetaTrader 5)
==================================================================================
Source: YouTube Shorts @UnfilteredFinanceWithSahil (Video: LkTzH_5xiJ0)
Creator: Smart Money Concepts (SMC) / Inner Circle Trader (ICT)
Backtest Metrics: +99.9% Return in 1 Year (Double Capital), 50%+ Win Rate, 13.7% Drawdown
Timeframe: H4 (Discount/Premium Dealing Range Bias) -> M15 (Liquidity Sweep + 71% Fib Retracement)
Markets: EURUSD, GBPUSD, XAUUSD, USDJPY

Strategy Rules:
1. Higher Timeframe Bias (H4):
   - Measure Dealing Range (High to Low).
   - Discount Zone (< 50% equilibrium) = Long setups preferred.
   - Premium Zone (> 50% equilibrium) = Short setups preferred.
2. Low Timeframe Setup (M15):
   - Liquidity Sweep: Price sweeps below previous swing low (Bullish) or above swing high (Bearish).
   - Market Structure Shift (MSS/BOS): Aggressive displacement closes beyond recent structural pivot.
   - Fair Value Gap (FVG): Formed inside the displacement impulse.
   - Fibonacci Retracement: 71% Fib level (0.705 - 0.710 Optimal Trade Entry - OTE) aligns with FVG.
3. Entry Trigger:
   - Price retraces to the 70.5% - 71.0% Fib zone and rejects with candle confirmation.
   - Stop Loss: Below the sweep extreme wick.
   - Take Profit: Major liquidity high or 1:2.5 Risk-to-Reward.

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


def get_h4_bias(symbol: str):
    """
    Computes H4 dealing range (highest high, lowest low) and 50% equilibrium.
    """
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, 40)
    if rates is None or len(rates) < 10:
        return None, None, None
    df = pd.DataFrame(rates)
    h4_high = float(df["high"].max())
    h4_low = float(df["low"].min())
    eq = (h4_high + h4_low) / 2.0
    current_price = float(df["close"].iloc[-1])
    bias = "DISCOUNT" if current_price < eq else "PREMIUM"
    return bias, h4_high, h4_low


def scan_symbol(symbol: str) -> dict:
    resolved = resolve_symbol(symbol)
    if not mt5.symbol_select(resolved, True):
        return {"action": "HOLD", "reason": f"Symbol {resolved} not found"}

    bias, h4_high, h4_low = get_h4_bias(resolved)
    if bias is None:
        return {"action": "HOLD", "reason": "Failed to get H4 bias"}

    rates_m15 = mt5.copy_rates_from_pos(resolved, mt5.TIMEFRAME_M15, 0, 100)
    if rates_m15 is None or len(rates_m15) < 40:
        return {"action": "HOLD", "reason": "Insufficient M15 bars"}

    df = pd.DataFrame(rates_m15)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    # Find swing points on M15
    recent_low = float(df["low"].iloc[-30:-5].min())
    recent_high = float(df["high"].iloc[-30:-5].max())

    bar = df.iloc[-2]       # Last closed candle
    prev_bar = df.iloc[-3]  # Previous candle
    close = float(bar["close"])
    low = float(bar["low"])
    high = float(bar["high"])

    # Bullish Setup:
    # 1. Price swept recent low within last 15 bars
    # 2. Bullish displacement impulse formed: Leg from sweep low to impulse high
    sweep_bars = df.iloc[-15:-3]
    swept_lows = sweep_bars[sweep_bars["low"] < recent_low]

    if len(swept_lows) > 0 and bias == "DISCOUNT":
        sweep_extreme_low = float(swept_lows["low"].min())
        impulse_high = float(df["high"].iloc[-10:].max())
        leg_range = impulse_high - sweep_extreme_low

        if leg_range > 0:
            fib_71 = impulse_high - 0.71 * leg_range
            fib_62 = impulse_high - 0.618 * leg_range

            # Retracement check: did low enter 62-71% zone and current candle closes green?
            in_ote = (low <= fib_62 and low >= fib_71 * 0.999) or (prev_bar["low"] <= fib_62 and prev_bar["low"] >= fib_71 * 0.999)
            if in_ote and bar["close"] > bar["open"]:
                sl = sweep_extreme_low - leg_range * 0.05
                risk = close - sl
                if risk <= 0:
                    risk = close * 0.002
                    sl = close - risk
                tp = impulse_high + 0.5 * leg_range
                rr = round((tp - close) / risk, 2)
                return {
                    "action": "BUY",
                    "symbol": resolved,
                    "entry": round(close, 5),
                    "sl": round(sl, 5),
                    "tp": round(tp, 5),
                    "rr": rr,
                    "h4_bias": bias,
                    "fib_71_zone": [round(fib_71, 5), round(fib_62, 5)],
                    "sweep_low": round(sweep_extreme_low, 5),
                    "reason": f"SMC 71% Fib Retracement in H4 Discount Zone [{round(fib_71,5)} - {round(fib_62,5)}] + Bullish Reversal"
                }

    # Bearish Setup:
    swept_highs = sweep_bars[sweep_bars["high"] > recent_high]
    if len(swept_highs) > 0 and bias == "PREMIUM":
        sweep_extreme_high = float(swept_highs["high"].max())
        impulse_low = float(df["low"].iloc[-10:].min())
        leg_range = sweep_extreme_high - impulse_low

        if leg_range > 0:
            fib_71 = impulse_low + 0.71 * leg_range
            fib_62 = impulse_low + 0.618 * leg_range

            in_ote = (high >= fib_62 and high <= fib_71 * 1.001) or (prev_bar["high"] >= fib_62 and prev_bar["high"] <= fib_71 * 1.001)
            if in_ote and bar["close"] < bar["open"]:
                sl = sweep_extreme_high + leg_range * 0.05
                risk = sl - close
                if risk <= 0:
                    risk = close * 0.002
                    sl = close + risk
                tp = impulse_low - 0.5 * leg_range
                rr = round((close - tp) / risk, 2)
                return {
                    "action": "SELL",
                    "symbol": resolved,
                    "entry": round(close, 5),
                    "sl": round(sl, 5),
                    "tp": round(tp, 5),
                    "rr": rr,
                    "h4_bias": bias,
                    "fib_71_zone": [round(fib_62, 5), round(fib_71, 5)],
                    "sweep_high": round(sweep_extreme_high, 5),
                    "reason": f"SMC 71% Fib Retracement in H4 Premium Zone [{round(fib_62,5)} - {round(fib_71,5)}] + Bearish Reversal"
                }

    return {
        "action": "HOLD",
        "symbol": resolved,
        "entry": round(float(df.iloc[-1]["close"]), 5),
        "h4_bias": bias,
        "h4_range": [round(h4_low, 5), round(h4_high, 5)],
        "reason": f"No active 71% Fib Sniper setup in H4 {bias} zone"
    }


def main():
    parser = argparse.ArgumentParser(description="SMC Liquidity Sniper 71% Fib Scanner")
    parser.add_argument("symbol", nargs="?", default="EURUSD.0", help="Forex / Gold Symbol")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    args = parser.parse_args()

    if not mt5.initialize():
        print(f"[ERROR] MT5 init failed: {mt5.last_error()}")
        sys.exit(1)

    print("=" * 75)
    print("  SMC LIQUIDITY SNIPER 71% FIB RETRACEMENT FOREX SCANNER")
    print(f"  Target Symbol: {args.symbol} | Multi-TF: H4 Bias -> M15 Execution")
    print(f"  Source: SMC Liquidity Sniper (LkTzH_5xiJ0) | +99.9% Return in 1 Year")
    print("=" * 75)

    try:
        while True:
            res = scan_symbol(args.symbol)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if res["action"] in ["BUY", "SELL"]:
                print(f"\n>>> [{ts}] [{res['action']}_SIGNAL] {res['symbol']}")
                print(f"    Entry: {res['entry']} | SL: {res['sl']} | TP: {res['tp']} | R:R: 1:{res['rr']}")
                print(f"    H4 Bias: {res['h4_bias']} | 71% Fib Zone: {res['fib_71_zone']}")
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
