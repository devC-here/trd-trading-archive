"""
Trade With Pat A+ Supply & Demand + FVG Forex Signal Scanner (MetaTrader 5)
==========================================================================
Source: YouTube Shorts @UnfilteredFinanceWithSahil (Video: YPAN6XgXbQM)
Creator: Trade With Pat (350k Subscribers, Canadian Pro Trader)
Backtest Metrics: +74.6% Return in 1 Year, 79% Claimed Win Rate, +2100% PnL Claim
Timeframe: M15 (Universal for Forex, Gold, Crypto, Indices)

Strategy Steps:
1. Market Structure & BOS:
   - Identify trend via Higher Highs/Higher Lows (Bullish) or Lower Highs/Lower Lows (Bearish)
   - Break of Structure (BOS) confirms institutional direction.
2. Impulsive Displacement + FVG:
   - Strong multi-candle expansion leaving an institutional Fair Value Gap (FVG).
   - Base Candle before displacement: Mark Open to Close as the A+ Demand Zone (or Supply Zone).
3. Retest & Execution:
   - Price returns to test the Demand Zone (wick penetrates or body touches zone).
   - Confirmation: Next candle closes Green (Bullish) off the Demand Zone.
   - Entry: Close of confirmation green candle.
   - Stop Loss: Slightly below the Demand Zone Low.
   - Take Profit: Recent Swing High (Liquidity Target) or min 1:2 RR.
4. Reverse completely for Supply Zone Short Trades.

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


def find_pat_setups(df: pd.DataFrame):
    """
    Identifies Pat A+ Demand & Supply Zones with FVG validation.
    """
    demand_zones = []
    supply_zones = []

    for i in range(5, len(df) - 4):
        # Look for multi-candle bullish displacement
        c_base = df.iloc[i]
        c1 = df.iloc[i + 1]
        c2 = df.iloc[i + 2]
        c3 = df.iloc[i + 3]

        # Bullish displacement: c1, c2, c3 mostly green and creating an FVG
        is_bull_displacement = (c1["close"] > c1["open"]) and (c2["close"] > c2["open"]) and (c3["close"] > c2["close"])
        has_bull_fvg = c1["high"] < c3["low"]

        if is_bull_displacement and has_bull_fvg:
            zone_low = min(c_base["low"], c_base["open"], c_base["close"])
            zone_high = max(c_base["open"], c_base["close"])
            demand_zones.append({
                "bar_idx": i,
                "low": float(zone_low),
                "high": float(zone_high),
                "fvg_low": float(c1["high"]),
                "fvg_high": float(c3["low"]),
                "swing_high": float(max(c1["high"], c2["high"], c3["high"])),
                "time": c_base["time"]
            })

        # Bearish displacement: c1, c2, c3 mostly red and creating an FVG
        is_bear_displacement = (c1["close"] < c1["open"]) and (c2["close"] < c2["open"]) and (c3["close"] < c2["close"])
        has_bear_fvg = c1["low"] > c3["high"]

        if is_bear_displacement and has_bear_fvg:
            zone_high = max(c_base["high"], c_base["open"], c_base["close"])
            zone_low = min(c_base["open"], c_base["close"])
            supply_zones.append({
                "bar_idx": i,
                "low": float(zone_low),
                "high": float(zone_high),
                "fvg_low": float(c3["high"]),
                "fvg_high": float(c1["low"]),
                "swing_low": float(min(c1["low"], c2["low"], c3["low"])),
                "time": c_base["time"]
            })

    return demand_zones, supply_zones


def scan_symbol(symbol: str, timeframe=mt5.TIMEFRAME_M15) -> dict:
    resolved = resolve_symbol(symbol)
    if not mt5.symbol_select(resolved, True):
        return {"action": "HOLD", "reason": f"Symbol {resolved} not found"}

    rates = mt5.copy_rates_from_pos(resolved, timeframe, 0, 150)
    if rates is None or len(rates) < 50:
        return {"action": "HOLD", "reason": "Insufficient bars retrieved"}

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    demand_zones, supply_zones = find_pat_setups(df)

    bar = df.iloc[-2]       # Last closed candle
    prev_bar = df.iloc[-3]  # Previous candle
    close = float(bar["close"])
    open_p = float(bar["open"])
    low = float(bar["low"])
    high = float(bar["high"])

    is_green = close > open_p
    is_red = close < open_p

    # Check Demand Zone triggers
    for dz in reversed(demand_zones[-5:]):
        # Did prev_bar touch/wick the zone, and current bar is green?
        zone_touched = (prev_bar["low"] <= dz["high"] and prev_bar["close"] >= dz["low"]) or \
                       (low <= dz["high"] and close >= dz["low"])
        if zone_touched and is_green:
            sl = dz["low"] - (dz["high"] - dz["low"]) * 0.15
            risk = close - sl
            if risk <= 0:
                risk = close * 0.002
                sl = close - risk
            target = max(dz["swing_high"], close + 2.0 * risk)
            tp = target
            rr = round((tp - close) / risk, 2)
            return {
                "action": "BUY",
                "symbol": resolved,
                "entry": round(close, 5),
                "sl": round(sl, 5),
                "tp": round(tp, 5),
                "rr": rr,
                "demand_zone": [round(dz["low"], 5), round(dz["high"], 5)],
                "target_high": round(dz["swing_high"], 5),
                "reason": f"Trade With Pat Demand Zone [{round(dz['low'],5)} - {round(dz['high'],5)}] Retest + Green Confirmation"
            }

    # Check Supply Zone triggers
    for sz in reversed(supply_zones[-5:]):
        zone_touched = (prev_bar["high"] >= sz["low"] and prev_bar["close"] <= sz["high"]) or \
                       (high >= sz["low"] and close <= sz["high"])
        if zone_touched and is_red:
            sl = sz["high"] + (sz["high"] - sz["low"]) * 0.15
            risk = sl - close
            if risk <= 0:
                risk = close * 0.002
                sl = close + risk
            target = min(sz["swing_low"], close - 2.0 * risk)
            tp = target
            rr = round((close - tp) / risk, 2)
            return {
                "action": "SELL",
                "symbol": resolved,
                "entry": round(close, 5),
                "sl": round(sl, 5),
                "tp": round(tp, 5),
                "rr": rr,
                "supply_zone": [round(sz["low"], 5), round(sz["high"], 5)],
                "target_low": round(sz["swing_low"], 5),
                "reason": f"Trade With Pat Supply Zone [{round(sz['low'],5)} - {round(sz['high'],5)}] Retest + Red Confirmation"
            }

    return {
        "action": "HOLD",
        "symbol": resolved,
        "entry": round(float(df.iloc[-1]["close"]), 5),
        "active_demand_zones": len(demand_zones),
        "active_supply_zones": len(supply_zones),
        "reason": f"Awaiting Demand/Supply zone touch & confirmation ({len(demand_zones)} DZ, {len(supply_zones)} SZ)"
    }


def main():
    parser = argparse.ArgumentParser(description="Trade With Pat A+ Demand/Supply + FVG Scanner")
    parser.add_argument("symbol", nargs="?", default="EURUSD.0", help="Forex / Gold Symbol")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    args = parser.parse_args()

    if not mt5.initialize():
        print(f"[ERROR] MT5 init failed: {mt5.last_error()}")
        sys.exit(1)

    print("=" * 75)
    print("  TRADE WITH PAT A+ DEMAND/SUPPLY + FVG FOREX SIGNAL SCANNER")
    print(f"  Target Symbol: {args.symbol} | Timeframe: M15")
    print(f"  Source: Trade With Pat (YPAN6XgXbQM) | +74.6% Return, 79% Win Rate Claim")
    print("=" * 75)

    try:
        while True:
            res = scan_symbol(args.symbol)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if res["action"] in ["BUY", "SELL"]:
                print(f"\n>>> [{ts}] [{res['action']}_SIGNAL] {res['symbol']}")
                print(f"    Entry: {res['entry']} | SL: {res['sl']} | TP: {res['tp']} | R:R: 1:{res['rr']}")
                print(f"    Zone: {res.get('demand_zone', res.get('supply_zone'))}")
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
