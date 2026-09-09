"""
Kojo Forex Pure Price Action Demand & Supply Forex Signal Scanner (MetaTrader 5)
================================================================================
Source: YouTube Shorts @UnfilteredFinanceWithSahil (Video: 7XtF2FH92MM)
Creator: Kojo Forex (Dubai Forex Trader, 1M+ Followers)
Backtest Metrics: +56.8% Return in 1 Year, 54% Win Rate, 15.4% Max Drawdown
Timeframe: M15 (Primary for Gold / Forex)
Markets: Gold (XAUUSD), EURUSD, GBPUSD, USDJPY

Rules:
- Zero Indicators - Pure Institutional Price Action:
1. Demand Zone Identification:
   - Locate a strong bullish displacement move (multi-candle impulse).
   - Zone is drawn from the Lowest Low to the Candle Close of the base candle.
2. Demand Zone Retest:
   - Price retraces back into the Demand Zone.
   - Price holds the zone (no candle closes below the zone low).
3. Buy Trigger:
   - A bullish confirmation candle closes inside or rejecting the Demand Zone.
   - Entry at candle close.
   - Stop Loss: Slightly below the lowest low of the Demand Zone.
   - Take Profit: Previous swing high / next resistance level (min 1:2 RR).
4. Reverse completely for Supply Zone Sell Trades.

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


def find_zones(df: pd.DataFrame, lookback: int = 60):
    """
    Scans for Kojo Forex Demand and Supply zones within the recent lookback window.
    """
    demand_zones = []
    supply_zones = []

    # Iterate through bars looking for impulsive displacement
    for i in range(10, len(df) - 5):
        # Check for strong bullish displacement (large green candles following a base)
        c0, c1, c2 = df.iloc[i], df.iloc[i + 1], df.iloc[i + 2]
        body1 = c1["close"] - c1["open"]
        body2 = c2["close"] - c2["open"]
        avg_body = (df["high"] - df["low"]).iloc[i - 10:i].mean()

        if body1 > avg_body * 1.2 and body2 > 0 and c2["close"] > c1["high"]:
            # Base candle is c0
            zone_low = min(c0["low"], c0["open"], c0["close"])
            zone_high = max(c0["open"], c0["close"])
            demand_zones.append({
                "bar_idx": i,
                "low": zone_low,
                "high": zone_high,
                "time": c0["time"],
                "displacement_high": max(c1["high"], c2["high"])
            })

        # Check for strong bearish displacement
        if -body1 > avg_body * 1.2 and body2 < 0 and c2["close"] < c1["low"]:
            zone_high = max(c0["high"], c0["open"], c0["close"])
            zone_low = min(c0["open"], c0["close"])
            supply_zones.append({
                "bar_idx": i,
                "low": zone_low,
                "high": zone_high,
                "time": c0["time"],
                "displacement_low": min(c1["low"], c2["low"])
            })

    return demand_zones, supply_zones


def scan_symbol(symbol: str, timeframe=mt5.TIMEFRAME_M15) -> dict:
    resolved = resolve_symbol(symbol)
    if not mt5.symbol_select(resolved, True):
        return {"action": "HOLD", "reason": f"Symbol {resolved} not found"}

    rates = mt5.copy_rates_from_pos(resolved, timeframe, 0, 150)
    if rates is None or len(rates) < 60:
        return {"action": "HOLD", "reason": "Insufficient bars retrieved"}

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    demand_zones, supply_zones = find_zones(df)

    bar = df.iloc[-2]  # last closed bar
    close = bar["close"]
    low = bar["low"]
    high = bar["high"]
    is_bullish = bar["close"] > bar["open"]
    is_bearish = bar["close"] < bar["open"]

    # Check Demand Zone reaction
    for dz in reversed(demand_zones[-5:]):
        # If price penetrated the zone recently without closing below it
        if low <= dz["high"] and close >= dz["low"]:
            # Check if bar is a bullish confirmation (bullish body with lower rejection wick)
            lower_wick = min(bar["open"], bar["close"]) - bar["low"]
            body = abs(bar["close"] - bar["open"])
            if is_bullish and (lower_wick > body * 0.4 or bar["close"] > df.iloc[-3]["high"]):
                sl = dz["low"] - (dz["high"] - dz["low"]) * 0.2
                target = dz.get("displacement_high", close + 2 * (close - sl))
                risk = close - sl
                if risk <= 0:
                    risk = close * 0.002
                    sl = close - risk
                tp = max(target, close + 2 * risk)
                rr = round((tp - close) / risk, 2)
                return {
                    "action": "BUY",
                    "symbol": resolved,
                    "entry": round(close, 5),
                    "sl": round(sl, 5),
                    "tp": round(tp, 5),
                    "rr": rr,
                    "zone_low": round(dz["low"], 5),
                    "zone_high": round(dz["high"], 5),
                    "reason": f"Kojo Forex Demand Zone Retest [{round(dz['low'],5)} - {round(dz['high'],5)}] + Bullish Rejection"
                }

    # Check Supply Zone reaction
    for sz in reversed(supply_zones[-5:]):
        if high >= sz["low"] and close <= sz["high"]:
            upper_wick = bar["high"] - max(bar["open"], bar["close"])
            body = abs(bar["close"] - bar["open"])
            if is_bearish and (upper_wick > body * 0.4 or bar["close"] < df.iloc[-3]["low"]):
                sl = sz["high"] + (sz["high"] - sz["low"]) * 0.2
                target = sz.get("displacement_low", close - 2 * (sl - close))
                risk = sl - close
                if risk <= 0:
                    risk = close * 0.002
                    sl = close + risk
                tp = min(target, close - 2 * risk)
                rr = round((close - tp) / risk, 2)
                return {
                    "action": "SELL",
                    "symbol": resolved,
                    "entry": round(close, 5),
                    "sl": round(sl, 5),
                    "tp": round(tp, 5),
                    "rr": rr,
                    "zone_low": round(sz["low"], 5),
                    "zone_high": round(sz["high"], 5),
                    "reason": f"Kojo Forex Supply Zone Retest [{round(sz['low'],5)} - {round(sz['high'],5)}] + Bearish Rejection"
                }

    return {
        "action": "HOLD",
        "symbol": resolved,
        "entry": round(df.iloc[-1]["close"], 5),
        "total_demand_zones": len(demand_zones),
        "total_supply_zones": len(supply_zones),
        "reason": f"Price not currently reacting to active Kojo Demand/Supply zones ({len(demand_zones)} DZ / {len(supply_zones)} SZ)"
    }


def main():
    parser = argparse.ArgumentParser(description="Kojo Forex Demand & Supply Signal Scanner")
    parser.add_argument("symbol", nargs="?", default="EURUSD.0", help="Forex / Gold Symbol")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    args = parser.parse_args()

    if not mt5.initialize():
        print(f"[ERROR] MT5 init failed: {mt5.last_error()}")
        sys.exit(1)

    print("=" * 70)
    print("  KOJO FOREX PURE PRICE ACTION DEMAND/SUPPLY SCANNER")
    print(f"  Target Symbol: {args.symbol} | Timeframe: M15")
    print(f"  Source: Kojo Forex (7XtF2FH92MM) | +56.8% Return, 54% Win Rate")
    print("=" * 70)

    try:
        while True:
            res = scan_symbol(args.symbol)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if res["action"] in ["BUY", "SELL"]:
                print(f"\n>>> [{ts}] [{res['action']}_SIGNAL] {res['symbol']}")
                print(f"    Entry: {res['entry']} | SL: {res['sl']} | TP: {res['tp']} | R:R: 1:{res['rr']}")
                print(f"    Zone: [{res['zone_low']} - {res['zone_high']}]")
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
