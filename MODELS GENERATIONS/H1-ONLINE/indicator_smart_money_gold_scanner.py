"""
========================================================================================
MT5 SIGNAL SCANNER: SMART MONEY GOLD & FOREX DEMAND/SUPPLY + 8 EMA STRATEGY
========================================================================================
Source: Short Video zBvUf9v4EtU | 1-Year Backtest: +63.8% Return, 49.2% Win Rate
Market: Gold (XAUUSD) / Major Forex (EURUSD, GBPUSD)
Timeframe: H1 Institutional Supply/Demand Zone + M15 8 EMA Cross
Indicators:
  1. H1 Institutional Base Origin Zones (Smart Money Demand/Supply)
  2. 8 EMA (Execution Trigger)
Logic:
  - Buy Setup:
      1. Institutional Demand: Aggressive buying origin identified on H1 chart.
      2. Liquidity Retest: Price pulls back to test into the fresh Demand Zone.
      3. Trigger: Price crosses and closes a strong green candle ABOVE the 8 EMA.
      4. Stop Loss: Placed slightly below the H1 Demand Zone.
      5. Take Profit: 1:2 to 1:2.5 Risk-to-Reward Ratio.
  - Sell Setup (Inverted):
      1. Institutional Supply: Aggressive selling origin identified on H1 chart.
      2. Liquidity Retest: Price pulls back to test into the fresh Supply Zone.
      3. Trigger: Price crosses and closes a strong red candle BELOW the 8 EMA.
      4. Stop Loss: Placed slightly above the H1 Supply Zone.
      5. Take Profit: 1:2 to 1:2.5 Risk-to-Reward Ratio.
========================================================================================
NOTE: SIGNAL ONLY - No trade execution or live orders are placed.
========================================================================================
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import time
import datetime
import pandas as pd
import numpy as np
import MetaTrader5 as mt5

# ==========================================
# 1. CONFIGURATION & STRATEGY SETTINGS
# ==========================================
DEFAULT_SYMBOLS = ["XAUUSD", "EURUSD.0", "EURUSD", "EURUSDm", "EURUSD.pro", "GBPUSD"]
TIMEFRAME_HTF = mt5.TIMEFRAME_H1   # 1-Hour Zone Detection
TIMEFRAME_LTF = mt5.TIMEFRAME_M15  # 15-Minute Trigger Execution
EMA_TRIGGER = 8
RISK_REWARD_RATIO = 2.0
POLL_INTERVAL = 5                 # Seconds between poll checks
LOOKBACK_BARS = 100

ACTIVE_SYMBOL = None
PIP_SIZE = 0.0001


def resolve_symbol():
    """Auto-detects active MT5 symbol and pip scaling."""
    global ACTIVE_SYMBOL, PIP_SIZE
    for s in DEFAULT_SYMBOLS:
        info = mt5.symbol_info(s)
        if info is not None:
            if not info.visible:
                mt5.symbol_select(s, True)
            ACTIVE_SYMBOL = s
            PIP_SIZE = 0.01 if ("XAU" in s or "GOLD" in s or info.digits in (2, 3)) else 0.0001
            return ACTIVE_SYMBOL
    ACTIVE_SYMBOL = "EURUSD"
    return ACTIVE_SYMBOL


def initialize_mt5():
    """Connects to MetaTrader 5 terminal."""
    if not mt5.initialize():
        print(f"[-] MT5 Connection failed: {mt5.last_error()}")
        return False
    symbol = resolve_symbol()
    print(f"[+] Connected to MT5. Monitoring symbol: {symbol}")
    return True


def fetch_rates(timeframe=None, count=LOOKBACK_BARS):
    """Fetches candle rates."""
    if timeframe is None:
        timeframe = TIMEFRAME_LTF
    rates = mt5.copy_rates_from_pos(ACTIVE_SYMBOL, timeframe, 0, count)
    if rates is None or len(rates) < 30:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


def find_h1_demand_supply(df_h1: pd.DataFrame):
    """Identifies recent aggressive H1 institutional buying/selling zones."""
    df_h1['body'] = (df_h1['close'] - df_h1['open']).abs()
    df_h1['avg_body'] = df_h1['body'].rolling(10).mean()

    recent_demand = None
    recent_supply = None

    # Search backwards for aggressive institutional moves
    for i in range(len(df_h1) - 4, 10, -1):
        # Demand: Aggressive green candle
        if df_h1['close'].iloc[i] > df_h1['open'].iloc[i] and df_h1['body'].iloc[i] > df_h1['avg_body'].iloc[i] * 1.3:
            if recent_demand is None:
                recent_demand = (df_h1['low'].iloc[i-1], df_h1['open'].iloc[i])

        # Supply: Aggressive red candle
        if df_h1['close'].iloc[i] < df_h1['open'].iloc[i] and df_h1['body'].iloc[i] > df_h1['avg_body'].iloc[i] * 1.3:
            if recent_supply is None:
                recent_supply = (df_h1['open'].iloc[i], df_h1['high'].iloc[i-1])

        if recent_demand and recent_supply:
            break

    return recent_demand, recent_supply


def emit_signal(sig_type: str, candle_time: datetime.datetime, entry_price: float, sl_price: float, tp_price: float, zone_top: float, zone_bot: float):
    """Prints a structured ASCII signal card to console."""
    risk_pips = abs(entry_price - sl_price) / PIP_SIZE
    reward_pips = abs(tp_price - entry_price) / PIP_SIZE

    print("\n" + "=" * 75)
    print(f" [ALERT] MT5 SIGNAL ALERT: SMART MONEY GOLD/FOREX ZONE ({sig_type.upper()})")
    print("=" * 75)
    print(f" Symbol:        {ACTIVE_SYMBOL} | Timeframe: M15 (with H1 Institutional Zone)")
    print(f" Bar Time:      {candle_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Signal Type:   {sig_type.upper()} (Zone Retest + 8 EMA Crossover)")
    print(f" H1 Zone:       {zone_bot:.5f} - {zone_top:.5f}")
    print(f" Entry Price:   {entry_price:.5f}")
    print(f" Stop Loss:     {sl_price:.5f}  ({risk_pips:.1f} pips)")
    print(f" Take Profit:   {tp_price:.5f}  ({reward_pips:.1f} pips)")
    print(f" Risk/Reward:   1:{RISK_REWARD_RATIO:.1f}")
    print(f" Status:        SIGNAL ONLY (Ready for manual or external execution)")
    print("=" * 75 + "\n")


def run_scanner():
    """Main live market scanner loop."""
    if not initialize_mt5():
        return

    print("\n" + "=" * 75)
    print(f"   SMART MONEY GOLD/FOREX DEMAND-SUPPLY SCANNER ({ACTIVE_SYMBOL})   ")
    print("=" * 75)
    print(" Strategy: H1 Institutional Zone Retest + M15 8-EMA Reclaim")
    print(" Status: Scanning live bars... Press Ctrl+C to stop.\n")

    last_processed_candle_time = None

    try:
        while True:
            df_h1 = fetch_rates(TIMEFRAME_HTF, count=50)
            df_m15 = fetch_rates(TIMEFRAME_LTF, count=100)

            if df_h1 is None or df_m15 is None:
                time.sleep(POLL_INTERVAL)
                continue

            current_candle_time = df_m15['time'].iloc[-1]
            closed_bar = df_m15.iloc[-2]
            prev_bar = df_m15.iloc[-3]

            if last_processed_candle_time != current_candle_time:
                last_processed_candle_time = current_candle_time

                df_m15['ema8'] = df_m15['close'].ewm(span=EMA_TRIGGER, adjust=False).mean()

                demand_zone, supply_zone = find_h1_demand_supply(df_h1)

                # ----------------------------------------------------
                # BUY LOGIC: Price inside/near H1 Demand Zone + 8 EMA cross
                # ----------------------------------------------------
                if demand_zone is not None:
                    z_bot, z_top = min(demand_zone), max(demand_zone)
                    tested_demand = any(df_m15['low'].iloc[-4:] <= z_top + (5 * PIP_SIZE))
                    ema8_bull_cross = (prev_bar['close'] <= prev_bar['ema8']) and (closed_bar['close'] > closed_bar['ema8']) and (closed_bar['close'] > closed_bar['open'])

                    if tested_demand and ema8_bull_cross:
                        entry = closed_bar['close']
                        sl = z_bot - (4.0 * PIP_SIZE)
                        risk = entry - sl
                        if risk > 0:
                            tp = entry + (risk * RISK_REWARD_RATIO)
                            emit_signal("BUY", current_candle_time, entry, sl, tp, z_top, z_bot)

                # ----------------------------------------------------
                # SELL LOGIC: Price inside/near H1 Supply Zone + 8 EMA cross
                # ----------------------------------------------------
                if supply_zone is not None:
                    z_bot, z_top = min(supply_zone), max(supply_zone)
                    tested_supply = any(df_m15['high'].iloc[-4:] >= z_bot - (5 * PIP_SIZE))
                    ema8_bear_cross = (prev_bar['close'] >= prev_bar['ema8']) and (closed_bar['close'] < closed_bar['ema8']) and (closed_bar['close'] < closed_bar['open'])

                    if tested_supply and ema8_bear_cross:
                        entry = closed_bar['close']
                        sl = z_top + (4.0 * PIP_SIZE)
                        risk = sl - entry
                        if risk > 0:
                            tp = entry - (risk * RISK_REWARD_RATIO)
                            emit_signal("SELL", current_candle_time, entry, sl, tp, z_top, z_bot)

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n[STOP] Scanner stopped by user.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    run_scanner()
