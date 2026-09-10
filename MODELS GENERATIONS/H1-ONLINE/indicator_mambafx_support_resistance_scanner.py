"""
========================================================================================
MT5 SIGNAL SCANNER: MAMBA FX FOREX S/R + 8/50 SMA SCALPING STRATEGY
========================================================================================
Source: Short Video TSRKVlz4tjo | 1-Year Backtest: +62% Return, 13% DD
Creator Attribution: Mamba FX (US Forex Trader)
Market: Forex (EURUSD, GBPUSD) / Universal MT5
Timeframe: H4 Support/Resistance + M15 SMA Breakout Execution
Indicators:
  1. H4 Structural Support & Resistance Zones
  2. 8 SMA (Simple Moving Average)
  3. 50 SMA (Simple Moving Average)
Logic:
  - Buy Setup:
      1. Structural Context: Price is rebounding from H4 Structural Support.
      2. Trend Transition: On M15, 8 SMA is above 50 SMA (or crosses above it).
      3. Trigger: A strong green candle breaks and closes above both 8 and 50 SMAs.
      4. Stop Loss: Placed below the 50 SMA and recent swing low.
      5. Take Profit: H4 Resistance target or 1:2.5 Risk-to-Reward Ratio.
  - Sell Setup (Inverted):
      1. Structural Context: Price is rejecting from H4 Structural Resistance.
      2. Trend Transition: On M15, 8 SMA is below 50 SMA.
      3. Trigger: A strong red candle breaks and closes below both 8 and 50 SMAs.
      4. Stop Loss: Placed above the 50 SMA and recent swing high.
      5. Take Profit: H4 Support target or 1:2.5 Risk-to-Reward Ratio.
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
DEFAULT_SYMBOLS = ["EURUSD.0", "EURUSD", "EURUSDm", "EURUSD.pro", "GBPUSD", "XAUUSD"]
TIMEFRAME_HTF = mt5.TIMEFRAME_H4   # 4-Hour S/R
TIMEFRAME_LTF = mt5.TIMEFRAME_M15  # 15-Minute Execution
SMA_FAST = 8
SMA_SLOW = 50
RISK_REWARD_RATIO = 2.5
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
            PIP_SIZE = 0.0001 if info.digits in (4, 5) else (0.01 if info.digits in (2, 3) else 1.0)
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


def find_h4_sr_levels(df_h4: pd.DataFrame):
    """Calculates major H4 Support and Resistance levels from recent swing points."""
    h4_highs = df_h4['high'].values
    h4_lows = df_h4['low'].values
    length = len(df_h4)

    recent_support = min(h4_lows[-15:])
    recent_resistance = max(h4_highs[-15:])
    return recent_support, recent_resistance


def emit_signal(sig_type: str, candle_time: datetime.datetime, entry_price: float, sl_price: float, tp_price: float, h4_level: float):
    """Prints a structured ASCII signal card to console."""
    risk_pips = abs(entry_price - sl_price) / PIP_SIZE
    reward_pips = abs(tp_price - entry_price) / PIP_SIZE

    print("\n" + "=" * 75)
    print(f" [ALERT] MT5 SIGNAL ALERT: MAMBA FX S/R + 8/50 SMA ({sig_type.upper()})")
    print("=" * 75)
    print(f" Symbol:        {ACTIVE_SYMBOL} | Timeframe: M15 (with H4 S/R)")
    print(f" Bar Time:      {candle_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Signal Type:   {sig_type.upper()} (H4 S/R Rebound + 8/50 SMA Break)")
    print(f" H4 Key Level:  {h4_level:.5f}")
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
    print(f"   MAMBA FX FOREX S/R + 8/50 SMA SCANNER ({ACTIVE_SYMBOL} M15)   ")
    print("=" * 75)
    print(" Strategy: H4 Structural S/R Bounce + M15 8/50 SMA Momentum Breakout")
    print(" Status: Scanning live bars... Press Ctrl+C to stop.\n")

    last_processed_candle_time = None

    try:
        while True:
            df_h4 = fetch_rates(TIMEFRAME_HTF, count=40)
            df_m15 = fetch_rates(TIMEFRAME_LTF, count=100)

            if df_h4 is None or df_m15 is None:
                time.sleep(POLL_INTERVAL)
                continue

            current_candle_time = df_m15['time'].iloc[-1]
            closed_bar = df_m15.iloc[-2]
            prev_bar = df_m15.iloc[-3]

            if last_processed_candle_time != current_candle_time:
                last_processed_candle_time = current_candle_time

                # SMAs on M15
                df_m15['sma8'] = df_m15['close'].rolling(window=SMA_FAST).mean()
                df_m15['sma50'] = df_m15['close'].rolling(window=SMA_SLOW).mean()

                h4_support, h4_resistance = find_h4_sr_levels(df_h4)

                # ----------------------------------------------------
                # BUY LOGIC: Bouncing off H4 Support + Break above 8/50 SMA
                # ----------------------------------------------------
                near_h4_support = any(df_m15['low'].iloc[-6:] <= h4_support + (15 * PIP_SIZE))
                sma_bull = df_m15['sma8'].iloc[-2] > df_m15['sma50'].iloc[-2]
                breakout_sma_bull = (prev_bar['close'] <= prev_bar['sma8']) and (closed_bar['close'] > closed_bar['sma8']) and (closed_bar['close'] > closed_bar['sma50'])

                if near_h4_support and sma_bull and breakout_sma_bull:
                    entry = closed_bar['close']
                    sl = closed_bar['sma50'] - (3.0 * PIP_SIZE)
                    risk = entry - sl
                    if risk > 0:
                        tp = h4_resistance if h4_resistance > entry else entry + (risk * RISK_REWARD_RATIO)
                        emit_signal("BUY", current_candle_time, entry, sl, tp, h4_support)

                # ----------------------------------------------------
                # SELL LOGIC: Rejecting H4 Resistance + Break below 8/50 SMA
                # ----------------------------------------------------
                near_h4_resistance = any(df_m15['high'].iloc[-6:] >= h4_resistance - (15 * PIP_SIZE))
                sma_bear = df_m15['sma8'].iloc[-2] < df_m15['sma50'].iloc[-2]
                breakdown_sma_bear = (prev_bar['close'] >= prev_bar['sma8']) and (closed_bar['close'] < closed_bar['sma8']) and (closed_bar['close'] < closed_bar['sma50'])

                if near_h4_resistance and sma_bear and breakdown_sma_bear:
                    entry = closed_bar['close']
                    sl = closed_bar['sma50'] + (3.0 * PIP_SIZE)
                    risk = sl - entry
                    if risk > 0:
                        tp = h4_support if h4_support < entry else entry - (risk * RISK_REWARD_RATIO)
                        emit_signal("SELL", current_candle_time, entry, sl, tp, h4_resistance)

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n[STOP] Scanner stopped by user.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    run_scanner()
