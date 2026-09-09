"""
========================================================================================
MT5 SIGNAL SCANNER: TG CAPITAL LONDON KILLZONE EMA STACK + FVG STRATEGY
========================================================================================
Source: Short Video _bdNNLDEyOU | 1-Year Backtest: +68.4% Return, 76% Win Rate, 9.2% DD
Creator Attribution: Tyler Gottele (TG Capital)
Market: Forex (EURUSD, GBPUSD) / Universal MT5
Timeframe: M30 Execution (London & NY Session)
Indicators:
  1. 200 EMA (Macro Baseline Filter)
  2. Stacked EMAs (5, 9, 13, 21 Periods)
  3. Fair Value Gap (FVG) 3-Candle Imbalance
  4. 50% Equilibrium FVG Retest Rejection
Logic:
  - Buy Setup:
      1. Macro Trend: Price > 200 EMA
      2. Micro Alignment: 5 EMA > 9 EMA > 13 EMA > 21 EMA (Sequential Bullish Stack)
      3. Imbalance: Bullish FVG forms (Candle 1 High < Candle 3 Low)
      4. Mitigation: Price pulls back into the 50% FVG equilibrium with a rejection candle
         whose lower wick tests <= 50% FVG level and closes back above it.
      5. Trigger: Next candle breaks and closes above rejection candle high.
      6. Stop Loss: Placed below FVG Candle 2 (Middle Candle) Low.
      7. Take Profit: 1:2 to 1:3 Risk-to-Reward Ratio.
  - Sell Setup (Inverted):
      1. Macro Trend: Price < 200 EMA
      2. Micro Alignment: 5 EMA < 9 EMA < 13 EMA < 21 EMA (Sequential Bearish Stack)
      3. Imbalance: Bearish FVG forms (Candle 1 Low > Candle 3 High)
      4. Mitigation: Price pulls back into 50% FVG equilibrium with upper wick testing >= 50% level.
      5. Trigger: Next candle breaks and closes below rejection candle low.
      6. Stop Loss: Placed above FVG Candle 2 High.
      7. Take Profit: 1:2 to 1:3 Risk-to-Reward Ratio.
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
TIMEFRAME = mt5.TIMEFRAME_M30    # 30-Minute Chart
EMA_TREND = 200
EMA_5 = 5
EMA_9 = 9
EMA_13 = 13
EMA_21 = 21
RISK_REWARD_RATIO = 2.5
POLL_INTERVAL = 10               # Seconds between poll checks
LOOKBACK_BARS = 250

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


def fetch_rates(count=LOOKBACK_BARS):
    """Fetches M30 candle rates and calculates 200 EMA + 5, 9, 13, 21 EMAs."""
    rates = mt5.copy_rates_from_pos(ACTIVE_SYMBOL, TIMEFRAME, 0, count)
    if rates is None or len(rates) < EMA_TREND + 10:
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    df['ema200'] = df['close'].ewm(span=EMA_TREND, adjust=False).mean()
    df['ema5'] = df['close'].ewm(span=EMA_5, adjust=False).mean()
    df['ema9'] = df['close'].ewm(span=EMA_9, adjust=False).mean()
    df['ema13'] = df['close'].ewm(span=EMA_13, adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=EMA_21, adjust=False).mean()

    return df


def emit_signal(sig_type: str, candle_time: datetime.datetime, entry_price: float, sl_price: float, tp_price: float, fvg_top: float, fvg_bot: float):
    """Prints a structured ASCII signal card to console."""
    risk_pips = abs(entry_price - sl_price) / PIP_SIZE
    reward_pips = abs(tp_price - entry_price) / PIP_SIZE

    print("\n" + "=" * 75)
    print(f" [ALERT] MT5 SIGNAL ALERT: TG CAPITAL LONDON FVG + EMA STACK ({sig_type.upper()})")
    print("=" * 75)
    print(f" Symbol:        {ACTIVE_SYMBOL} | Timeframe: M30")
    print(f" Bar Time:      {candle_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Signal Type:   {sig_type.upper()} (EMA Stack + 50% FVG Mitigation)")
    print(f" FVG Zone:      {fvg_bot:.5f} - {fvg_top:.5f} | 50% Eq: {((fvg_top+fvg_bot)/2):.5f}")
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
    print(f"   TG CAPITAL LONDON EMA STACK + FVG SCANNER ({ACTIVE_SYMBOL} M30)   ")
    print("=" * 75)
    print(" Strategy: 200 EMA + 5/9/13/21 Stack + FVG 50% Equilibrium Retest")
    print(" Status: Scanning live bars... Press Ctrl+C to stop.\n")

    last_processed_candle_time = None

    try:
        while True:
            df = fetch_rates()
            if df is None:
                time.sleep(POLL_INTERVAL)
                continue

            current_candle_time = df['time'].iloc[-1]
            closed_bar = df.iloc[-2]       # Trigger confirmation bar
            retest_bar = df.iloc[-3]       # Candle that wicked into FVG

            if last_processed_candle_time != current_candle_time:
                last_processed_candle_time = current_candle_time

                # ----------------------------------------------------
                # BUY LOGIC
                # ----------------------------------------------------
                # 1. Macro: Price > 200 EMA
                macro_bull = closed_bar['close'] > closed_bar['ema200']
                # 2. Sequential EMA Stack
                ema_stack_bull = (closed_bar['ema5'] > closed_bar['ema9'] > closed_bar['ema13'] > closed_bar['ema21'])

                # 3. Check for Bullish FVG in preceding 6 bars
                # A bullish FVG occurs when Bar i-2 High < Bar i Low
                bull_fvg = None
                for idx in range(len(df) - 6, len(df) - 3):
                    c1_high = df['high'].iloc[idx - 2]
                    c2_low = df['low'].iloc[idx - 1]
                    c3_low = df['low'].iloc[idx]
                    if c3_low > c1_high: # FVG gap exists
                        fvg_top = c3_low
                        fvg_bot = c1_high
                        eq = (fvg_top + fvg_bot) / 2.0
                        # Test if retest_bar tested 50% eq and closed above
                        if retest_bar['low'] <= eq and retest_bar['close'] > eq:
                            bull_fvg = (fvg_top, fvg_bot, c2_low)
                            break

                if macro_bull and ema_stack_bull and bull_fvg is not None:
                    # 4. Trigger: closed_bar breaks retest_bar high
                    if closed_bar['high'] > retest_bar['high'] and closed_bar['close'] > retest_bar['high']:
                        entry = closed_bar['close']
                        sl = bull_fvg[2] - (2.0 * PIP_SIZE)
                        risk = entry - sl
                        if risk > 0:
                            tp = entry + (risk * RISK_REWARD_RATIO)
                            emit_signal("BUY", current_candle_time, entry, sl, tp, bull_fvg[0], bull_fvg[1])

                # ----------------------------------------------------
                # SELL LOGIC
                # ----------------------------------------------------
                macro_bear = closed_bar['close'] < closed_bar['ema200']
                ema_stack_bear = (closed_bar['ema5'] < closed_bar['ema9'] < closed_bar['ema13'] < closed_bar['ema21'])

                bear_fvg = None
                for idx in range(len(df) - 6, len(df) - 3):
                    c1_low = df['low'].iloc[idx - 2]
                    c2_high = df['high'].iloc[idx - 1]
                    c3_high = df['high'].iloc[idx]
                    if c3_high < c1_low: # Bearish FVG gap
                        fvg_top = c1_low
                        fvg_bot = c3_high
                        eq = (fvg_top + fvg_bot) / 2.0
                        if retest_bar['high'] >= eq and retest_bar['close'] < eq:
                            bear_fvg = (fvg_top, fvg_bot, c2_high)
                            break

                if macro_bear and ema_stack_bear and bear_fvg is not None:
                    if closed_bar['low'] < retest_bar['low'] and closed_bar['close'] < retest_bar['low']:
                        entry = closed_bar['close']
                        sl = bear_fvg[2] + (2.0 * PIP_SIZE)
                        risk = sl - entry
                        if risk > 0:
                            tp = entry - (risk * RISK_REWARD_RATIO)
                            emit_signal("SELL", current_candle_time, entry, sl, tp, bear_fvg[0], bear_fvg[1])

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n[STOP] Scanner stopped by user.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    run_scanner()
