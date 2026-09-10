"""
========================================================================================
MT5 SIGNAL SCANNER: MULTI-TIMEFRAME MACD MOMENTUM STRATEGY
========================================================================================
Source: Short Video aUKmS6ZDeq4 | 1-Year Backtest: 62% Win Rate
Market: Universal (MT5 Forex EURUSD, Gold XAUUSD, Crypto, Indices)
Timeframe: M15 (configurable to H1 or M5)
Indicators:
  1. 200 EMA (Exponential Moving Average - Macro Trend Baseline)
  2. MACD (Fast: 12, Slow: 26, Signal: 9, Histogram)
Logic:
  - Buy Setup:
      1. Trend: Price > 200 EMA (Bullish Market Regime).
      2. Momentum Thrust: MACD Line crosses ABOVE Signal Line.
      3. Quality Filter: Crossover occurs near or below the zero line (fresh momentum start).
      4. Trigger: Confirmed on candle close.
      5. Stop Loss: Below recent swing low; Take Profit: 1:2 Risk-Reward.
  - Sell Setup:
      1. Trend: Price < 200 EMA (Bearish Market Regime).
      2. Momentum Thrust: MACD Line crosses BELOW Signal Line.
      3. Quality Filter: Crossover occurs near or above the zero line (fresh momentum start).
      4. Trigger: Confirmed on candle close.
      5. Stop Loss: Above recent swing high; Take Profit: 1:2 Risk-Reward.
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
TIMEFRAME = mt5.TIMEFRAME_M15    # 15-Minute Chart
EMA_TREND = 200
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
RISK_REWARD_RATIO = 2.0
POLL_INTERVAL = 5                # Seconds between poll checks
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
    """Fetches candle rates and calculates 200 EMA and MACD."""
    rates = mt5.copy_rates_from_pos(ACTIVE_SYMBOL, TIMEFRAME, 0, count)
    if rates is None or len(rates) < EMA_TREND + 20:
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # 200 EMA
    df['ema200'] = df['close'].ewm(span=EMA_TREND, adjust=False).mean()

    # MACD Calculation
    fast_ema = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
    slow_ema = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    df['macd_line'] = fast_ema - slow_ema
    df['signal_line'] = df['macd_line'].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df['macd_hist'] = df['macd_line'] - df['signal_line']

    return df


def emit_signal(sig_type: str, candle_time: datetime.datetime, entry_price: float, sl_price: float, tp_price: float, macd_val: float, sig_val: float):
    """Prints a structured ASCII signal card to console."""
    risk_pips = abs(entry_price - sl_price) / PIP_SIZE
    reward_pips = abs(tp_price - entry_price) / PIP_SIZE

    print("\n" + "=" * 75)
    print(f" [ALERT] MT5 SIGNAL ALERT: MACD MOMENTUM SYSTEM ({sig_type.upper()})")
    print("=" * 75)
    print(f" Symbol:        {ACTIVE_SYMBOL} | Timeframe: M15")
    print(f" Bar Time:      {candle_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Signal Type:   {sig_type.upper()} (Trend Momentum Crossover)")
    print(f" MACD Line:     {macd_val:.5f} | Signal Line: {sig_val:.5f}")
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
    print(f"   MACD 62% WIN-RATE MOMENTUM SIGNAL SCANNER ({ACTIVE_SYMBOL} M15)   ")
    print("=" * 75)
    print(" Strategy: 200 EMA Baseline + MACD Fast/Signal Crossover Thrust")
    print(" Status: Scanning live bars... Press Ctrl+C to stop.\n")

    last_processed_candle_time = None

    try:
        while True:
            df = fetch_rates()
            if df is None:
                time.sleep(POLL_INTERVAL)
                continue

            current_candle_time = df['time'].iloc[-1]
            closed_bar = df.iloc[-2]       # Newly completed bar
            prev_bar = df.iloc[-3]         # Prior bar for crossover verification

            if last_processed_candle_time != current_candle_time:
                last_processed_candle_time = current_candle_time

                # ----------------------------------------------------
                # BUY LOGIC (Bullish Trend Momentum)
                # ----------------------------------------------------
                # 1. Macro Trend: Price above 200 EMA
                trend_bullish = closed_bar['close'] > closed_bar['ema200']
                
                # 2. Bullish MACD Crossover on closed bar
                bull_cross = (prev_bar['macd_line'] <= prev_bar['signal_line']) and (closed_bar['macd_line'] > closed_bar['signal_line'])
                
                # 3. Quality Filter: Crossover is fresh (MACD line not extremely extended)
                fresh_cross = closed_bar['macd_line'] < (0.0010)

                if trend_bullish and bull_cross and fresh_cross:
                    entry = closed_bar['close']
                    recent_low = df['low'].iloc[-6:-1].min()
                    sl = recent_low - (2.0 * PIP_SIZE)
                    risk = entry - sl
                    if risk > 0:
                        tp = entry + (risk * RISK_REWARD_RATIO)
                        emit_signal("BUY", current_candle_time, entry, sl, tp, closed_bar['macd_line'], closed_bar['signal_line'])

                # ----------------------------------------------------
                # SELL LOGIC (Bearish Trend Momentum)
                # ----------------------------------------------------
                # 1. Macro Trend: Price below 200 EMA
                trend_bearish = closed_bar['close'] < closed_bar['ema200']
                
                # 2. Bearish MACD Crossover on closed bar
                bear_cross = (prev_bar['macd_line'] >= prev_bar['signal_line']) and (closed_bar['macd_line'] < closed_bar['signal_line'])
                
                # 3. Quality Filter: Crossover is fresh
                fresh_cross_bear = closed_bar['macd_line'] > (-0.0010)

                if trend_bearish and bear_cross and fresh_cross_bear:
                    entry = closed_bar['close']
                    recent_high = df['high'].iloc[-6:-1].max()
                    sl = recent_high + (2.0 * PIP_SIZE)
                    risk = sl - entry
                    if risk > 0:
                        tp = entry - (risk * RISK_REWARD_RATIO)
                        emit_signal("SELL", current_candle_time, entry, sl, tp, closed_bar['macd_line'], closed_bar['signal_line'])

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n[STOP] Scanner stopped by user.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    run_scanner()
