"""
========================================================================================
MT5 SIGNAL SCANNER: 8/20 DUAL EMA VALUE-ZONE SCALPING STRATEGY
========================================================================================
Source: Short Video uDGXfF1kjHI | 1-Year Backtest: +41.8% Return, 55% Win Rate, 14.3% DD
Market: Forex (EURUSD, GBPUSD) / Universal MT5
Timeframe: M5 Execution
Indicators:
  1. 8 EMA (Fast Momentum Guide)
  2. 20 EMA (Intraday Value Baseline - Forex 24h equivalent of session VWAP)
Logic:
  - Buy Setup:
      1. Trend Regime: Price > 20 EMA and 8 EMA > 20 EMA (Bullish Momentum).
      2. Pullback: Price retraces back to touch the 20/8 EMA value pocket (Low <= 20 EMA).
      3. Rejection & Reclaim: A solid green candle forms and closes back ABOVE the 8 EMA.
      4. Entry: Confirmed on close of the bullish reclaim candle.
      5. Stop Loss: Placed below the low of the pullback candle.
      6. Take Profit: 1:2 Risk-to-Reward Ratio.
  - Sell Setup (Inverted):
      1. Trend Regime: Price < 20 EMA and 8 EMA < 20 EMA (Bearish Momentum).
      2. Pullback: Price retraces back to touch the 20/8 EMA value pocket (High >= 20 EMA).
      3. Rejection & Reclaim: A solid red candle forms and closes back BELOW the 8 EMA.
      4. Entry: Confirmed on close of the bearish reclaim candle.
      5. Stop Loss: Placed above the high of the pullback candle.
      6. Take Profit: 1:2 Risk-to-Reward Ratio.
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
TIMEFRAME = mt5.TIMEFRAME_M5     # 5-Minute Chart
EMA_FAST = 8
EMA_SLOW = 20
RISK_REWARD_RATIO = 2.0
POLL_INTERVAL = 5                # Seconds between poll checks
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


def fetch_rates(count=LOOKBACK_BARS):
    """Fetches candle rates and calculates 8 and 20 EMAs."""
    rates = mt5.copy_rates_from_pos(ACTIVE_SYMBOL, TIMEFRAME, 0, count)
    if rates is None or len(rates) < EMA_SLOW + 10:
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    df['ema8'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
    df['ema20'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()

    return df


def emit_signal(sig_type: str, candle_time: datetime.datetime, entry_price: float, sl_price: float, tp_price: float, ema8_val: float, ema20_val: float):
    """Prints a structured ASCII signal card to console."""
    risk_pips = abs(entry_price - sl_price) / PIP_SIZE
    reward_pips = abs(tp_price - entry_price) / PIP_SIZE

    print("\n" + "=" * 75)
    print(f" [ALERT] MT5 SIGNAL ALERT: 8/20 EMA VALUE-ZONE SCALP ({sig_type.upper()})")
    print("=" * 75)
    print(f" Symbol:        {ACTIVE_SYMBOL} | Timeframe: M5")
    print(f" Bar Time:      {candle_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Signal Type:   {sig_type.upper()} (Value Pocket Pullback & Reclaim)")
    print(f" EMAs:          8 EMA = {ema8_val:.5f} | 20 EMA = {ema20_val:.5f}")
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
    print(f"   8/20 EMA VALUE-ZONE SCALPER SCANNER ({ACTIVE_SYMBOL} M5)   ")
    print("=" * 75)
    print(" Strategy: 8/20 EMA Alignment + Pullback to 20 EMA + 8 EMA Reclaim")
    print(" Status: Scanning live bars... Press Ctrl+C to stop.\n")

    last_processed_candle_time = None

    try:
        while True:
            df = fetch_rates()
            if df is None:
                time.sleep(POLL_INTERVAL)
                continue

            current_candle_time = df['time'].iloc[-1]
            closed_bar = df.iloc[-2]       # Reclaim candle
            pullback_bar = df.iloc[-3]     # Touch candle

            if last_processed_candle_time != current_candle_time:
                last_processed_candle_time = current_candle_time

                # ----------------------------------------------------
                # BUY LOGIC (8/20 Bullish Alignment + Pullback)
                # ----------------------------------------------------
                bull_alignment = closed_bar['ema8'] > closed_bar['ema20']
                # Pulled back to touch 20 EMA
                tested_20 = (pullback_bar['low'] <= pullback_bar['ema20']) or (closed_bar['low'] <= closed_bar['ema20'])
                # Reclaim: Closed bar is green and closed above 8 EMA
                green_reclaim = (closed_bar['close'] > closed_bar['open']) and (closed_bar['close'] > closed_bar['ema8'])

                if bull_alignment and tested_20 and green_reclaim:
                    entry = closed_bar['close']
                    recent_low = min(pullback_bar['low'], closed_bar['low'])
                    sl = recent_low - (2.0 * PIP_SIZE)
                    risk = entry - sl
                    if risk > 0:
                        tp = entry + (risk * RISK_REWARD_RATIO)
                        emit_signal("BUY", current_candle_time, entry, sl, tp, closed_bar['ema8'], closed_bar['ema20'])

                # ----------------------------------------------------
                # SELL LOGIC (8/20 Bearish Alignment + Pullback)
                # ----------------------------------------------------
                bear_alignment = closed_bar['ema8'] < closed_bar['ema20']
                tested_20_bear = (pullback_bar['high'] >= pullback_bar['ema20']) or (closed_bar['high'] >= closed_bar['ema20'])
                red_reclaim = (closed_bar['close'] < closed_bar['open']) and (closed_bar['close'] < closed_bar['ema8'])

                if bear_alignment and tested_20_bear and red_reclaim:
                    entry = closed_bar['close']
                    recent_high = max(pullback_bar['high'], closed_bar['high'])
                    sl = recent_high + (2.0 * PIP_SIZE)
                    risk = sl - entry
                    if risk > 0:
                        tp = entry - (risk * RISK_REWARD_RATIO)
                        emit_signal("SELL", current_candle_time, entry, sl, tp, closed_bar['ema8'], closed_bar['ema20'])

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n[STOP] Scanner stopped by user.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    run_scanner()
