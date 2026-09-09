"""
========================================================================================
MT5 SIGNAL SCANNER: EZEKIEL CHEW (PROP FIRM) 50 EMA REJECTION STRATEGY
========================================================================================
Source: Short Video AKuM1cTgPks | Tested by Sahil
Creator Attribution: Ezekiel Chew (Asia Forex Mentor / Prop Firm Trainer)
Market: Universal (MT5 Forex EURUSD, Gold XAUUSD, Crypto, Indices)
Timeframe: H1 (configurable to M15)
Indicators: 50 EMA (Exponential Moving Average)
Logic:
  - Trend Alignment:
      - Uptrend: 50 EMA sloping upwards, price has been trading above 50 EMA.
      - Downtrend: 50 EMA sloping downwards, price has been trading below 50 EMA.
  - Pullback & Rejection:
      - Bullish Setup: Price pulls back to touch the 50 EMA. Forms a rejection candle
        with a long lower wick (>= 40% of total candle range) testing the 50 EMA and closing bullish.
      - Bearish Setup: Price pulls back to touch the 50 EMA. Forms a rejection candle
        with a long upper wick (>= 40% of total candle range) testing the 50 EMA and closing bearish.
  - Trigger & Risk Management:
      - Trigger: Confirmation bar breaks the extreme of the rejection candle.
      - Stop Loss: Placed beyond the rejection candle's wick.
      - Take Profit: 1:2 Risk-to-Reward Ratio.
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
TIMEFRAME = mt5.TIMEFRAME_H1     # 1-Hour Chart (Standard Prop Firm TF)
EMA_PERIOD = 50
RISK_REWARD_RATIO = 2.0
POLL_INTERVAL = 10               # Seconds between poll checks
LOOKBACK_BARS = 150

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
    """Fetches candle rates and calculates 50 EMA and candle wick metrics."""
    rates = mt5.copy_rates_from_pos(ACTIVE_SYMBOL, TIMEFRAME, 0, count)
    if rates is None or len(rates) < EMA_PERIOD + 15:
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # 50 EMA
    df['ema50'] = df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()

    # Candle metrics
    df['range'] = df['high'] - df['low']
    df['body'] = (df['close'] - df['open']).abs()
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)

    return df


def emit_signal(sig_type: str, candle_time: datetime.datetime, entry_price: float, sl_price: float, tp_price: float, ema_val: float):
    """Prints a structured ASCII signal card to console."""
    risk_pips = abs(entry_price - sl_price) / PIP_SIZE
    reward_pips = abs(tp_price - entry_price) / PIP_SIZE

    print("\n" + "=" * 75)
    print(f" [ALERT] MT5 SIGNAL ALERT: EZEKIEL CHEW 50 EMA REJECTION ({sig_type.upper()})")
    print("=" * 75)
    print(f" Symbol:        {ACTIVE_SYMBOL} | Timeframe: H1")
    print(f" Bar Time:      {candle_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Signal Type:   {sig_type.upper()} (50 EMA Dynamic Rejection)")
    print(f" 50 EMA Level:  {ema_val:.5f}")
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
    print(f"   EZEKIEL CHEW 50 EMA REJECTION SCANNER ({ACTIVE_SYMBOL} H1)   ")
    print("=" * 75)
    print(" Strategy: 50 EMA Trend Slope + Pullback Wick Rejection + Breakout")
    print(" Status: Scanning live bars... Press Ctrl+C to stop.\n")

    last_processed_candle_time = None

    try:
        while True:
            df = fetch_rates()
            if df is None:
                time.sleep(POLL_INTERVAL)
                continue

            current_candle_time = df['time'].iloc[-1]
            closed_bar = df.iloc[-2]       # Last completed bar
            rejection_bar = df.iloc[-3]    # Candidate rejection candle

            if last_processed_candle_time != current_candle_time:
                last_processed_candle_time = current_candle_time

                # ----------------------------------------------------
                # BUY LOGIC (Uptrend 50 EMA Rejection)
                # ----------------------------------------------------
                # 1. 50 EMA slope is positive
                ema_rising = rejection_bar['ema50'] > df['ema50'].iloc[-8]
                # 2. Rejection candle touched 50 EMA with its low
                tested_ema_buy = rejection_bar['low'] <= rejection_bar['ema50'] <= rejection_bar['high']
                # 3. Strong lower wick (rejection) >= 35% of total candle range & closed above EMA
                bull_wick = (rejection_bar['range'] > 0) and (rejection_bar['lower_wick'] >= 0.35 * rejection_bar['range'])
                closed_above = rejection_bar['close'] > rejection_bar['ema50']
                # 4. Confirmation: Closed bar broke rejection bar high
                breakout_buy = (closed_bar['high'] > rejection_bar['high']) and (closed_bar['close'] > rejection_bar['high'])

                if ema_rising and tested_ema_buy and bull_wick and closed_above and breakout_buy:
                    entry = closed_bar['close']
                    sl = rejection_bar['low'] - (2.0 * PIP_SIZE)
                    risk = entry - sl
                    if risk > 0:
                        tp = entry + (risk * RISK_REWARD_RATIO)
                        emit_signal("BUY", current_candle_time, entry, sl, tp, rejection_bar['ema50'])

                # ----------------------------------------------------
                # SELL LOGIC (Downtrend 50 EMA Rejection)
                # ----------------------------------------------------
                # 1. 50 EMA slope is negative
                ema_falling = rejection_bar['ema50'] < df['ema50'].iloc[-8]
                # 2. Rejection candle touched 50 EMA with its high
                tested_ema_sell = rejection_bar['low'] <= rejection_bar['ema50'] <= rejection_bar['high']
                # 3. Strong upper wick (rejection) >= 35% of total candle range & closed below EMA
                bear_wick = (rejection_bar['range'] > 0) and (rejection_bar['upper_wick'] >= 0.35 * rejection_bar['range'])
                closed_below = rejection_bar['close'] < rejection_bar['ema50']
                # 4. Confirmation: Closed bar broke rejection bar low
                breakout_sell = (closed_bar['low'] < rejection_bar['low']) and (closed_bar['close'] < rejection_bar['low'])

                if ema_falling and tested_ema_sell and bear_wick and closed_below and breakout_sell:
                    entry = closed_bar['close']
                    sl = rejection_bar['high'] + (2.0 * PIP_SIZE)
                    risk = sl - entry
                    if risk > 0:
                        tp = entry - (risk * RISK_REWARD_RATIO)
                        emit_signal("SELL", current_candle_time, entry, sl, tp, rejection_bar['ema50'])

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n[STOP] Scanner stopped by user.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    run_scanner()
