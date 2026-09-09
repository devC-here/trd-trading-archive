"""
========================================================================================
MT5 SIGNAL SCANNER: ANANT LADHA 10 EMA FLOATING ALERT STRATEGY
========================================================================================
Source: Short Video FYfQqZ3bM6g | 1-Year Backtest: +38.6% Return, 54% Win Rate, 13.4% DD
Creator Attribution: Anant Ladha (Invest Aaj For Kal - 3M+ Subs)
Market: Forex (EURUSD, GBPUSD) / Universal MT5
Timeframe: M5 / M15 Execution
Indicators: 10 EMA (Exponential Moving Average)
Logic:
  - Buy Setup:
      1. Trend: Price is in an uptrend trading above 10 EMA.
      2. Alert Candle: A RED candle forms that floats completely above the 10 EMA without
         touching it (Alert Candle Low > 10 EMA).
      3. Trigger: The subsequent candle breaks and closes above the High of the Red Alert Candle.
      4. Stop Loss: Placed at the 10 EMA level (below alert candle low).
      5. Take Profit: 1:3 Risk-to-Reward Ratio (High RR trend continuation).
  - Sell Setup:
      1. Trend: Price is in a downtrend trading below 10 EMA.
      2. Alert Candle: A GREEN candle forms that floats completely below the 10 EMA without
         touching it (Alert Candle High < 10 EMA).
      3. Trigger: The subsequent candle breaks and closes below the Low of the Green Alert Candle.
      4. Stop Loss: Placed at the 10 EMA level (above alert candle high).
      5. Take Profit: 1:3 Risk-to-Reward Ratio.
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
TIMEFRAME = mt5.TIMEFRAME_M5     # 5-Minute Execution (or M15)
EMA_PERIOD = 10
RISK_REWARD_RATIO = 3.0          # High RR trend runner (1:3 to 1:5)
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
    """Fetches candle rates and calculates 10 EMA."""
    rates = mt5.copy_rates_from_pos(ACTIVE_SYMBOL, TIMEFRAME, 0, count)
    if rates is None or len(rates) < EMA_PERIOD + 10:
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # 10 EMA
    df['ema10'] = df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()

    return df


def emit_signal(sig_type: str, candle_time: datetime.datetime, entry_price: float, sl_price: float, tp_price: float, ema_val: float, alert_candle_time):
    """Prints a structured ASCII signal card to console."""
    risk_pips = abs(entry_price - sl_price) / PIP_SIZE
    reward_pips = abs(tp_price - entry_price) / PIP_SIZE

    print("\n" + "=" * 75)
    print(f" [ALERT] MT5 SIGNAL ALERT: ANANT LADHA 10 EMA FLOATING SYSTEM ({sig_type.upper()})")
    print("=" * 75)
    print(f" Symbol:        {ACTIVE_SYMBOL} | Timeframe: M5")
    print(f" Bar Time:      {candle_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Signal Type:   {sig_type.upper()} (Non-Touching Alert Candle Breakout)")
    print(f" Alert Candle:  {alert_candle_time.strftime('%Y-%m-%d %H:%M:%S')} (Detached from 10 EMA)")
    print(f" 10 EMA Level:  {ema_val:.5f}")
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
    print(f"   ANANT LADHA 10 EMA SIGNAL SCANNER ({ACTIVE_SYMBOL} M5)   ")
    print("=" * 75)
    print(" Strategy: Floating Non-Touching Pullback Candle -> Next Candle Breakout")
    print(" Status: Scanning live bars... Press Ctrl+C to stop.\n")

    last_processed_candle_time = None

    try:
        while True:
            df = fetch_rates()
            if df is None:
                time.sleep(POLL_INTERVAL)
                continue

            current_candle_time = df['time'].iloc[-1]
            closed_bar = df.iloc[-2]       # Trigger candle
            alert_bar = df.iloc[-3]        # Floating alert candle

            if last_processed_candle_time != current_candle_time:
                last_processed_candle_time = current_candle_time

                # ----------------------------------------------------
                # BUY LOGIC (Uptrend Floating Red Candle Breakout)
                # ----------------------------------------------------
                # 1. Alert bar is RED
                is_red_alert = alert_bar['close'] < alert_bar['open']
                # 2. Alert bar floats completely above 10 EMA without touching it
                floats_above_ema = alert_bar['low'] > alert_bar['ema10']
                # 3. Trigger: closed_bar breaks and closes above alert_bar High
                bull_breakout = (closed_bar['high'] > alert_bar['high']) and (closed_bar['close'] > alert_bar['high'])

                if is_red_alert and floats_above_ema and bull_breakout:
                    entry = closed_bar['close']
                    sl = min(alert_bar['low'], alert_bar['ema10']) - (2.0 * PIP_SIZE)
                    risk = entry - sl
                    if risk > 0:
                        tp = entry + (risk * RISK_REWARD_RATIO)
                        emit_signal("BUY", current_candle_time, entry, sl, tp, closed_bar['ema10'], alert_bar['time'])

                # ----------------------------------------------------
                # SELL LOGIC (Downtrend Floating Green Candle Breakdown)
                # ----------------------------------------------------
                # 1. Alert bar is GREEN
                is_green_alert = alert_bar['close'] > alert_bar['open']
                # 2. Alert bar floats completely below 10 EMA without touching it
                floats_below_ema = alert_bar['high'] < alert_bar['ema10']
                # 3. Trigger: closed_bar breaks and closes below alert_bar Low
                bear_breakdown = (closed_bar['low'] < alert_bar['low']) and (closed_bar['close'] < alert_bar['low'])

                if is_green_alert and floats_below_ema and bear_breakdown:
                    entry = closed_bar['close']
                    sl = max(alert_bar['high'], alert_bar['ema10']) + (2.0 * PIP_SIZE)
                    risk = sl - entry
                    if risk > 0:
                        tp = entry - (risk * RISK_REWARD_RATIO)
                        emit_signal("SELL", current_candle_time, entry, sl, tp, closed_bar['ema10'], alert_bar['time'])

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n[STOP] Scanner stopped by user.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    run_scanner()
