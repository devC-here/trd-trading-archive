"""
========================================================================================
MT5 SIGNAL SCANNER: RAYNER TEO 10 EMA + INSIDE BAR MOMENTUM STRATEGY
========================================================================================
Source: Short Video r-0XAsrQR-I | 1-Year Backtest: +52% Return, 19% DD
Creator Attribution: Rayner Teo (2M+ Subscribers)
Market: Forex (EURUSD, GBPUSD) / Universal MT5
Timeframe: H1 Execution (configurable to M15)
Indicators: 10 EMA (Exponential Moving Average)
Logic:
  - Buy Setup:
      1. Trend Regime: Price is trading above the 10 EMA.
      2. Inside Bar Setup: An Inside Bar forms inside the Mother Bar
         (Inside Bar High < Mother Bar High AND Inside Bar Low > Mother Bar Low).
      3. Trigger: A subsequent candle breaks and closes ABOVE the Mother Bar's High.
      4. Stop Loss: Placed slightly below the Inside Bar's Low.
      5. Take Profit: 1:2.5 Risk-to-Reward Ratio.
  - Sell Setup (Inverted):
      1. Trend Regime: Price is trading below the 10 EMA.
      2. Inside Bar Setup: An Inside Bar forms inside the Mother Bar.
      3. Trigger: A subsequent candle breaks and closes BELOW the Mother Bar's Low.
      4. Stop Loss: Placed slightly above the Inside Bar's High.
      5. Take Profit: 1:2.5 Risk-to-Reward Ratio.
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
TIMEFRAME = mt5.TIMEFRAME_H1     # 1-Hour Chart (or M15)
EMA_PERIOD = 10
RISK_REWARD_RATIO = 2.5
POLL_INTERVAL = 10               # Seconds between poll checks
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
    df['ema10'] = df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    return df


def emit_signal(sig_type: str, candle_time: datetime.datetime, entry_price: float, sl_price: float, tp_price: float, mother_high: float, mother_low: float):
    """Prints a structured ASCII signal card to console."""
    risk_pips = abs(entry_price - sl_price) / PIP_SIZE
    reward_pips = abs(tp_price - entry_price) / PIP_SIZE

    print("\n" + "=" * 75)
    print(f" [ALERT] MT5 SIGNAL ALERT: RAYNER TEO INSIDE BAR BREAKOUT ({sig_type.upper()})")
    print("=" * 75)
    print(f" Symbol:        {ACTIVE_SYMBOL} | Timeframe: H1")
    print(f" Bar Time:      {candle_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Signal Type:   {sig_type.upper()} (10 EMA Trend + Inside Bar Contraction Break)")
    print(f" Mother Bar:    High = {mother_high:.5f} | Low = {mother_low:.5f}")
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
    print(f"   RAYNER TEO 10 EMA INSIDE BAR SCANNER ({ACTIVE_SYMBOL} H1)   ")
    print("=" * 75)
    print(" Strategy: 10 EMA Trend + Volatility Contraction Inside Bar Breakout")
    print(" Status: Scanning live bars... Press Ctrl+C to stop.\n")

    last_processed_candle_time = None

    try:
        while True:
            df = fetch_rates()
            if df is None:
                time.sleep(POLL_INTERVAL)
                continue

            current_candle_time = df['time'].iloc[-1]
            closed_bar = df.iloc[-2]       # Trigger bar
            inside_bar = df.iloc[-3]       # Inside bar candidate
            mother_bar = df.iloc[-4]       # Mother bar

            if last_processed_candle_time != current_candle_time:
                last_processed_candle_time = current_candle_time

                # Check if inside_bar is genuinely inside mother_bar
                is_inside = (inside_bar['high'] < mother_bar['high']) and (inside_bar['low'] > mother_bar['low'])

                if is_inside:
                    # ----------------------------------------------------
                    # BUY LOGIC: Price > 10 EMA + Break above Mother High
                    # ----------------------------------------------------
                    bull_trend = closed_bar['close'] > closed_bar['ema10']
                    bull_breakout = (closed_bar['high'] > mother_bar['high']) and (closed_bar['close'] > mother_bar['high'])

                    if bull_trend and bull_breakout:
                        entry = closed_bar['close']
                        sl = inside_bar['low'] - (2.0 * PIP_SIZE)
                        risk = entry - sl
                        if risk > 0:
                            tp = entry + (risk * RISK_REWARD_RATIO)
                            emit_signal("BUY", current_candle_time, entry, sl, tp, mother_bar['high'], mother_bar['low'])

                    # ----------------------------------------------------
                    # SELL LOGIC: Price < 10 EMA + Break below Mother Low
                    # ----------------------------------------------------
                    bear_trend = closed_bar['close'] < closed_bar['ema10']
                    bear_breakdown = (closed_bar['low'] < mother_bar['low']) and (closed_bar['close'] < mother_bar['low'])

                    if bear_trend and bear_breakdown:
                        entry = closed_bar['close']
                        sl = inside_bar['high'] + (2.0 * PIP_SIZE)
                        risk = sl - entry
                        if risk > 0:
                            tp = entry - (risk * RISK_REWARD_RATIO)
                            emit_signal("SELL", current_candle_time, entry, sl, tp, mother_bar['high'], mother_bar['low'])

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n[STOP] Scanner stopped by user.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    run_scanner()
