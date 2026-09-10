"""
========================================================================================
MT5 SIGNAL SCANNER: RAYNER TEO 200 SMA + RSI(10) SWING PULLBACK STRATEGY
========================================================================================
Source: Short Video xLTliEACIQc | Tested by Sahil
Creator Attribution: Rayner Teo (TradingwithRayner)
Market: Universal (Forex EURUSD, Gold XAUUSD, Crypto, Indices)
Timeframe: H1 Execution (configurable to M15 or Daily)
Indicators:
  1. 200 SMA (Simple Moving Average - Macro Regime Filter)
  2. 10 RSI (Relative Strength Index - Pullback Extremes)
Logic:
  - Buy Setup:
      1. Price is ABOVE 200 SMA (Bullish Macro Trend).
      2. RSI(10) pulls back and drops below 30 (Oversold condition in uptrend).
      3. Trigger: Price forms a bullish candle while RSI hooks back up above 30.
      4. Stop Loss: Placed below the recent swing low.
      5. Take Profit: 1:2 Risk-to-Reward Ratio.
  - Sell Setup:
      1. Price is BELOW 200 SMA (Bearish Macro Trend).
      2. RSI(10) pulls back and climbs above 70 (Overbought condition in downtrend).
      3. Trigger: Price forms a bearish candle while RSI hooks back down below 70.
      4. Stop Loss: Placed above the recent swing high.
      5. Take Profit: 1:2 Risk-to-Reward Ratio.
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
TIMEFRAME = mt5.TIMEFRAME_H1     # 1-Hour Trend Swing (Can also be M15)
SMA_PERIOD = 200
RSI_PERIOD = 10
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
RISK_REWARD_RATIO = 2.0
POLL_INTERVAL = 10               # Seconds between poll checks
LOOKBACK_BARS = 300

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


def calculate_rsi(series: pd.Series, period: int = 10) -> pd.Series:
    """Calculates Wilder's Relative Strength Index."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def fetch_rates(count=LOOKBACK_BARS):
    """Fetches candle rates and calculates 200 SMA and 10 RSI."""
    rates = mt5.copy_rates_from_pos(ACTIVE_SYMBOL, TIMEFRAME, 0, count)
    if rates is None or len(rates) < SMA_PERIOD + 10:
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # 200 SMA
    df['sma200'] = df['close'].rolling(window=SMA_PERIOD).mean()

    # 10 RSI
    df['rsi10'] = calculate_rsi(df['close'], period=RSI_PERIOD)

    return df


def emit_signal(sig_type: str, candle_time: datetime.datetime, entry_price: float, sl_price: float, tp_price: float, rsi_val: float):
    """Prints a structured ASCII signal card to console."""
    risk_pips = abs(entry_price - sl_price) / PIP_SIZE
    reward_pips = abs(tp_price - entry_price) / PIP_SIZE

    print("\n" + "=" * 75)
    print(f" [ALERT] MT5 SIGNAL ALERT: RAYNER TEO 200 SMA + RSI ({sig_type.upper()})")
    print("=" * 75)
    print(f" Symbol:        {ACTIVE_SYMBOL} | Timeframe: H1")
    print(f" Bar Time:      {candle_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Signal Type:   {sig_type.upper()} (Trend Pullback)")
    print(f" 10-RSI Value:  {rsi_val:.2f}")
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
    print(f"   RAYNER TEO 200 SMA + 10 RSI SIGNAL SCANNER ({ACTIVE_SYMBOL} H1)   ")
    print("=" * 75)
    print(" Strategy: 200 SMA Trend + RSI(10) Extreme Pullback Reversal")
    print(" Status: Scanning live bars... Press Ctrl+C to stop.\n")

    last_processed_candle_time = None

    try:
        while True:
            df = fetch_rates()
            if df is None:
                time.sleep(POLL_INTERVAL)
                continue

            current_candle_time = df['time'].iloc[-1]
            closed_bar = df.iloc[-2]       # Last closed bar
            prev_bar = df.iloc[-3]         # Prior bar for RSI hook detection

            if last_processed_candle_time != current_candle_time:
                last_processed_candle_time = current_candle_time

                # ----------------------------------------------------
                # BUY LOGIC (Uptrend Pullback)
                # ----------------------------------------------------
                # 1. Macro Trend: Price above 200 SMA
                trend_bullish = closed_bar['close'] > closed_bar['sma200']
                # 2. RSI Oversold recovery: Previous bar was <= 30, closed bar hooked above 30
                rsi_oversold_hook = (prev_bar['rsi10'] <= RSI_OVERSOLD) and (closed_bar['rsi10'] > RSI_OVERSOLD)
                # 3. Candle Confirmation: Bullish candle body
                is_bull_candle = closed_bar['close'] > closed_bar['open']

                if trend_bullish and rsi_oversold_hook and is_bull_candle:
                    entry = closed_bar['close']
                    recent_low = df['low'].iloc[-6:-1].min()
                    sl = recent_low - (2.0 * PIP_SIZE)
                    risk = entry - sl
                    if risk > 0:
                        tp = entry + (risk * RISK_REWARD_RATIO)
                        emit_signal("BUY", current_candle_time, entry, sl, tp, closed_bar['rsi10'])

                # ----------------------------------------------------
                # SELL LOGIC (Downtrend Pullback)
                # ----------------------------------------------------
                # 1. Macro Trend: Price below 200 SMA
                trend_bearish = closed_bar['close'] < closed_bar['sma200']
                # 2. RSI Overbought recovery: Previous bar was >= 70, closed bar hooked below 70
                rsi_overbought_hook = (prev_bar['rsi10'] >= RSI_OVERBOUGHT) and (closed_bar['rsi10'] < RSI_OVERBOUGHT)
                # 3. Candle Confirmation: Bearish candle body
                is_bear_candle = closed_bar['close'] < closed_bar['open']

                if trend_bearish and rsi_overbought_hook and is_bear_candle:
                    entry = closed_bar['close']
                    recent_high = df['high'].iloc[-6:-1].max()
                    sl = recent_high + (2.0 * PIP_SIZE)
                    risk = sl - entry
                    if risk > 0:
                        tp = entry - (risk * RISK_REWARD_RATIO)
                        emit_signal("SELL", current_candle_time, entry, sl, tp, closed_bar['rsi10'])

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n[STOP] Scanner stopped by user.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    run_scanner()
