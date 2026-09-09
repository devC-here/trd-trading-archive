"""
========================================================================================
FOREX MT5 SIGNAL SCANNER: SUPERTREND SWING STRATEGY
========================================================================================
Source: Short Video YwDleTVqM5c | Pure Trend-Following Swing Strategy
Market: Forex (EURUSD / Major Pairs)
Timeframe: M15 Execution (or H1)
Indicators: Supertrend (ATR Period: 7, Multiplier: 2.5), 200 EMA Baseline
Logic:
  1. Macro Regime: Price positioned relative to 200 EMA (Price > 200 EMA for Long)
  2. Polarity Flip: Supertrend flips from Red (Sell) to Green (Buy)
  3. Confirmation: Candle closes firmly above the green Supertrend line
  4. Risk Management: SL at Supertrend level; TP at 1:2 Risk-to-Reward
========================================================================================
NOTE: SIGNAL ONLY - No trade execution or live orders are placed.
========================================================================================
"""

import time
import datetime
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ==========================================
# 1. CONFIGURATION & STRATEGY SETTINGS
# ==========================================
DEFAULT_SYMBOLS = ["EURUSD.0", "EURUSD", "EURUSDm", "EURUSD.pro"]
TIMEFRAME = mt5.TIMEFRAME_M15     # 15-Minute Chart
ATR_PERIOD = 7
SUPERTREND_MULT = 2.5
EMA_MACRO = 200
RISK_REWARD_RATIO = 2.0           # 1:2 Risk-to-Reward
POLL_INTERVAL = 5                 # Seconds between poll checks
LOOKBACK_BARS = 250

ACTIVE_SYMBOL = None
PIP_SIZE = 0.0001


def resolve_forex_symbol():
    """Auto-detects broker-specific EURUSD symbol format."""
    global ACTIVE_SYMBOL, PIP_SIZE
    for s in DEFAULT_SYMBOLS:
        info = mt5.symbol_info(s)
        if info is not None:
            if not info.visible:
                mt5.symbol_select(s, True)
            ACTIVE_SYMBOL = s
            PIP_SIZE = 0.0001 if info.digits in (4, 5) else 0.01
            return ACTIVE_SYMBOL
    ACTIVE_SYMBOL = "EURUSD"
    return ACTIVE_SYMBOL


def initialize_mt5():
    """Connects to MetaTrader 5 terminal."""
    if not mt5.initialize():
        print(f"[-] MT5 Connection failed: {mt5.last_error()}")
        return False
    
    symbol = resolve_forex_symbol()
    print(f"[+] Connected to MT5. Monitoring symbol: {symbol}")
    return True


def calculate_supertrend(df: pd.DataFrame, period=7, multiplier=2.5):
    """Calculates true mathematical Supertrend indicator."""
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Average True Range (ATR)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    
    hl2 = (high + low) / 2
    upper_basic = hl2 + (multiplier * atr)
    lower_basic = hl2 - (multiplier * atr)
    
    upper_band = upper_basic.copy()
    lower_band = lower_basic.copy()
    trend = [1] * len(df)
    st = [0.0] * len(df)
    
    for i in range(1, len(df)):
        # Calculate trailing bands
        if close.iloc[i-1] > lower_band.iloc[i-1]:
            lower_band.iloc[i] = max(lower_basic.iloc[i], lower_band.iloc[i-1])
        else:
            lower_band.iloc[i] = lower_basic.iloc[i]
            
        if close.iloc[i-1] < upper_band.iloc[i-1]:
            upper_band.iloc[i] = min(upper_basic.iloc[i], upper_band.iloc[i-1])
        else:
            upper_band.iloc[i] = upper_basic.iloc[i]
            
        # Determine trend direction
        if close.iloc[i] > upper_band.iloc[i-1]:
            trend[i] = 1
        elif close.iloc[i] < lower_band.iloc[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
            
        st[i] = lower_band.iloc[i] if trend[i] == 1 else upper_band.iloc[i]
        
    df['st'] = st
    df['st_trend'] = trend
    df['atr'] = atr
    return df


def fetch_rates():
    """Fetches rates and computes Supertrend & 200 EMA."""
    rates = mt5.copy_rates_from_pos(ACTIVE_SYMBOL, TIMEFRAME, 0, LOOKBACK_BARS)
    if rates is None or len(rates) < 100:
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['ema200'] = df['close'].ewm(span=EMA_MACRO, adjust=False).mean()
    df = calculate_supertrend(df, period=ATR_PERIOD, multiplier=SUPERTREND_MULT)
    return df


def emit_signal(sig_type: str, candle_time: datetime.datetime, entry_price: float, sl_price: float, tp_price: float, st_level: float):
    """Prints a structured ASCII signal card to console."""
    risk_pips = abs(entry_price - sl_price) / PIP_SIZE
    reward_pips = abs(tp_price - entry_price) / PIP_SIZE
    
    print("\n" + "=" * 70)
    print(f" [ALERT] FOREX SIGNAL ALERT: SUPERTREND SWING ({sig_type.upper()})")
    print("=" * 70)
    print(f" Symbol:           {ACTIVE_SYMBOL} | Timeframe: M15")
    print(f" Bar Time:         {candle_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Signal Type:      {sig_type.upper()}")
    print(f" Supertrend Level: {st_level:.5f}")
    print(f" Entry Price:      {entry_price:.5f}")
    print(f" Stop Loss:        {sl_price:.5f}  ({risk_pips:.1f} pips)")
    print(f" Take Profit:      {tp_price:.5f}  ({reward_pips:.1f} pips)")
    print(f" Risk/Reward:      1:{RISK_REWARD_RATIO:.1f}")
    print(f" Status:           SIGNAL ONLY (Ready for manual or external execution)")
    print("=" * 70 + "\n")


def run_scanner():
    """Main live market scanner loop."""
    if not initialize_mt5():
        return

    print("\n" + "=" * 70)
    print(f"   SUPERTREND SWING FOREX SIGNAL SCANNER ({ACTIVE_SYMBOL} M15)   ")
    print("=" * 70)
    print(" Strategy Logic: 200 EMA Baseline -> Supertrend (7, 2.5) Polarity Flip -> Close Confirmation")
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
            prev_bar = df.iloc[-3]         # Bar before
            
            if last_processed_candle_time != current_candle_time:
                last_processed_candle_time = current_candle_time

                # ----------------------------------------------------
                # BUY LOGIC (Supertrend Red -> Green Flip)
                # ----------------------------------------------------
                macro_bull = closed_bar['close'] > closed_bar['ema200']
                flip_to_green = (prev_bar['st_trend'] == -1) and (closed_bar['st_trend'] == 1)
                closed_above_st = closed_bar['close'] > closed_bar['st']

                if macro_bull and flip_to_green and closed_above_st:
                    entry = closed_bar['close']
                    sl = closed_bar['st'] - (2.0 * PIP_SIZE)
                    risk = entry - sl
                    if risk > 0:
                        tp = entry + (risk * RISK_REWARD_RATIO)
                        emit_signal("BUY", current_candle_time, entry, sl, tp, closed_bar['st'])

                # ----------------------------------------------------
                # SELL LOGIC (Supertrend Green -> Red Flip - Inverted)
                # ----------------------------------------------------
                macro_bear = closed_bar['close'] < closed_bar['ema200']
                flip_to_red = (prev_bar['st_trend'] == 1) and (closed_bar['st_trend'] == -1)
                closed_below_st = closed_bar['close'] < closed_bar['st']

                if macro_bear and flip_to_red and closed_below_st:
                    entry = closed_bar['close']
                    sl = closed_bar['st'] + (2.0 * PIP_SIZE)
                    risk = sl - entry
                    if risk > 0:
                        tp = entry - (risk * RISK_REWARD_RATIO)
                        emit_signal("SELL", current_candle_time, entry, sl, tp, closed_bar['st'])

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n[STOP] Scanner stopped by user.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    run_scanner()
