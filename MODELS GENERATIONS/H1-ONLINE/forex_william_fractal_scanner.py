"""
========================================================================================
FOREX MT5 SIGNAL SCANNER: WILLIAM FRACTAL SCALPING (Neeraj Joshi Setup)
========================================================================================
Source: Short Video _0d1qw3p-wQ | 1-Year Backtest: +35.0% Return, 55.0% Win Rate
Market: Forex (EURUSD / Major Pairs)
Timeframe: Multi-Timeframe (H4 Context -> M5 Execution)
Indicators: William Fractal High / Low Levels
Logic:
  1. 4H Chart: Mark the active William Fractal High & Fractal Low key levels
  2. 5M Chart: Wait for market price to sweep beyond the 4H Fractal High/Low
  3. Purge & Rejection: M5 candle pierces the 4H level but closes back INSIDE
  4. Trigger: Confirmed rejection candle closing back below 4H High (Sell) / above 4H Low (Buy)
  5. Risk Management: SL just beyond highest/lowest wick of the breakout; TP at 1:2 RR
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
TIMEFRAME_HTF = mt5.TIMEFRAME_H4  # 4-Hour Macro Fractal Reference
TIMEFRAME_LTF = mt5.TIMEFRAME_M5  # 5-Minute Entry Execution
RISK_REWARD_RATIO = 2.0           # 1:2 Risk-to-Reward
POLL_INTERVAL = 5                 # Seconds between poll checks

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


def fetch_rates(timeframe=None, count=100):
    """Fetches candle rates for a given timeframe."""
    if timeframe is None:
        timeframe = TIMEFRAME_LTF
    rates = mt5.copy_rates_from_pos(ACTIVE_SYMBOL, timeframe, 0, count)
    if rates is None or len(rates) < 30:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


def get_latest_h4_fractals(df_h4: pd.DataFrame):
    """Calculates William 5-bar Fractals on 4H chart."""
    fractal_highs = []
    fractal_lows = []
    n = len(df_h4)
    
    # 5-bar William Fractal
    for i in range(2, n - 2):
        c_high = df_h4['high'].iloc[i]
        c_low = df_h4['low'].iloc[i]
        
        # High fractal: higher than 2 bars before and 2 bars after
        if (c_high > df_h4['high'].iloc[i-1] and c_high > df_h4['high'].iloc[i-2] and
            c_high > df_h4['high'].iloc[i+1] and c_high > df_h4['high'].iloc[i+2]):
            fractal_highs.append(c_high)
            
        # Low fractal: lower than 2 bars before and 2 bars after
        if (c_low < df_h4['low'].iloc[i-1] and c_low < df_h4['low'].iloc[i-2] and
            c_low < df_h4['low'].iloc[i+1] and c_low < df_h4['low'].iloc[i+2]):
            fractal_lows.append(c_low)
            
    latest_fh = fractal_highs[-1] if fractal_highs else None
    latest_fl = fractal_lows[-1] if fractal_lows else None
    return latest_fh, latest_fl


def emit_signal(sig_type: str, candle_time: datetime.datetime, entry_price: float, sl_price: float, tp_price: float, fractal_level: float):
    """Prints a structured ASCII signal card to console."""
    risk_pips = abs(entry_price - sl_price) / PIP_SIZE
    reward_pips = abs(tp_price - entry_price) / PIP_SIZE
    
    print("\n" + "=" * 70)
    print(f" [ALERT] FOREX SIGNAL ALERT: WILLIAM FRACTAL SWEEP ({sig_type.upper()})")
    print("=" * 70)
    print(f" Symbol:          {ACTIVE_SYMBOL} | Macro TF: H4 -> Entry TF: M5")
    print(f" Bar Time:        {candle_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Signal Type:     {sig_type.upper()}")
    print(f" 4H Fractal Level:{fractal_level:.5f}")
    print(f" Entry Price:     {entry_price:.5f}")
    print(f" Stop Loss:       {sl_price:.5f}  ({risk_pips:.1f} pips)")
    print(f" Take Profit:     {tp_price:.5f}  ({reward_pips:.1f} pips)")
    print(f" Risk/Reward:     1:{RISK_REWARD_RATIO:.1f}")
    print(f" Status:          SIGNAL ONLY (Ready for manual or external execution)")
    print("=" * 70 + "\n")


def run_scanner():
    """Main live market scanner loop."""
    if not initialize_mt5():
        return

    print("\n" + "=" * 70)
    print(f"   WILLIAM FRACTAL FOREX SIGNAL SCANNER ({ACTIVE_SYMBOL})   ")
    print("=" * 70)
    print(" Strategy Logic: 4H Fractal High/Low -> 5M Sweep Rejection Close Back Inside")
    print(" Status: Scanning live bars... Press Ctrl+C to stop.\n")

    last_processed_candle_time = None

    try:
        while True:
            df_h4 = fetch_rates(TIMEFRAME_HTF, count=100)
            df_m5 = fetch_rates(TIMEFRAME_LTF, count=60)
            
            if df_h4 is None or df_m5 is None:
                time.sleep(POLL_INTERVAL)
                continue

            fh_level, fl_level = get_latest_h4_fractals(df_h4)
            current_candle_time = df_m5['time'].iloc[-1]
            closed_bar = df_m5.iloc[-2]       # Last completed M5 bar
            
            if last_processed_candle_time != current_candle_time and fh_level and fl_level:
                last_processed_candle_time = current_candle_time

                # ----------------------------------------------------
                # BEARISH FRACTAL SWEEP (Sell Signal)
                # ----------------------------------------------------
                # Bar pierced above 4H Fractal High, but closed back below it
                swept_high = closed_bar['high'] > fh_level
                closed_below = closed_bar['close'] < fh_level
                is_bearish = closed_bar['close'] < closed_bar['open']
                
                if swept_high and closed_below and is_bearish:
                    entry = closed_bar['close']
                    highest_wick = df_m5['high'].iloc[-4:-1].max()
                    sl = highest_wick + (2.0 * PIP_SIZE)
                    risk = sl - entry
                    if risk > 0:
                        tp = entry - (risk * RISK_REWARD_RATIO)
                        emit_signal("SELL", current_candle_time, entry, sl, tp, fh_level)

                # ----------------------------------------------------
                # BULLISH FRACTAL SWEEP (Buy Signal - Inverted)
                # ----------------------------------------------------
                # Bar pierced below 4H Fractal Low, but closed back above it
                swept_low = closed_bar['low'] < fl_level
                closed_above = closed_bar['close'] > fl_level
                is_bullish = closed_bar['close'] > closed_bar['open']
                
                if swept_low and closed_above and is_bullish:
                    entry = closed_bar['close']
                    lowest_wick = df_m5['low'].iloc[-4:-1].min()
                    sl = lowest_wick - (2.0 * PIP_SIZE)
                    risk = entry - sl
                    if risk > 0:
                        tp = entry + (risk * RISK_REWARD_RATIO)
                        emit_signal("BUY", current_candle_time, entry, sl, tp, fl_level)

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n[STOP] Scanner stopped by user.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    run_scanner()
