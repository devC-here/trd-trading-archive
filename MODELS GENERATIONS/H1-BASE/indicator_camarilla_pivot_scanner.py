"""
========================================================================================
MT5 SIGNAL SCANNER: CAMARILLA PIVOT REVERSAL & BREAKOUT SYSTEM
========================================================================================
Source: Short Video GPSvVOZv-No | King Research Academy Camarilla Backtest
Market: Universal (MT5 Forex EURUSD, Gold XAUUSD, Crypto, Indices)
Timeframe: M5 / M15 Execution with Daily Pivot Derivation
Indicators: Camarilla Pivot Equation (H4, H3, L3, L4 derived from Previous Day H/L/C)
Logic:
  - Camarilla Formula:
      Range = High_prev - Low_prev
      H4 = Close_prev + (Range * 1.1 / 2)
      H3 = Close_prev + (Range * 1.1 / 4)
      L3 = Close_prev - (Range * 1.1 / 4)
      L4 = Close_prev - (Range * 1.1 / 2)
  - Setups:
      1. L3 Bullish Reversal: Price tests L3 support, rejects, and closes back above L3.
         -> Buy Signal | SL: Below L4 | TP: Target H3 or 1:2 RR.
      2. H3 Bearish Reversal: Price tests H3 resistance, rejects, and closes back below H3.
         -> Sell Signal | SL: Above H4 | TP: Target L3 or 1:2 RR.
      3. H4 Bullish Breakout: Candle closes decisively above H4.
         -> Buy Signal | SL: Below H3 | TP: 1:2 RR.
      4. L4 Bearish Breakdown: Candle closes decisively below L4.
         -> Sell Signal | SL: Above H3 | TP: 1:2 RR.
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
TIMEFRAME_EXEC = mt5.TIMEFRAME_M5    # 5-Minute Execution
TIMEFRAME_DAILY = mt5.TIMEFRAME_D1   # Daily for Camarilla calculations
RISK_REWARD_RATIO = 2.0
POLL_INTERVAL = 5                    # Seconds between poll checks
LOOKBACK_BARS = 120

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


def fetch_camarilla_levels():
    """Calculates Camarilla H4, H3, L3, L4 from completed previous day bar."""
    daily_rates = mt5.copy_rates_from_pos(ACTIVE_SYMBOL, TIMEFRAME_DAILY, 1, 1)
    if daily_rates is None or len(daily_rates) == 0:
        return None

    prev_day = daily_rates[0]
    high = prev_day['high']
    low = prev_day['low']
    close = prev_day['close']
    rng = high - low

    h4 = close + (rng * 1.1 / 2.0)
    h3 = close + (rng * 1.1 / 4.0)
    l3 = close - (rng * 1.1 / 4.0)
    l4 = close - (rng * 1.1 / 2.0)

    return {
        "H4": h4,
        "H3": h3,
        "L3": l3,
        "L4": l4,
        "date": datetime.datetime.fromtimestamp(prev_day['time']).strftime('%Y-%m-%d')
    }


def fetch_rates(count=LOOKBACK_BARS):
    """Fetches execution candle rates."""
    rates = mt5.copy_rates_from_pos(ACTIVE_SYMBOL, TIMEFRAME_EXEC, 0, count)
    if rates is None or len(rates) < 30:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


def emit_signal(sig_type: str, setup_name: str, candle_time: datetime.datetime, entry_price: float, sl_price: float, tp_price: float, levels: dict):
    """Prints a structured ASCII signal card to console."""
    risk_pips = abs(entry_price - sl_price) / PIP_SIZE
    reward_pips = abs(tp_price - entry_price) / PIP_SIZE

    print("\n" + "=" * 75)
    print(f" [ALERT] MT5 SIGNAL ALERT: CAMARILLA PIVOT ({sig_type.upper()})")
    print("=" * 75)
    print(f" Symbol:        {ACTIVE_SYMBOL} | Timeframe: M5")
    print(f" Bar Time:      {candle_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Setup:         {setup_name}")
    print(f" Daily Pivots:  H4={levels['H4']:.5f} | H3={levels['H3']:.5f} | L3={levels['L3']:.5f} | L4={levels['L4']:.5f}")
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
    print(f"   CAMARILLA PIVOT REVERSAL & BREAKOUT SCANNER ({ACTIVE_SYMBOL} M5)   ")
    print("=" * 75)
    print(" Strategy: H3/L3 Mean Reversion & H4/L4 Structural Expansion")
    print(" Status: Scanning live bars... Press Ctrl+C to stop.\n")

    last_processed_candle_time = None

    try:
        while True:
            levels = fetch_camarilla_levels()
            df = fetch_rates()
            if df is None or levels is None:
                time.sleep(POLL_INTERVAL)
                continue

            current_candle_time = df['time'].iloc[-1]
            closed_bar = df.iloc[-2]       # Last completed bar
            prev_bar = df.iloc[-3]

            if last_processed_candle_time != current_candle_time:
                last_processed_candle_time = current_candle_time

                # ----------------------------------------------------
                # SETUP 1: L3 BULLISH REVERSAL
                # ----------------------------------------------------
                # Price touched below L3 and closed back above L3
                tested_l3 = (prev_bar['low'] <= levels['L3']) or (closed_bar['low'] <= levels['L3'])
                reclaimed_l3 = (closed_bar['close'] > levels['L3']) and (closed_bar['close'] > closed_bar['open'])

                if tested_l3 and reclaimed_l3:
                    entry = closed_bar['close']
                    sl = levels['L4'] - (2.0 * PIP_SIZE)
                    risk = entry - sl
                    if risk > 0:
                        tp = levels['H3'] if levels['H3'] > entry else entry + (risk * RISK_REWARD_RATIO)
                        emit_signal("BUY", "L3 Bullish Support Reversal", current_candle_time, entry, sl, tp, levels)

                # ----------------------------------------------------
                # SETUP 2: H3 BEARISH REVERSAL
                # ----------------------------------------------------
                # Price touched above H3 and closed back below H3
                tested_h3 = (prev_bar['high'] >= levels['H3']) or (closed_bar['high'] >= levels['H3'])
                rejected_h3 = (closed_bar['close'] < levels['H3']) and (closed_bar['close'] < closed_bar['open'])

                if tested_h3 and rejected_h3:
                    entry = closed_bar['close']
                    sl = levels['H4'] + (2.0 * PIP_SIZE)
                    risk = sl - entry
                    if risk > 0:
                        tp = levels['L3'] if levels['L3'] < entry else entry - (risk * RISK_REWARD_RATIO)
                        emit_signal("SELL", "H3 Bearish Resistance Reversal", current_candle_time, entry, sl, tp, levels)

                # ----------------------------------------------------
                # SETUP 3: H4 BREAKOUT (Bullish Expansion)
                # ----------------------------------------------------
                if prev_bar['close'] <= levels['H4'] and closed_bar['close'] > levels['H4']:
                    entry = closed_bar['close']
                    sl = levels['H3']
                    risk = entry - sl
                    if risk > 0:
                        tp = entry + (risk * RISK_REWARD_RATIO)
                        emit_signal("BUY", "H4 Bullish Range Breakout", current_candle_time, entry, sl, tp, levels)

                # ----------------------------------------------------
                # SETUP 4: L4 BREAKDOWN (Bearish Expansion)
                # ----------------------------------------------------
                if prev_bar['close'] >= levels['L4'] and closed_bar['close'] < levels['L4']:
                    entry = closed_bar['close']
                    sl = levels['L3']
                    risk = sl - entry
                    if risk > 0:
                        tp = entry - (risk * RISK_REWARD_RATIO)
                        emit_signal("SELL", "L4 Bearish Range Breakdown", current_candle_time, entry, sl, tp, levels)

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n[STOP] Scanner stopped by user.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    run_scanner()
