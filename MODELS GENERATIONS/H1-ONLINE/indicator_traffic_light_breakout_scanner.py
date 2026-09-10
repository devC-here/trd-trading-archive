"""
========================================================================================
MT5 SIGNAL SCANNER: SUBASISH PANI 'TRAFFIC LIGHT' BREAKOUT STRATEGY
========================================================================================
Source: Short Video _e9_VfItAFE | 1-Year Backtest: +267.95% Return, 51.2% WR, 10.88% DD
Creator Attribution: Subasish Pani (Power of Stocks - 2M+ Subscribers)
Market: Forex (EURUSD, GBPUSD) / Universal MT5
Timeframe: M5 Execution
Indicators: Pure Price Action Opposite Color 2-Candle Traffic Light Compression
Logic:
  - Traffic Light Zone Formation:
      Two consecutive candles of opposite color form:
      (Green Candle followed by Red Candle, or Red Candle followed by Green Candle).
      Traffic Light High = max(Candle 1 High, Candle 2 High)
      Traffic Light Low  = min(Candle 1 Low, Candle 2 Low)
  - Buy Trigger:
      A subsequent candle breaks and closes decisively ABOVE the Traffic Light High.
      Stop Loss: Placed below the Traffic Light Low.
      Take Profit: 1:2 Risk-to-Reward Ratio.
  - Sell Trigger:
      A subsequent candle breaks and closes decisively BELOW the Traffic Light Low.
      Stop Loss: Placed above the Traffic Light High.
      Take Profit: 1:2 Risk-to-Reward Ratio.
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
RISK_REWARD_RATIO = 2.0
POLL_INTERVAL = 5                # Seconds between poll checks
LOOKBACK_BARS = 60

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
    """Fetches M5 candle rates."""
    rates = mt5.copy_rates_from_pos(ACTIVE_SYMBOL, TIMEFRAME, 0, count)
    if rates is None or len(rates) < 20:
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


def emit_signal(sig_type: str, candle_time: datetime.datetime, entry_price: float, sl_price: float, tp_price: float, tl_high: float, tl_low: float):
    """Prints a structured ASCII signal card to console."""
    risk_pips = abs(entry_price - sl_price) / PIP_SIZE
    reward_pips = abs(tp_price - entry_price) / PIP_SIZE

    print("\n" + "=" * 75)
    print(f" [ALERT] MT5 SIGNAL ALERT: SUBASISH PANI TRAFFIC LIGHT ({sig_type.upper()})")
    print("=" * 75)
    print(f" Symbol:        {ACTIVE_SYMBOL} | Timeframe: M5")
    print(f" Bar Time:      {candle_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Signal Type:   {sig_type.upper()} (Opposite Color Compression Breakout)")
    print(f" Traffic Light: High = {tl_high:.5f} | Low = {tl_low:.5f}")
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
    print(f"   SUBASISH PANI 'TRAFFIC LIGHT' 267% SCANNER ({ACTIVE_SYMBOL} M5)   ")
    print("=" * 75)
    print(" Strategy: Red-Green 2-Bar Range Compression Breakout")
    print(" Status: Scanning live bars... Press Ctrl+C to stop.\n")

    last_processed_candle_time = None

    try:
        while True:
            df = fetch_rates()
            if df is None:
                time.sleep(POLL_INTERVAL)
                continue

            current_candle_time = df['time'].iloc[-1]
            closed_bar = df.iloc[-2]       # Breakout bar
            c2 = df.iloc[-3]               # Traffic light bar 2
            c1 = df.iloc[-4]               # Traffic light bar 1

            if last_processed_candle_time != current_candle_time:
                last_processed_candle_time = current_candle_time

                # Check if c1 and c2 are opposite colors (Red-Green or Green-Red)
                c1_green = c1['close'] > c1['open']
                c2_green = c2['close'] > c2['open']

                if c1_green != c2_green:
                    tl_high = max(c1['high'], c2['high'])
                    tl_low = min(c1['low'], c2['low'])

                    # ----------------------------------------------------
                    # BUY BREAKOUT: closed_bar breaks above tl_high
                    # ----------------------------------------------------
                    if closed_bar['high'] > tl_high and closed_bar['close'] > tl_high:
                        entry = closed_bar['close']
                        sl = tl_low - (2.0 * PIP_SIZE)
                        risk = entry - sl
                        if risk > 0:
                            tp = entry + (risk * RISK_REWARD_RATIO)
                            emit_signal("BUY", current_candle_time, entry, sl, tp, tl_high, tl_low)

                    # ----------------------------------------------------
                    # SELL BREAKOUT: closed_bar breaks below tl_low
                    # ----------------------------------------------------
                    elif closed_bar['low'] < tl_low and closed_bar['close'] < tl_low:
                        entry = closed_bar['close']
                        sl = tl_high + (2.0 * PIP_SIZE)
                        risk = sl - entry
                        if risk > 0:
                            tp = entry - (risk * RISK_REWARD_RATIO)
                            emit_signal("SELL", current_candle_time, entry, sl, tp, tl_high, tl_low)

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n[STOP] Scanner stopped by user.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    run_scanner()
