"""
========================================================================================
MT5 SIGNAL SCANNER: SUBASISH PANI (POWER OF STOCKS) BOLLINGER BAND STRATEGY
========================================================================================
Source: Short Video QZMvJrxS_HQ | Tested by Sahil
Creator Attribution: Subasish Pani (Power of Stocks)
Market: Universal (MT5 Forex, Gold, Crypto, Indices)
Timeframe: M15 (also applicable to M5)
Indicators: Bollinger Bands (Period: 20, Standard Deviation: 2.0)
Logic:
  1. Mean Reversion Alert Candle:
     - Bearish Alert: A candle forms whose LOW is strictly GREATER than the Upper Bollinger Band
       (the entire candle is completely detached outside the upper band).
     - Bullish Alert: A candle forms whose HIGH is strictly LESS than the Lower Bollinger Band
       (the entire candle is completely detached outside the lower band).
  2. Trigger Confirmation:
     - Sell Signal: The next candle breaks and closes below the Alert Candle's Low.
     - Buy Signal: The next candle breaks and closes above the Alert Candle's High.
  3. Risk Management:
     - Stop Loss: Placed directly at the Alert Candle's High (Sell) / Low (Buy).
     - Take Profit: 1:3 Risk-to-Reward Ratio (High RR Mean Reversion).
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
BB_PERIOD = 20
BB_STD = 2.0
RISK_REWARD_RATIO = 3.0          # Subasish Pani uses 1:3 to 1:4 RR
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
    """Fetches M15 candle rates and calculates Bollinger Bands."""
    rates = mt5.copy_rates_from_pos(ACTIVE_SYMBOL, TIMEFRAME, 0, count)
    if rates is None or len(rates) < BB_PERIOD + 5:
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # Calculate Bollinger Bands
    df['bb_mid'] = df['close'].rolling(window=BB_PERIOD).mean()
    df['bb_std'] = df['close'].rolling(window=BB_PERIOD).std()
    df['bb_upper'] = df['bb_mid'] + (BB_STD * df['bb_std'])
    df['bb_lower'] = df['bb_mid'] - (BB_STD * df['bb_std'])

    return df


def emit_signal(sig_type: str, candle_time: datetime.datetime, entry_price: float, sl_price: float, tp_price: float, alert_candle_time):
    """Prints a structured ASCII signal card to console."""
    risk_pips = abs(entry_price - sl_price) / PIP_SIZE
    reward_pips = abs(tp_price - entry_price) / PIP_SIZE

    print("\n" + "=" * 75)
    print(f" [ALERT] MT5 SIGNAL ALERT: SUBASISH PANI BOLLINGER BAND ({sig_type.upper()})")
    print("=" * 75)
    print(f" Symbol:        {ACTIVE_SYMBOL} | Timeframe: M15")
    print(f" Bar Time:      {candle_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Signal Type:   {sig_type.upper()} (Mean Reversion)")
    print(f" Alert Candle:  {alert_candle_time.strftime('%Y-%m-%d %H:%M:%S')} (Fully Outside Band)")
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
    print(f"   SUBASISH PANI BOLLINGER BAND SIGNAL SCANNER ({ACTIVE_SYMBOL} M15)   ")
    print("=" * 75)
    print(" Strategy: Detached Candle Outside Upper/Lower Band -> Next Bar Breakout")
    print(" Status: Scanning live bars... Press Ctrl+C to stop.\n")

    last_processed_candle_time = None

    try:
        while True:
            df = fetch_rates()
            if df is None:
                time.sleep(POLL_INTERVAL)
                continue

            current_candle_time = df['time'].iloc[-1]
            closed_bar = df.iloc[-2]       # Most recently closed bar
            alert_bar = df.iloc[-3]        # Previous bar to test for detachment

            # Evaluate only once per newly closed candle
            if last_processed_candle_time != current_candle_time:
                last_processed_candle_time = current_candle_time

                # ----------------------------------------------------
                # SHORT SETUP (Overbought Mean Reversion)
                # ----------------------------------------------------
                # 1. Alert candle is completely detached above Upper Band
                is_bear_alert = alert_bar['low'] > alert_bar['bb_upper']

                # 2. Closed bar breaks below Alert candle's low
                bear_breakout = (closed_bar['low'] < alert_bar['low']) and (closed_bar['close'] < alert_bar['low'])

                if is_bear_alert and bear_breakout:
                    entry = closed_bar['close']
                    sl = alert_bar['high'] + (2.0 * PIP_SIZE)
                    risk = sl - entry
                    if risk > 0:
                        tp = entry - (risk * RISK_REWARD_RATIO)
                        emit_signal("SELL", current_candle_time, entry, sl, tp, alert_bar['time'])

                # ----------------------------------------------------
                # LONG SETUP (Oversold Mean Reversion)
                # ----------------------------------------------------
                # 1. Alert candle is completely detached below Lower Band
                is_bull_alert = alert_bar['high'] < alert_bar['bb_lower']

                # 2. Closed bar breaks above Alert candle's high
                bull_breakout = (closed_bar['high'] > alert_bar['high']) and (closed_bar['close'] > alert_bar['high'])

                if is_bull_alert and bull_breakout:
                    entry = closed_bar['close']
                    sl = alert_bar['low'] - (2.0 * PIP_SIZE)
                    risk = entry - sl
                    if risk > 0:
                        tp = entry + (risk * RISK_REWARD_RATIO)
                        emit_signal("BUY", current_candle_time, entry, sl, tp, alert_bar['time'])

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n[STOP] Scanner stopped by user.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    run_scanner()
