"""
========================================================================================
MT5 SIGNAL SCANNER: TANISHA GARG CAMARILLA PIVOTS + RSI(21) STRATEGY
========================================================================================
Source: Short Video Vm3nN5biOjI | 1-Year Backtest: +74.5% Return, 57% Win Rate, 9.7% DD
Creator Attribution: Tanisha Garg
Market: Forex (EURUSD, GBPUSD) / Universal MT5
Timeframe: M5 Execution with Daily Camarilla Levels
Indicators:
  1. Camarilla Pivots (S2, S4, R2, R4 derived from Daily H/L/C)
  2. 21-Period RSI (Relative Strength Index)
Logic:
  - Buy Setup:
      1. S/R Location: Price tests near Camarilla Support S2 or S4.
      2. RSI Momentum Recovery: RSI(21) dipped below 40 and crosses back ABOVE 40.
      3. Trigger: A bullish reversal candle forms and closes green at the support zone.
      4. Stop Loss: Placed below S4 (or recent swing low).
      5. Take Profit: Target R2 level or 1:2 Risk-to-Reward Ratio.
  - Sell Setup (Inverted):
      1. S/R Location: Price tests near Camarilla Resistance R2 or R4.
      2. RSI Momentum Recovery: RSI(21) climbed above 60 and crosses back BELOW 60.
      3. Trigger: A bearish reversal candle forms and closes red at the resistance zone.
      4. Stop Loss: Placed above R4 (or recent swing high).
      5. Take Profit: Target S2 level or 1:2 Risk-to-Reward Ratio.
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
RSI_PERIOD = 21
RISK_REWARD_RATIO = 2.0
POLL_INTERVAL = 5                    # Seconds between poll checks
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


def fetch_camarilla_levels():
    """Calculates Camarilla R2, R4, S2, S4 from previous day."""
    rates = mt5.copy_rates_from_pos(ACTIVE_SYMBOL, TIMEFRAME_DAILY, 1, 1)
    if rates is None or len(rates) == 0:
        return None

    prev = rates[0]
    high = prev['high']
    low = prev['low']
    close = prev['close']
    rng = high - low

    r4 = close + (rng * 1.1 / 2.0)
    r2 = close + (rng * 1.1 / 6.0)
    s2 = close - (rng * 1.1 / 6.0)
    s4 = close - (rng * 1.1 / 2.0)

    return {"R4": r4, "R2": r2, "S2": s2, "S4": s4}


def calculate_rsi(series: pd.Series, period: int = 21) -> pd.Series:
    """Calculates Wilder's RSI."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def fetch_rates(count=LOOKBACK_BARS):
    """Fetches M5 candle rates and calculates 21 RSI."""
    rates = mt5.copy_rates_from_pos(ACTIVE_SYMBOL, TIMEFRAME_EXEC, 0, count)
    if rates is None or len(rates) < RSI_PERIOD + 10:
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['rsi21'] = calculate_rsi(df['close'], period=RSI_PERIOD)
    return df


def emit_signal(sig_type: str, candle_time: datetime.datetime, entry_price: float, sl_price: float, tp_price: float, pivot_level: str, rsi_val: float):
    """Prints a structured ASCII signal card to console."""
    risk_pips = abs(entry_price - sl_price) / PIP_SIZE
    reward_pips = abs(tp_price - entry_price) / PIP_SIZE

    print("\n" + "=" * 75)
    print(f" [ALERT] MT5 SIGNAL ALERT: TANISHA GARG CAMARILLA + RSI ({sig_type.upper()})")
    print("=" * 75)
    print(f" Symbol:        {ACTIVE_SYMBOL} | Timeframe: M5")
    print(f" Bar Time:      {candle_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Signal Type:   {sig_type.upper()} (Zone: {pivot_level})")
    print(f" 21-RSI Value:  {rsi_val:.2f}")
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
    print(f"   TANISHA GARG CAMARILLA + RSI(21) SCANNER ({ACTIVE_SYMBOL} M5)   ")
    print("=" * 75)
    print(" Strategy: S2/S4 Support + RSI <40 Recovery Hook -> Target R2")
    print(" Status: Scanning live bars... Press Ctrl+C to stop.\n")

    last_processed_candle_time = None

    try:
        while True:
            cam = fetch_camarilla_levels()
            df = fetch_rates()
            if df is None or cam is None:
                time.sleep(POLL_INTERVAL)
                continue

            current_candle_time = df['time'].iloc[-1]
            closed_bar = df.iloc[-2]
            prev_bar = df.iloc[-3]

            if last_processed_candle_time != current_candle_time:
                last_processed_candle_time = current_candle_time

                # ----------------------------------------------------
                # BUY LOGIC: Test S2 or S4 + RSI crosses above 40
                # ----------------------------------------------------
                tested_s2_or_s4 = (
                    abs(closed_bar['low'] - cam['S2']) <= (10 * PIP_SIZE) or 
                    abs(closed_bar['low'] - cam['S4']) <= (10 * PIP_SIZE) or
                    closed_bar['low'] <= cam['S2']
                )
                rsi_buy_hook = (prev_bar['rsi21'] <= 40) and (closed_bar['rsi21'] > 40)
                bull_candle = closed_bar['close'] > closed_bar['open']

                if tested_s2_or_s4 and rsi_buy_hook and bull_candle:
                    entry = closed_bar['close']
                    sl = cam['S4'] - (3.0 * PIP_SIZE)
                    risk = entry - sl
                    if risk > 0:
                        tp = cam['R2'] if cam['R2'] > entry else entry + (risk * RISK_REWARD_RATIO)
                        emit_signal("BUY", current_candle_time, entry, sl, tp, "S2/S4 Support Zone", closed_bar['rsi21'])

                # ----------------------------------------------------
                # SELL LOGIC: Test R2 or R4 + RSI crosses below 60
                # ----------------------------------------------------
                tested_r2_or_r4 = (
                    abs(closed_bar['high'] - cam['R2']) <= (10 * PIP_SIZE) or 
                    abs(closed_bar['high'] - cam['R4']) <= (10 * PIP_SIZE) or
                    closed_bar['high'] >= cam['R2']
                )
                rsi_sell_hook = (prev_bar['rsi21'] >= 60) and (closed_bar['rsi21'] < 60)
                bear_candle = closed_bar['close'] < closed_bar['open']

                if tested_r2_or_r4 and rsi_sell_hook and bear_candle:
                    entry = closed_bar['close']
                    sl = cam['R4'] + (3.0 * PIP_SIZE)
                    risk = sl - entry
                    if risk > 0:
                        tp = cam['S2'] if cam['S2'] < entry else entry - (risk * RISK_REWARD_RATIO)
                        emit_signal("SELL", current_candle_time, entry, sl, tp, "R2/R4 Resistance Zone", closed_bar['rsi21'])

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n[STOP] Scanner stopped by user.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    run_scanner()
