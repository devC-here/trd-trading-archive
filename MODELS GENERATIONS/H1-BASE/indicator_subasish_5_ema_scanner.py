"""
========================================================================================
MT5 SIGNAL SCANNER: SUBASISH PANI 5 EMA SCALPING STRATEGY
========================================================================================
Source: Short Video ttSbBzKmXyc | 1-Year Backtest: +64% Return, 1:3 RR
Creator Attribution: Subasish Pani (Power of Stocks - 2M+ Subscribers)
Market: Forex (EURUSD, GBPUSD) / Universal MT5
Timeframe: M5 (Selling) / M15 (Buying)
Indicators: 5 EMA (Exponential Moving Average)
Logic:
  - Sell Setup (M5 Chart):
      1. An Alert Candle forms whose LOW does NOT touch the 5 EMA
         (Alert Candle Low > 5 EMA, candle is detached above 5 EMA).
      2. Trigger: The subsequent candle breaks and closes below the Alert Candle's Low.
      3. Stop Loss: Placed at the Alert Candle's High.
      4. Take Profit: 1:3 Risk-to-Reward Ratio.
  - Buy Setup (M15 Chart):
      1. An Alert Candle forms whose HIGH does NOT touch the 5 EMA
         (Alert Candle High < 5 EMA, candle is detached below 5 EMA).
      2. Trigger: The subsequent candle breaks and closes above the Alert Candle's High.
      3. Stop Loss: Placed at the Alert Candle's Low.
      4. Take Profit: 1:3 Risk-to-Reward Ratio.
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
TIMEFRAME_SELL = mt5.TIMEFRAME_M5    # 5-Minute for Sell alerts
TIMEFRAME_BUY = mt5.TIMEFRAME_M15    # 15-Minute for Buy alerts
EMA_PERIOD = 5
RISK_REWARD_RATIO = 3.0              # Fixed 1:3 RR
POLL_INTERVAL = 5                    # Seconds between poll checks
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


def fetch_rates(timeframe=None, count=LOOKBACK_BARS):
    """Fetches candle rates and calculates 5 EMA."""
    if timeframe is None:
        timeframe = TIMEFRAME_SELL
    rates = mt5.copy_rates_from_pos(ACTIVE_SYMBOL, timeframe, 0, count)
    if rates is None or len(rates) < EMA_PERIOD + 5:
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['ema5'] = df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    return df


def emit_signal(sig_type: str, candle_time: datetime.datetime, entry_price: float, sl_price: float, tp_price: float, alert_time: datetime.datetime, tf_str: str):
    """Prints a structured ASCII signal card to console."""
    risk_pips = abs(entry_price - sl_price) / PIP_SIZE
    reward_pips = abs(tp_price - entry_price) / PIP_SIZE

    print("\n" + "=" * 75)
    print(f" [ALERT] MT5 SIGNAL ALERT: SUBASISH PANI 5 EMA SCALP ({sig_type.upper()})")
    print("=" * 75)
    print(f" Symbol:        {ACTIVE_SYMBOL} | Timeframe: {tf_str}")
    print(f" Bar Time:      {candle_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Signal Type:   {sig_type.upper()} (5 EMA Detached Alert Breakout)")
    print(f" Alert Candle:  {alert_time.strftime('%Y-%m-%d %H:%M:%S')}")
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
    print(f"   SUBASISH PANI 5 EMA SIGNAL SCANNER ({ACTIVE_SYMBOL})   ")
    print("=" * 75)
    print(" Strategy: Detached 5-EMA Candle -> Next Candle Breakout (1:3 Target)")
    print(" Status: Scanning live bars... Press Ctrl+C to stop.\n")

    last_sell_candle = None
    last_buy_candle = None

    try:
        while True:
            # 1. Check M5 Sell Setups
            df_m5 = fetch_rates(TIMEFRAME_SELL)
            if df_m5 is not None:
                current_time_m5 = df_m5['time'].iloc[-1]
                closed_bar_m5 = df_m5.iloc[-2]
                alert_bar_m5 = df_m5.iloc[-3]

                if last_sell_candle != current_time_m5:
                    last_sell_candle = current_time_m5

                    # Alert: Low did not touch 5 EMA (completely above 5 EMA)
                    is_sell_alert = alert_bar_m5['low'] > alert_bar_m5['ema5']
                    # Breakout: closed bar breaks below alert low
                    breakout_sell = (closed_bar_m5['low'] < alert_bar_m5['low']) and (closed_bar_m5['close'] < alert_bar_m5['low'])

                    if is_sell_alert and breakout_sell:
                        entry = closed_bar_m5['close']
                        sl = alert_bar_m5['high'] + (2.0 * PIP_SIZE)
                        risk = sl - entry
                        if risk > 0:
                            tp = entry - (risk * RISK_REWARD_RATIO)
                            emit_signal("SELL", current_time_m5, entry, sl, tp, alert_bar_m5['time'], "M5")

            # 2. Check M15 Buy Setups
            df_m15 = fetch_rates(TIMEFRAME_BUY)
            if df_m15 is not None:
                current_time_m15 = df_m15['time'].iloc[-1]
                closed_bar_m15 = df_m15.iloc[-2]
                alert_bar_m15 = df_m15.iloc[-3]

                if last_buy_candle != current_time_m15:
                    last_buy_candle = current_time_m15

                    # Alert: High did not touch 5 EMA (completely below 5 EMA)
                    is_buy_alert = alert_bar_m15['high'] < alert_bar_m15['ema5']
                    # Breakout: closed bar breaks above alert high
                    breakout_buy = (closed_bar_m15['high'] > alert_bar_m15['high']) and (closed_bar_m15['close'] > alert_bar_m15['high'])

                    if is_buy_alert and breakout_buy:
                        entry = closed_bar_m15['close']
                        sl = alert_bar_m15['low'] - (2.0 * PIP_SIZE)
                        risk = entry - sl
                        if risk > 0:
                            tp = entry + (risk * RISK_REWARD_RATIO)
                            emit_signal("BUY", current_time_m15, entry, sl, tp, alert_bar_m15['time'], "M15")

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n[STOP] Scanner stopped by user.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    run_scanner()
