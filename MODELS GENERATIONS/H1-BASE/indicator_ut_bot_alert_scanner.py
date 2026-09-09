"""
========================================================================================
MT5 SIGNAL SCANNER: UT BOT ALERT + 200 EMA TRAILING STRATEGY
========================================================================================
Source: Short Video SggjoXJiNXw | 1-Year Backtest: +58% Return, 11% DD
Market: Forex (EURUSD, GBPUSD) / Universal MT5
Timeframe: M15 Execution
Indicators:
  1. UT Bot Alert Trailing ATR Algorithm (Key Value: 1.0, ATR Period: 10)
  2. 200 EMA (Macro Trend Baseline)
Logic:
  - Concept:
      UT Bot calculates dynamic volatility trailing bands using ATR.
      Crosses represent rapid momentum shifts.
  - Buy Setup:
      1. Trend Regime: Price > 200 EMA (Bullish Macro).
      2. Volatility Shift: Price crosses and closes ABOVE the UT Bot ATR trailing stop.
      3. Stop Loss: Placed at the UT Bot Trailing Stop level.
      4. Take Profit: 1:2 Risk-to-Reward Ratio.
  - Sell Setup (Inverted):
      1. Trend Regime: Price < 200 EMA (Bearish Macro).
      2. Volatility Shift: Price crosses and closes BELOW the UT Bot ATR trailing stop.
      3. Stop Loss: Placed at the UT Bot Trailing Stop level.
      4. Take Profit: 1:2 Risk-to-Reward Ratio.
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
EMA_TREND = 200
ATR_PERIOD = 10
KEY_VALUE = 1.0                  # UT Bot Sensitivity
RISK_REWARD_RATIO = 2.0
POLL_INTERVAL = 5                # Seconds between poll checks
LOOKBACK_BARS = 250

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


def calculate_ut_bot(df: pd.DataFrame, key_val=KEY_VALUE, atr_p=ATR_PERIOD):
    """Calculates UT Bot trailing stop line."""
    high = df['high']
    low = df['low']
    close = df['close']

    # ATR
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_p).mean()

    n_loss = key_val * atr

    trail_stop = [0.0] * len(df)
    for i in range(1, len(df)):
        c_price = close.iloc[i]
        p_price = close.iloc[i-1]
        p_stop = trail_stop[i-1]
        loss = n_loss.iloc[i]

        if c_price > p_stop:
            trail_stop[i] = max(p_stop, c_price - loss)
        else:
            trail_stop[i] = min(p_stop, c_price + loss)

    df['ut_stop'] = trail_stop
    return df


def fetch_rates(count=LOOKBACK_BARS):
    """Fetches candle rates and calculates 200 EMA and UT Bot."""
    rates = mt5.copy_rates_from_pos(ACTIVE_SYMBOL, TIMEFRAME, 0, count)
    if rates is None or len(rates) < EMA_TREND + 20:
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['ema200'] = df['close'].ewm(span=EMA_TREND, adjust=False).mean()
    df = calculate_ut_bot(df)
    return df


def emit_signal(sig_type: str, candle_time: datetime.datetime, entry_price: float, sl_price: float, tp_price: float, ut_stop: float):
    """Prints a structured ASCII signal card to console."""
    risk_pips = abs(entry_price - sl_price) / PIP_SIZE
    reward_pips = abs(tp_price - entry_price) / PIP_SIZE

    print("\n" + "=" * 75)
    print(f" [ALERT] MT5 SIGNAL ALERT: UT BOT ALERT SYSTEM ({sig_type.upper()})")
    print("=" * 75)
    print(f" Symbol:        {ACTIVE_SYMBOL} | Timeframe: M15")
    print(f" Bar Time:      {candle_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Signal Type:   {sig_type.upper()} (UT Bot Trailing Stop Flip)")
    print(f" UT Stop Line:  {ut_stop:.5f}")
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
    print(f"   UT BOT ALERT SIGNAL SCANNER ({ACTIVE_SYMBOL} M15)   ")
    print("=" * 75)
    print(" Strategy: 200 EMA Baseline + UT Bot ATR Dynamic Trailing Stop Flip")
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
            prev_bar = df.iloc[-3]         # Prior bar

            if last_processed_candle_time != current_candle_time:
                last_processed_candle_time = current_candle_time

                # ----------------------------------------------------
                # BUY LOGIC: Price > 200 EMA + Price crosses above UT Stop
                # ----------------------------------------------------
                macro_bull = closed_bar['close'] > closed_bar['ema200']
                ut_buy_flip = (prev_bar['close'] <= prev_bar['ut_stop']) and (closed_bar['close'] > closed_bar['ut_stop'])

                if macro_bull and ut_buy_flip:
                    entry = closed_bar['close']
                    sl = closed_bar['ut_stop'] - (2.0 * PIP_SIZE)
                    risk = entry - sl
                    if risk > 0:
                        tp = entry + (risk * RISK_REWARD_RATIO)
                        emit_signal("BUY", current_candle_time, entry, sl, tp, closed_bar['ut_stop'])

                # ----------------------------------------------------
                # SELL LOGIC: Price < 200 EMA + Price crosses below UT Stop
                # ----------------------------------------------------
                macro_bear = closed_bar['close'] < closed_bar['ema200']
                ut_sell_flip = (prev_bar['close'] >= prev_bar['ut_stop']) and (closed_bar['close'] < closed_bar['ut_stop'])

                if macro_bear and ut_sell_flip:
                    entry = closed_bar['close']
                    sl = closed_bar['ut_stop'] + (2.0 * PIP_SIZE)
                    risk = sl - entry
                    if risk > 0:
                        tp = entry - (risk * RISK_REWARD_RATIO)
                        emit_signal("SELL", current_candle_time, entry, sl, tp, closed_bar['ut_stop'])

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n[STOP] Scanner stopped by user.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    run_scanner()
