"""
========================================================================================
MT5 SIGNAL SCANNER: 200 EMA + STOCHASTIC OVERSOLD/OVERBOUGHT PULLBACK STRATEGY
========================================================================================
Source: Short Video _cMfadzVxpo | 1-Year Backtest: +68% Return, 44% Win Rate
Market: Forex (EURUSD, GBPUSD) / Universal MT5
Timeframe: M15 Execution
Indicators:
  1. 200 EMA (Macro Trend Baseline)
  2. Stochastic Oscillator (%K: 14, %D: 3, Smooth: 3)
Logic:
  - Buy Setup:
      1. Macro Trend: Price > 200 EMA (Bullish Trend).
      2. Stochastic Oversold Pullback: Both %K and %D lines dip below 20 and then cross back ABOVE 20.
      3. Trigger: Confirmed on candle close as Stochastic reclaims the >20 zone.
      4. Stop Loss: Placed below the recent swing low.
      5. Take Profit: 1:2 Risk-to-Reward Ratio.
  - Sell Setup (Inverted):
      1. Macro Trend: Price < 200 EMA (Bearish Trend).
      2. Stochastic Overbought Pullback: Both %K and %D lines rise above 80 and then cross back BELOW 80.
      3. Trigger: Confirmed on candle close as Stochastic re-enters the <80 zone.
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
TIMEFRAME = mt5.TIMEFRAME_M15    # 15-Minute Chart
EMA_TREND = 200
STOCH_K = 14
STOCH_D = 3
STOCH_SMOOTH = 3
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


def fetch_rates(count=LOOKBACK_BARS):
    """Fetches candle rates and calculates 200 EMA and Stochastic %K / %D."""
    rates = mt5.copy_rates_from_pos(ACTIVE_SYMBOL, TIMEFRAME, 0, count)
    if rates is None or len(rates) < EMA_TREND + 20:
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # 200 EMA
    df['ema200'] = df['close'].ewm(span=EMA_TREND, adjust=False).mean()

    # Stochastic
    low_min = df['low'].rolling(window=STOCH_K).min()
    high_max = df['high'].rolling(window=STOCH_K).max()
    raw_k = 100 * ((df['close'] - low_min) / (high_max - low_min + 1e-9))
    df['stoch_k'] = raw_k.rolling(window=STOCH_SMOOTH).mean()
    df['stoch_d'] = df['stoch_k'].rolling(window=STOCH_D).mean()

    return df


def emit_signal(sig_type: str, candle_time: datetime.datetime, entry_price: float, sl_price: float, tp_price: float, k_val: float, d_val: float):
    """Prints a structured ASCII signal card to console."""
    risk_pips = abs(entry_price - sl_price) / PIP_SIZE
    reward_pips = abs(tp_price - entry_price) / PIP_SIZE

    print("\n" + "=" * 75)
    print(f" [ALERT] MT5 SIGNAL ALERT: 200 EMA + STOCHASTIC PULLBACK ({sig_type.upper()})")
    print("=" * 75)
    print(f" Symbol:        {ACTIVE_SYMBOL} | Timeframe: M15")
    print(f" Bar Time:      {candle_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Signal Type:   {sig_type.upper()} (200 EMA Trend + Stoch Extreme Reversal)")
    print(f" Stochastic:    %K = {k_val:.2f} | %D = {d_val:.2f}")
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
    print(f"   200 EMA + STOCHASTIC PULLBACK SCANNER ({ACTIVE_SYMBOL} M15)   ")
    print("=" * 75)
    print(" Strategy: 200 EMA Trend + Stoch <20 Reclaim (Buy) / >80 Breakdown (Sell)")
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
            prev_bar = df.iloc[-3]         # Prior bar for crossover

            if last_processed_candle_time != current_candle_time:
                last_processed_candle_time = current_candle_time

                # ----------------------------------------------------
                # BUY LOGIC: Trend Bullish + Stoch Oversold Reclaim (>20)
                # ----------------------------------------------------
                trend_bullish = closed_bar['close'] > closed_bar['ema200']
                stoch_oversold_cross = (prev_bar['stoch_k'] <= 20 or prev_bar['stoch_d'] <= 20) and (closed_bar['stoch_k'] > 20 and closed_bar['stoch_d'] > 20)

                if trend_bullish and stoch_oversold_cross:
                    entry = closed_bar['close']
                    recent_low = df['low'].iloc[-6:-1].min()
                    sl = recent_low - (2.0 * PIP_SIZE)
                    risk = entry - sl
                    if risk > 0:
                        tp = entry + (risk * RISK_REWARD_RATIO)
                        emit_signal("BUY", current_candle_time, entry, sl, tp, closed_bar['stoch_k'], closed_bar['stoch_d'])

                # ----------------------------------------------------
                # SELL LOGIC: Trend Bearish + Stoch Overbought Breakdown (<80)
                # ----------------------------------------------------
                trend_bearish = closed_bar['close'] < closed_bar['ema200']
                stoch_overbought_cross = (prev_bar['stoch_k'] >= 80 or prev_bar['stoch_d'] >= 80) and (closed_bar['stoch_k'] < 80 and closed_bar['stoch_d'] < 80)

                if trend_bearish and stoch_overbought_cross:
                    entry = closed_bar['close']
                    recent_high = df['high'].iloc[-6:-1].max()
                    sl = recent_high + (2.0 * PIP_SIZE)
                    risk = sl - entry
                    if risk > 0:
                        tp = entry - (risk * RISK_REWARD_RATIO)
                        emit_signal("SELL", current_candle_time, entry, sl, tp, closed_bar['stoch_k'], closed_bar['stoch_d'])

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n[STOP] Scanner stopped by user.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    run_scanner()
