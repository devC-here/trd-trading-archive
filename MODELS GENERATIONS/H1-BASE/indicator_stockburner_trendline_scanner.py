"""
========================================================================================
MT5 SIGNAL SCANNER: STOCK BURNER MULTI-TIMEFRAME TRENDLINE BREAKOUT STRATEGY
========================================================================================
Source: Short Video Q2EyAXyrcV0 | 1-Year Backtest: +47.4% Return, 48% Win Rate, 21% DD
Creator Attribution: Stock Burner (1.27M Subscribers)
Market: Forex (EURUSD, GBPUSD) / Universal MT5
Timeframe: H1 Trend Filter + M15 Pullback Breakout Execution
Indicators:
  1. 50 EMA on H1 (Macro Regime)
  2. M15 Dynamic Swing Trendline Equation (y = mx + c)
Logic:
  - Buy Setup:
      1. Macro Trend: Price > 50 EMA on H1 (Bullish Trend).
      2. M15 Pullback: Price pulls back against the trend, forming two descending swing highs.
      3. Trendline: A downward trendline connects the two lower swing highs.
      4. Breakout Trigger: An M15 candle breaks and closes decisively ABOVE the trendline.
      5. Stop Loss: Placed below the lowest point of the pullback swing.
      6. Take Profit: 1:2 to 1:3 Risk-to-Reward Ratio.
  - Sell Setup (Inverted):
      1. Macro Trend: Price < 50 EMA on H1 (Bearish Trend).
      2. M15 Pullback: Price pulls back, forming two ascending swing lows.
      3. Trendline: An upward trendline connects the two higher swing lows.
      4. Breakdown Trigger: An M15 candle breaks and closes decisively BELOW the trendline.
      5. Stop Loss: Placed above the highest point of the pullback swing.
      6. Take Profit: 1:2 to 1:3 Risk-to-Reward Ratio.
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
TIMEFRAME_HTF = mt5.TIMEFRAME_H1  # 1-Hour Macro Trend
TIMEFRAME_LTF = mt5.TIMEFRAME_M15 # 15-Minute Breakout Execution
EMA_TREND = 50
SWING_N = 4                      # Pivot detection window
RISK_REWARD_RATIO = 2.5
POLL_INTERVAL = 5                # Seconds between poll checks
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


def fetch_rates(timeframe=None, count=LOOKBACK_BARS):
    """Fetches candle rates for a given timeframe."""
    if timeframe is None:
        timeframe = TIMEFRAME_LTF
    rates = mt5.copy_rates_from_pos(ACTIVE_SYMBOL, timeframe, 0, count)
    if rates is None or len(rates) < 30:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


def find_swing_pivots(df: pd.DataFrame, n=SWING_N):
    """Finds recent swing highs and swing lows."""
    highs = df['high'].values
    lows = df['low'].values
    length = len(df)

    swing_highs = [] # (index, price)
    swing_lows = []

    for i in range(n, length - n):
        if highs[i] == max(highs[i-n:i+n+1]):
            swing_highs.append((i, highs[i]))
        if lows[i] == min(lows[i-n:i+n+1]):
            swing_lows.append((i, lows[i]))

    return swing_highs, swing_lows


def emit_signal(sig_type: str, candle_time: datetime.datetime, entry_price: float, sl_price: float, tp_price: float, tl_val: float):
    """Prints a structured ASCII signal card to console."""
    risk_pips = abs(entry_price - sl_price) / PIP_SIZE
    reward_pips = abs(tp_price - entry_price) / PIP_SIZE

    print("\n" + "=" * 75)
    print(f" [ALERT] MT5 SIGNAL ALERT: STOCK BURNER TRENDLINE BREAKOUT ({sig_type.upper()})")
    print("=" * 75)
    print(f" Symbol:        {ACTIVE_SYMBOL} | Timeframe: M15 (with H1 Trend)")
    print(f" Bar Time:      {candle_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Signal Type:   {sig_type.upper()} (Pullback Trendline Breakout)")
    print(f" Trendline Val: {tl_val:.5f}")
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
    print(f"   STOCK BURNER TRENDLINE BREAKOUT SCANNER ({ACTIVE_SYMBOL} M15)   ")
    print("=" * 75)
    print(" Strategy: H1 50-EMA Trend + M15 Lower Highs Pullback Trendline Break")
    print(" Status: Scanning live bars... Press Ctrl+C to stop.\n")

    last_processed_candle_time = None

    try:
        while True:
            df_h1 = fetch_rates(TIMEFRAME_HTF, count=70)
            df_m15 = fetch_rates(TIMEFRAME_LTF, count=100)

            if df_h1 is None or df_m15 is None:
                time.sleep(POLL_INTERVAL)
                continue

            current_candle_time = df_m15['time'].iloc[-1]
            closed_bar = df_m15.iloc[-2]
            prev_bar = df_m15.iloc[-3]

            if last_processed_candle_time != current_candle_time:
                last_processed_candle_time = current_candle_time

                # H1 Trend: 50 EMA
                df_h1['ema50'] = df_h1['close'].ewm(span=EMA_TREND, adjust=False).mean()
                h1_bull = df_h1['close'].iloc[-1] > df_h1['ema50'].iloc[-1]
                h1_bear = df_h1['close'].iloc[-1] < df_h1['ema50'].iloc[-1]

                swing_highs, swing_lows = find_swing_pivots(df_m15)

                # ----------------------------------------------------
                # BUY SETUP: H1 Bullish + Breakout of Downward Trendline
                # ----------------------------------------------------
                if h1_bull and len(swing_highs) >= 2:
                    p1 = swing_highs[-2] # Earlier swing high (idx1, price1)
                    p2 = swing_highs[-1] # Later swing high (idx2, price2)

                    # Ensure descending lower highs (pullback)
                    if p2[1] < p1[1] and (p2[0] - p1[0]) >= 3:
                        slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
                        current_idx = len(df_m15) - 2 # closed_bar index
                        tl_price = p2[1] + slope * (current_idx - p2[0])
                        prev_tl_price = p2[1] + slope * (current_idx - 1 - p2[0])

                        # Breakout condition
                        if prev_bar['close'] <= prev_tl_price and closed_bar['close'] > tl_price:
                            entry = closed_bar['close']
                            recent_low = df_m15['low'].iloc[p2[0]:].min()
                            sl = recent_low - (2.0 * PIP_SIZE)
                            risk = entry - sl
                            if risk > 0:
                                tp = entry + (risk * RISK_REWARD_RATIO)
                                emit_signal("BUY", current_candle_time, entry, sl, tp, tl_price)

                # ----------------------------------------------------
                # SELL SETUP: H1 Bearish + Breakdown of Upward Trendline
                # ----------------------------------------------------
                if h1_bear and len(swing_lows) >= 2:
                    p1 = swing_lows[-2]
                    p2 = swing_lows[-1]

                    # Ascending higher lows (pullback up)
                    if p2[1] > p1[1] and (p2[0] - p1[0]) >= 3:
                        slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
                        current_idx = len(df_m15) - 2
                        tl_price = p2[1] + slope * (current_idx - p2[0])
                        prev_tl_price = p2[1] + slope * (current_idx - 1 - p2[0])

                        if prev_bar['close'] >= prev_tl_price and closed_bar['close'] < tl_price:
                            entry = closed_bar['close']
                            recent_high = df_m15['high'].iloc[p2[0]:].max()
                            sl = recent_high + (2.0 * PIP_SIZE)
                            risk = sl - entry
                            if risk > 0:
                                tp = entry - (risk * RISK_REWARD_RATIO)
                                emit_signal("SELL", current_candle_time, entry, sl, tp, tl_price)

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n[STOP] Scanner stopped by user.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    run_scanner()
