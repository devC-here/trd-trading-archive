"""
========================================================================================
FOREX MT5 SIGNAL SCANNER: 20/200 DUAL EMA + 2X ATR VOLATILITY ENVELOPE
========================================================================================
Source: Short Video R7kX_T4ZtNo | 1-Year Backtest: +54.0% Return, 58.0% Win Rate
Market: Forex (EURUSD / Major Pairs)
Timeframe: M5 Execution
Indicators: 20 EMA, 200 EMA, ATR (14)
Logic:
  1. Macro Trend Bias: Price and 20 EMA must be above 200 EMA (Buy) / below 200 EMA (Sell)
  2. Pullback: Price pulls back to test the 20 EMA
  3. Confirmation: Candle closes back in trend direction above 20 EMA (Buy)
  4. Risk Management: Stop Loss placed at 2.0x ATR distance; Take Profit at 1:2 RR
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
TIMEFRAME = mt5.TIMEFRAME_M5      # 5-Minute Chart
EMA_FAST = 20
EMA_SLOW = 200
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0              # 2x ATR dynamic Stop Loss buffer
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


def fetch_rates():
    """Fetches candle rates and calculates 20 EMA, 200 EMA, and ATR."""
    rates = mt5.copy_rates_from_pos(ACTIVE_SYMBOL, TIMEFRAME, 0, LOOKBACK_BARS)
    if rates is None or len(rates) < 100:
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # EMAs
    df['ema20'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()
    
    # True Range & ATR
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/ATR_PERIOD, adjust=False).mean()
    
    return df


def emit_signal(sig_type: str, candle_time: datetime.datetime, entry_price: float, sl_price: float, tp_price: float, atr_val: float):
    """Prints a structured ASCII signal card to console."""
    risk_pips = abs(entry_price - sl_price) / PIP_SIZE
    reward_pips = abs(tp_price - entry_price) / PIP_SIZE
    
    print("\n" + "=" * 70)
    print(f" [ALERT] FOREX SIGNAL ALERT: 20/200 EMA + 2X ATR ({sig_type.upper()})")
    print("=" * 70)
    print(f" Symbol:        {ACTIVE_SYMBOL} | Timeframe: M5")
    print(f" Bar Time:      {candle_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Signal Type:   {sig_type.upper()}")
    print(f" ATR (14):      {atr_val / PIP_SIZE:.1f} pips")
    print(f" Entry Price:   {entry_price:.5f}")
    print(f" Stop Loss:     {sl_price:.5f}  ({risk_pips:.1f} pips)")
    print(f" Take Profit:   {tp_price:.5f}  ({reward_pips:.1f} pips)")
    print(f" Risk/Reward:   1:{RISK_REWARD_RATIO:.1f}")
    print(f" Status:        SIGNAL ONLY (Ready for manual or external execution)")
    print("=" * 70 + "\n")


def run_scanner():
    """Main live market scanner loop."""
    if not initialize_mt5():
        return

    print("\n" + "=" * 70)
    print(f"   20/200 EMA PULLBACK FOREX SIGNAL SCANNER ({ACTIVE_SYMBOL} M5)   ")
    print("=" * 70)
    print(" Strategy Logic: 20/200 EMA Macro Trend -> 20 EMA Pullback -> 2x ATR Dynamic SL")
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
            prev_bar = df.iloc[-3]         # Prior bar
            
            if last_processed_candle_time != current_candle_time:
                last_processed_candle_time = current_candle_time
                atr_current = closed_bar['atr']

                # ----------------------------------------------------
                # BUY LOGIC: Price & 20 EMA > 200 EMA + 20 EMA Pullback
                # ----------------------------------------------------
                trend_bull = (closed_bar['ema20'] > closed_bar['ema200']) and (closed_bar['close'] > closed_bar['ema200'])
                # Price pulled back to touch 20 EMA in recent bars
                pullback_touch = any(df['low'].iloc[-4:-1] <= df['ema20'].iloc[-4:-1])
                # Closed bar closed back above 20 EMA with bullish body
                rebound_bull = (closed_bar['close'] > closed_bar['ema20']) and (closed_bar['close'] > closed_bar['open'])

                if trend_bull and pullback_touch and rebound_bull:
                    entry = closed_bar['close']
                    sl_dist = max(atr_current * ATR_MULTIPLIER, 2.0 * PIP_SIZE)
                    sl = entry - sl_dist
                    tp = entry + (sl_dist * RISK_REWARD_RATIO)
                    emit_signal("BUY", current_candle_time, entry, sl, tp, atr_current)

                # ----------------------------------------------------
                # SELL LOGIC: Price & 20 EMA < 200 EMA + 20 EMA Pullback
                # ----------------------------------------------------
                trend_bear = (closed_bar['ema20'] < closed_bar['ema200']) and (closed_bar['close'] < closed_bar['ema200'])
                pullback_touch_bear = any(df['high'].iloc[-4:-1] >= df['ema20'].iloc[-4:-1])
                rebound_bear = (closed_bar['close'] < closed_bar['ema20']) and (closed_bar['close'] < closed_bar['open'])

                if trend_bear and pullback_touch_bear and rebound_bear:
                    entry = closed_bar['close']
                    sl_dist = max(atr_current * ATR_MULTIPLIER, 2.0 * PIP_SIZE)
                    sl = entry + sl_dist
                    tp = entry - (sl_dist * RISK_REWARD_RATIO)
                    emit_signal("SELL", current_candle_time, entry, sl, tp, atr_current)

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n[STOP] Scanner stopped by user.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    run_scanner()
