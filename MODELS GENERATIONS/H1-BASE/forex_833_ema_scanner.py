"""
========================================================================================
FOREX MT5 SIGNAL SCANNER: 833 EMA STRATEGY (Wizard Trader)
========================================================================================
Source: Short Video FaHKCazt49A | 1-Year Backtest: +63.2% Return, 57.8% Win Rate
Market: Forex (EURUSD / Major Pairs)
Timeframe: M5 Execution (with H1/H4 swing option)
Indicators: 8 EMA (Fast), 33 EMA (Slow)
Logic:
  1. Trend Alignment: 8 EMA > 33 EMA and Price > both (for Buy) / 8 EMA < 33 EMA (for Sell)
  2. Pullback: Price pulls back to test 33 EMA
  3. Dominance Candle: Strong bullish/bearish candle with large body forms at 33 EMA
  4. Trigger: Next candle breaks the Dominance Candle high (Buy) / low (Sell)
  5. Risk Management: SL below recent swing low; TP at 1:2 Risk-Reward
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
EMA_FAST = 8
EMA_SLOW = 33
RISK_REWARD_RATIO = 2.0           # 1:2 Risk-to-Reward
POLL_INTERVAL = 5                 # Seconds between poll checks
LOOKBACK_BARS = 120

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
    """Fetches M5 candle rates and calculates indicators."""
    rates = mt5.copy_rates_from_pos(ACTIVE_SYMBOL, TIMEFRAME, 0, LOOKBACK_BARS)
    if rates is None or len(rates) < 50:
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Calculate 8 and 33 EMAs
    df['ema8'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
    df['ema33'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()

    # Candle properties
    df['body'] = (df['close'] - df['open']).abs()
    df['range'] = df['high'] - df['low']
    df['avg_body'] = df['body'].rolling(10).mean()
    
    return df


def emit_signal(sig_type: str, candle_time: datetime.datetime, entry_price: float, sl_price: float, tp_price: float):
    """Prints a structured ASCII signal card to console."""
    risk_pips = abs(entry_price - sl_price) / PIP_SIZE
    reward_pips = abs(tp_price - entry_price) / PIP_SIZE
    
    print("\n" + "=" * 70)
    print(f" [ALERT] FOREX SIGNAL ALERT: 833 EMA STRATEGY ({sig_type.upper()})")
    print("=" * 70)
    print(f" Symbol:        {ACTIVE_SYMBOL} | Timeframe: M5")
    print(f" Bar Time:      {candle_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Signal Type:   {sig_type.upper()}")
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
    print(f"   833 EMA FOREX SIGNAL SCANNER ({ACTIVE_SYMBOL} M5)   ")
    print("=" * 70)
    print(" Strategy Logic: 8/33 EMA Trend -> 33 EMA Pullback -> Dominance Candle Break")
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
            prev_bar = df.iloc[-3]         # Potential Dominance candle
            
            # Evaluate only once per newly closed candle
            if last_processed_candle_time != current_candle_time:
                last_processed_candle_time = current_candle_time

                # ----------------------------------------------------
                # BUY LOGIC (Long Setup)
                # ----------------------------------------------------
                # Step 1: Trend Alignment
                trend_bullish = (closed_bar['ema8'] > closed_bar['ema33']) and (closed_bar['close'] > closed_bar['ema8'])
                
                # Step 2: Pullback test to 33 EMA (within last 3 bars)
                pullback_bullish = any(df['low'].iloc[-4:-1] <= df['ema33'].iloc[-4:-1])
                
                # Step 3: Dominance Candle (strong green body at 33 EMA)
                is_bull_dominance = (
                    (prev_bar['close'] > prev_bar['open']) and 
                    (prev_bar['body'] > prev_bar['avg_body'] * 1.1) and 
                    (prev_bar['body'] >= prev_bar['range'] * 0.55)
                )
                
                # Step 4: Closed bar broke Dominance Candle high
                bull_breakout = (closed_bar['high'] > prev_bar['high']) and (closed_bar['close'] > prev_bar['high'])

                if trend_bullish and pullback_bullish and is_bull_dominance and bull_breakout:
                    entry = closed_bar['close']
                    recent_low = df['low'].iloc[-5:-1].min()
                    sl = recent_low - (2.0 * PIP_SIZE)
                    risk = entry - sl
                    if risk > 0:
                        tp = entry + (risk * RISK_REWARD_RATIO)
                        emit_signal("BUY", current_candle_time, entry, sl, tp)

                # ----------------------------------------------------
                # SELL LOGIC (Short Setup - Inverted)
                # ----------------------------------------------------
                trend_bearish = (closed_bar['ema8'] < closed_bar['ema33']) and (closed_bar['close'] < closed_bar['ema8'])
                pullback_bearish = any(df['high'].iloc[-4:-1] >= df['ema33'].iloc[-4:-1])
                
                is_bear_dominance = (
                    (prev_bar['close'] < prev_bar['open']) and 
                    (prev_bar['body'] > prev_bar['avg_body'] * 1.1) and 
                    (prev_bar['body'] >= prev_bar['range'] * 0.55)
                )
                
                bear_breakout = (closed_bar['low'] < prev_bar['low']) and (closed_bar['close'] < prev_bar['low'])

                if trend_bearish and pullback_bearish and is_bear_dominance and bear_breakout:
                    entry = closed_bar['close']
                    recent_high = df['high'].iloc[-5:-1].max()
                    sl = recent_high + (2.0 * PIP_SIZE)
                    risk = sl - entry
                    if risk > 0:
                        tp = entry - (risk * RISK_REWARD_RATIO)
                        emit_signal("SELL", current_candle_time, entry, sl, tp)

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n[STOP] Scanner stopped by user.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    run_scanner()
