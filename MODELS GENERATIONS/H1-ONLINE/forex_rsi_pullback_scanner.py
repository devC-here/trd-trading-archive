"""
========================================================================================
FOREX MT5 SIGNAL SCANNER: RSI PULLBACK IN TREND (50 EMA + RSI 30/70)
========================================================================================
Source: Short Video ryoDybQXEs4 | 1-Year Backtest: +47.6% Return, 60.7% Win Rate, 2.36 PF
Market: Forex (EURUSD / Major Pairs)
Timeframe: M5 or M15 Execution
Indicators: 50 EMA, RSI (14)
Logic:
  1. Trend Alignment: Price above 50 EMA for Long trades (Trend with you, not against)
  2. Oversold Pullback: RSI(14) dips below 30 into oversold territory
  3. Confirmation Trigger: RSI crosses back above 30 with a bullish confirmation candle
  4. Risk Management: Stop Loss below pullback low; Target prior swing high or 1:2 RR
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
EMA_PERIOD = 50
RSI_PERIOD = 14
RSI_OVERSOLD = 30.0
RSI_OVERBOUGHT = 70.0
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
    """Fetches rates and calculates 50 EMA + RSI (14)."""
    rates = mt5.copy_rates_from_pos(ACTIVE_SYMBOL, TIMEFRAME, 0, LOOKBACK_BARS)
    if rates is None or len(rates) < 60:
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # 50 EMA
    df['ema50'] = df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    
    # Wilder's RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    return df


def emit_signal(sig_type: str, candle_time: datetime.datetime, entry_price: float, sl_price: float, tp_price: float, rsi_val: float):
    """Prints a structured ASCII signal card to console."""
    risk_pips = abs(entry_price - sl_price) / PIP_SIZE
    reward_pips = abs(tp_price - entry_price) / PIP_SIZE
    
    print("\n" + "=" * 70)
    print(f" [ALERT] FOREX SIGNAL ALERT: RSI PULLBACK STRATEGY ({sig_type.upper()})")
    print("=" * 70)
    print(f" Symbol:        {ACTIVE_SYMBOL} | Timeframe: M5")
    print(f" Bar Time:      {candle_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Signal Type:   {sig_type.upper()}")
    print(f" RSI Value:     {rsi_val:.2f}")
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
    print(f"   RSI PULLBACK FOREX SIGNAL SCANNER ({ACTIVE_SYMBOL} M5)   ")
    print("=" * 70)
    print(" Strategy Logic: 50 EMA Trend Filter -> RSI 30/70 Pullback -> Confirmation Close")
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

                # ----------------------------------------------------
                # BUY LOGIC (Long Setup)
                # ----------------------------------------------------
                # 1. Price above 50 EMA
                uptrend = closed_bar['close'] > closed_bar['ema50']
                # 2. RSI was oversold (<30) in last 4 bars
                was_oversold = any(df['rsi'].iloc[-5:-2] < RSI_OVERSOLD)
                # 3. RSI crossed back above 30 on closed bar
                rsi_crossed_up = (prev_bar['rsi'] <= RSI_OVERSOLD) and (closed_bar['rsi'] > RSI_OVERSOLD)
                # 4. Bullish confirmation candle body
                bullish_candle = closed_bar['close'] > closed_bar['open']

                if uptrend and was_oversold and rsi_crossed_up and bullish_candle:
                    entry = closed_bar['close']
                    pullback_low = df['low'].iloc[-6:-1].min()
                    sl = pullback_low - (2.0 * PIP_SIZE)
                    risk = entry - sl
                    if risk > 0:
                        tp = entry + (risk * RISK_REWARD_RATIO)
                        emit_signal("BUY", current_candle_time, entry, sl, tp, closed_bar['rsi'])

                # ----------------------------------------------------
                # SELL LOGIC (Short Setup - Inverted)
                # ----------------------------------------------------
                downtrend = closed_bar['close'] < closed_bar['ema50']
                was_overbought = any(df['rsi'].iloc[-5:-2] > RSI_OVERBOUGHT)
                rsi_crossed_down = (prev_bar['rsi'] >= RSI_OVERBOUGHT) and (closed_bar['rsi'] < RSI_OVERBOUGHT)
                bearish_candle = closed_bar['close'] < closed_bar['open']

                if downtrend and was_overbought and rsi_crossed_down and bearish_candle:
                    entry = closed_bar['close']
                    pullback_high = df['high'].iloc[-6:-1].max()
                    sl = pullback_high + (2.0 * PIP_SIZE)
                    risk = sl - entry
                    if risk > 0:
                        tp = entry - (risk * RISK_REWARD_RATIO)
                        emit_signal("SELL", current_candle_time, entry, sl, tp, closed_bar['rsi'])

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n[STOP] Scanner stopped by user.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    run_scanner()
