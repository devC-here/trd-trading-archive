"""
========================================================================================
FOREX MT5 SIGNAL SCANNER: ICT BREAKER BLOCK STRATEGY
========================================================================================
Source: Short Video B-Y_5S0kVpQ | 1-Year Backtest: +61.0% Return, 47.9% Win Rate
Market: Forex (EURUSD / Major Pairs)
Timeframe: M15 Execution
Indicators: Swing Structure, Market Structure Shift (MSS), Failed Order Block / Breaker Block
Logic:
  1. Trend Identification: Clean swing structure (Higher Highs & Higher Lows)
  2. Market Structure Shift (MSS): Price closes candle body beyond structural swing point
  3. Breaker Block Identification: The failed order block zone that led to the sweep
  4. Retest / Mitigation: Price pulls back into the Breaker Block zone and closes inside
  5. Risk Management: SL beyond Breaker Block zone; TP at 1:2 Risk-Reward
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
TIMEFRAME = mt5.TIMEFRAME_M15     # 15-Minute Chart
SWING_LOOKBACK = 5                # Bars left/right for swing high/low
RISK_REWARD_RATIO = 2.0           # 1:2 Risk-to-Reward
POLL_INTERVAL = 5                 # Seconds between poll checks
LOOKBACK_BARS = 150

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
    """Fetches M15 candle rates."""
    rates = mt5.copy_rates_from_pos(ACTIVE_SYMBOL, TIMEFRAME, 0, LOOKBACK_BARS)
    if rates is None or len(rates) < 60:
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


def find_swings(df: pd.DataFrame):
    """Finds local swing highs and swing lows."""
    highs = []
    lows = []
    n = len(df)
    
    for i in range(SWING_LOOKBACK, n - SWING_LOOKBACK):
        current_high = df['high'].iloc[i]
        current_low = df['low'].iloc[i]
        
        is_swing_high = all(current_high > df['high'].iloc[i - j] for j in range(1, SWING_LOOKBACK + 1)) and \
                        all(current_high > df['high'].iloc[i + j] for j in range(1, SWING_LOOKBACK + 1))
        
        is_swing_low = all(current_low < df['low'].iloc[i - j] for j in range(1, SWING_LOOKBACK + 1)) and \
                       all(current_low < df['low'].iloc[i + j] for j in range(1, SWING_LOOKBACK + 1))
        
        if is_swing_high:
            highs.append((i, current_high, df['time'].iloc[i]))
        if is_swing_low:
            lows.append((i, current_low, df['time'].iloc[i]))
            
    return highs, lows


def emit_signal(sig_type: str, candle_time: datetime.datetime, entry_price: float, sl_price: float, tp_price: float, zone_top: float, zone_bot: float):
    """Prints a structured ASCII signal card to console."""
    risk_pips = abs(entry_price - sl_price) / PIP_SIZE
    reward_pips = abs(tp_price - entry_price) / PIP_SIZE
    
    print("\n" + "=" * 70)
    print(f" [ALERT] FOREX SIGNAL ALERT: ICT BREAKER BLOCK ({sig_type.upper()})")
    print("=" * 70)
    print(f" Symbol:        {ACTIVE_SYMBOL} | Timeframe: M15")
    print(f" Bar Time:      {candle_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Signal Type:   {sig_type.upper()}")
    print(f" Breaker Zone:  {zone_bot:.5f} - {zone_top:.5f}")
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
    print(f"   ICT BREAKER BLOCK FOREX SIGNAL SCANNER ({ACTIVE_SYMBOL} M15)   ")
    print("=" * 70)
    print(" Strategy Logic: Swing Trend -> Market Structure Shift (MSS) -> Breaker Retest")
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
            
            if last_processed_candle_time != current_candle_time:
                last_processed_candle_time = current_candle_time
                swing_highs, swing_lows = find_swings(df)

                if len(swing_highs) >= 2 and len(swing_lows) >= 2:
                    # ----------------------------------------------------
                    # BEARISH BREAKER BLOCK (Sell Setup)
                    # ----------------------------------------------------
                    # Recent structure was Higher Highs (HH) & Higher Lows (HL)
                    sh2 = swing_highs[-2]
                    sh1 = swing_highs[-1]
                    sl1 = swing_lows[-1]
                    
                    if sh1[1] > sh2[1]:  # Uptrend leg
                        # Market Structure Shift (MSS): Price broke below Higher Low (sl1)
                        mss_happened = any(df['close'].iloc[sh1[0]:-2] < sl1[1])
                        
                        if mss_happened:
                            # Breaker Block: The highest bullish candle before the higher high
                            idx_range = df.iloc[sh2[0]:sh1[0]]
                            bull_candles = idx_range[idx_range['close'] > idx_range['open']]
                            if not bull_candles.empty:
                                breaker_candle = bull_candles.iloc[-1]
                                zone_top = breaker_candle['high']
                                zone_bot = breaker_candle['low']
                                
                                # Current bar mitigated the breaker block zone
                                in_zone = (closed_bar['high'] >= zone_bot) and (closed_bar['close'] <= zone_top)
                                is_bearish_rejection = closed_bar['close'] < closed_bar['open']
                                
                                if in_zone and is_bearish_rejection:
                                    entry = closed_bar['close']
                                    sl = zone_top + (2.5 * PIP_SIZE)
                                    risk = sl - entry
                                    if risk > 0:
                                        tp = entry - (risk * RISK_REWARD_RATIO)
                                        emit_signal("SELL", current_candle_time, entry, sl, tp, zone_top, zone_bot)

                    # ----------------------------------------------------
                    # BULLISH BREAKER BLOCK (Buy Setup - Inverted)
                    # ----------------------------------------------------
                    sl2 = swing_lows[-2]
                    sl1 = swing_lows[-1]
                    sh1 = swing_highs[-1]
                    
                    if sl1[1] < sl2[1]:  # Downtrend leg
                        mss_happened = any(df['close'].iloc[sl1[0]:-2] > sh1[1])
                        
                        if mss_happened:
                            idx_range = df.iloc[sl2[0]:sl1[0]]
                            bear_candles = idx_range[idx_range['close'] < idx_range['open']]
                            if not bear_candles.empty:
                                breaker_candle = bear_candles.iloc[-1]
                                zone_top = breaker_candle['high']
                                zone_bot = breaker_candle['low']
                                
                                in_zone = (closed_bar['low'] <= zone_top) and (closed_bar['close'] >= zone_bot)
                                is_bullish_rejection = closed_bar['close'] > closed_bar['open']
                                
                                if in_zone and is_bullish_rejection:
                                    entry = closed_bar['close']
                                    sl = zone_bot - (2.5 * PIP_SIZE)
                                    risk = entry - sl
                                    if risk > 0:
                                        tp = entry + (risk * RISK_REWARD_RATIO)
                                        emit_signal("BUY", current_candle_time, entry, sl, tp, zone_top, zone_bot)

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n[STOP] Scanner stopped by user.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    run_scanner()
