"""
========================================================================================
FOREX MT5 SIGNAL SCANNER: ICT ORDER BLOCK & FVG RETEST STRATEGY
========================================================================================
Source: Short Video 4W3SUtxl0u8 | 1-Year Backtest: +74.1% Return, 53.2% Win Rate
Market: Forex (EURUSD / Major Pairs)
Timeframe: Multi-Timeframe (H4 Institutional Context -> M15 Execution)
Indicators: Order Block (OB), Fair Value Gap (FVG), Market Structure Shift (MSS)
Logic:
  1. H4 Context: Identify valid institutional Order Block (OB) zone
  2. M15 Reaction: Price taps into the H4 Order Block zone
  3. M15 Imbalance: 3-candle Fair Value Gap (FVG) forms with Market Structure Shift
  4. Retest / Mitigation: Price pulls back into the M15 FVG and confirms direction
  5. Risk Management: SL beyond local OB/FVG origin; TP at 1:2 Risk-Reward
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
TIMEFRAME_HTF = mt5.TIMEFRAME_H4  # 4-Hour Order Block Context
TIMEFRAME_LTF = mt5.TIMEFRAME_M15 # 15-Minute FVG Execution
MIN_FVG_PIPS = 2.0                # Minimum FVG gap size in pips
RISK_REWARD_RATIO = 2.0           # 1:2 Risk-to-Reward
POLL_INTERVAL = 5                 # Seconds between poll checks

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


def fetch_rates(timeframe=None, count=100):
    """Fetches candle rates for a given timeframe."""
    if timeframe is None:
        timeframe = TIMEFRAME_LTF
    rates = mt5.copy_rates_from_pos(ACTIVE_SYMBOL, timeframe, 0, count)
    if rates is None or len(rates) < 30:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


def find_h4_order_blocks(df_h4: pd.DataFrame):
    """Identifies institutional Order Blocks on H4 chart."""
    bullish_obs = []
    bearish_obs = []
    n = len(df_h4)
    
    for i in range(5, n - 2):
        candle = df_h4.iloc[i]
        next_candle = df_h4.iloc[i+1]
        
        # Bullish OB: Last down-candle before large displacement up
        if candle['close'] < candle['open'] and (next_candle['close'] - next_candle['open']) > (4.0 * PIP_SIZE):
            bullish_obs.append((candle['low'], candle['high'], df_h4['time'].iloc[i]))
            
        # Bearish OB: Last up-candle before large displacement down
        if candle['close'] > candle['open'] and (next_candle['open'] - next_candle['close']) > (4.0 * PIP_SIZE):
            bearish_obs.append((candle['low'], candle['high'], df_h4['time'].iloc[i]))
            
    recent_bull_ob = bullish_obs[-1] if bullish_obs else None
    recent_bear_ob = bearish_obs[-1] if bearish_obs else None
    return recent_bull_ob, recent_bear_ob


def find_m15_fvg(df_m15: pd.DataFrame):
    """Detects 3-candle Fair Value Gap (imbalance) on M15."""
    # Bullish FVG: Low of candle 3 > High of candle 1
    c1 = df_m15.iloc[-4]
    c2 = df_m15.iloc[-3]
    c3 = df_m15.iloc[-2]
    
    bull_gap = c3['low'] - c1['high']
    if bull_gap >= (MIN_FVG_PIPS * PIP_SIZE) and c2['close'] > c2['open']:
        return "BULLISH_FVG", c1['high'], c3['low'], c2['low']
        
    bear_gap = c1['low'] - c3['high']
    if bear_gap >= (MIN_FVG_PIPS * PIP_SIZE) and c2['close'] < c2['open']:
        return "BEARISH_FVG", c3['high'], c1['low'], c2['high']
        
    return None, 0.0, 0.0, 0.0


def emit_signal(sig_type: str, candle_time: datetime.datetime, entry_price: float, sl_price: float, tp_price: float, fvg_bot: float, fvg_top: float):
    """Prints a structured ASCII signal card to console."""
    risk_pips = abs(entry_price - sl_price) / PIP_SIZE
    reward_pips = abs(tp_price - entry_price) / PIP_SIZE
    
    print("\n" + "=" * 70)
    print(f" [ALERT] FOREX SIGNAL ALERT: ICT ORDER BLOCK & FVG ({sig_type.upper()})")
    print("=" * 70)
    print(f" Symbol:        {ACTIVE_SYMBOL} | Macro H4 -> Entry M15")
    print(f" Bar Time:      {candle_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Signal Type:   {sig_type.upper()}")
    print(f" FVG Zone:      {fvg_bot:.5f} - {fvg_top:.5f}")
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
    print(f"   ICT ORDER BLOCK & FVG FOREX SIGNAL SCANNER ({ACTIVE_SYMBOL})   ")
    print("=" * 70)
    print(" Strategy Logic: H4 Order Block Tap -> M15 FVG Imbalance -> Mitigation Retest")
    print(" Status: Scanning live bars... Press Ctrl+C to stop.\n")

    last_processed_candle_time = None

    try:
        while True:
            df_h4 = fetch_rates(TIMEFRAME_HTF, count=60)
            df_m15 = fetch_rates(TIMEFRAME_LTF, count=80)
            
            if df_h4 is None or df_m15 is None:
                time.sleep(POLL_INTERVAL)
                continue

            bull_ob, bear_ob = find_h4_order_blocks(df_h4)
            current_candle_time = df_m15['time'].iloc[-1]
            closed_bar = df_m15.iloc[-2]       # Last completed M15 bar
            
            if last_processed_candle_time != current_candle_time:
                last_processed_candle_time = current_candle_time
                fvg_type, fvg_bot, fvg_top, invalidation = find_m15_fvg(df_m15)

                # ----------------------------------------------------
                # BUY LOGIC: Bullish OB Context + Bullish FVG Mitigation
                # ----------------------------------------------------
                if bull_ob and fvg_type == "BULLISH_FVG":
                    ob_bot, ob_top, _ = bull_ob
                    # Price reacted off H4 Bullish OB recently
                    reacted_ob = any((df_m15['low'].iloc[-6:-1] <= ob_top) & (df_m15['close'].iloc[-6:-1] >= ob_bot))
                    # M15 closed bar mitigated the FVG zone and rejected upwards
                    mitigated_fvg = (closed_bar['low'] <= fvg_top) and (closed_bar['close'] >= fvg_bot)
                    is_bull = closed_bar['close'] > closed_bar['open']

                    if reacted_ob and mitigated_fvg and is_bull:
                        entry = closed_bar['close']
                        sl = invalidation - (2.0 * PIP_SIZE)
                        risk = entry - sl
                        if risk > 0:
                            tp = entry + (risk * RISK_REWARD_RATIO)
                            emit_signal("BUY", current_candle_time, entry, sl, tp, fvg_bot, fvg_top)

                # ----------------------------------------------------
                # SELL LOGIC: Bearish OB Context + Bearish FVG Mitigation
                # ----------------------------------------------------
                if bear_ob and fvg_type == "BEARISH_FVG":
                    ob_bot, ob_top, _ = bear_ob
                    reacted_ob = any((df_m15['high'].iloc[-6:-1] >= ob_bot) & (df_m15['close'].iloc[-6:-1] <= ob_top))
                    mitigated_fvg = (closed_bar['high'] >= fvg_bot) and (closed_bar['close'] <= fvg_top)
                    is_bear = closed_bar['close'] < closed_bar['open']

                    if reacted_ob and mitigated_fvg and is_bear:
                        entry = closed_bar['close']
                        sl = invalidation + (2.0 * PIP_SIZE)
                        risk = sl - entry
                        if risk > 0:
                            tp = entry - (risk * RISK_REWARD_RATIO)
                            emit_signal("SELL", current_candle_time, entry, sl, tp, fvg_bot, fvg_top)

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n[STOP] Scanner stopped by user.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    run_scanner()
