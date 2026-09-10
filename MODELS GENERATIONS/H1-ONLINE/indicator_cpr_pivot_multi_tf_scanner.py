"""
========================================================================================
MT5 SIGNAL SCANNER: GOMATHI SHANKAR MULTI-TIMEFRAME CPR PIVOT STRATEGY
========================================================================================
Source: Short Video i5eNsT9GFDM | 1-Year Backtest: +28.4% Return, 54% Win Rate, 18.6% DD
Creator Attribution: Gomathi Shankar (Pioneer of CPR / Central Pivot Range)
Market: Forex (EURUSD, GBPUSD) / Universal MT5
Timeframe: M5 Execution with Daily CPR Derivation
Indicators:
  - Central Pivot Range (CPR):
      Pivot (P) = (High + Low + Close) / 3
      Bottom Central (BC) = (High + Low) / 2
      Top Central (TC) = (P - BC) + P
  - S/R Levels:
      R1 = (2 * P) - Low | S1 = (2 * P) - High
      R2 = P + (High - Low) | S2 = P - (High - Low)
      PDH (Previous Day High), PDL (Previous Day Low)
Logic:
  - Buy Setup:
      1. Regime: Price is above Daily CPR (Bullish Bias).
      2. Trigger: An M5 candle breaks and closes decisively ABOVE BOTH R1 and PDH.
      3. Stop Loss: Placed below R1 / PDH.
      4. Take Profit: R2 Level or 1:2 Risk-to-Reward Ratio.
  - Sell Setup:
      1. Regime: Price is below Daily CPR (Bearish Bias).
      2. Trigger: An M5 candle breaks and closes decisively BELOW BOTH S1 and PDL.
      3. Stop Loss: Placed above S1 / PDL.
      4. Take Profit: S2 Level or 1:2 Risk-to-Reward Ratio.
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
TIMEFRAME_EXEC = mt5.TIMEFRAME_M5    # 5-Minute Chart
TIMEFRAME_DAILY = mt5.TIMEFRAME_D1   # Daily for CPR levels
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


def fetch_cpr_levels():
    """Calculates CPR and pivot levels from the prior daily bar."""
    daily_rates = mt5.copy_rates_from_pos(ACTIVE_SYMBOL, TIMEFRAME_DAILY, 1, 1)
    if daily_rates is None or len(daily_rates) == 0:
        return None

    prev = daily_rates[0]
    high = prev['high']
    low = prev['low']
    close = prev['close']

    p = (high + low + close) / 3.0
    bc = (high + low) / 2.0
    tc = (p - bc) + p

    # Ensure TC is top, BC is bottom
    cpr_top = max(tc, bc)
    cpr_bot = min(tc, bc)

    r1 = (2.0 * p) - low
    s1 = (2.0 * p) - high
    r2 = p + (high - low)
    s2 = p - (high - low)

    return {
        "P": p,
        "CPR_TOP": cpr_top,
        "CPR_BOT": cpr_bot,
        "R1": r1,
        "S1": s1,
        "R2": r2,
        "S2": s2,
        "PDH": high,
        "PDL": low
    }


def fetch_rates(count=LOOKBACK_BARS):
    """Fetches M5 execution candles."""
    rates = mt5.copy_rates_from_pos(ACTIVE_SYMBOL, TIMEFRAME_EXEC, 0, count)
    if rates is None or len(rates) < 30:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


def emit_signal(sig_type: str, candle_time: datetime.datetime, entry_price: float, sl_price: float, tp_price: float, r_level: float, pd_level: float):
    """Prints a structured ASCII signal card to console."""
    risk_pips = abs(entry_price - sl_price) / PIP_SIZE
    reward_pips = abs(tp_price - entry_price) / PIP_SIZE

    print("\n" + "=" * 75)
    print(f" [ALERT] MT5 SIGNAL ALERT: GOMATHI SHANKAR CPR PIVOT ({sig_type.upper()})")
    print("=" * 75)
    print(f" Symbol:        {ACTIVE_SYMBOL} | Timeframe: M5")
    print(f" Bar Time:      {candle_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Signal Type:   {sig_type.upper()} (CPR Bias + R1/PDH Breakout)")
    print(f" Key Levels:    Pivot/S1/R1 = {r_level:.5f} | PDH/PDL = {pd_level:.5f}")
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
    print(f"   GOMATHI SHANKAR CPR PIVOT SIGNAL SCANNER ({ACTIVE_SYMBOL} M5)   ")
    print("=" * 75)
    print(" Strategy: Above/Below CPR Bias + R1/PDH Breakout & S1/PDL Breakdown")
    print(" Status: Scanning live bars... Press Ctrl+C to stop.\n")

    last_processed_candle_time = None

    try:
        while True:
            cpr = fetch_cpr_levels()
            df = fetch_rates()
            if df is None or cpr is None:
                time.sleep(POLL_INTERVAL)
                continue

            current_candle_time = df['time'].iloc[-1]
            closed_bar = df.iloc[-2]       # Trigger bar
            prev_bar = df.iloc[-3]

            if last_processed_candle_time != current_candle_time:
                last_processed_candle_time = current_candle_time

                # ----------------------------------------------------
                # BUY LOGIC: Price > CPR Top AND Breaks R1 + PDH
                # ----------------------------------------------------
                above_cpr = closed_bar['close'] > cpr['CPR_TOP']
                break_r1_pdh = (
                    (prev_bar['close'] <= max(cpr['R1'], cpr['PDH'])) and 
                    (closed_bar['close'] > max(cpr['R1'], cpr['PDH']))
                )

                if above_cpr and break_r1_pdh:
                    entry = closed_bar['close']
                    sl = max(cpr['R1'], cpr['PDH']) - (3.0 * PIP_SIZE)
                    risk = entry - sl
                    if risk > 0:
                        tp = cpr['R2'] if cpr['R2'] > entry else entry + (risk * RISK_REWARD_RATIO)
                        emit_signal("BUY", current_candle_time, entry, sl, tp, cpr['R1'], cpr['PDH'])

                # ----------------------------------------------------
                # SELL LOGIC: Price < CPR Bot AND Breaks S1 + PDL
                # ----------------------------------------------------
                below_cpr = closed_bar['close'] < cpr['CPR_BOT']
                break_s1_pdl = (
                    (prev_bar['close'] >= min(cpr['S1'], cpr['PDL'])) and 
                    (closed_bar['close'] < min(cpr['S1'], cpr['PDL']))
                )

                if below_cpr and break_s1_pdl:
                    entry = closed_bar['close']
                    sl = min(cpr['S1'], cpr['PDL']) + (3.0 * PIP_SIZE)
                    risk = sl - entry
                    if risk > 0:
                        tp = cpr['S2'] if cpr['S2'] < entry else entry - (risk * RISK_REWARD_RATIO)
                        emit_signal("SELL", current_candle_time, entry, sl, tp, cpr['S1'], cpr['PDL'])

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n[STOP] Scanner stopped by user.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    run_scanner()
