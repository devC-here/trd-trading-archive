"""
========================================================================================
FOREX MT5 SIGNAL SCANNER: SLC SUPPLY DEMAND + STOCHASTIC STRATEGY
========================================================================================
Source: Short Video snzJ4zWZXno | 1-Year Backtest: +39.1% Return, 57.8% Win Rate
Market: Forex (EURUSD / Major Pairs)
Timeframe: Multi-Timeframe (H1 Trend Bias -> M5 Execution)
Indicators: Demand/Supply Zones, Stochastic Oscillator (14, 3, 3), 50 EMA
Logic:
  1. H1 Trend: 50 EMA on 1H chart determines macro bias (Uptrend / Downtrend)
  2. Demand/Supply Zone: Mark origin candle of aggressive momentum moves
  3. Zone Retest: Price pulls back into the unmitigated Demand (Buy) or Supply (Sell) zone
  4. Stochastic Confirmation: %K crosses above %D from below 20 (Buy) / below %D from >80 (Sell)
  5. Risk Management: SL beyond Zone boundary; TP fixed 1:2 Risk-Reward
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
TIMEFRAME_HTF = mt5.TIMEFRAME_H1  # 1-Hour Trend
TIMEFRAME_LTF = mt5.TIMEFRAME_M5  # 5-Minute Entry
STOCH_K = 14
STOCH_D = 3
STOCH_SMOOTH = 3
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


def calculate_stochastic(df: pd.DataFrame, k_period=14, d_period=3, smooth=3):
    """Calculates full Stochastic Oscillator (%K, %D)."""
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    
    raw_k = 100 * ((df['close'] - low_min) / (high_max - low_min + 1e-9))
    df['stoch_k'] = raw_k.rolling(window=smooth).mean()
    df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
    return df


def find_demand_supply_zones(df_ltf: pd.DataFrame):
    """Identifies recent demand and supply zones from explosive displacement candles."""
    demand_zones = []
    supply_zones = []
    
    n = len(df_ltf)
    for i in range(10, n - 2):
        candle = df_ltf.iloc[i]
        next_candle = df_ltf.iloc[i+1]
        
        # Bullish Displacement (Demand Base)
        if (next_candle['close'] - next_candle['open']) > (3.0 * PIP_SIZE) and (next_candle['close'] > candle['high']):
            demand_zones.append((candle['low'], candle['high'], df_ltf['time'].iloc[i]))
            
        # Bearish Displacement (Supply Base)
        if (next_candle['open'] - next_candle['close']) > (3.0 * PIP_SIZE) and (next_candle['close'] < candle['low']):
            supply_zones.append((candle['low'], candle['high'], df_ltf['time'].iloc[i]))
            
    recent_demand = demand_zones[-1] if demand_zones else None
    recent_supply = supply_zones[-1] if supply_zones else None
    return recent_demand, recent_supply


def emit_signal(sig_type: str, candle_time: datetime.datetime, entry_price: float, sl_price: float, tp_price: float, zone_bot: float, zone_top: float):
    """Prints a structured ASCII signal card to console."""
    risk_pips = abs(entry_price - sl_price) / PIP_SIZE
    reward_pips = abs(tp_price - entry_price) / PIP_SIZE
    
    print("\n" + "=" * 70)
    print(f" [ALERT] FOREX SIGNAL ALERT: SLC SUPPLY/DEMAND + STOCHASTIC ({sig_type.upper()})")
    print("=" * 70)
    print(f" Symbol:        {ACTIVE_SYMBOL} | Macro H1 -> Entry M5")
    print(f" Bar Time:      {candle_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Signal Type:   {sig_type.upper()}")
    print(f" Zone Range:    {zone_bot:.5f} - {zone_top:.5f}")
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
    print(f"   SLC SUPPLY/DEMAND FOREX SIGNAL SCANNER ({ACTIVE_SYMBOL})   ")
    print("=" * 70)
    print(" Strategy Logic: H1 Trend Alignment -> M5 Zone Mitigation -> Stochastic Reversal Cross")
    print(" Status: Scanning live bars... Press Ctrl+C to stop.\n")

    last_processed_candle_time = None

    try:
        while True:
            df_h1 = fetch_rates(TIMEFRAME_HTF, count=60)
            df_m5 = fetch_rates(TIMEFRAME_LTF, count=80)
            
            if df_h1 is None or df_m5 is None:
                time.sleep(POLL_INTERVAL)
                continue

            df_h1['ema50'] = df_h1['close'].ewm(span=50, adjust=False).mean()
            df_m5 = calculate_stochastic(df_m5, k_period=STOCH_K, d_period=STOCH_D, smooth=STOCH_SMOOTH)
            
            recent_demand, recent_supply = find_demand_supply_zones(df_m5)
            
            current_candle_time = df_m5['time'].iloc[-1]
            closed_bar = df_m5.iloc[-2]       # Last completed M5 bar
            prev_bar = df_m5.iloc[-3]         # Prior M5 bar
            h1_bar = df_h1.iloc[-2]           # Completed H1 bar
            
            if last_processed_candle_time != current_candle_time:
                last_processed_candle_time = current_candle_time

                # ----------------------------------------------------
                # BUY LOGIC: H1 Uptrend + Demand Zone + Stoch Cross Up < 25
                # ----------------------------------------------------
                h1_uptrend = h1_bar['close'] > h1_bar['ema50']
                if h1_uptrend and recent_demand:
                    zone_bot, zone_top, _ = recent_demand
                    # Price pierced into demand zone
                    in_demand = (closed_bar['low'] <= zone_top) and (closed_bar['close'] >= zone_bot)
                    # Stochastic crossed up from oversold territory
                    stoch_cross_up = (prev_bar['stoch_k'] <= prev_bar['stoch_d']) and \
                                     (closed_bar['stoch_k'] > closed_bar['stoch_d']) and \
                                     (prev_bar['stoch_k'] < 25.0)

                    if in_demand and stoch_cross_up:
                        entry = closed_bar['close']
                        sl = zone_bot - (2.0 * PIP_SIZE)
                        risk = entry - sl
                        if risk > 0:
                            tp = entry + (risk * RISK_REWARD_RATIO)
                            emit_signal("BUY", current_candle_time, entry, sl, tp, zone_bot, zone_top)

                # ----------------------------------------------------
                # SELL LOGIC: H1 Downtrend + Supply Zone + Stoch Cross Down > 75
                # ----------------------------------------------------
                h1_downtrend = h1_bar['close'] < h1_bar['ema50']
                if h1_downtrend and recent_supply:
                    zone_bot, zone_top, _ = recent_supply
                    in_supply = (closed_bar['high'] >= zone_bot) and (closed_bar['close'] <= zone_top)
                    stoch_cross_down = (prev_bar['stoch_k'] >= prev_bar['stoch_d']) and \
                                       (closed_bar['stoch_k'] < closed_bar['stoch_d']) and \
                                       (prev_bar['stoch_k'] > 75.0)

                    if in_supply and stoch_cross_down:
                        entry = closed_bar['close']
                        sl = zone_top + (2.0 * PIP_SIZE)
                        risk = sl - entry
                        if risk > 0:
                            tp = entry - (risk * RISK_REWARD_RATIO)
                            emit_signal("SELL", current_candle_time, entry, sl, tp, zone_bot, zone_top)

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n[STOP] Scanner stopped by user.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    run_scanner()
