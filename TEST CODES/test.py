import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime

# --- Settings ---
# CHECK YOUR BROKER: Some use "EURUSD", others use "EURUSD.m" or "EURUSD.ecn"
LEAD_SYMBOL = "EURUSD.e" 
LAG_SYMBOL = "XAUUSD.w"
TIMEFRAME = mt5.TIMEFRAME_M15
EMA_PERIOD = 10
LOOKBACK_WINDOW = 3 
BACKTEST_BARS = 1000 

def mt5_style_ema(closes, period=10):
    if len(closes) == 0: return np.array([])
    alpha = 2 / (period + 1)
    ema = np.zeros(len(closes))
    ema[0] = closes[0]
    for i in range(1, len(closes)):
        ema[i] = ema[i - 1] + alpha * (closes[i] - ema[i - 1])
    return ema

def run_backtest():
    if not mt5.initialize():
        print("MT5 Initialization Failed")
        return

    # 1. Fetch data and check for errors
    eur_rates = mt5.copy_rates_from_pos(LEAD_SYMBOL, TIMEFRAME, 0, BACKTEST_BARS)
    gold_rates = mt5.copy_rates_from_pos(LAG_SYMBOL, TIMEFRAME, 0, BACKTEST_BARS)

    if eur_rates is None or len(eur_rates) == 0:
        print(f"❌ Error: Could not get data for {LEAD_SYMBOL}. Check the symbol name.")
        mt5.shutdown()
        return
    
    if gold_rates is None or len(gold_rates) == 0:
        print(f"❌ Error: Could not get data for {LAG_SYMBOL}. Check the symbol name.")
        mt5.shutdown()
        return

    df_eur = pd.DataFrame(eur_rates)
    df_gold = pd.DataFrame(gold_rates)

    # 2. Calculate EMAs
    eur_ema = mt5_style_ema(df_eur['close'].values, EMA_PERIOD)
    gold_ema = mt5_style_ema(df_gold['close'].values, EMA_PERIOD)

    total_signals = 0
    success_moves = 0 # To track if Gold actually followed the direction
    
    print(f"--- Backtesting {len(df_gold)} bars on {TIMEFRAME} ---")

    # 3. Loop through history
    for i in range(LOOKBACK_WINDOW + 1, len(df_gold)):
        # Gold Crossover at index 'i' (Current)
        gold_crossed_up = (df_gold['close'][i-1] < gold_ema[i-1]) and (df_gold['close'][i] > gold_ema[i])
        gold_crossed_down = (df_gold['close'][i-1] > gold_ema[i-1]) and (df_gold['close'][i] < gold_ema[i])

        if gold_crossed_up or gold_crossed_down:
            eur_led = False
            # Check window: Did EURUSD cross in the bars leading up to this?
            for j in range(1, LOOKBACK_WINDOW + 1):
                idx = i - j
                if gold_crossed_up:
                    if (df_eur['close'][idx-1] < eur_ema[idx-1]) and (df_eur['close'][idx] > eur_ema[idx]):
                        eur_led = True
                elif gold_crossed_down:
                    if (df_eur['close'][idx-1] > eur_ema[idx-1]) and (df_eur['close'][idx] < eur_ema[idx]):
                        eur_led = True
            
            if eur_led:
                total_signals += 1
                sig_time = datetime.fromtimestamp(df_gold['time'][i])
                
                # Check if price was higher/lower 3 bars LATER (Simple result check)
                future_idx = min(i + 3, len(df_gold) - 1)
                is_profitable = False
                if gold_crossed_up and df_gold['close'][future_idx] > df_gold['close'][i]:
                    is_profitable = True
                elif gold_crossed_down and df_gold['close'][future_idx] < df_gold['close'][i]:
                    is_profitable = True
                
                if is_profitable: success_moves += 1
                
                status = "✅ WIN" if is_profitable else "❌ FAIL"
                print(f"[{sig_time}] EUR Lead -> Gold {'UP' if gold_crossed_up else 'DOWN'} | Result: {status}")

    print(f"\n--- Backtest Results ---")
    print(f"Total Signals: {total_signals}")
    if total_signals > 0:
        print(f"Success Rate (3 bars later): {(success_moves/total_signals)*100:.2f}%")
    
    mt5.shutdown()

if __name__ == "__main__":
    run_backtest()