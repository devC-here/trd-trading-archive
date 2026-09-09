"""
========================================================================================
UNIFIED MT5 STRATEGY RUNNER (SIGNAL-ONLY - 70 VERIFIED STRATEGIES)
========================================================================================
Master launcher for all 70 MT5 Signal Scanners derived from the creator's 215+ tested videos.
All scanners are strictly SIGNAL ONLY (calculating Entry, SL, TP, Risk/Reward).
No automated trade placement is executed.

Available Scanners:
  [A] Institutional Market Structure, SMC & Liquidity:
    1.  ict_breaker          - ICT Market Structure Shift & Breaker Block Mitigation (M15)
    2.  ict_order_block      - H4 Institutional Order Block + M15 FVG Retest (H4/M15)
    3.  tg_capital_london    - TG Capital London Killzone EMA Stack + FVG 50% Eq (M30)
    4.  william_fractal      - Neeraj Joshi 4H Fractal Sweep Reversal (H4/M5)
    5.  slc_supply_demand    - SLC Supply/Demand Zones + Stochastic %K/%D (H1/M5)
    6.  fibonacci_golden     - The Trading School 0.618 - 0.786 Golden Phantom Zone (M15)
    7.  stockburner_tl       - Stock Burner H1 Trend + M15 Pullback Trendline Break (M15)
    8.  smart_money_gold     - Smart Money H1 Demand/Supply Zone + 8 EMA Cross (M15)

  [B] Moving Average & Trend Following Systems:
    9.  833_ema              - Wizard Trader 8/33 EMA + Dominance Breakout (M5)
    10. 20_200_ema           - 20/200 Dual EMA Macro Trend + 20 EMA Pullback (M5)
    11. 50_ema_rejection     - Ezekiel Chew (Prop Firm) 50 EMA Dynamic Rejection (H1)
    12. subasish_5_ema       - Subasish Pani 5 EMA Scalping Strategy (M5/M15)
    13. 10_ema_anant_ladha   - Anant Ladha 10 EMA Floating Alert Candle Breakout (M5)
    14. rayner_inside_bar    - Rayner Teo 10 EMA + Inside Bar Momentum Breakout (H1)
    15. 8_20_ema_scalp       - 8/20 Dual EMA Value-Zone Scalping Strategy (M5)

  [C] Oscillators, Reversals & Mean Reversion:
    16. stoch_200_ema        - 200 EMA + Stochastic Oversold/Overbought Pullback (M15)
    17. rayner_teo_sma_rsi   - Rayner Teo 200 SMA + 10 RSI Swing Pullback (H1)
    18. macd_momentum        - Multi-Timeframe MACD 62% Win Rate Momentum (M15)
    19. rsi_pullback         - 50 EMA Trend Filter + RSI 30/70 Pullback Reversal (M5)
    20. subasish_bollinger   - Subasish Pani (Power of Stocks) Bollinger Band (M15)
    21. tanisha_garg_cpr_rsi - Tanisha Garg Camarilla Pivots (S2/S4) + 21 RSI (M5)

  [D] Mathematical Pivots, Volatility & Breakout Systems:
    22. camarilla_pivot      - Camarilla Daily Pivot Points Reversal & Breakout (M5)
    23. cpr_pivot_multi_tf   - Gomathi Shankar Central Pivot Range + R1/PDH Break (M5)
    24. supertrend_swing     - Supertrend (7, 2.5) + 200 EMA Baseline Polarity Flip (M15)
    25. ut_bot_alert         - UT Bot Alert ATR Trailing + 200 EMA Baseline (M15)
    26. mambafx_sr_sma       - Mamba FX Forex H4 S/R + 8/50 SMA Scalper (M15)
    27. traffic_light        - Subasish Pani 'Traffic Light' 2-Candle Breakout (M5)

  [E] Advanced Price Action & Institutional Supply/Demand Systems:
    28. chandelier_zlsma     - Chandelier Exit (22, 3.0) + 32 ZLSMA Trend Rider (H1)
    29. kojo_demand_supply   - Kojo Forex Dubai Pure Price Action Demand/Supply (M15)
    30. trading_school_mbp   - The Trading School Secret MBP Session Setup (H1/M5)
    31. trade_with_pat       - Trade With Pat A+ Supply & Demand + FVG Sniper (M15)
    32. smc_liquidity_sniper - SMC Liquidity Sniper 71% Fib Retracement + FVG (H4/M15)
    33. double_bo_retest     - Pure Price Action Double Breakout Retest Strategy (M15)
    34. asia_forex_mentor    - Asia Forex Mentor Ezekiel Chew Smoothed Heikin Ashi (H1)
    35. bharat_rsi_bb        - Bharat Jhunjhunwala RSI 60/40 Momentum + Bollinger Band (H1)

  [F] Precision Momentum, Pullback & Retracement Systems:
    36. berlin_7_ema         - Trade Like Berlin 'Final Boss' 7 EMA System (H1/M15)
    37. ezekiel_50_ema_pb    - Ezekiel Chew 50 EMA Pullback Engulfing Trigger (H1)
    38. trading_geek_sr      - The Trading Geek (Brad Goh) S/R Exhaustion Price Action (H4/H1)
    39. tori_trendline       - Tori Trades Multi-Touch Structural Trendline Breakout (M15)
    40. priyank_21_ema       - Hold with Priyank 21 EMA Pullback Breakout Strategy (M15)
    41. jenfer_impulse       - Jenfer 'One-Candle' 50-75% Equilibrium Scalping (M5)
    42. trading_lab_dema     - Trading Lab 200 DEMA + Supertrend Algorithmic Scanner (M15)

  [G] Multi-Confluence, Renko, Traps & Trend Reclaim Systems:
    43. half_trend_atr       - HalfTrend (2, 2.0) ATR Dynamic Channel Breakout (M15)
    44. umer_sandwich        - Umer Akhtar 3-Candle Doji Sandwich Reversal (M15/H1)
    45. 3tf_macd_momentum    - 3-Timeframe MACD Momentum Confluence (D1/H4/H1)
    46. ict_double_sweep     - ICT Double Sweep Sniper Entry Strategy (H1/M15)
    47. ict_daily_ob_fvg     - MS Trading ICT Daily OB + H1 MSS/FVG Retest (D1/H1)
    48. tg_ema_stack_doji    - TG Capital (Tyler Gottle) EMA Stack + FVG 50% Doji (M30)
    49. rayner_50_ema_swing  - Rayner Teo 'Catch The Wave' 50 EMA Pullback Swing (D1)
    50. ema_rsi_reclaim      - 50 EMA Trend + RSI 30/70 Reclaim Confirmation (H1)
    51. booming_halftrend    - Booming Bulls (Anish Singh) HalfTrend + DMI Intraday (M15)
    52. priyank_weekly_trap  - Priyank Sharma MTF Weekly Trap + M15 MSS (W1/D1/M15)
    53. paradise_redbox      - Traders Paradise 'Red Box' S/R Fakeout Reversal (M15)
    54. renko_10_sma_obv     - Renko Brick Trend + 10 SMA + OBV Alignment (Renko/M15)

  [H] Institutional Imbalance, Pro Breakouts & Trend Confirmation Systems:
    55. umar_gold_london     - Umar Punjabi Gold London Session 1st Hour Killzone (M15/M5)
    56. bill_alligator       - Bill Williams Alligator Momentum & Teeth Cross (H1)
    57. ict_po3_amd          - ICT PO3 (Accumulation Manipulation Distribution) / CRT (M15/M5)
    58. oliver_elephant_180  - Oliver Velez Bull/Bear 180 Elephant Candle Reversal (M5)
    59. mov_avg_fib_scalp    - The Moving Average Trendline + Fib Golden Zone Scalper (M5)
    60. jesse_smc_fvg        - Jesse Rogers SMC FVG Sniper Scalping Scanner (M5)
    61. theta_ema_rsi        - Theta Gainers 50/200 EMA + RSI 40-55 Value Pullback (M15)
    62. gold_smc_imbalance   - Gold (XAUUSD) SMC Institutional Imbalance Zone (M30)
    63. dual_ema_stoch_rsi   - 50/200 Dual EMA + Stochastic RSI Momentum Scalper (M5)
    64. smc_breaker_v2       - SMC Failed Order Block Breaker Retest Scanner (M15)
    65. steven_hart_4h       - Steven Hart 4H Market Structure + H1 Pullback Breakout (H4/H1)
    66. bhanushali_44_ma     - Siddharth Bhanushali 44 Moving Average Dynamic Swing (D1)
    67. mukul_10_ema_vwap    - Mukul Chowdhary 10 EMA + Session VWAP 1:4 RR System (M15)
    68. luxalgo_confirm      - LuxAlgo Confirmation Trend & Baseline Momentum (H1)
    69. uk_break_bounce      - UK 20-Year Pro Trader Break and Bounce Scanner (M15)
    70. pb_multi_tf_fvg      - PB Trading Multi-Timeframe H1 + M5 FVG Confluence (H1/M5)
========================================================================================
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import time
import argparse
import subprocess
import threading
from typing import List

SCANNERS = {
    # Group A: Institutional Market Structure, SMC & Liquidity
    "1":  ("ict_breaker", "forex_ict_breaker_block_scanner.py", "ICT Market Structure Shift & Breaker Block (M15)"),
    "2":  ("ict_order_block", "forex_ict_order_block_scanner.py", "H4 Institutional Order Block + M15 FVG Retest (H4/M15)"),
    "3":  ("tg_capital_london", "indicator_tg_capital_london_fvg_scanner.py", "TG Capital London Killzone EMA Stack + FVG 50% Eq (M30)"),
    "4":  ("william_fractal", "forex_william_fractal_scanner.py", "Neeraj Joshi 4H Fractal Sweep Reversal (H4/M5)"),
    "5":  ("slc_supply_demand", "forex_slc_supply_demand_scanner.py", "SLC Supply/Demand Zones + Stochastic %K/%D (H1/M5)"),
    "6":  ("fibonacci_golden", "indicator_fibonacci_golden_zone_scanner.py", "The Trading School 0.618 - 0.786 Golden Phantom Zone (M15)"),
    "7":  ("stockburner_tl", "indicator_stockburner_trendline_scanner.py", "Stock Burner H1 Trend + M15 Pullback Trendline Break (M15)"),
    "8":  ("smart_money_gold", "indicator_smart_money_gold_scanner.py", "Smart Money H1 Demand/Supply Zone + 8 EMA Cross (M15)"),

    # Group B: Moving Average & Trend Following Systems
    "9":  ("833_ema", "forex_833_ema_scanner.py", "Wizard Trader 8/33 EMA + Dominance Breakout (M5)"),
    "10": ("20_200_ema", "forex_20_200_ema_scanner.py", "20/200 Dual EMA Macro Trend + 20 EMA Pullback (M5)"),
    "11": ("50_ema_rejection", "indicator_50_ema_rejection_scanner.py", "Ezekiel Chew (Prop Firm) 50 EMA Dynamic Rejection (H1)"),
    "12": ("subasish_5_ema", "indicator_subasish_5_ema_scanner.py", "Subasish Pani 5 EMA Scalping Strategy (M5/M15)"),
    "13": ("10_ema_anant_ladha", "indicator_10_ema_anant_ladha_scanner.py", "Anant Ladha 10 EMA Floating Alert Candle Breakout (M5)"),
    "14": ("rayner_inside_bar", "indicator_rayner_teo_inside_bar_scanner.py", "Rayner Teo 10 EMA + Inside Bar Momentum Breakout (H1)"),
    "15": ("8_20_ema_scalp", "indicator_8_20_ema_scalp_scanner.py", "8/20 Dual EMA Value-Zone Scalping Strategy (M5)"),

    # Group C: Oscillators, Reversals & Mean Reversion
    "16": ("stoch_200_ema", "indicator_stochastic_200_ema_scanner.py", "200 EMA + Stochastic Oversold/Overbought Pullback (M15)"),
    "17": ("rayner_teo_sma_rsi", "indicator_rayner_teo_sma_rsi_scanner.py", "Rayner Teo 200 SMA + 10 RSI Swing Pullback (H1)"),
    "18": ("macd_momentum", "indicator_macd_momentum_scanner.py", "Multi-Timeframe MACD 62% Win Rate Momentum (M15)"),
    "19": ("rsi_pullback", "forex_rsi_pullback_scanner.py", "50 EMA Trend Filter + RSI 30/70 Pullback Reversal (M5)"),
    "20": ("subasish_bollinger", "indicator_subasish_bollinger_scanner.py", "Subasish Pani (Power of Stocks) Bollinger Band (M15)"),
    "21": ("tanisha_garg_cpr_rsi", "indicator_tanisha_garg_cpr_rsi_scanner.py", "Tanisha Garg Camarilla Pivots (S2/S4) + 21 RSI (M5)"),

    # Group D: Mathematical Pivots, Volatility & Breakout Systems
    "22": ("camarilla_pivot", "indicator_camarilla_pivot_scanner.py", "Camarilla Daily Pivot Points Reversal & Breakout (M5)"),
    "23": ("cpr_pivot_multi_tf", "indicator_cpr_pivot_multi_tf_scanner.py", "Gomathi Shankar Central Pivot Range + R1/PDH Break (M5)"),
    "24": ("supertrend_swing", "forex_supertrend_swing_scanner.py", "Supertrend (7, 2.5) + 200 EMA Baseline Polarity Flip (M15)"),
    "25": ("ut_bot_alert", "indicator_ut_bot_alert_scanner.py", "UT Bot Alert ATR Trailing + 200 EMA Baseline (M15)"),
    "26": ("mambafx_sr_sma", "indicator_mambafx_support_resistance_scanner.py", "Mamba FX Forex H4 S/R + 8/50 SMA Scalper (M15)"),
    "27": ("traffic_light", "indicator_traffic_light_breakout_scanner.py", "Subasish Pani 'Traffic Light' 2-Candle Breakout (M5)"),

    # Group E: Advanced Price Action & Institutional Supply/Demand Systems
    "28": ("chandelier_zlsma", "indicator_chandelier_exit_zlsma_scanner.py", "Chandelier Exit (22, 3.0) + 32 ZLSMA Trend Rider (H1)"),
    "29": ("kojo_demand_supply", "forex_kojo_price_action_demand_scanner.py", "Kojo Forex Dubai Pure Price Action Demand/Supply (M15)"),
    "30": ("trading_school_mbp", "forex_trading_school_mbp_gold_scanner.py", "The Trading School Secret MBP Session Setup (H1/M5)"),
    "31": ("trade_with_pat", "forex_trade_with_pat_demand_fvg_scanner.py", "Trade With Pat A+ Supply & Demand + FVG Sniper (M15)"),
    "32": ("smc_liquidity_sniper", "forex_smc_liquidity_sniper_71fib_scanner.py", "SMC Liquidity Sniper 71% Fib Retracement + FVG (H4/M15)"),
    "33": ("double_bo_retest", "indicator_double_breakout_retest_scanner.py", "Pure Price Action Double Breakout Retest Strategy (M15)"),
    "34": ("asia_forex_mentor", "forex_asia_forex_mentor_heikin_ashi_scanner.py", "Asia Forex Mentor Smoothed Heikin Ashi (H1)"),
    "35": ("bharat_rsi_bb", "indicator_bharat_jhunjhunwala_rsi_bb_scanner.py", "Bharat Jhunjhunwala RSI 60/40 Momentum + BB (H1)"),

    # Group F: Precision Momentum, Pullback & Retracement Systems
    "36": ("berlin_7_ema", "forex_trade_like_berlin_7_ema_scanner.py", "Trade Like Berlin 'Final Boss' 7 EMA System (H1/M15)"),
    "37": ("ezekiel_50_ema_pb", "forex_ezekiel_chew_50_ema_pullback_scanner.py", "Ezekiel Chew 50 EMA Pullback Engulfing Trigger (H1)"),
    "38": ("trading_geek_sr", "forex_trading_geek_exhaustion_scanner.py", "The Trading Geek (Brad Goh) S/R Exhaustion Price Action (H4/H1)"),
    "39": ("tori_trendline", "indicator_tori_trades_trendline_scanner.py", "Tori Trades Multi-Touch Structural Trendline Breakout (M15)"),
    "40": ("priyank_21_ema", "indicator_hold_with_priyank_21_ema_scanner.py", "Hold with Priyank 21 EMA Pullback Breakout Strategy (M15)"),
    "41": ("jenfer_impulse", "forex_jenfer_impulse_scalp_scanner.py", "Jenfer 'One-Candle' 50-75% Equilibrium Scalping (M5)"),
    "42": ("trading_lab_dema", "indicator_trading_lab_dema_supertrend_scanner.py", "Trading Lab 200 DEMA + Supertrend Algorithmic Scanner (M15)"),

    # Group G: Multi-Confluence, Renko, Traps & Trend Reclaim Systems
    "43": ("half_trend_atr", "indicator_half_trend_atr_scanner.py", "HalfTrend (2, 2.0) ATR Dynamic Channel Breakout (M15)"),
    "44": ("umer_sandwich", "forex_umer_akhtar_sandwich_scanner.py", "Umer Akhtar 3-Candle Doji Sandwich Reversal (M15/H1)"),
    "45": ("3tf_macd_momentum", "indicator_3tf_macd_momentum_scanner.py", "3-Timeframe MACD Momentum Confluence (D1/H4/H1)"),
    "46": ("ict_double_sweep", "forex_ict_double_sweep_sniper_scanner.py", "ICT Double Sweep Sniper Entry Strategy (H1/M15)"),
    "47": ("ict_daily_ob_fvg", "forex_ict_daily_ob_h1_fvg_scanner.py", "MS Trading ICT Daily OB + H1 MSS/FVG Retest (D1/H1)"),
    "48": ("tg_ema_stack_doji", "forex_tg_capital_ema_stack_doji_scanner.py", "TG Capital (Tyler Gottle) EMA Stack + FVG 50% Doji (M30)"),
    "49": ("rayner_50_ema_swing", "forex_rayner_teo_50_ema_swing_scanner.py", "Rayner Teo 'Catch The Wave' 50 EMA Pullback Swing (D1)"),
    "50": ("ema_rsi_reclaim", "forex_50_ema_rsi_reclaim_scanner.py", "50 EMA Trend + RSI 30/70 Reclaim Confirmation (H1)"),
    "51": ("booming_halftrend", "forex_booming_bulls_halftrend_dmi_scanner.py", "Booming Bulls (Anish Singh) HalfTrend + DMI Intraday (M15)"),
    "52": ("priyank_weekly_trap", "forex_priyank_weekly_trap_mss_scanner.py", "Priyank Sharma MTF Weekly Trap + M15 MSS (W1/D1/M15)"),
    "53": ("paradise_redbox", "forex_traders_paradise_redbox_scanner.py", "Traders Paradise 'Red Box' S/R Fakeout Reversal (M15)"),
    "54": ("renko_10_sma_obv", "indicator_renko_brick_trend_scanner.py", "Renko Brick Trend + 10 SMA + OBV Alignment (Renko/M15)"),

    # Group H: Institutional Imbalance, Pro Breakouts & Trend Confirmation Systems
    "55": ("umar_gold_london", "forex_umar_punjabi_gold_london_scanner.py", "Umar Punjabi Gold London Session 1st Hour Killzone (M15/M5)"),
    "56": ("bill_alligator", "indicator_bill_williams_alligator_scanner.py", "Bill Williams Alligator Momentum & Teeth Cross (H1)"),
    "57": ("ict_po3_amd", "forex_ict_po3_amd_framework_scanner.py", "ICT PO3 (Accumulation Manipulation Distribution) / CRT (M15/M5)"),
    "58": ("oliver_elephant_180", "forex_oliver_velez_elephant_180_scanner.py", "Oliver Velez Bull/Bear 180 Elephant Candle Reversal (M5)"),
    "59": ("mov_avg_fib_scalp", "forex_the_moving_average_fib_scalp_scanner.py", "The Moving Average Trendline + Fib Golden Zone Scalper (M5)"),
    "60": ("jesse_smc_fvg", "forex_jesse_rogers_smc_fvg_scanner.py", "Jesse Rogers SMC FVG Sniper Scalping Scanner (M5)"),
    "61": ("theta_ema_rsi", "forex_theta_gainers_ema_rsi_scanner.py", "Theta Gainers 50/200 EMA + RSI 40-55 Value Pullback (M15)"),
    "62": ("gold_smc_imbalance", "forex_gold_smc_imbalance_zone_scanner.py", "Gold (XAUUSD) SMC Institutional Imbalance Zone (M30)"),
    "63": ("dual_ema_stoch_rsi", "forex_dual_ema_stoch_rsi_scanner.py", "50/200 Dual EMA + Stochastic RSI Momentum Scalper (M5)"),
    "64": ("smc_breaker_v2", "forex_smc_breaker_block_v2_scanner.py", "SMC Failed Order Block Breaker Retest Scanner (M15)"),
    "65": ("steven_hart_4h", "forex_steven_hart_4h_trend_pullback_scanner.py", "Steven Hart 4H Market Structure + H1 Pullback Breakout (H4/H1)"),
    "66": ("bhanushali_44_ma", "forex_bhanushali_44_ma_rejection_scanner.py", "Siddharth Bhanushali 44 Moving Average Dynamic Swing (D1)"),
    "67": ("mukul_10_ema_vwap", "forex_mukul_chowdhary_10_ema_vwap_scanner.py", "Mukul Chowdhary 10 EMA + Session VWAP 1:4 RR System (M15)"),
    "68": ("luxalgo_confirm", "indicator_luxalgo_confirmation_trend_scanner.py", "LuxAlgo Confirmation Trend & Baseline Momentum (H1)"),
    "69": ("uk_break_bounce", "forex_uk_trader_break_and_bounce_scanner.py", "UK 20-Year Pro Trader Break and Bounce Scanner (M15)"),
    "70": ("pb_multi_tf_fvg", "forex_pb_trading_h1_m5_fvg_scanner.py", "PB Trading Multi-Timeframe H1 + M5 FVG Confluence (H1/M5)"),
}


def test_mt5_connection():
    """Verifies MT5 connection and active symbol resolution."""
    print("=" * 75)
    print(" [TEST] TESTING MT5 TERMINAL CONNECTION & SYMBOL DISCOVERY")
    print("=" * 75)
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            print(f"[-] MT5 Terminal initialization failed: {mt5.last_error()}")
            print("   Please ensure MetaTrader 5 desktop app is running and logged in.")
            return False
        
        terminal_info = mt5.terminal_info()
        account_info = mt5.account_info()
        print(f"[+] Connected to MT5 Terminal Build: {terminal_info.build if terminal_info else 'Unknown'}")
        if account_info:
            print(f"   Account: {account_info.login} ({account_info.server}) | Currency: {account_info.currency}")
        
        candidates = ["EURUSD.0", "EURUSD", "EURUSDm", "EURUSD.pro", "GBPUSD", "XAUUSD"]
        found_symbol = None
        for s in candidates:
            info = mt5.symbol_info(s)
            if info is not None:
                found_symbol = s
                digits = info.digits
                pip_size = 0.0001 if digits in (4, 5) else 0.01
                print(f"   Matched Active Symbol: {s} (Digits: {digits}, Pip Size: {pip_size})")
                break
        
        if not found_symbol:
            print("   [!] Note: No standard default symbol found automatically; custom broker symbol names can be set.")
        
        mt5.shutdown()
        return True
    except ImportError:
        print("[-] MetaTrader5 python library is not installed in current environment.")
        return False
    except Exception as e:
        print(f"[-] Unexpected error testing MT5: {e}")
        return False


def run_single_script(script_name: str):
    """Runs a single scanner script in foreground."""
    print(f"\n[LAUNCH] Launching {script_name}...")
    try:
        subprocess.run([sys.executable, script_name], check=True)
    except KeyboardInterrupt:
        print(f"\n[STOP] Stopped {script_name}")


def stream_process(proc, name):
    for line in iter(proc.stdout.readline, ''):
        if line:
            print(f"[{name}] {line}", end="")
    proc.stdout.close()


def run_all_scanners():
    """Launches all strategy scanners concurrently with prefixed console output."""
    print("=" * 75)
    print(" [LAUNCH] LAUNCHING ALL 70 STRATEGY SCANNERS CONCURRENTLY")
    print("=" * 75)
    print(" Press Ctrl+C at any time to shut down all scanners.\n")

    processes: List[subprocess.Popen] = []
    threads: List[threading.Thread] = []

    try:
        for key, (short_name, script_path, desc) in SCANNERS.items():
            print(f" [WAIT] Starting [{short_name.upper()}]: {desc}")
            proc = subprocess.Popen(
                [sys.executable, "-u", script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            processes.append(proc)
            t = threading.Thread(target=stream_process, args=(proc, short_name.upper()), daemon=True)
            t.start()
            threads.append(t)
            time.sleep(0.20)

        print("\n[+] All 70 scanners running. Awaiting market signals...\n")
        while any(p.poll() is None for p in processes):
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[STOP] Shutting down all background scanners...")
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        for proc in processes:
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        print("[+] All scanners cleanly terminated.")


def show_menu():
    print("\n" + "=" * 80)
    print("                 MT5 SIGNAL SCANNERS (70 STRATEGIES)")
    print("=" * 80)
    print("  --- [A] Institutional Market Structure & SMC ---")
    for k in ["1", "2", "3", "4", "5", "6", "7", "8"]:
        print(f"  [{k:>2}] {SCANNERS[k][2]}")
    
    print("\n  --- [B] Moving Averages & Trend Systems ---")
    for k in ["9", "10", "11", "12", "13", "14", "15"]:
        print(f"  [{k:>2}] {SCANNERS[k][2]}")

    print("\n  --- [C] Oscillators, Reversals & Mean Reversion ---")
    for k in ["16", "17", "18", "19", "20", "21"]:
        print(f"  [{k:>2}] {SCANNERS[k][2]}")

    print("\n  --- [D] Mathematical Pivots, Volatility & Breakout Systems ---")
    for k in ["22", "23", "24", "25", "26", "27"]:
        print(f"  [{k:>2}] {SCANNERS[k][2]}")

    print("\n  --- [E] Advanced Price Action & Institutional Supply/Demand Systems ---")
    for k in ["28", "29", "30", "31", "32", "33", "34", "35"]:
        print(f"  [{k:>2}] {SCANNERS[k][2]}")

    print("\n  --- [F] Precision Momentum, Pullback & Retracement Systems ---")
    for k in ["36", "37", "38", "39", "40", "41", "42"]:
        print(f"  [{k:>2}] {SCANNERS[k][2]}")

    print("\n  --- [G] Multi-Confluence, Renko, Traps & Trend Reclaim Systems ---")
    for k in ["43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54"]:
        print(f"  [{k:>2}] {SCANNERS[k][2]}")

    print("\n  --- [H] Institutional Imbalance, Pro Breakouts & Trend Confirmation Systems ---")
    for k in ["55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70"]:
        print(f"  [{k:>2}] {SCANNERS[k][2]}")

    print("\n  [ALL] Run ALL 70 Scanners Concurrently (Multi-Threaded)")
    print("  [C]   Test MT5 Terminal Connection & Symbol Discovery")
    print("  [Q]   Exit")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Unified MT5 Signal Scanner Launcher")
    parser.add_argument("--all", action="store_true", help="Run all 70 scanners concurrently")
    parser.add_argument("--list", action="store_true", help="Print table of all 70 scanners and exit")
    parser.add_argument("--check", action="store_true", help="Check MT5 terminal connection and exit")
    parser.add_argument("--scanner", type=str, choices=[s[0] for s in SCANNERS.values()], help="Run specific scanner by code name")
    args = parser.parse_args()

    if args.list:
        show_menu()
        return

    if args.check:
        test_mt5_connection()
        return

    if args.all:
        run_all_scanners()
        return

    if args.scanner:
        for key, (s_name, script_path, desc) in SCANNERS.items():
            if s_name == args.scanner:
                run_single_script(script_path)
                return

    # Interactive menu
    while True:
        show_menu()
        choice = input("Select an option (1-70, ALL, C, Q): ").strip().upper()
        if choice in SCANNERS:
            run_single_script(SCANNERS[choice][1])
        elif choice == "ALL":
            run_all_scanners()
        elif choice == "C":
            test_mt5_connection()
        elif choice == "Q":
            print("Exiting. Happy trading!")
            break
        else:
            print("[-] Invalid option. Please select 1-70, ALL, C, or Q.")

if __name__ == "__main__":
    main()
