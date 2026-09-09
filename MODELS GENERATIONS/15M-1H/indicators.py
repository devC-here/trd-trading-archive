import numpy as np
import pandas as pd
import scipy.stats as stats  # Handles skewness and kurtosis calculations

# --- LOAD HIGH-RESOLUTION SOURCE DATA ---
df_m15 = pd.read_csv("ohlcv_40_m15.csv")

def calculate_padded_105_indicators(sub_block, block_open_price, block_mean_volume):
    """
    Takes a live lookback sub-block from step 1 to current t. Pads the lookback
    with synthetic neutral baseline candles if lookback length < 16 to guarantee
    a warm, non-zero starting state. Computes exactly 105 indicators.
    """
    features = {}
    
    live_n = len(sub_block)
    target_lookback = 16
    
    # Isolate underlying arrays for rapid processing
    live_o = sub_block['open'].values
    live_h = sub_block['high'].values
    live_l = sub_block['low'].values
    live_c = sub_block['close'].values
    live_v = sub_block['volume'].values
    
    # 1. ENFORCE NEUTRAL PADDING TO SOLVE COLD-START ZERO PAST PROBLEM
    if live_n < target_lookback:
        pad_size = target_lookback - live_n
        pad_o = np.full(pad_size, block_open_price)
        pad_c = np.full(pad_size, block_open_price)
        # Symmetrical mini-wicks provide baseline structural bounding variance
        pad_h = np.full(pad_size, block_open_price + 0.0001)
        pad_l = np.full(pad_size, block_open_price - 0.0001)
        pad_v = np.full(pad_size, block_mean_volume)
        
        opens = np.concatenate([pad_o, live_o])
        highs = np.concatenate([pad_h, live_h])
        lows = np.concatenate([pad_l, live_l])
        closes = np.concatenate([pad_c, live_c])
        volumes = np.concatenate([pad_v, live_v])
    else:
        opens, highs, lows, closes, volumes = live_o, live_h, live_l, live_c, live_v
        
    N = len(closes) # Hard-bounded to exactly 16 steps
    x = np.arange(N)
    
    # Core series vectors 
    log_returns = np.log(closes[1:] / closes[:-1])
    typical_prices = (highs + lows + closes) / 3.0
    money_flow = typical_prices * volumes
    prev_closes = np.roll(closes, 1); prev_closes[0] = opens[0]
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_closes), np.abs(lows - prev_closes)))
    
    # Core Exponential Moving Average (EMA) mathematical filter
    def ema(s, p):
        alpha = 2 / (p + 1); r = np.zeros_like(s); r[0] = s[0]
        for i in range(1, len(s)): r[i] = alpha * s[i] + (1 - alpha) * r[i-1]
        return r

    # Core Weighted Moving Average (WMA) mathematical filter
    def wma(s, p):
        w = np.arange(1, p + 1); r = np.zeros_like(s)
        for i in range(len(s)):
            if i < p - 1: r[i] = s[i]
            else: r[i] = np.dot(s[i - p + 1 : i + 1], w) / w.sum()
        return r

    # =========================================================================
    # DOMAIN 1: MOMENTUM & VELOCITY OSCILLATORS (Features 1-25)
    # =========================================================================
    deltas = closes[1:] - closes[:-1]
    gains = np.sum(deltas[deltas > 0]); losses = np.sum(np.abs(deltas[deltas < 0]))
    
    features['1_cmo_16'] = ((gains - losses) / (gains + losses + 1e-9)) * 100
    features['2_rsi_16'] = (gains / (gains + losses + 1e-9)) * 100
    
    g_7 = np.sum(deltas[-7:][deltas[-7:] > 0]); l_7 = np.sum(np.abs(deltas[-7:][deltas[-7:] < 0]))
    features['3_cmo_7'] = ((g_7 - l_7) / (g_7 + l_7 + 1e-9)) * 100
    features['4_rsi_7'] = (g_7 / (g_7 + l_7 + 1e-9)) * 100
    
    g_3 = np.sum(deltas[-3:][deltas[-3:] > 0]); l_3 = np.sum(np.abs(deltas[-3:][deltas[-3:] < 0]))
    features['5_cmo_3'] = ((g_3 - l_3) / (g_3 + l_3 + 1e-9)) * 100
    features['6_rsi_3'] = (g_3 / (g_3 + l_3 + 1e-9)) * 100
    
    stoch_k_16 = []
    for i in range(N):
        h_sub, l_sub = np.max(highs[max(0, i-2):i+1]), np.min(lows[max(0, i-2):i+1])
        stoch_k_16.append(((closes[i] - l_sub) / (h_sub - l_sub + 1e-9)) * 100)
    features['7_stochastic_k_16'] = stoch_k_16[-1]
    features['8_stochastic_d_16'] = np.mean(stoch_k_16[-3:])
    
    stoch_k_5 = [((closes[i] - np.min(lows[max(0, i-1):i+1])) / (np.max(highs[max(0, i-1):i+1]) - np.min(lows[max(0, i-1):i+1]) + 1e-9)) * 100 for i in range(N)]
    features['9_stochastic_k_5'] = stoch_k_5[-1]
    features['10_stochastic_d_5'] = np.mean(stoch_k_5[-2:])
    
    features['11_roc_1'] = ((closes[-1] - closes[-2]) / (closes[-2] + 1e-9)) * 100
    features['12_roc_2'] = ((closes[-1] - closes[-3]) / (closes[-3] + 1e-9)) * 100
    features['13_roc_4'] = ((closes[-1] - closes[-5]) / (closes[-5] + 1e-9)) * 100
    features['14_roc_8'] = ((closes[-1] - closes[-9]) / (closes[-9] + 1e-9)) * 100
    features['15_roc_acceleration'] = features['11_roc_1'] - ((closes[-2] - closes[-3]) / (closes[-3] + 1e-9)) * 100
    
    m_fast, m_slow = ema(closes, 3), ema(closes, 8)
    m_line = m_fast - m_slow; m_sig = ema(m_line, 3); m_hist = m_line - m_sig
    features['16_macd_line'] = m_line[-1]
    features['17_macd_signal'] = m_sig[-1]
    features['18_macd_hist'] = m_hist[-1]
    features['19_macd_hist_acceleration'] = m_hist[-1] - m_hist[-2]
    
    h_max, l_min = np.max(highs), np.min(lows)
    features['20_williams_r_16'] = ((h_max - closes[-1]) / (h_max - l_min + 1e-9)) * -100
    features['21_williams_r_5'] = ((np.max(highs[-5:]) - closes[-1]) / (np.max(highs[-5:]) - np.min(lows[-5:]) + 1e-9)) * -100
    features['22_cci'] = (typical_prices[-1] - np.mean(typical_prices)) / (0.015 * np.mean(np.abs(typical_prices - np.mean(typical_prices))) + 1e-9)
    features['23_tsi'] = (ema(ema(closes[1:] - closes[:-1], 5), 3)[-1] / (ema(ema(np.abs(closes[1:] - closes[:-1]), 5), 3)[-1] + 1e-9)) * 100
    features['24_awesome_oscillator'] = np.mean(typical_prices[-3:]) - np.mean(typical_prices)
    features['25_rvi'] = wma((closes - opens) / (highs - lows + 1e-9), 4)[-1]

    # =========================================================================
    # DOMAIN 2: LOCATION, BASELINES & CHANNEL FIBS (Features 26-51)
    # =========================================================================
    features['26_sma_16_dev'] = ((closes[-1] - np.mean(closes)) / (np.mean(closes) + 1e-9)) * 100
    features['27_sma_5_dev'] = ((closes[-1] - np.mean(closes[-5:])) / (np.mean(closes[-5:]) + 1e-9)) * 100
    features['28_ema_16_dev'] = ((closes[-1] - ema(closes, 16)[-1]) / (ema(closes, 16)[-1] + 1e-9)) * 100
    features['29_ema_5_dev'] = ((closes[-1] - ema(closes, 5)[-1]) / (ema(closes, 5)[-1] + 1e-9)) * 100
    
    features['30_donchian_position'] = (closes[-1] - l_min) / (h_max - l_min + 1e-9)
    features['31_donchian_upper_delta'] = h_max - closes[-1]
    features['32_donchian_lower_delta'] = closes[-1] - l_min
    features['33_donchian_5_position'] = (closes[-1] - np.min(lows[-5:])) / (np.max(highs[-5:]) - np.min(lows[-5:]) + 1e-9)
    
    slope, intercept = np.polyfit(x, closes, 1)
    features['34_lin_reg_slope'] = slope
    features['35_lin_reg_intercept'] = intercept
    preds = slope * x + intercept
    features['36_r_squared'] = 1 - (np.sum((closes - preds) ** 2) / (np.sum((closes - np.mean(closes)) ** 2) + 1e-9))
    
    slope_v, _ = np.polyfit(x, volumes, 1)
    features['37_volume_trend_slope'] = slope_v
    
    hma_s = wma(2 * wma(closes, 4) - wma(closes, 8), 3)
    features['38_hma_slope'] = hma_s[-1] - hma_s[-2]
    
    kama = np.zeros_like(closes); kama[0] = closes[0]
    for i in range(1, N):
        sc = ( (abs(closes[i] - closes[max(0, i-4)]) / (np.sum(np.abs(closes[max(1, i-3):i+1] - closes[max(0, i-4):i])) + 1e-9)) * (2/3 - 2/31) + 2/31 ) ** 2
        kama[i] = kama[i-1] + sc * (closes[i] - kama[i-1])
    features['39_kama_delta'] = closes[-1] - kama[-1]
    features['40_vwap_z_score'] = (closes[-1] - (np.sum(closes * volumes) / (np.sum(volumes) + 1e-9))) / (np.std(closes) + 1e-9)
    
    fib_range = h_max - l_min + 1e-9
    features['41_fib_0_dist'] = closes[-1] - l_min
    features['42_fib_236_dist'] = closes[-1] - (l_min + 0.236 * fib_range)
    features['43_fib_382_dist'] = closes[-1] - (l_min + 0.382 * fib_range)
    features['44_fib_500_dist'] = closes[-1] - (l_min + 0.500 * fib_range)
    features['45_fib_618_dist'] = closes[-1] - (l_min + 0.618 * fib_range)
    features['46_fib_100_dist'] = h_max - closes[-1]
    
    pivot = (h_max + l_min + closes[-1]) / 3.0
    features['47_pivot_dist'] = closes[-1] - pivot
    features['48_r1_dist'] = (2 * pivot - l_min) - closes[-1]
    features['49_s1_dist'] = closes[-1] - (2 * pivot - h_max)
    features['50_r2_dist'] = (pivot + h_max - l_min) - closes[-1]
    features['51_s2_dist'] = closes[-1] - (pivot - h_max + l_min)

    # =========================================================================
    # DOMAIN 3: VOLATILITY, CHAOS & STATISTICAL REGIMES (Features 52-72)
    # =========================================================================
    features['52_atr_16'] = np.mean(tr)
    features['53_atr_5'] = np.mean(tr[-5:])
    features['54_atr_3'] = np.mean(tr[-3:])
    features['55_atr_expansion_ratio'] = features['53_atr_5'] / (features['52_atr_16'] + 1e-9)
    
    features['56_std_closes'] = np.std(closes)
    features['57_variance_log_returns'] = np.var(log_returns)
    
    # SAFE GUARD: Protect advanced statistics from flat segments with zero standard deviation
    features['58_skewness_returns'] = stats.skew(log_returns) if (len(log_returns) > 2 and np.std(log_returns) > 0) else 0.0
    features['59_kurtosis_returns'] = stats.kurtosis(log_returns) if (len(log_returns) > 2 and np.std(log_returns) > 0) else 0.0
    features['60_historical_volatility'] = np.std(log_returns) * np.sqrt(252 * 24 * 4)
    
    b_bandwidth = (4 * np.std(closes)) / (np.mean(closes) + 1e-9)
    features['61_bollinger_bandwidth'] = b_bandwidth
    features['62_bollinger_bandwidth_z_score'] = (b_bandwidth - np.mean(tr)) / (np.std(tr) + 1e-9)
    features['63_keltner_channel_squeeze'] = b_bandwidth / (((2 * features['52_atr_16']) / (np.mean(closes) + 1e-9)) + 1e-9)
    features['64_kaufman_efficiency_ratio'] = np.abs(closes[-1] - closes[0]) / (np.sum(np.abs(closes[1:] - closes[:-1])) + 1e-9)
    
    features['65_fractal_dimension_index'] = 1.0 + (np.log(np.sum(np.abs(closes[1:] - closes[:-1])) / (fib_range) + 1e-9) / np.log(N))
    features['66_choppiness_index'] = (np.log10(np.sum(tr) / (fib_range) + 1e-9) / np.log10(N)) * 100
    features['67_range_pips'] = fib_range * 10000
    features['68_final_candle_range_pips'] = tr[-1] * 10000
    features['69_high_low_ratio'] = h_max / (l_min + 1e-9)
    features['70_close_open_log_ratio'] = np.log(closes[-1] / (opens[0] + 1e-9))
    features['71_intraday_variance_ratio'] = np.var(highs - lows) / (np.var(closes) + 1e-9)
    features['72_true_range_z_score'] = (tr[-1] - np.mean(tr)) / (np.std(tr) + 1e-9)

    # =========================================================================
    # DOMAIN 4: VOLUME FLOW & LIQUIDITY CONVICTION (Features 73-93)
    # =========================================================================
    obv = np.zeros(N); obv[0] = volumes[0]
    for i in range(1, N): obv[i] = obv[i-1] + volumes[i] if closes[i] > closes[i-1] else (obv[i-1] - volumes[i] if closes[i] < closes[i-1] else obv[i-1])
    features['73_obv_final'] = obv[-1]
    features['74_obv_slope'] = np.polyfit(x, obv, 1)[0]
    
    clv = ((closes - lows) - (highs - closes)) / (highs - lows + 1e-9)
    features['75_chaikin_money_flow_16'] = np.sum(clv * volumes) / (np.sum(volumes) + 1e-9)
    features['76_chaikin_money_flow_5'] = np.sum(clv[-5:] * volumes[-5:]) / (np.sum(volumes[-5:]) + 1e-9)
    
    pos_mf = np.sum([money_flow[i] for i in range(1, N) if typical_prices[i] > typical_prices[i-1]])
    neg_mf = np.sum([money_flow[i] for i in range(1, N) if typical_prices[i] <= typical_prices[i-1]])
    features['77_money_flow_index'] = (pos_mf / (pos_mf + neg_mf + 1e-9)) * 100
    
    features['78_accumulation_distribution_line'] = np.cumsum(clv * volumes)[-1]
    
    vpt = np.zeros(N)
    for i in range(1, N): vpt[i] = vpt[i-1] + volumes[i] * ((closes[i] - closes[i-1]) / (closes[i-1] + 1e-9))
    features['79_volume_price_trend'] = vpt[-1]
    features['80_relative_volume_z_score'] = (volumes[-1] - np.mean(volumes)) / (np.std(volumes) + 1e-9)
    features['81_volume_rate_of_change'] = ((volumes[-1] - volumes[-4]) / (volumes[-4] + 1e-9)) * 100
    features['82_force_index_final'] = (closes[-1] - closes[-2]) * volumes[-1]
    features['83_force_index_smoothed'] = np.mean((closes[1:] - closes[:-1]) * volumes[1:])
    
    nvi, pvi = np.zeros(N), np.zeros(N); nvi[0], pvi[0] = 1000, 1000
    for i in range(1, N):
        nvi[i] = nvi[i-1] + ((closes[i] - closes[i-1]) / (closes[i-1] + 1e-9)) * nvi[i-1] if volumes[i] < volumes[i-1] else nvi[i-1]
        pvi[i] = pvi[i-1] + ((closes[i] - closes[i-1]) / (closes[i-1] + 1e-9)) * pvi[i-1] if volumes[i] > volumes[i-1] else pvi[i-1]
    features['84_negative_volume_index'] = nvi[-1]
    features['85_positive_volume_index'] = pvi[-1]
    
    vol_fast = ema(volumes.astype(float), 3)
    vol_slow = ema(volumes.astype(float), 8)
    features['86_volume_oscillator'] = ((vol_fast[-1] - vol_slow[-1]) / (vol_slow[-1] + 1e-9)) * 100
    
    # SAFE GUARD: Guarantee correlation coefficient only triggers if both arrays have non-zero variance
    features['87_price_volume_correlation'] = np.corrcoef(closes, volumes)[0, 1] if (np.std(closes) > 0 and np.std(volumes) > 0) else 0.0
    features['88_range_volume_correlation'] = np.corrcoef(highs - lows, volumes)[0, 1] if (np.std(highs - lows) > 0 and np.std(volumes) > 0) else 0.0
    features['89_volume_climax_filter'] = 1 if volumes[-1] > (np.mean(volumes) + 2 * np.std(volumes)) else 0
    features['90_money_flow_acceleration'] = money_flow[-1] - money_flow[-2]
    features['91_volume_weighted_returns'] = np.sum(log_returns * volumes[1:]) / (np.sum(volumes[1:]) + 1e-9)
    features['92_volume_to_range_ratio_final'] = volumes[-1] / ((highs[-1] - lows[-1]) * 10000 + 1e-9)
    features['93_mean_volume_to_range_ratio'] = np.mean(volumes / ((highs - lows) * 10000 + 1e-9))

    # =========================================================================
    # DOMAIN 5: CANDLESTICK GEOMETRY & PRICE ACTION STRUCTURE (Features 94-105)
    # =========================================================================
    c_ranges = highs - lows + 1e-9; c_bodies = np.abs(closes - opens)
    features['94_relative_body_to_range_ratio'] = np.mean(c_bodies / c_ranges)
    features['95_final_body_to_range_ratio'] = c_bodies[-1] / c_ranges[-1]
    
    u_wicks = highs - np.maximum(opens, closes); l_wicks = np.minimum(opens, closes) - lows
    features['96_upper_wick_overhead_ratio'] = np.mean(u_wicks / c_ranges)
    features['97_final_upper_wick_ratio'] = u_wicks[-1] / c_ranges[-1]
    features['98_lower_wick_absorption_ratio'] = np.mean(l_wicks / c_ranges)
    features['99_final_lower_wick_ratio'] = l_wicks[-1] / c_ranges[-1]
    features['100_wick_asymmetry_ratio'] = np.sum(u_wicks) / (np.sum(l_wicks) + 1e-9)
    
    counter = 0
    for i in range(1, N):
        if closes[i] > closes[i-1]: counter = counter + 1 if counter >= 0 else 1
        elif closes[i] < closes[i-1]: counter = counter - 1 if counter <= 0 else -1
    features['101_consecutive_direction_counter'] = counter
    features['102_intra_block_open_close_gaps'] = np.sum(np.abs(opens[1:] - closes[:-1]))
    features['103_final_candle_color'] = 1 if closes[-1] > opens[-1] else (0 if closes[-1] == opens[-1] else -1)
    features['104_body_size_to_atr_ratio'] = c_bodies[-1] / (np.mean(tr) + 1e-9)
    features['105_gap_to_range_ratio'] = features['102_intra_block_open_close_gaps'] / (fib_range + 1e-9)
    
    return features


# =============================================================================
# MAIN TIMELINE ENGINE EXECUTION BLOCK
# =============================================================================
processed_rows = []

for (cond_id, sample_id), block in df_m15.groupby(['condition_id', 'sample']):
    block_sorted = block.sort_values('m15_sequence')
    
    anchor_open = block_sorted['open'].iloc[0]
    anchor_mean_volume = block_sorted['volume'].mean()
    
    for idx in range(len(block_sorted)):
        sub_block_lookback = block_sorted.iloc[:idx+1]
        row_dict = block_sorted.iloc