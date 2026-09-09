# synthetic_breakout.py
import pandas as pd
import numpy as np
import random
from tqdm import trange

N_SAMPLES = 15000
SEQ_LEN = 20
OUTPUT_PATH = "synthetic_breakout_dataset.csv"
BASE_PRICE = 1.6000
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ------------------------------------------------------------
# Candle generator (realistic for EURUSD M15)
# ------------------------------------------------------------
def generate_candle(base_price, body_scale=0.0005, wick_mult=(0.2,0.9), vol_range=(300,3000), bias=0.5):
    direction = 1 if random.random() < bias else -1
    body = float(np.random.uniform(0.35, 1.1) * body_scale)
    wick = float(body * np.random.uniform(*wick_mult))

    o = float(base_price)
    c = float(o + direction * body)
    h = max(o, c) + wick
    l = min(o, c) - wick
    v = int(np.random.uniform(*vol_range))

    return {"open": o, "high": float(h), "low": float(l), "close": float(c), "volume": v}, c


# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------
def rolling_z(x, w):
    return (x - x.rolling(w).mean()) / (x.rolling(w).std() + 1e-12)


def compute_vwap(df):
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = (tp * df["volume"]).cumsum()
    vv = df["volume"].cumsum() + 1e-12
    return pv / vv


def anchored_vwap_from_synthetic(df, start_idx):
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = (tp * df["volume"]).copy().values
    vv = df["volume"].copy().values

    pv[:start_idx] = 0.0
    vv[:start_idx] = 0.0

    cpv = np.cumsum(pv)
    cv = np.cumsum(vv) + 1e-12
    return pd.Series(cpv / cv, index=df.index)


# ------------------------------------------------------------
# FEATURE ENGINEERING (Volatility–Breakout Model)
# ------------------------------------------------------------
def compute_breakout_features(df):
    df = df.copy()
    eps = 1e-12

    # Basics
    df["range"] = df["high"] - df["low"]
    df["body"] = (df["close"] - df["open"]).abs()
    df["dir"] = np.sign(df["close"] - df["open"]).fillna(0).astype(int)

    # ATR14 proxy
    df["atr14"] = df["range"].rolling(14, min_periods=1).mean()
    df["atr_z"] = rolling_z(df["atr14"], 40).fillna(0.0)

    # Multi-horizon ranges
    df["range_3"] = df["range"].rolling(3, min_periods=1).mean()
    df["range_12"] = df["range"].rolling(12, min_periods=1).mean()

    df["compression_ratio"] = df["range_3"] / (df["range_12"] + eps)
    df["compression_zscore"] = rolling_z(df["compression_ratio"], 40).fillna(0.0)

    # Bollinger Bands (20,2)
    mavg = df["close"].rolling(20, min_periods=1).mean()
    std = df["close"].rolling(20, min_periods=1).std().fillna(0.0)

    df["bb_upper"] = mavg + 2 * std
    df["bb_lower"] = mavg - 2 * std

    # Bandwidth normalized by ATR
    df["bb_bandwidth"] = (df["bb_upper"] - df["bb_lower"]) / (df["atr14"] + eps)
    df["bb_bandwidth_z"] = rolling_z(df["bb_bandwidth"], 40).fillna(0.0)

    # bb_position: normalized inside band
    rng = (df["bb_upper"] - df["bb_lower"]).replace(0.0, eps)
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (rng + eps)
    df["bb_position"] = df["bb_position"].clip(0.0, 1.0)

    # MACD (EMA-based)
    df["ema_fast"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=26, adjust=False).mean()
    df["macd_line"] = df["ema_fast"] - df["ema_slow"]
    df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]
    df["macd_hist_slope"] = df["macd_hist"].diff().fillna(0.0)

    # macd_cross_flag
    df["macd_cross_flag"] = 1
    up_cross = (df["macd_line"] > df["macd_signal"]) & (df["macd_line"].shift(1) <= df["macd_signal"].shift(1))
    down_cross = (df["macd_line"] < df["macd_signal"]) & (df["macd_line"].shift(1) >= df["macd_signal"].shift(1))
    df.loc[up_cross, "macd_cross_flag"] = 2
    df.loc[down_cross, "macd_cross_flag"] = 0

    # RSI(14)
    delta = df["close"].diff().fillna(0.0)
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.rolling(14, min_periods=1).mean()
    roll_down = down.rolling(14, min_periods=1).mean() + eps
    rs = roll_up / roll_down
    df["rsi14"] = 100.0 - (100.0 / (1.0 + rs))
    df["rsi_z"] = rolling_z(df["rsi14"], 40).fillna(0.0)

    # Interactions
    df["rsi_x_macd"] = (df["rsi_z"] * df["macd_hist"]).clip(-3, 3)
    df["bbpos_x_macd"] = (df["bb_position"] * df["macd_hist"]).clip(-3, 3)

    # Extract last-bar features
    last = df.iloc[-1]
    features = {
        "atr14": float(last["atr14"]),
        "atr_z": float(last["atr_z"]),
        "range_3": float(last["range_3"]),
        "range_12": float(last["range_12"]),
        "compression_ratio": float(last["compression_ratio"]),
        "compression_zscore": float(last["compression_zscore"]),
        "bb_position": float(last["bb_position"]),
        "bb_bandwidth": float(last["bb_bandwidth"]),
        "bb_bandwidth_z": float(last["bb_bandwidth_z"]),
        "macd_hist": float(last["macd_hist"]),
        "macd_hist_slope": float(last["macd_hist_slope"]),
        "macd_cross_flag": int(last["macd_cross_flag"]),
        "rsi14": float(last["rsi14"]),
        "rsi_z": float(last["rsi_z"]),
        "rsi_x_macd": float(last["rsi_x_macd"]),
        "bbpos_x_macd": float(last["bbpos_x_macd"]),
    }

    return features, df


# ------------------------------------------------------------
# LABELING LOGIC
# ------------------------------------------------------------
def label_breakout_sample(features, df):
    cz = features["compression_zscore"]
    atr_z = features["atr_z"]
    macd_hist = features["macd_hist"]
    bbpos = features["bb_position"]

    # vol_regime
    if cz < -0.9 and atr_z < -0.5:
        vol_regime = 1       # compressing
    elif cz > 0.9 or atr_z > 0.9:
        vol_regime = 2       # expanding
    else:
        vol_regime = 0       # stable

    # breakout bias
    bias_score = macd_hist * (bbpos - 0.5)
    if abs(bias_score) < 0.0005:
        breakout_bias = 1
    elif bias_score > 0:
        breakout_bias = 2
    else:
        breakout_bias = 0

    # breakout strength
    if (cz < -1.2) and (abs(macd_hist) > 0.0006):
        breakout_strength = 1
    else:
        if vol_regime == 2 and abs(macd_hist) > 0.0008:
            breakout_strength = 1
        else:
            breakout_strength = 0

    return int(vol_regime), int(breakout_bias), int(breakout_strength)


# ------------------------------------------------------------
# SEQUENCE GENERATOR
# ------------------------------------------------------------
def generate_sequence(base_price=BASE_PRICE):
    scenario = random.choices(
        ["calm_trend","vol_rise","range","drift","sudden_spike","multi_tf_trend"],
        weights=[0.28,0.12,0.28,0.12,0.10,0.10], k=1
    )[0]

    prices = [base_price]
    seq = []

    # Scenario parameters
    if scenario == "calm_trend":
        bias = random.choice([0.65, 0.7]); vol_rng = (350,900); body_scale = 0.00045
    elif scenario == "vol_rise":
        bias = 0.5; vol_rng = (800,3000); body_scale = 0.0008
    elif scenario == "range":
        bias = 0.5; vol_rng = (200,700); body_scale = 0.00025
    elif scenario == "drift":
        bias = 0.55; vol_rng = (250,900); body_scale = 0.0003
    elif scenario == "sudden_spike":
        bias = 0.5; vol_rng = (1000,6000); body_scale = 0.0012
    else:
        bias = 0.7; vol_rng = (500,2000); body_scale = 0.0006

    # Build candles
    for _ in range(SEQ_LEN):
        local_bias = bias if random.random() > 0.08 else 1 - bias
        cndl, new_close = generate_candle(prices[-1], body_scale=body_scale,
                                          vol_range=vol_rng, bias=local_bias)
        seq.append(cndl)
        prices.append(new_close)

    df = pd.DataFrame(seq)
    features, df_full = compute_breakout_features(df)
    labels = label_breakout_sample(features, df_full)

    flat = {}

    # Add last-bar features
    for k, v in features.items():
        flat[k] = v

    # --- FIXED HERE: use df_full ---
    flat["seq_mean_range"] = float(df_full["range"].mean())
    flat["seq_std_range"] = float(df_full["range"].std())
    flat["seq_dir_ratio"] = float((df_full["dir"] > 0).mean())
    flat["seq_mean_volume"] = float(df_full["volume"].mean())

    vol_regime, breakout_bias, breakout_strength = labels
    flat["vol_regime"] = vol_regime
    flat["breakout_bias"] = breakout_bias
    flat["breakout_strength"] = breakout_strength

    return flat


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    print(f"Generating {N_SAMPLES} Breakout sequences (EURUSD M15 style)...")
    rows = []
    for _ in trange(N_SAMPLES):
        rows.append(generate_sequence(BASE_PRICE + np.random.normal(0, 0.001)))

    df_out = pd.DataFrame(rows)
    df_out = df_out[sorted(df_out.columns)]
    df_out.to_csv(OUTPUT_PATH, index=False)

    print(f"✅ Saved to {OUTPUT_PATH} — shape={df_out.shape}")
    print(df_out.head())
