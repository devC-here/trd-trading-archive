# synthetic_momentum.py
import pandas as pd
import numpy as np
import random
from tqdm import trange

N_SAMPLES = 15000
SEQ_LEN = 20
OUTPUT_PATH = "synthetic_momentum_dataset.csv"
BASE_PRICE = 1.1000
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def generate_candle(base_price, body_scale=0.0005, wick_mult=(0.15,0.85), vol_range=(300,3000), bias=0.5):
    direction = 1 if random.random() < bias else -1
    body = float(np.random.uniform(0.3, 1.1) * body_scale)
    wick = float(body * np.random.uniform(*wick_mult))
    o = float(base_price)
    c = float(o + direction * body)
    h = max(o, c) + wick
    l = min(o, c) - wick
    v = int(np.random.uniform(*vol_range))
    return {"open": o, "high": float(h), "low": float(l), "close": float(c), "volume": v}, c

def rolling_z(x, w):
    return (x - x.rolling(w).mean()) / (x.rolling(w).std() + 1e-12)

def compute_momentum_features(df):
    df = df.copy()
    eps = 1e-12

    # basic
    df["close_prev1"] = df["close"].shift(1)
    df["logret1"] = np.log(df["close"] / df["close_prev1"]).fillna(0.0)
    df["logret3"] = np.log(df["close"] / df["close"].shift(3)).fillna(0.0)

    # ROC
    df["roc3"] = df["close"].pct_change(3).fillna(0.0)
    df["roc5"] = df["close"].pct_change(5).fillna(0.0)

    # RSI 14 lightweight
    delta = df["close"].diff().fillna(0.0)
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.rolling(14, min_periods=1).mean()
    roll_dn = down.rolling(14, min_periods=1).mean() + eps
    rs = roll_up / roll_dn
    df["rsi14"] = 100 - (100 / (1 + rs))
    df["rsi_ema5"] = df["rsi14"].ewm(span=5, adjust=False).mean()
    df["rsi_z"] = rolling_z(df["rsi14"], 40).fillna(0.0)

    # MACD (12,26,9 approx)
    ema_fast = df["close"].ewm(span=12, adjust=False).mean()
    ema_slow = df["close"].ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_sig = macd_line.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = (macd_line - macd_sig).fillna(0.0)
    df["macd_hist_slope"] = df["macd_hist"].diff().fillna(0.0)
    df["macd_ema5"] = df["macd_hist"].ewm(span=5, adjust=False).mean()

    # return zscore (long window)
    df["return_zscore"] = rolling_z(df["logret3"], 40).fillna(0.0)

    # close slope regression (simple linear slope on last 10 closes)
    closes = df["close"].rolling(10, min_periods=3)
    def slope(s):
        arr = s.values
        if len(arr) < 3:
            return 0.0
        x = np.arange(len(arr))
        a = np.polyfit(x, arr, 1)[0]
        return a
    df["close_slope_10"] = closes.apply(slope).fillna(0.0)

    # trend strength = abs(slope) * (atr proxy)
    df["range"] = df["high"] - df["low"]
    df["atr14"] = df["range"].rolling(14, min_periods=1).mean().fillna(0.0)
    df["trend_strength_score"] = (df["close_slope_10"].abs() * (df["atr14"] / (df["close"] + eps))).fillna(0.0)

    # interactions
    df["rsi_x_macd"] = (df["rsi_z"] * df["macd_hist"]).clip(-3, 3)
    df["roc3_x_macd"] = (df["roc3"] * df["macd_hist"]).clip(-3, 3)

    # sequence stats
    df["seq_ret_sum"] = df["logret1"].rolling(SEQ_LEN, min_periods=1).sum()
    df["seq_volatility"] = df["logret1"].rolling(SEQ_LEN, min_periods=1).std().fillna(0.0)

    # final feature pick (last bar)
    last = df.iloc[-1]
    feats = {
        "logret1": float(last["logret1"]),
        "logret3": float(last["logret3"]),
        "roc3": float(last["roc3"]),
        "roc5": float(last["roc5"]),
        "rsi14": float(last["rsi14"]),
        "rsi_ema5": float(last["rsi_ema5"]),
        "rsi_z": float(last["rsi_z"]),
        "macd_hist": float(last["macd_hist"]),
        "macd_hist_slope": float(last["macd_hist_slope"]),
        "macd_ema5": float(last["macd_ema5"]),
        "return_zscore": float(last["return_zscore"]),
        "close_slope_10": float(last["close_slope_10"]),
        "trend_strength_score": float(last["trend_strength_score"]),
        "rsi_x_macd": float(last["rsi_x_macd"]),
        "roc3_x_macd": float(last["roc3_x_macd"]),
        # sequence-level
        "seq_ret_sum": float(last["seq_ret_sum"]),
        "seq_volatility": float(last["seq_volatility"]),
    }
    return feats, df

def label_momentum_sample(feats, df):
    """
    Labels:
      - momentum_state (0 down,1 neutral,2 up)
      - slope_zone (0 decreasing,1 flat,2 increasing) based on close_slope_10
      - entry_bias (0 avoid,1 neutral,2 long/short bias) using momentum + trend_strength
    """
    lr3 = feats["logret3"]
    slope = feats["close_slope_10"]
    trend_strength = feats["trend_strength_score"]
    macd = feats["macd_hist"]

    # momentum_state: combine recent returns and macd
    score = (lr3 * 1.0) + (macd * 5.0)
    if score > 0.0008:
        momentum_state = 2
    elif score < -0.0008:
        momentum_state = 0
    else:
        momentum_state = 1

    # slope zone thresholds (calibrated for M15)
    if slope > 0.00005:
        slope_zone = 2
    elif slope < -0.00005:
        slope_zone = 0
    else:
        slope_zone = 1

    # entry_bias: requires momentum_state strong + trend_strength not overextended
    if momentum_state == 2 and trend_strength > 0 and trend_strength < 1e-3:
        entry_bias = 2
    elif momentum_state == 0 and trend_strength > 0 and trend_strength < 1e-3:
        entry_bias = 0
    else:
        entry_bias = 1

    return int(momentum_state), int(slope_zone), int(entry_bias)

def generate_sequence(base_price=BASE_PRICE):
    scenario = random.choices(
        ["trend_strong","trend_weak","mean_revert","chop","momentum_spike"],
        weights=[0.25,0.25,0.2,0.2,0.1], k=1
    )[0]

    prices = [base_price]
    seq = []
    if scenario == "trend_strong":
        bias = random.choice([0.7,0.75]); vol_rng = (400,1400); body_scale = 0.0006
    elif scenario == "trend_weak":
        bias = random.choice([0.6,0.65]); vol_rng = (300,900); body_scale = 0.00045
    elif scenario == "mean_revert":
        bias = 0.5; vol_rng = (200,800); body_scale = 0.00035
    elif scenario == "chop":
        bias = 0.5; vol_rng = (150,700); body_scale = 0.00025
    else:  # momentum_spike
        bias = 0.6; vol_rng = (800,3500); body_scale = 0.0009

    for i in range(SEQ_LEN):
        local_bias = bias if random.random() > 0.12 else 1 - bias
        cndl, new_close = generate_candle(prices[-1], body_scale=body_scale, vol_range=vol_rng, bias=local_bias)
        seq.append(cndl)
        prices.append(new_close)

    df = pd.DataFrame(seq)
    feats, df_full = compute_momentum_features(df)
    labels = label_momentum_sample(feats, df_full)

    flat = {}
    for k, v in feats.items():
        flat[k] = v

    # add a couple sequence stats
    flat["seq_dir_ratio"] = float((df_full["close"].diff().fillna(0.0) > 0).mean())
    flat["seq_mean_range"] = float(df_full["range"].mean())
    flat["seq_std_range"] = float(df_full["range"].std())

    mstate, slope_zone, entry_bias = labels
    flat["momentum_state"] = mstate
    flat["slope_zone"] = slope_zone
    flat["entry_bias"] = entry_bias

    return flat

if __name__ == "__main__":
    print(f"Generating {N_SAMPLES} Momentum sequences (EURUSD M15 style)...")
    rows = []
    for _ in trange(N_SAMPLES):
        rows.append(generate_sequence(BASE_PRICE + np.random.normal(0, 0.001)))
    df_out = pd.DataFrame(rows)
    df_out = df_out[sorted(df_out.columns)]
    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved to {OUTPUT_PATH} — shape={df_out.shape}")
    print(df_out.head())
