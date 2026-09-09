# synthetic_macro_noevents.py
"""
Macro / Session Context generator WITHOUT any news/event features.
Creates session-level synthetic sequences focused on:
 - session phase (open/mid/close)
 - multi-timeframe alignment (M15 -> H1 proxy)
 - regime transitions and intra-session volatility shifts
No event_shock or any news-impact features or labels.
"""

import pandas as pd
import numpy as np
import random
from tqdm import trange

N_SAMPLES = 15000
SEQ_LEN = 20
OUTPUT_PATH = "synthetic_macro_noevents_dataset.csv"
BASE_PRICE = 1.1000

def generate_candle(base_price, body_scale=0.0005, wick_mult=(0.2,0.9), vol_range=(300,3000), bias=0.5):
    direction = 1 if random.random() < bias else -1
    body = np.random.uniform(0.3, 1.2) * body_scale
    wick = body * np.random.uniform(*wick_mult)
    o = base_price
    c = o + direction * body
    h = max(o,c) + wick
    l = min(o,c) - wick
    v = int(np.random.uniform(*vol_range))
    return {"open": o, "high": h, "low": l, "close": c, "volume": v}, c

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

def compute_macro_features(df):
    df = df.copy()
    # basics
    df["range"] = df["high"] - df["low"]
    df["body"] = (df["close"] - df["open"]).abs()
    df["dir"] = np.sign(df["close"] - df["open"]).fillna(0).astype(int)
    df["mid"] = (df["high"] + df["low"]) / 2
    df["typ"] = (df["high"] + df["low"] + df["close"]) / 3

    # ATRs and volatility multi-horizon
    df["atr5"] = df["range"].rolling(5).mean()
    df["atr20"] = df["range"].rolling(20).mean()
    df["atr_ratio"] = df["atr5"] / (df["atr20"] + 1e-12)
    df["vol_z20"] = rolling_z(df["range"], 20).fillna(0.0)

    # VWAPs and anchored VWAPs
    df["vwap"] = compute_vwap(df)
    df["awap_session_start"] = ( ( (df["high"] + df["low"] + df["close"]) / 3.0 ) * df["volume"] ).cumsum() / (df["volume"].cumsum()+1e-12)
    idx_max = df["high"].idxmax() if len(df)>0 else 0
    idx_min = df["low"].idxmin() if len(df)>0 else 0
    df["awap_from_high"] = anchored_vwap_from_synthetic(df, idx_max)
    df["awap_from_low"] = anchored_vwap_from_synthetic(df, idx_min)

    # multi-timeframe proxies (H1 from M15)
    df["h1_close_proxy"] = df["close"].rolling(4, min_periods=1).mean()
    df["h1_slope"] = df["h1_close_proxy"].diff().fillna(0.0)
    df["m15_slope"] = df["close"].diff().fillna(0.0)
    df["tf_alignment"] = np.sign(df["h1_slope"].rolling(5, min_periods=1).mean()) * np.sign(df["m15_slope"])

    # session phase / position
    df["bar_idx"] = np.arange(len(df))
    df["session_pos"] = df["bar_idx"] / max(1, (len(df)-1))
    df["phase_open"] = (df["session_pos"] < 0.2).astype(int)
    df["phase_mid"] = ((df["session_pos"] >= 0.2) & (df["session_pos"] < 0.8)).astype(int)
    df["phase_close"] = (df["session_pos"] >= 0.8).astype(int)

    # liquidity proxies
    df["vol_mean5"] = df["volume"].rolling(5).mean()
    df["vol_z5"] = rolling_z(df["volume"], 5).fillna(0.0)
    df["liquidity_score"] = df["vol_mean5"] / (df["atr5"] + 1e-12)

    # regime / entropy
    # compute sign-entropy over last 10 bars
    rr = df["dir"].rolling(10).apply(lambda x: (x==1).mean() if len(x)>0 else 0.0, raw=True)
    p_up = rr.fillna(0.0)
    p_dn = (1 - rr).fillna(0.0)
    df["dir_entropy_10"] = - (p_up * np.log2(p_up + 1e-9) + p_dn * np.log2(p_dn + 1e-9))

    # vwap devs
    df["vwap_dev"] = df["close"] - df["vwap"]
    df["vwap_dev_z"] = rolling_z(df["vwap_dev"], 20).fillna(0.0)
    df["awap_gap_high"] = df["close"] - df["awap_from_high"]
    df["awap_gap_low"] = df["close"] - df["awap_from_low"]

    df["liquidity_x_vwapdev"] = df["liquidity_score"] * df["vwap_dev_z"]

    return df.fillna(0.0)

def label_sequence(df):
    last = df.iloc[-1]
    # session_bias: 0 neutral,1 risk_on(trend),2 risk_off (volatility-driven reversion)
    if last["tf_alignment"] > 0 and last["vwap_dev_z"] > 0.2:
        session_bias = 1
    elif last["vol_z20"] > 1.2:
        session_bias = 2
    else:
        session_bias = 0

    # tf_alignment_head: 0 mismatch,1 weak_align,2 strong_align
    align_score = (abs(last["h1_slope"]) + abs(last["m15_slope"]))
    if last["tf_alignment"] > 0 and align_score > np.percentile((abs(df["h1_slope"])+abs(df["m15_slope"])).fillna(0.0), 70):
        tf_alignment = 2
    elif last["tf_alignment"] > 0:
        tf_alignment = 1
    else:
        tf_alignment = 0

    # volatility_shift: 0 stable,1 rising_vol,2 falling_vol (recent vs previous)
    recent = df["atr5"].tail(5).mean()
    prev = df["atr5"].tail(10).head(5).mean() if len(df)>=10 else df["atr5"].mean()
    if recent > prev * 1.25:
        volatility_shift = 1
    elif recent < prev * 0.8:
        volatility_shift = 2
    else:
        volatility_shift = 0

    # session_phase: 0 open,1 mid,2 close
    if last["session_pos"] < 0.2:
        session_phase = 0
    elif last["session_pos"] < 0.8:
        session_phase = 1
    else:
        session_phase = 2

    # expected_session_move: categorize relative to atr20 (no event multiplier)
    expected_move = last["atr20"]
    ratio = expected_move / (last["atr20"] + 1e-12)
    expected_move_cat = 0 if ratio <= 1.0 else (1 if ratio <= 1.5 else 2)

    # continuity: if tf_alignment strong & low recent vol -> likely continue
    continuity = 1 if (tf_alignment==2 and volatility_shift==2 and session_bias==1) else 0

    # vol_regime (0 calm,1 normal,2 expanding) from vol_z20
    vol_regime = 2 if last["vol_z20"] > 1.2 else (1 if last["vol_z20"] > 0.3 else 0)

    return int(session_bias), int(tf_alignment), int(volatility_shift), int(session_phase), int(expected_move_cat), int(continuity), int(vol_regime)

def generate_sequence(base_price=BASE_PRICE):
    scenario = random.choices(
        ["calm_trend","vol_rise","range","drift","multi_tf_trend"],
        weights=[0.3,0.15,0.25,0.15,0.15], k=1
    )[0]

    prices = [base_price]
    seq = []
    if scenario == "calm_trend":
        bias = random.choice([0.7, 0.75]); vol_rng = (400,1200); body_scale = 0.00045
    elif scenario == "vol_rise":
        bias = 0.5; vol_rng = (800,3000); body_scale = 0.0007
    elif scenario == "range":
        bias = 0.5; vol_rng = (200,900); body_scale = 0.0003
    elif scenario == "drift":
        bias = 0.55; vol_rng = (200,900); body_scale = 0.00025
    else:
        bias = 0.7; vol_rng = (600,2500); body_scale = 0.0006

    for i in range(SEQ_LEN):
        local_bias = bias if random.random() > 0.08 else 1-bias
        cndl, new_close = generate_candle(prices[-1], body_scale=body_scale, vol_range=vol_rng, bias=local_bias)
        seq.append(cndl)
        prices.append(new_close)

    df = pd.DataFrame(seq)
    df = compute_macro_features(df)

    (session_bias, tf_alignment, volatility_shift, session_phase,
     expected_move_cat, continuity, vol_regime) = label_sequence(df)

    flat = {}
    per_bar_cols = ["close","vwap","mid","typ","range","body","dir","volume","session_pos","phase_open","phase_mid","phase_close"]
    for i in range(SEQ_LEN):
        for c in per_bar_cols:
            flat[f"{c}_t-{i}"] = float(df.iloc[i][c])

    last_cols = [
        "atr5","atr20","atr_ratio","vol_z20","vwap_dev","vwap_dev_z",
        "h1_slope","m15_slope","tf_alignment","liquidity_score",
        "vol_mean5","vol_z5","dir_entropy_10","awap_gap_high","awap_gap_low",
        "liquidity_x_vwapdev"
    ]
    for c in last_cols:
        flat[c] = float(df.iloc[-1].get(c, 0.0))

    flat["mean_close"] = float(df["close"].mean())
    flat["std_range"] = float(df["range"].std())
    flat["pct_positive"] = float((df["dir"]>0).mean())

    flat["session_bias"] = session_bias
    flat["tf_alignment_head"] = tf_alignment
    flat["volatility_shift"] = volatility_shift
    flat["session_phase"] = session_phase
    flat["expected_move_cat"] = expected_move_cat
    flat["continuity"] = continuity
    flat["vol_regime"] = vol_regime

    return flat

if __name__ == "__main__":
    print(f"Generating {N_SAMPLES} Macro (no-events) sequences...")
    rows = []
    for _ in trange(N_SAMPLES):
        rows.append(generate_sequence(BASE_PRICE + np.random.normal(0, 0.001)))
    df_out = pd.DataFrame(rows)
    df_out = df_out[sorted(df_out.columns)]
    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved to {OUTPUT_PATH} — shape={df_out.shape}")
    print(df_out.head())
