# synthetic_microstructure.py  ·  v2  (next-candle edition)
# ─────────────────────────────────────────────────────────────────────
# Generates training data for the Microstructure model
# 32 features | 6 heads — ALL labels predict the NEXT 1-2 candles
# FX M15 auction physics
# ─────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import random
try:
    from tqdm import trange
except ImportError:
    def trange(n, **kw):
        return range(n)

N_SAMPLES    = 20000
SEQ_LEN      = 25           # context bars for features
FUTURE_BARS  = 2            # bars labelled
OUTPUT_PATH  = "synthetic_microstructure_dataset.csv"
BASE_PRICE   = 1.2000
SEED         = 42
random.seed(SEED)
np.random.seed(SEED)

EPS = 1e-12


# ─────────────────────────────────────────────────────────────────────
# Candle generator
# ─────────────────────────────────────────────────────────────────────

def generate_candle(base_price, body_scale=0.00045, wick_mult=(0.1, 0.8),
                    vol_range=(200, 2000), bias=0.5):
    direction = 1 if random.random() < bias else -1
    body      = float(np.random.uniform(0.15, 1.0) * body_scale)
    wick      = float(body * np.random.uniform(*wick_mult))
    tail      = float(body * np.random.uniform(*wick_mult))
    o  = float(base_price)
    c  = float(o + direction * body)
    h  = max(o, c) + wick
    l  = min(o, c) - tail
    v  = int(np.random.uniform(*vol_range))
    return {"open": o, "high": float(h), "low": float(l),
            "close": float(c), "volume": v}, c


def rolling_z(x, w):
    return (x - x.rolling(w).mean()) / (x.rolling(w).std() + EPS)


# ─────────────────────────────────────────────────────────────────────
# Feature computation — unchanged from v1, on context bars only
# ─────────────────────────────────────────────────────────────────────

def compute_features(df):
    df  = df.copy().reset_index(drop=True)

    df["range"]       = df["high"] - df["low"]
    df["body"]        = df["close"] - df["open"]
    df["abs_body"]    = df["body"].abs()
    df["dir"]         = np.sign(df["body"]).fillna(0).astype(int)
    df["upper_wick"]  = df["high"]  - df[["open","close"]].max(axis=1)
    df["lower_wick"]  = df[["open","close"]].min(axis=1) - df["low"]

    df["body_ratio"]     = df["abs_body"]   / (df["range"] + EPS)
    df["upper_ratio"]    = df["upper_wick"] / (df["range"] + EPS)
    df["lower_ratio"]    = df["lower_wick"] / (df["range"] + EPS)
    df["body_ratio_s5"]  = df["body_ratio"].rolling(5, min_periods=1).mean()
    df["upper_ratio_s5"] = df["upper_ratio"].rolling(5, min_periods=1).mean()
    df["lower_ratio_s5"] = df["lower_ratio"].rolling(5, min_periods=1).mean()

    df["body_z"]         = rolling_z(df["abs_body"], 40).fillna(0.0)
    df["range_z"]        = rolling_z(df["range"],   40).fillna(0.0)
    df["tail_asym"]      = (df["upper_wick"] - df["lower_wick"]) / (df["range"] + EPS)
    df["tail_to_body"]   = (df["upper_wick"] + df["lower_wick"]) / (df["abs_body"] + EPS)

    df["signed_vol"]     = df["volume"] * df["dir"]
    df["orderflow_5"]    = df["signed_vol"].rolling(5,  min_periods=1).sum()
    df["orderflow_10"]   = df["signed_vol"].rolling(10, min_periods=1).sum()

    df["small_body_flag"] = (df["abs_body"] < df["abs_body"].rolling(5,min_periods=1).mean()*0.6).astype(int)
    df["opp_wick"]        = 0.0
    df.loc[df["dir"] >  0, "opp_wick"] = df["lower_wick"]
    df.loc[df["dir"] <  0, "opp_wick"] = df["upper_wick"]
    df["absorption_raw"]  = (df["small_body_flag"] * df["opp_wick"]).rolling(5, min_periods=1).sum()

    df["atr5"]            = df["range"].rolling(5,  min_periods=1).mean()
    df["atr10"]           = df["range"].rolling(10, min_periods=1).mean()
    df["range_mean5"]     = df["atr5"]
    df["pullback_depth"]  = df["range"] / (df["range_mean5"] + EPS)
    df["pullback_z"]      = rolling_z(df["pullback_depth"], 20).fillna(0.0)
    df["exhaustion"]      = ((df["upper_wick"] + df["lower_wick"]) / (df["abs_body"] + EPS)).fillna(0.0)
    df["expected_move_raw"] = df["atr5"] * (1 + df["orderflow_5"].abs() / (df["volume"].rolling(5,min_periods=1).mean() + EPS))

    df["wick_body_inter"] = ((df["upper_wick"] - df["lower_wick"]) * df["abs_body"]).clip(-2.0, 2.0)
    df["body_vol_inter"]  = (df["abs_body"] * df["volume"]).clip(-1e6, 1e6)

    cons, cur, prev = [], 0, 0
    for d in df["dir"].fillna(0).tolist():
        if d == prev and d != 0:  cur += 1
        elif d != 0:               cur  = 1
        else:                      cur  = 0
        cons.append(cur)
        prev = d
    df["consec_len"] = cons

    df["close_gravity"]  = (df["close"] - df["low"]) / (df["range"] + EPS)
    df["body_mid_disp"]  = (((df["open"]+df["close"])/2) - ((df["high"]+df["low"])/2)) / (df["range"] + EPS)
    df["tick_delta"]     = df["close_gravity"] * 2 - 1
    delta_vol            = df["tick_delta"] * df["volume"]
    df["cum_delta_5z"]   = rolling_z(delta_vol.rolling(5,min_periods=1).sum(), 20).fillna(0.0)
    vol_avg10            = df["volume"].rolling(10, min_periods=1).mean()
    dom_wick             = df[["upper_wick","lower_wick"]].max(axis=1)
    rej_dir              = np.where(df["lower_wick"] > df["upper_wick"], 1, -1)
    df["rejection_qual"] = ((dom_wick / (df["range"]+EPS)) *
                            (df["volume"] / (vol_avg10+EPS)) *
                            rej_dir).clip(-4, 4)
    comp_a               = (df["body_ratio"] < 0.25).astype(int)
    comp_b               = (df["range"] / (df["atr10"]+EPS) > 0.80).astype(int)
    comp_c               = (df["volume"] / (vol_avg10+EPS) > 1.20).astype(int)
    df["absorption_fp"]  = (comp_a + comp_b + comp_c).astype(float)
    bull = df["dir"] >= 0
    df["dir_efficiency"] = np.where(bull,
                                    df["body_ratio"] * df["close_gravity"],
                                    df["body_ratio"] * (1 - df["close_gravity"]))
    vol_avg5             = df["volume"].rolling(5, min_periods=1).mean()
    df["bar_energy"]     = (df["range"] / (df["atr5"]+EPS) *
                            df["volume"] / (vol_avg5+EPS)).clip(0, 6)
    df["gap_pressure"]   = ((df["open"] - df["close"].shift(1).fillna(df["open"])) /
                             (df["atr5"]+EPS)).clip(-3, 3)

    return df


# ─────────────────────────────────────────────────────────────────────
# FORWARD LABEL GENERATION — 6 heads, all about NEXT 1-2 candles
# ─────────────────────────────────────────────────────────────────────

def compute_labels(df, future_candles: list) -> tuple:
    """
    df              : full context DataFrame (features already computed)
    future_candles  : list of 2 raw dicts {open, high, low, close, volume}
    """
    last  = df.iloc[-1]
    atr   = float(last["atr5"]) + EPS
    ref   = float(last["close"])

    c1    = future_candles[0]
    c2    = future_candles[1] if len(future_candles) >= 2 else c1

    # ── Future bar measurements ───────────────────────────────────────
    c1_body_atr  = abs(c1["close"] - c1["open"]) / atr
    c2_body_atr  = abs(c2["close"] - c2["open"]) / atr
    c1_bull      = c1["close"] > c1["open"]
    c2_bull      = c2["close"] > c2["open"]
    net_atr      = (c2["close"] - ref) / atr
    seq_high     = max(c1["high"], c2["high"])
    seq_low      = min(c1["low"],  c2["low"])
    seq_range    = (seq_high - seq_low) / atr
    c1_cg        = (c1["close"] - c1["low"]) / (c1["high"] - c1["low"] + EPS)
    c1_uw        = (c1["high"] - max(c1["open"], c1["close"])) / (c1["high"] - c1["low"] + EPS)
    c1_lw        = (min(c1["open"], c1["close"]) - c1["low"]) / (c1["high"] - c1["low"] + EPS)

    # ── Context signals ───────────────────────────────────────────────
    tick_d   = float(last["tick_delta"])
    cum_d5   = float(last["cum_delta_5z"])
    bar_e    = float(last["bar_energy"])
    rq       = float(last["rejection_qual"])
    afp      = float(last["absorption_fp"])

    # ── HEAD 1: next_flow_direction [0=bearish 1=contested 2=bullish]
    if   net_atr > 0.30:    flow_dir = 2
    elif net_atr < -0.30:   flow_dir = 0
    else:                    flow_dir = 1

    # ── HEAD 2: next_bar_intent [0=absorption 1=rejection 2=impulse 3=continuation_fade]
    if c1_body_atr < 0.15 and c1_uw + c1_lw > 0.5:
        nxt_intent = 0   # absorption / indecision
    elif (c1_lw > 0.40 and c1_uw < 0.20) or (c1_uw > 0.40 and c1_lw < 0.20):
        nxt_intent = 1   # rejection / pin
    elif c1_body_atr > 0.8 and c1_cg > 0.65:
        nxt_intent = 2   # impulse
    else:
        # continuation with fade check
        if c2_body_atr < c1_body_atr * 0.60:   nxt_intent = 3  # fade
        else:                                    nxt_intent = 2  # impulse

    # ── HEAD 3: next_magnitude [0=thin 1=normal 2=strong]
    avg_body = (c1_body_atr + c2_body_atr) / 2
    if   avg_body > 1.5:    nxt_mag = 2
    elif avg_body > 0.5:    nxt_mag = 1
    else:                    nxt_mag = 0

    # ── HEAD 4: expected_rejection [0=no 1=partial 2=strong]
    # Did a wick form in direction opposite to entry?
    if c1_bull:
        rej_score = c1_uw   # upper wick on a bull bar = rejection of highs
    else:
        rej_score = c1_lw   # lower wick on bear bar = rejection of lows
    if   rej_score > 0.45:   nxt_rej = 2
    elif rej_score > 0.20:   nxt_rej = 1
    else:                     nxt_rej = 0

    # ── HEAD 5: momentum_follow_through [0=failed 1=partial 2=confirmed]
    # Does next candle follow the microstructure signal in the context bar?
    # tick_delta > 0 and cum_delta > 0 = bullish orderflow context
    flow_bullish = (tick_d > 0.15 and cum_d5 > 0.2)
    flow_bearish = (tick_d < -0.15 and cum_d5 < -0.2)
    if flow_bullish:
        if net_atr > 0.60:   follow = 2
        elif net_atr > 0.10: follow = 1
        else:                 follow = 0
    elif flow_bearish:
        if net_atr < -0.60:  follow = 2
        elif net_atr < -0.10:follow = 1
        else:                 follow = 0
    else:
        follow = 1   # no clear signal

    # ── HEAD 6: next_range_type [0=inside_compress 1=normal 2=outside_expand]
    last_range_atr = float(last["range"]) / atr
    if   seq_range < last_range_atr * 0.7:   rng_type = 0
    elif seq_range > last_range_atr * 1.5:   rng_type = 2
    else:                                      rng_type = 1

    return (int(flow_dir), int(nxt_intent), int(nxt_mag),
            int(nxt_rej), int(follow), int(rng_type))


# ─────────────────────────────────────────────────────────────────────
# Scenario generator
# ─────────────────────────────────────────────────────────────────────

SCENARIOS = {
    "impulse_clean":     dict(bias=0.75, vol=(600,2200), body=0.0009, wicks=(0.05,0.35)),
    "impulse_messy":     dict(bias=0.70, vol=(500,2000), body=0.0008, wicks=(0.15,0.60)),
    "wick_rejection":    dict(bias=0.50, vol=(400,1600), body=0.0004, wicks=(0.40,1.20)),
    "absorption_slow":   dict(bias=0.55, vol=(300,1400), body=0.0003, wicks=(0.20,0.70)),
    "absorption_active": dict(bias=0.52, vol=(800,2800), body=0.0002, wicks=(0.30,0.80)),
    "chop":              dict(bias=0.50, vol=(150, 700), body=0.0003, wicks=(0.10,0.50)),
    "spike_news":        dict(bias=0.50, vol=(1200,6000),body=0.0012, wicks=(0.05,0.30)),
    "exhaustion_run":    dict(bias=0.72, vol=(400,2000), body=0.0007, wicks=(0.10,0.45)),
    "dual_rejection":    dict(bias=0.50, vol=(500,1800), body=0.0003, wicks=(0.50,1.50)),
}
WEIGHTS = [0.12, 0.10, 0.14, 0.10, 0.08, 0.16, 0.06, 0.14, 0.10]


def generate_sequence(base_price=BASE_PRICE):
    sc_name = random.choices(list(SCENARIOS.keys()), weights=WEIGHTS, k=1)[0]
    sc      = SCENARIOS[sc_name]
    prices  = [base_price + np.random.normal(0, 0.0005)]
    candles = []
    total   = SEQ_LEN + FUTURE_BARS

    for i in range(total):
        local_bias = sc["bias"] if random.random() > 0.10 else 1 - sc["bias"]

        if sc_name == "exhaustion_run" and i > SEQ_LEN // 2:
            fade       = 1 - (i - SEQ_LEN//2) / (SEQ_LEN//2 + 1) * 0.6
            body_scale = sc["body"] * fade
        else:
            body_scale = sc["body"]

        c, new_close = generate_candle(
            prices[-1], body_scale=body_scale,
            wick_mult=sc["wicks"], vol_range=sc["vol"], bias=local_bias)

        if sc_name == "dual_rejection" and random.random() < 0.30 and i < SEQ_LEN:
            extra = abs(np.random.normal(0.0005, 0.0002))
            c["high"] += extra
            c["low"]  -= extra

        candles.append(c)
        prices.append(new_close)

    df_context = pd.DataFrame(candles[:SEQ_LEN])
    future     = candles[SEQ_LEN:]

    df_context = compute_features(df_context)
    labels     = compute_labels(df_context, future)
    last       = df_context.iloc[-1]

    feature_names = [
        "abs_body","upper_wick","lower_wick","range",
        "body_ratio","upper_ratio","lower_ratio",
        "body_ratio_s5","upper_ratio_s5","lower_ratio_s5",
        "body_z","range_z","tail_asym","tail_to_body",
        "orderflow_5","orderflow_10",
        "absorption_raw","pullback_z","exhaustion","expected_move_raw",
        "wick_body_inter","body_vol_inter","consec_len",
        "close_gravity","body_mid_disp","tick_delta","cum_delta_5z",
        "rejection_qual","absorption_fp","dir_efficiency",
        "bar_energy","gap_pressure",
    ]

    row = {fn: float(last[fn]) for fn in feature_names}
    (row["label_next_flow_dir"],   row["label_next_bar_intent"],
     row["label_next_magnitude"],  row["label_next_rejection"],
     row["label_momentum_follow"], row["label_next_range_type"]) = labels
    return row


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Generating {N_SAMPLES} microstructure samples — NEXT-CANDLE labels (FX M15) ...")
    rows = []
    for _ in trange(N_SAMPLES):
        rows.append(generate_sequence())

    df_out = pd.DataFrame(rows)
    label_cols = [c for c in df_out.columns if c.startswith("label_")]
    feat_cols  = [c for c in df_out.columns if not c.startswith("label_")]
    df_out     = df_out[sorted(feat_cols) + label_cols]
    df_out.to_csv(OUTPUT_PATH, index=False)

    label_maps = {
        "label_next_flow_dir":    {0:"bearish", 1:"contested", 2:"bullish"},
        "label_next_bar_intent":  {0:"absorption", 1:"rejection", 2:"impulse", 3:"fade"},
        "label_next_magnitude":   {0:"thin", 1:"normal", 2:"strong"},
        "label_next_rejection":   {0:"none", 1:"partial", 2:"strong"},
        "label_momentum_follow":  {0:"failed", 1:"partial", 2:"confirmed"},
        "label_next_range_type":  {0:"inside_compress", 1:"normal", 2:"outside_expand"},
    }
    print(f"\n✅  Saved → {OUTPUT_PATH}  shape={df_out.shape}")
    print(f"    Features : {len(feat_cols)}")
    for lc in label_cols:
        counts = df_out[lc].value_counts().sort_index()
        named  = {label_maps[lc].get(k,k): f"{100*v/len(df_out):.1f}%" for k, v in counts.items()}
        print(f"    {lc}: {named}")
