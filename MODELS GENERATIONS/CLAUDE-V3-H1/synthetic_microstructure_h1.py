# synthetic_microstructure_h1.py  ·  v4  (market-realistic + sensitive)
# ─────────────────────────────────────────────────────────────────────
# H1 FX Microstructure — uses shared market_physics_h1 engine
# 32 features | 6 heads | 3-bar path-aware labels
# ─────────────────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from market_physics_h1 import VolState, gen_candle_h1, evolve_bias, future_path_metrics, EPS
import pandas as pd
import numpy as np
import random

try:
    from tqdm import trange
except ImportError:
    def trange(n, **kw): return range(n)

N_SAMPLES   = 20000
SEQ_LEN     = 25
FUTURE_BARS = 3
OUTPUT_PATH = "synthetic_microstructure_h1_dataset.csv"
BASE_PRICE  = 1.2000
SEED        = 42
random.seed(SEED); np.random.seed(SEED)
EPS_L       = EPS


def rolling_z(x, w):
    return (x - x.rolling(w).mean()) / (x.rolling(w).std() + EPS_L)


# ─────────────────────────────────────────────────────────────────────
# Features (32) — same schema
# ─────────────────────────────────────────────────────────────────────

def compute_features(df):
    df = df.copy().reset_index(drop=True)
    df["range"]      = df["high"]-df["low"]
    df["body"]       = df["close"]-df["open"]
    df["abs_body"]   = df["body"].abs()
    df["dir"]        = np.sign(df["body"]).fillna(0).astype(int)
    df["upper_wick"] = df["high"] - df[["open","close"]].max(axis=1)
    df["lower_wick"] = df[["open","close"]].min(axis=1) - df["low"]

    df["body_ratio"]     = df["abs_body"]/(df["range"]+EPS_L)
    df["upper_ratio"]    = df["upper_wick"]/(df["range"]+EPS_L)
    df["lower_ratio"]    = df["lower_wick"]/(df["range"]+EPS_L)
    df["body_ratio_s5"]  = df["body_ratio"].rolling(5,min_periods=1).mean()
    df["upper_ratio_s5"] = df["upper_ratio"].rolling(5,min_periods=1).mean()
    df["lower_ratio_s5"] = df["lower_ratio"].rolling(5,min_periods=1).mean()

    df["body_z"]       = rolling_z(df["abs_body"],min(40,len(df))).fillna(0.0)
    df["range_z"]      = rolling_z(df["range"],  min(40,len(df))).fillna(0.0)
    df["tail_asym"]    = (df["upper_wick"]-df["lower_wick"])/(df["range"]+EPS_L)
    df["tail_to_body"] = (df["upper_wick"]+df["lower_wick"])/(df["abs_body"]+EPS_L)

    df["signed_vol"]  = df["volume"]*df["dir"]
    df["orderflow_5"] = df["signed_vol"].rolling(5, min_periods=1).sum()
    df["orderflow_10"]= df["signed_vol"].rolling(10,min_periods=1).sum()

    df["small_body_flag"] = (df["abs_body"]<df["abs_body"].rolling(5,min_periods=1).mean()*0.6).astype(int)
    df["opp_wick"]        = 0.0
    df.loc[df["dir"]>0,"opp_wick"] = df["lower_wick"]
    df.loc[df["dir"]<0,"opp_wick"] = df["upper_wick"]
    df["absorption_raw"]  = (df["small_body_flag"]*df["opp_wick"]).rolling(5,min_periods=1).sum()

    df["atr5"]  = df["range"].rolling(5, min_periods=1).mean()
    df["atr10"] = df["range"].rolling(10,min_periods=1).mean()
    df["pullback_depth"]  = df["range"]/(df["atr5"]+EPS_L)
    df["pullback_z"]      = rolling_z(df["pullback_depth"],min(20,len(df))).fillna(0.0)
    df["exhaustion"]      = ((df["upper_wick"]+df["lower_wick"])/(df["abs_body"]+EPS_L)).fillna(0.0)
    df["expected_move_raw"] = df["atr5"]*(1+df["orderflow_5"].abs()/(df["volume"].rolling(5,min_periods=1).mean()+EPS_L))

    df["wick_body_inter"] = ((df["upper_wick"]-df["lower_wick"])*df["abs_body"]).clip(-2.0,2.0)
    df["body_vol_inter"]  = (df["abs_body"]*df["volume"]).clip(-1e6,1e6)

    cons,cur,prev = [],[],0
    for d in df["dir"].fillna(0).tolist():
        if d==prev and d!=0: cur=cur+1 if cur else 1
        elif d!=0: cur=1
        else: cur=0
        cons.append(cur); prev=d
    df["consec_len"] = cons

    df["close_gravity"]  = (df["close"]-df["low"])/(df["range"]+EPS_L)
    df["body_mid_disp"]  = (((df["open"]+df["close"])/2)-((df["high"]+df["low"])/2))/(df["range"]+EPS_L)
    df["tick_delta"]     = df["close_gravity"]*2-1
    delta_vol            = df["tick_delta"]*df["volume"]
    df["cum_delta_5z"]   = rolling_z(delta_vol.rolling(5,min_periods=1).sum(),min(20,len(df))).fillna(0.0)

    vol_avg10 = df["volume"].rolling(10,min_periods=1).mean()
    dom_wick  = df[["upper_wick","lower_wick"]].max(axis=1)
    rej_dir   = np.where(df["lower_wick"]>df["upper_wick"],1,-1)
    df["rejection_qual"] = ((dom_wick/(df["range"]+EPS_L))*(df["volume"]/(vol_avg10+EPS_L))*rej_dir).clip(-4,4)

    df["absorption_fp"] = ((df["body_ratio"]<0.25).astype(int)+
                           (df["range"]/(df["atr10"]+EPS_L)>0.80).astype(int)+
                           (df["volume"]/(vol_avg10+EPS_L)>1.20).astype(int)).astype(float)
    bull = df["dir"]>=0
    df["dir_efficiency"] = np.where(bull, df["body_ratio"]*df["close_gravity"],
                                           df["body_ratio"]*(1-df["close_gravity"]))
    vol_avg5  = df["volume"].rolling(5,min_periods=1).mean()
    df["bar_energy"]   = (df["range"]/(df["atr5"]+EPS_L)*df["volume"]/(vol_avg5+EPS_L)).clip(0,6)
    df["gap_pressure"] = ((df["open"]-df["close"].shift(1).fillna(df["open"]))/(df["atr5"]+EPS_L)).clip(-3,3)
    return df


# ─────────────────────────────────────────────────────────────────────
# Labels — 3-bar path-aware
# ─────────────────────────────────────────────────────────────────────

def compute_labels(df, future_candles: list) -> tuple:
    last  = df.iloc[-1]
    atr   = float(last["atr5"]) + EPS_L
    pm    = future_path_metrics(future_candles, atr)
    net3=pm["net3"]; avg_body=pm["avg_body"]; mfe=pm["mfe"]
    pullback=pm["pullback"]; path_range=pm["path_range"]; close_grav=pm["close_grav"]
    b1,b2=pm["b1"],pm["b2"]; d1,d2=pm["d1"],pm["d2"]
    c1=future_candles[0]
    c1_r=(c1["high"]-c1["low"]+EPS_L)
    c1_cg=(c1["close"]-c1["low"])/c1_r
    c1_uw=(c1["high"]-max(c1["open"],c1["close"]))/c1_r
    c1_lw=(min(c1["open"],c1["close"])-c1["low"])/c1_r
    c1_ba=abs(c1["close"]-c1["open"])/atr
    tick_d=float(last["tick_delta"]); cum_d5=float(last["cum_delta_5z"])
    # HEAD 1: flow direction — p33≈0.48
    if   net3 > 0.50:    flow_dir = 2
    elif net3 < -0.50:   flow_dir = 0
    else:                 flow_dir = 1
    # HEAD 2: bar intent — specific patterns first, then default
    if c1_ba < 0.22 and (c1_uw+c1_lw) > 0.48:
        nxt_intent = 0   # absorption
    elif (c1_lw>0.35 and c1_uw<0.22) or (c1_uw>0.35 and c1_lw<0.22):
        nxt_intent = 1   # rejection
    elif c1_ba > 0.60 and c1_cg > 0.58:
        nxt_intent = 2   # impulse
    else:
        nxt_intent = 3   # fade
    # HEAD 3: magnitude — p33/p75 = 0.39/0.74
    if   avg_body > 0.74:    nxt_mag = 2
    elif avg_body > 0.39:    nxt_mag = 1
    else:                    nxt_mag = 0
    # HEAD 4: rejection quality — wick on c1 vs direction
    c1_bull=(c1["close"]>c1["open"]); rej_score=c1_uw if c1_bull else c1_lw
    if   rej_score > 0.30:   nxt_rej = 2
    elif rej_score > 0.14:   nxt_rej = 1
    else:                     nxt_rej = 0
    # HEAD 5: momentum follow-through — MFE/pullback
    flow_bull=(tick_d>0.18 and cum_d5>0.25); flow_bear=(tick_d<-0.18 and cum_d5<-0.25)
    if flow_bull:
        follow = 2 if (net3>0.55 and mfe>pullback*1.8) else (1 if net3>0.15 else 0)
    elif flow_bear:
        follow = 2 if (net3<-0.55 and mfe>pullback*1.8) else (1 if net3<-0.15 else 0)
    else:
        follow = 1
    # HEAD 6: range type — use absolute path_range (atr5-normalised)
    # p25=1.19, p75=2.55 → use p25 for compress, p75 for expand → ~25%/50%/25%
    if   path_range < 1.20:    rng_type = 0   # inside compress ~25%
    elif path_range > 2.55:    rng_type = 2   # outside expand  ~25%
    else:                       rng_type = 1   # normal          ~50%
    return (int(flow_dir), int(nxt_intent), int(nxt_mag),
            int(nxt_rej), int(follow), int(rng_type))


# ─────────────────────────────────────────────────────────────────────
# Scenarios
# ─────────────────────────────────────────────────────────────────────
SCENARIOS = {
    "impulse_clean_bull":  (0.75, 0.0034, 8000,  0.05, 0.30, 0.18, 0.09),
    "impulse_clean_bear":  (0.25, 0.0034, 8000,  0.05, 0.30, 0.18, 0.09),
    "impulse_messy_bull":  (0.70, 0.0030, 7000,  0.15, 0.58, 0.16, 0.07),
    "impulse_messy_bear":  (0.30, 0.0030, 7000,  0.15, 0.58, 0.16, 0.07),
    "wick_rejection":      (0.50, 0.0012, 5500,  0.55, 1.60, 0.14, 0.13),
    "absorption_slow":     (0.54, 0.0008, 10000, 0.25, 0.80, 0.09, 0.10),
    "absorption_active":   (0.52, 0.0007, 14000, 0.30, 0.90, 0.09, 0.08),
    "chop":                (0.50, 0.0010, 2000,  0.10, 0.52, 0.09, 0.13),
    "spike_news":          (0.50, 0.0055, 24000, 0.04, 0.22, 0.35, 0.06),
    "exhaustion_run_bull": (0.70, 0.0026, 7000,  0.10, 0.45, 0.16, 0.09),
    "exhaustion_run_bear": (0.30, 0.0026, 7000,  0.10, 0.45, 0.16, 0.09),
}
SC_KEYS    = list(SCENARIOS.keys())
SC_WEIGHTS = [SCENARIOS[k][6] for k in SC_KEYS]


def generate_sequence(base_price=BASE_PRICE):
    sc_name = random.choices(SC_KEYS, weights=SC_WEIGHTS, k=1)[0]
    tb, base_atr, base_vol, wl, wh, g_alpha, _ = SCENARIOS[sc_name]

    price = base_price + np.random.normal(0, base_atr*1.2)
    vs    = VolState(base_atr, alpha=g_alpha)
    bias  = float(np.clip(0.50+np.random.normal(0,0.06), 0.15, 0.85))
    start_hour = random.randint(0, 23)

    news_mode = sc_name == "spike_news"
    exh_mode  = "exhaustion" in sc_name

    candles = []
    for i in range(SEQ_LEN + FUTURE_BARS):
        hour  = (start_hour+i)%24
        ar    = 0.75 if news_mode else 0.82
        bias  = evolve_bias(bias, tb, ar_coef=ar, noise_std=0.06)

        if news_mode:
            ph = i/(SEQ_LEN+FUTURE_BARS)
            vs.base = base_atr*(3.8 if 0.6<ph<0.72 else 0.55)
        if exh_mode:
            ph = i/(SEQ_LEN+FUTURE_BARS)
            vs.base = base_atr*(1.0+ph if ph<0.55 else max(0.35,1.55-(ph-0.55)*3.5))

        c, price = gen_candle_h1(price, vs, bias, hour, base_vol, wl, wh,
                                  news_prob=0.06 if news_mode else 0.03)
        candles.append(c)

    df_ctx  = pd.DataFrame(candles[:SEQ_LEN])
    future  = candles[SEQ_LEN:]
    df_ctx  = compute_features(df_ctx)
    labels  = compute_labels(df_ctx, future)
    last    = df_ctx.iloc[-1]

    feat_names = [
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
    row = {fn: float(last[fn]) for fn in feat_names}
    (row["label_next_flow_dir"],   row["label_next_bar_intent"],
     row["label_next_magnitude"],  row["label_next_rejection"],
     row["label_momentum_follow"], row["label_next_range_type"]) = labels
    return row


if __name__ == "__main__":
    print(f"Generating {N_SAMPLES} Microstructure H1 samples (v4) ...")
    rows = [generate_sequence() for _ in trange(N_SAMPLES)]
    df_out = pd.DataFrame(rows)
    lc = [c for c in df_out.columns if c.startswith("label_")]
    fc = [c for c in df_out.columns if not c.startswith("label_")]
    df_out[sorted(fc)+lc].to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved → {OUTPUT_PATH}  shape={df_out.shape}")
    for c in lc:
        print(f"  {c}: {dict(df_out[c].value_counts(normalize=True).sort_index().round(3))}")
