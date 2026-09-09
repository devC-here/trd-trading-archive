# synthetic_candle_h1.py  ·  v4  (market-realistic + sensitive)
# ─────────────────────────────────────────────────────────────────────
# H1 FX Candle Sequence — uses shared market_physics_h1 engine
# 30 features | 6 heads | 3-bar path-aware labels
# ─────────────────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from market_physics_h1 import VolState, gen_candle_h1, evolve_bias, future_path_metrics, EPS
import pandas as pd
import numpy as np
import random
from scipy import stats as sp_stats

try:
    from tqdm import trange
except ImportError:
    def trange(n, **kw): return range(n)

N_SAMPLES   = 20000
WIN         = 10
LOOKBACK    = 50
FUTURE_BARS = 3
OUTPUT_PATH = "synthetic_candle_h1_dataset.csv"
BASE_PRICE  = 1.2000
SEED        = 42
random.seed(SEED); np.random.seed(SEED)


def _ols_slope(arr):
    n = len(arr)
    if n < 2: return 0.0
    s = np.std(arr)
    if s < EPS: return 0.0
    slope, *_ = np.polyfit(np.arange(n, dtype=float), arr, 1)
    return float(slope/(s+EPS))

def _safe_corr(a, b):
    if np.std(a) < EPS or np.std(b) < EPS: return 0.0
    r, _ = sp_stats.pearsonr(a, b)
    return float(r) if np.isfinite(r) else 0.0


# ─────────────────────────────────────────────────────────────────────
# Features (30) — same schema
# ─────────────────────────────────────────────────────────────────────

def compute_candle_features(df: pd.DataFrame) -> dict:
    df = df.copy().reset_index(drop=True)
    df["range"]      = df["high"]-df["low"]
    df["body"]       = df["close"]-df["open"]
    df["abs_body"]   = df["body"].abs()
    df["dir"]        = np.sign(df["body"]).fillna(0)
    df["upper_wick"] = df["high"]-df[["open","close"]].max(axis=1)
    df["lower_wick"] = df[["open","close"]].min(axis=1)-df["low"]
    df["atr14"]      = df["range"].rolling(14,min_periods=1).mean()
    df["abs_body_avg30"] = df["abs_body"].rolling(30,min_periods=1).mean()
    df["range_avg30"]    = df["range"].rolling(30,min_periods=1).mean()
    df["vol_avg20"]      = df["volume"].rolling(20,min_periods=1).mean()
    df["body_ratio"]     = df["abs_body"]/(df["range"]+EPS)
    df["body_rat_avg5"]  = df["body_ratio"].rolling(5,min_periods=1).mean()
    df["close_grav"]     = (df["close"]-df["low"])/(df["range"]+EPS)
    df["shadow_dom"]     = (df["lower_wick"]-df["upper_wick"])/(df["range"]+EPS)
    bull = df["dir"]>=0
    df["dir_eff"]    = np.where(bull, df["body_ratio"]*df["close_grav"],
                                       df["body_ratio"]*(1-df["close_grav"]))
    df["vol_weight"] = (df["volume"]/(df["vol_avg20"]+EPS)).clip(0,4)
    ha_close = (df["open"]+df["high"]+df["low"]+df["close"])/4
    ha_open  = ha_close.ewm(span=3,adjust=False).mean()
    df["ha_body"]    = (ha_close-ha_open)/(df["atr14"]+EPS)
    cons,cur,prev = [],[],0
    for d in df["dir"].fillna(0).tolist():
        if d==prev and d!=0: cur+=1
        elif d!=0: cur=1
        else: cur=0
        cons.append(cur); prev=d
    df["consec_len"] = cons

    w       = df.iloc[-WIN:].reset_index(drop=True)
    atr_val = float(df["atr14"].iloc[-1]) + EPS

    f = {}
    f["lb_close_grav"]    = float(w["close_grav"].iloc[-1])
    f["lb_body_ratio"]    = float(w["body_ratio"].iloc[-1])
    f["lb_shadow_dom"]    = float(w["shadow_dom"].iloc[-1])
    f["lb_vol_weight"]    = float(min(w["vol_weight"].iloc[-1], 4.0))
    f["lb_dir_eff"]       = float(w["dir_eff"].iloc[-1])
    f["seq_cg_slope"]     = _ols_slope(w["close_grav"].values)
    bm = float(w["body_ratio"].mean()); bs = float(w["body_ratio"].std())
    f["body_entropy"]     = float(bs/(bm+EPS))
    f["wick_press_slope"] = _ols_slope(w["shadow_dom"].values)
    bar_conv = w["dir"]*w["body_ratio"]*w["vol_weight"]
    f["vol_narrative"]    = _ols_slope(bar_conv.values)
    f["seq_harmony"]      = _safe_corr(w["close_grav"].values, w["range"].values)
    early_dir = np.sign(w["body"].iloc[:5].sum()); late_dir = np.sign(w["body"].iloc[5:].sum())
    f["handoff"]          = float(early_dir != late_dir)
    early_b = float(w["body_ratio"].iloc[:5].mean()); late_b = float(w["body_ratio"].iloc[5:].mean())
    f["handoff_str"]      = float((late_b-early_b)/(atr_val))
    f["seq_accel"]        = float((late_b-early_b)/(float(w["body_ratio"].mean())+EPS))
    curr_body = float(w["abs_body"].iloc[-1]); prior_body = float(w["abs_body"].iloc[-2]) if len(w)>=2 else curr_body
    f["engulf_qual"]      = float((curr_body-prior_body)/atr_val*(float(w["dir"].iloc[-1]) or 1))
    inside_count=0
    for ii in range(1,len(w)):
        if w["high"].iloc[ii]<w["high"].iloc[ii-1] and w["low"].iloc[ii]>w["low"].iloc[ii-1]: inside_count+=1
        else: break
    f["inside_energy"]    = float(inside_count*(float(w["range"].iloc[0]) if inside_count>0 else 0.0)/atr_val)
    f["ha_body_last"]     = float(w["ha_body"].iloc[-1])
    ha_arr = w["ha_body"].values
    f["ha_body_slope"]    = float((ha_arr[-1]-ha_arr[max(0,len(ha_arr)-4)])/atr_val)
    uw=float(w["upper_wick"].iloc[-1]); lw=float(w["lower_wick"].iloc[-1])
    rng=float(w["range"].iloc[-1]); br=float(w["body_ratio"].iloc[-1]); cg=float(w["close_grav"].iloc[-1])
    body_sm = 1.0 if br<0.35 else 0.5
    f["pin_bull_qual"]    = float((lw/(rng+EPS))*body_sm*cg)
    f["pin_bear_qual"]    = float((uw/(rng+EPS))*body_sm*(1-cg))
    closes = w["close"].values; cmin,cmax = float(closes.min()),float(closes.max())
    seq_pos = (closes[-1]-cmin)/(cmax-cmin+EPS)
    f["doji_context"]     = float(max(0,0.15-br)/0.15*(seq_pos-0.5))
    f["body_vs_avg"]      = float(min(float(df["abs_body"].iloc[-1])/(float(df["abs_body_avg30"].iloc[-1])+EPS),4.0))
    f["range_vs_avg"]     = float(min(float(df["range"].iloc[-1])/(float(df["range_avg30"].iloc[-1])+EPS),4.0))
    f["body_accel"]       = float(br-float(df["body_rat_avg5"].iloc[-1]))
    br1=br; br3=float(w["body_ratio"].iloc[-3:].mean()) if len(w)>=3 else br1; br10=float(w["body_ratio"].mean())
    price_up = closes[-1]>closes[0]
    f["fractal_bull"]     = float((br1>br3>br10) and price_up)
    f["fractal_bear"]     = float((br1>br3>br10) and not price_up)
    ha_trend = float(w["ha_body"].mean()); sdr = float((w["dir"]>0).sum()/max(len(w),1))
    f["_atr"]           = float(df["atr14"].iloc[-1])
    f["_ref_close"]     = float(df["close"].iloc[-1])
    f["_ha_trend_str"]  = ha_trend
    f["_seq_dir_ratio"] = sdr
    return f


# ─────────────────────────────────────────────────────────────────────
# Labels — 3-bar path-aware
# ─────────────────────────────────────────────────────────────────────

def compute_labels(f: dict, future_candles: list) -> tuple:
    atr = max(f["_atr"], EPS)
    pm  = future_path_metrics(future_candles, atr)
    net3=pm["net3"]; avg_body=pm["avg_body"]; path_range=pm["path_range"]
    close_grav=pm["close_grav"]; pullback=pm["pullback"]
    b1,b2,b3=pm["b1"],pm["b2"],pm["b3"]; d1,d2,d3=pm["d1"],pm["d2"],pm["d3"]
    c1=future_candles[0]
    c1_ba=abs(c1["close"]-c1["open"])/atr
    c1_r=(c1["high"]-c1["low"]+EPS)
    c1_uw=(c1["high"]-max(c1["open"],c1["close"]))/c1_r
    c1_lw=(min(c1["open"],c1["close"])-c1["low"])/c1_r
    ha=f["_ha_trend_str"]; sdr=f["_seq_dir_ratio"]
    # HEAD 1: direction — p33≈0.50
    if   net3 > 0.50:    nxt_dir = 2
    elif net3 < -0.50:   nxt_dir = 0
    else:                 nxt_dir = 1
    # HEAD 2: body size — p33/p75 = 0.38/0.80
    if   avg_body > 0.80:    body_sz = 2
    elif avg_body > 0.38:    body_sz = 1
    else:                     body_sz = 0
    # HEAD 3: candle pattern — rebalanced for real H1 frequency
    # impulse: ~40%, engulf: ~20%, doji+pin: ~20%, inside: ~20%
    if c1_ba < 0.18:
        nxt_pat = 2   # doji/inside ~15%
    elif (c1_lw>0.30 and c1_uw<0.30 and c1_ba<0.50):
        nxt_pat = 1   # pin bull ~10%
    elif (c1_uw>0.30 and c1_lw<0.30 and c1_ba<0.50):
        nxt_pat = 1   # pin bear ~10%
    elif c1_ba > 0.55 and c1_ba > b2*1.10:
        nxt_pat = 3   # engulf/marubozu ~25%
    else:
        nxt_pat = 0   # impulse/normal ~50% (dominant, real market)
    # HEAD 4: sequence continuation — symmetric thresholds, balanced classes
    # Use absolute ha + equal sdr windows around 0.50
    # strong_bull: ha>0.08 AND sdr>0.58; strong_bear: ha<-0.08 AND sdr<0.42
    # reversal: context strongly DISAGREES with nxt_dir
    # Target: continuation~30%, indecision~45%, reversal~25%
    strong_bull = (ha > 0.08 and sdr > 0.58)
    strong_bear = (ha < -0.08 and sdr < 0.42)
    if strong_bull:
        if   nxt_dir == 2:  seq_cont = 2   # continuation
        elif nxt_dir == 0:  seq_cont = 0   # reversal (context bull, price fell)
        else:                seq_cont = 1
    elif strong_bear:
        if   nxt_dir == 0:  seq_cont = 2   # continuation (bear context + price fell)
        elif nxt_dir == 2:  seq_cont = 0   # reversal
        else:                seq_cont = 1
    else:
        seq_cont = 1   # no strong context = indecision
    # HEAD 5: close gravity — bimodal: 0.25/0.75
    if   close_grav > 0.75:  nxt_cg = 2
    elif close_grav < 0.25:  nxt_cg = 0
    else:                     nxt_cg = 1
    # HEAD 6: volatility — p25/p75 = 1.27/2.64
    if   path_range > 2.50:    nxt_vol = 2
    elif path_range < 1.30:    nxt_vol = 0
    else:                       nxt_vol = 1
    return (int(nxt_dir), int(body_sz), int(nxt_pat),
            int(seq_cont), int(nxt_cg), int(nxt_vol))


# ─────────────────────────────────────────────────────────────────────
# Scenarios
# ─────────────────────────────────────────────────────────────────────
SCENARIOS = {
    "bull_impulse":   (0.75, 0.0034, 8000,  0.05, 0.32, 0.18, 0.09),
    "bear_impulse":   (0.25, 0.0034, 8000,  0.05, 0.32, 0.18, 0.09),
    "pin_bull":       (0.52, 0.0008, 6000,  2.20, 5.00, 0.12, 0.07),
    "pin_bear":       (0.48, 0.0008, 6000,  2.20, 5.00, 0.12, 0.07),
    "inside_coil":    (0.50, 0.0010, 2000,  0.10, 0.55, 0.09, 0.09),
    "bull_trend":     (0.68, 0.0026, 6000,  0.08, 0.45, 0.14, 0.10),
    "bear_trend":     (0.32, 0.0026, 6000,  0.08, 0.45, 0.14, 0.10),
    "exhaustion_bull":(0.68, 0.0022, 6500,  0.08, 0.40, 0.16, 0.045),
    "exhaustion_bear":(0.32, 0.0022, 6500,  0.08, 0.40, 0.16, 0.045),
    "chop":           (0.50, 0.0012, 2200,  0.10, 0.52, 0.09, 0.11),
    "reversal_bull":  (0.62, 0.0028, 7000,  0.12, 0.55, 0.15, 0.09),
}
SC_KEYS    = list(SCENARIOS.keys())
SC_WEIGHTS = [SCENARIOS[k][6] for k in SC_KEYS]


def generate_sequence(base_price=BASE_PRICE):
    sc_name = random.choices(SC_KEYS, weights=SC_WEIGHTS, k=1)[0]
    tb, base_atr, base_vol, wl, wh, g_alpha, _ = SCENARIOS[sc_name]

    price = base_price + np.random.normal(0, base_atr*1.2)
    vs    = VolState(base_atr, alpha=g_alpha)
    bias  = float(np.clip(0.50+np.random.normal(0,0.07), 0.15, 0.85))
    start_hour = random.randint(0, 23)

    inside_coil = sc_name == "inside_coil"
    exhaustion  = sc_name == "exhaustion"

    candles = []
    for i in range(LOOKBACK + FUTURE_BARS):
        hour  = (start_hour+i)%24
        ar    = 0.88 if "trend" in sc_name else 0.80
        bias  = evolve_bias(bias, tb, ar_coef=ar, noise_std=0.06)

        if inside_coil:
            ph = i/(LOOKBACK+FUTURE_BARS)
            vs.base = base_atr*max(0.25, 1.0-ph*0.75)
        if exhaustion:
            ph = i/(LOOKBACK+FUTURE_BARS)
            vs.base = base_atr*(1.0+ph*0.8 if ph<0.6 else max(0.30,1.8-(ph-0.6)*4.0))
        if sc_name in ("pin_bull","pin_bear") and i == LOOKBACK-1:
            # force a pin bar just before future window
            wl2,wh2 = 3.0,6.0
            c, price = gen_candle_h1(price, vs, bias, hour, base_vol, wl2, wh2)
            candles.append(c); continue

        c, price = gen_candle_h1(price, vs, bias, hour, base_vol, wl, wh)
        candles.append(c)

    df_ctx  = pd.DataFrame(candles[:LOOKBACK])
    future  = candles[LOOKBACK:]

    df_ctx["range"]      = df_ctx["high"]-df_ctx["low"]
    df_ctx["body"]       = df_ctx["close"]-df_ctx["open"]
    df_ctx["abs_body"]   = df_ctx["body"].abs()
    df_ctx["dir"]        = np.sign(df_ctx["body"]).fillna(0)
    df_ctx["upper_wick"] = df_ctx["high"]-df_ctx[["open","close"]].max(axis=1)
    df_ctx["lower_wick"] = df_ctx[["open","close"]].min(axis=1)-df_ctx["low"]
    df_ctx["atr14"]      = df_ctx["range"].rolling(14,min_periods=1).mean()
    df_ctx["abs_body_avg30"] = df_ctx["abs_body"].rolling(30,min_periods=1).mean()
    df_ctx["range_avg30"]    = df_ctx["range"].rolling(30,min_periods=1).mean()
    df_ctx["vol_avg20"]      = df_ctx["volume"].rolling(20,min_periods=1).mean()
    df_ctx["body_ratio"]     = df_ctx["abs_body"]/(df_ctx["range"]+EPS)
    df_ctx["body_rat_avg5"]  = df_ctx["body_ratio"].rolling(5,min_periods=1).mean()
    df_ctx["close_grav"]     = (df_ctx["close"]-df_ctx["low"])/(df_ctx["range"]+EPS)
    df_ctx["shadow_dom"]     = (df_ctx["lower_wick"]-df_ctx["upper_wick"])/(df_ctx["range"]+EPS)
    bull2 = df_ctx["dir"]>=0
    df_ctx["dir_eff"]    = np.where(bull2,df_ctx["body_ratio"]*df_ctx["close_grav"],df_ctx["body_ratio"]*(1-df_ctx["close_grav"]))
    df_ctx["vol_weight"] = (df_ctx["volume"]/(df_ctx["vol_avg20"]+EPS)).clip(0,4)
    hc2 = (df_ctx["open"]+df_ctx["high"]+df_ctx["low"]+df_ctx["close"])/4
    df_ctx["ha_body"] = (hc2-hc2.ewm(span=3,adjust=False).mean())/(df_ctx["atr14"]+EPS)
    cons2,cur2,prev2=[],0,0
    for d in df_ctx["dir"].fillna(0).tolist():
        if d==prev2 and d!=0: cur2+=1
        elif d!=0: cur2=1
        else: cur2=0
        cons2.append(cur2); prev2=d
    df_ctx["consec_len"] = cons2

    feat   = compute_candle_features(df_ctx)
    labels = compute_labels(feat, future)

    row = {k: v for k, v in feat.items() if not k.startswith("_")}
    row["label_next_direction"],  row["label_next_body_size"], \
    row["label_next_pattern"],    row["label_seq_continuation"], \
    row["label_next_close_grav"], row["label_next_volatility"] = labels
    return row


if __name__ == "__main__":
    print(f"Generating {N_SAMPLES} Candle H1 samples (v4) ...")
    rows = [generate_sequence() for _ in trange(N_SAMPLES)]
    df_out = pd.DataFrame(rows)
    lc = [c for c in df_out.columns if c.startswith("label_")]
    fc = [c for c in df_out.columns if not c.startswith("label_")]
    df_out[sorted(fc)+lc].to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved → {OUTPUT_PATH}  shape={df_out.shape}")
    for c in lc:
        print(f"  {c}: {dict(df_out[c].value_counts(normalize=True).sort_index().round(3))}")
