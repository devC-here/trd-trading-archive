# synthetic_breakout_h1.py  ·  v4  (market-realistic + sensitive)
# ─────────────────────────────────────────────────────────────────────
# H1 FX Breakout/Squeeze — uses shared market_physics_h1 engine
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

N_SAMPLES    = 20000
CONTEXT_BARS = 120
FUTURE_BARS  = 3
OUTPUT_PATH  = "synthetic_breakout_h1_dataset.csv"
BASE_PRICE   = 1.2000
SEED         = 42
random.seed(SEED); np.random.seed(SEED)


def _rz(s, w):
    return (s - s.rolling(w).mean()) / (s.rolling(w).std() + EPS)


# ─────────────────────────────────────────────────────────────────────
# Features (32) — same schema
# ─────────────────────────────────────────────────────────────────────

def compute_breakout_features(df: pd.DataFrame):
    df = df.copy().reset_index(drop=True)
    df["range"]      = df["high"]-df["low"]
    df["body"]       = df["close"]-df["open"]
    df["abs_body"]   = df["body"].abs()
    df["dir"]        = np.sign(df["body"]).fillna(0)
    df["upper_wick"] = df["high"]-df[["open","close"]].max(axis=1)
    df["lower_wick"] = df[["open","close"]].min(axis=1)-df["low"]
    df["atr14"]      = df["range"].rolling(14,min_periods=1).mean()
    df["atr5"]       = df["range"].rolling(5, min_periods=1).mean()

    for w in [8,14,20]:
        mid  = df["close"].rolling(w,min_periods=1).mean()
        std  = df["close"].rolling(w,min_periods=1).std().fillna(0)
        atrw = df["range"].rolling(w,min_periods=1).mean()
        df[f"bb_upper_{w}"] = mid+2*std; df[f"bb_lower_{w}"] = mid-2*std
        df[f"bb_width_{w}"] = 4*std;     df[f"kc_width_{w}"] = 3*atrw
        df[f"kc_mid_{w}"]   = mid
        df[f"kc_upper_{w}"] = mid+1.5*atrw; df[f"kc_lower_{w}"] = mid-1.5*atrw

    df["bb_width_z"]  = _rz(df["bb_width_14"],100).fillna(0.0)
    df["kc_width_n"]  = df["kc_width_14"]/(df["atr14"]+EPS)
    sq8  = (df["bb_width_8"] <df["kc_width_8"]).astype(int)
    sq14 = (df["bb_width_14"]<df["kc_width_14"]).astype(int)
    sq20 = (df["bb_width_20"]<df["kc_width_20"]).astype(int)
    df["squeeze_stack"] = (sq8+sq14+sq20).astype(float)
    df["kb_ratio"]  = df["bb_width_14"]/(df["kc_width_14"]+EPS)
    df["kb_slope"]  = _rz(df["kb_ratio"].diff(1).rolling(10,min_periods=1).mean(),20).fillna(0.0)

    sq_s = (df["kb_ratio"]<1.0).astype(int)
    cs=[]; c=0
    for s in sq_s.tolist(): c=c+1 if s==1 else 0; cs.append(c)
    df["consec_squeeze"]   = cs
    df["sq_active"]        = sq_s
    avg_sq = max(float(pd.Series(cs).rolling(100,min_periods=5).mean().iloc[-1]),1.0)
    df["squeeze_maturity"] = (df["consec_squeeze"]/avg_sq).clip(0,4)
    hv_pct = df["bb_width_14"].rank(pct=True).fillna(0.5)
    df["vol_potential_e"]  = ((1-hv_pct)*(1-df["kb_ratio"]).clip(0,2)).clip(0,2)

    dh   = df["high"].rolling(20,min_periods=1).max(); dl = df["low"].rolling(20,min_periods=1).min()
    dm   = (dh+dl)/2; dw = (dh-dl)+EPS
    lean = ((df["close"]-dm)/dw).rolling(3,min_periods=1).mean()
    df["sq_lean"]     = (lean*sq_s).fillna(0.0)
    df["atr_ratio"]   = df["atr5"]/(df["atr14"]+EPS)
    df["atr_ratio_z"] = _rz(df["atr_ratio"],50).fillna(0.0)
    df["exp_velocity"]= _rz(df["bb_width_14"].diff(3).fillna(0),50).fillna(0.0)
    abv = (df["close"]-df["bb_upper_14"]).clip(lower=0)/(df["atr14"]+EPS)
    blw = (df["bb_lower_14"]-df["close"]).clip(lower=0)/(df["atr14"]+EPS)
    df["band_walk_bull"] = abv.rolling(5,min_periods=1).sum()
    df["band_walk_bear"] = blw.rolling(5,min_periods=1).sum()
    df["hv_percentile"]  = hv_pct
    bbr = (df["bb_upper_14"]-df["bb_lower_14"])+EPS
    df["bb_position"] = (df["close"]-df["bb_lower_14"])/bbr
    df["bb_pos_z"]    = _rz(df["bb_position"],50).fillna(0.0)
    df["kc_position"] = (df["close"]-df["kc_lower_14"])/(df["kc_width_14"]+EPS)
    mid14 = df["close"].rolling(14,min_periods=1).mean()
    df["mr_dist"]         = ((df["close"]-mid14)/(df["atr14"]+EPS)).clip(-3,3)
    df["coil_tightness"]  = (1-df["kb_ratio"]).clip(-1,1)
    ba20  = df["abs_body"].rolling(20,min_periods=1).mean()
    df["body_comp_r"]     = (df["abs_body"]/(ba20+EPS)).clip(0,3)
    df["close_grav"]      = (df["close"]-df["low"])/(df["range"]+EPS)
    va20  = df["volume"].rolling(20,min_periods=1).mean()
    df["vol_z20"]         = _rz(df["volume"],20).fillna(0.0)
    df["prebreak_qual"]   = (df["close_grav"]*df["body_comp_r"]*df["vol_z20"].clip(0,3)).clip(-3,3)
    df["intrabar_fill"]   = df["abs_body"]/(df["range"]+EPS)
    df["vol_sq_div"]      = (df["vol_z20"]*(df["squeeze_stack"]/3.0)).clip(-3,3)
    ca = (df["close"]>df["bb_upper_14"]).astype(float); cb = (df["close"]<df["bb_lower_14"]).astype(float)
    df["bkout_confirm"]   = ((ca-cb)*(df["vol_z20"]+1).clip(0,3)).clip(-3,3)
    df["don_bias"]        = ((df["close"]-dm)/dw).clip(-1,1)
    df["vol_z30"]         = _rz(df["volume"],30).fillna(0.0)
    df["vol_z100"]        = _rz(df["volume"],min(100,len(df))).fillna(0.0)
    df["vol_trend"]       = _rz(df["volume"].rolling(5,min_periods=1).mean()/(va20+EPS),30).fillna(0.0)

    last = df.iloc[-1]
    feat = {k: float(last[k]) for k in [
        "bb_width_z","kc_width_n","squeeze_stack","kb_ratio","kb_slope","squeeze_maturity",
        "vol_potential_e","sq_lean","atr_ratio_z","exp_velocity","band_walk_bull","band_walk_bear",
        "hv_percentile","bb_position","bb_pos_z","kc_position","mr_dist","coil_tightness",
        "body_comp_r","prebreak_qual","intrabar_fill","vol_sq_div","bkout_confirm","don_bias",
        "vol_z30","vol_z100","vol_trend","close_grav","atr14_val" if "atr14_val" in df.columns else "atr14",
    ]}
    # ensure atr14_val exists
    feat["atr14_val"]     = float(last["atr14"])
    feat["body_comp_r_z"] = float(_rz(df["body_comp_r"],50).iloc[-1]) if len(df)>=50 else 0.0
    feat["vol_z20_last"]  = float(last["vol_z20"])
    feat["_atr14"]        = float(last["atr14"])
    feat["_sq_lean"]      = float(last["sq_lean"])
    feat["_bkout_confirm"]= float(last["bkout_confirm"])
    return feat, df


# ─────────────────────────────────────────────────────────────────────
# Labels — 3-bar path-aware
# ─────────────────────────────────────────────────────────────────────

def compute_labels(f: dict, future_candles: list, context_close: float) -> tuple:
    atr = max(f["_atr14"], EPS)
    pm  = future_path_metrics(future_candles, atr)
    net3=pm["net3"]; avg_body=pm["avg_body"]; path_range=pm["path_range"]
    close_grav=pm["close_grav"]; mfe=pm["mfe"]; pullback=pm["pullback"]
    d1,d2,d3=pm["d1"],pm["d2"],pm["d3"]; b1,b2,b3=pm["b1"],pm["b2"],pm["b3"]
    sq_lean=f["_sq_lean"]; bk_confirm=f["_bkout_confirm"]
    # HEAD 1: direction — p33≈0.56
    if   net3 > 0.56:    next_dir = 2
    elif net3 < -0.56:   next_dir = 0
    else:                 next_dir = 1
    # HEAD 2: magnitude — p33/p75 = 0.43/0.85
    if   avg_body > 0.85:    next_mag = 2
    elif avg_body > 0.43:    next_mag = 1
    else:                     next_mag = 0
    # HEAD 3: pattern — 3-bar direction flow
    all_same=(d1==d2==d3)
    if all_same:
        next_pat = 0 if avg_body > 0.35 else 2
    else:
        flip12=(d1!=d2); flip23=(d2!=d3)
        if (flip12 and b2>b1*0.65) or (flip23 and b3>b2*0.65):
            next_pat = 1
        else:
            next_pat = 2
    # HEAD 4: volatility — breakout: p30/p70 for balance
    if   path_range > 2.60:    next_vol = 2   # expanding  ~30%
    elif path_range < 1.30:    next_vol = 0   # contracting ~25%
    else:                       next_vol = 1   # normal      ~45%
    # HEAD 5: close location — bimodal: 0.25/0.75
    if   close_grav > 0.75:  next_loc = 2
    elif close_grav < 0.25:  next_loc = 0
    else:                     next_loc = 1
    # HEAD 6: follow-through — MFE/pullback quality
    exp_bull=(sq_lean>0.05 or bk_confirm>0.30)
    exp_bear=(sq_lean<-0.05 or bk_confirm<-0.30)
    if exp_bull:
        follow = 2 if (net3>0.55 and mfe>pullback*1.8) else (1 if net3>0.15 else 0)
    elif exp_bear:
        follow = 2 if (net3<-0.55 and mfe>pullback*1.8) else (1 if net3<-0.15 else 0)
    else:
        follow = 2 if (avg_body>0.75 and pullback<1.0) else (1 if avg_body>0.40 else 0)
    return (int(next_dir), int(next_mag), int(next_pat),
            int(next_vol), int(next_loc), int(follow))


# ─────────────────────────────────────────────────────────────────────
# Scenarios
# (target_bias, base_atr, base_vol, wl, wh, garch_alpha, weight, sqb_frac)
# sqb_frac: fraction of CONTEXT_BARS to spend in squeeze compression
# ─────────────────────────────────────────────────────────────────────
SCENARIOS = {
    "deep_squeeze":         (0.50, 0.0008, 1500,  0.05, 0.28, 0.08, 0.08, 0.70),
    "squeeze_forming":      (0.50, 0.0016, 3000,  0.10, 0.48, 0.11, 0.09, 0.25),
    "bull_breakout":        (0.74, 0.0055, 12000, 0.05, 0.26, 0.28, 0.11, 0.20),
    "bear_breakout":        (0.26, 0.0055, 12000, 0.05, 0.26, 0.28, 0.11, 0.20),
    "false_bull_break":     (0.58, 0.0030, 5000,  0.45, 1.40, 0.18, 0.07, 0.15),
    "false_bear_break":     (0.42, 0.0030, 5000,  0.45, 1.40, 0.18, 0.07, 0.15),
    "trending_expansion":   (0.67, 0.0034, 7500,  0.10, 0.38, 0.16, 0.09, 0.00),
    "overdue_squeeze":      (0.50, 0.0007, 1200,  0.04, 0.20, 0.07, 0.09, 0.90),
    "chop":                 (0.50, 0.0018, 2500,  0.15, 0.60, 0.10, 0.11, 0.00),
    "vol_climax_bull":      (0.60, 0.0060, 22000, 0.07, 0.32, 0.32, 0.09, 0.10),
    "vol_climax_bear":      (0.40, 0.0060, 22000, 0.07, 0.32, 0.32, 0.09, 0.10),
}
SC_KEYS    = list(SCENARIOS.keys())
SC_WEIGHTS = [SCENARIOS[k][6] for k in SC_KEYS]


def generate_sequence(base_price=BASE_PRICE):
    sc_name = random.choices(SC_KEYS, weights=SC_WEIGHTS, k=1)[0]
    tb, base_atr, base_vol, wl, wh, g_alpha, _, sqb_frac = SCENARIOS[sc_name]
    sqb = int(sqb_frac * CONTEXT_BARS)

    price = base_price + np.random.normal(0, base_atr*1.5)
    vs    = VolState(base_atr, alpha=g_alpha)
    bias  = float(np.clip(0.50+np.random.normal(0,0.07), 0.15, 0.85))
    start_hour = random.randint(0, 23)

    false_break  = "false" in sc_name
    vol_climax   = "vol_climax" in sc_name
    breakout     = "breakout" in sc_name and "false" not in sc_name

    candles = []
    for i in range(CONTEXT_BARS + FUTURE_BARS):
        hour = (start_hour+i)%24
        ar   = 0.88 if "trending" in sc_name else 0.80
        bias = evolve_bias(bias, tb, ar_coef=ar, noise_std=0.055)

        # Squeeze compression phase
        if sqb > 0 and i < sqb:
            compress = max(0.18, 1.0 - i/sqb*0.70)
            vs.base  = base_atr*compress
        elif sqb > 0:
            # Post-squeeze expansion
            ph = (i-sqb)/(CONTEXT_BARS-sqb+1)
            vs.base = base_atr*(1.0+min(ph*3.0, 2.5))

        if vol_climax:
            ph = i/(CONTEXT_BARS+FUTURE_BARS)
            vs.base = base_atr*(max(0.5,3.5-ph*5) if ph>0.5 else (1.0+ph*4))
        if breakout:
            ph = i/(CONTEXT_BARS+FUTURE_BARS)
            vs.base = base_atr*(0.4 if ph<0.55 else min(3.2,0.4+(ph-0.55)*12))
        if false_break:
            ph = i/(CONTEXT_BARS+FUTURE_BARS)
            if ph>0.75:   # reversal zone — bigger wicks
                wl2,wh2 = wl*1.6, wh*2.4
                c, price = gen_candle_h1(price, vs, bias, hour, base_vol, wl2, wh2)
                candles.append(c); continue

        c, price = gen_candle_h1(price, vs, bias, hour, base_vol, wl, wh)
        candles.append(c)

    df_ctx  = pd.DataFrame(candles[:CONTEXT_BARS])
    future  = candles[CONTEXT_BARS:]
    f, _    = compute_breakout_features(df_ctx)
    labels  = compute_labels(f, future, df_ctx["close"].iloc[-1])

    feat = {k: v for k, v in f.items() if not k.startswith("_")}
    row  = dict(feat)
    (row["label_next_direction"], row["label_next_magnitude"],  row["label_next_pattern"],
     row["label_next_volatility"],row["label_next_close_loc"],  row["label_bk_follow_through"]) = labels
    return row


if __name__ == "__main__":
    print(f"Generating {N_SAMPLES} Breakout H1 samples (v4) ...")
    rows = [generate_sequence() for _ in trange(N_SAMPLES)]
    df_out = pd.DataFrame(rows)
    lc = [c for c in df_out.columns if c.startswith("label_")]
    fc = [c for c in df_out.columns if not c.startswith("label_")]
    df_out[sorted(fc)+lc].to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved → {OUTPUT_PATH}  shape={df_out.shape}")
    for c in lc:
        print(f"  {c}: {dict(df_out[c].value_counts(normalize=True).sort_index().round(3))}")
