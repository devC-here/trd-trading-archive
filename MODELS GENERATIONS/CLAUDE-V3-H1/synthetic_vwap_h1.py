# synthetic_vwap_h1.py  ·  v4  (market-realistic + sensitive)
# ─────────────────────────────────────────────────────────────────────
# H1 FX VWAP/Volume — uses shared market_physics_h1 engine
# Improvements over v3:
#   • GARCH vol clustering — quiet/explosive regimes
#   • AR(1) bias path — autocorrelated direction, natural trend/reversal
#   • Session volume profile — London/NY peaks realistically
#   • Liquidity-sweep wicks + micro-gaps
#   • 3-bar future window with MFE/MAE/pullback path metrics
#   • Path-aware labels — distinguishes "clean" from "messy" follow-through
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
LOOKBACK    = 100
FUTURE_BARS = 3
OUTPUT_PATH = "synthetic_vwap_h1_dataset.csv"
BASE_PRICE  = 1.2000
SEED        = 42
random.seed(SEED); np.random.seed(SEED)


def _rz(s, w):
    return (s - s.rolling(w).mean()) / (s.rolling(w).std() + EPS)


# ─────────────────────────────────────────────────────────────────────
# Feature computation — 32 features, same schema as v3
# ─────────────────────────────────────────────────────────────────────

def compute_vwap_features(df: pd.DataFrame, hour_vol_avg: dict = None) -> dict:
    df = df.copy().reset_index(drop=True)
    df["range"]      = df["high"] - df["low"]
    df["body"]       = df["close"] - df["open"]
    df["abs_body"]   = df["body"].abs()
    df["dir"]        = np.sign(df["body"]).fillna(0)
    df["upper_wick"] = df["high"] - df[["open","close"]].max(axis=1)
    df["lower_wick"] = df[["open","close"]].min(axis=1) - df["low"]

    hl = df["range"]
    hc = (df["high"] - df["close"].shift(1)).abs().fillna(0)
    lc = (df["low"]  - df["close"].shift(1)).abs().fillna(0)
    df["atr14"] = pd.concat([hl,hc,lc],axis=1).max(axis=1).rolling(14,min_periods=1).mean()

    tp      = (df["high"]+df["low"]+df["close"])/3
    tp_vol  = tp * df["volume"]
    vwap_50 = tp_vol.rolling(50,min_periods=1).sum()/(df["volume"].rolling(50,min_periods=1).sum()+EPS)
    df["vwap"] = vwap_50
    vwap_var   = ((tp-vwap_50)**2*df["volume"]).rolling(50,min_periods=1).sum()/(df["volume"].rolling(50,min_periods=1).sum()+EPS)
    df["vwap_std"]     = np.sqrt(vwap_var.clip(lower=0))/(df["atr14"]+EPS)
    df["vwap_dev_atr"] = (df["close"]-df["vwap"])/(df["atr14"]+EPS)
    df["vwap_sigma"]   = df["vwap_dev_atr"]/(df["vwap_std"]+EPS)
    df["vwap_slope_z"] = _rz((df["vwap"].diff(5)/(df["atr14"]+EPS)).fillna(0),30).fillna(0.0)

    vol_avg20   = df["volume"].rolling(20,min_periods=1).mean()
    vol_z_cross = (_rz(df["volume"],20)+1).clip(0,3)
    cross_bull  = ((df["close"]>df["vwap"])&(df["close"].shift(1)<=df["vwap"].shift(1))).astype(float)
    cross_bear  = ((df["close"]<df["vwap"])&(df["close"].shift(1)>=df["vwap"].shift(1))).astype(float)
    df["vwap_reclaim_qual"]   = (df["abs_body"]/(df["atr14"]+EPS)*vol_z_cross*cross_bull).fillna(0).clip(0,3)
    df["vwap_rejection_qual"] = (df["abs_body"]/(df["atr14"]+EPS)*vol_z_cross*cross_bear).fillna(0).clip(0,3)

    near_vwap = (df["vwap_dev_atr"].abs()<0.50).astype(int)
    df["vwap_touch_density"] = near_vwap.rolling(20,min_periods=1).sum()/20.0

    close_grav = (df["close"]-df["low"])/(df["range"]+EPS)
    df["tick_delta"]    = close_grav*2-1
    delta_vol           = df["tick_delta"]*df["volume"]
    df["cum_delta_20z"] = _rz(delta_vol.rolling(20,min_periods=1).sum(),50).fillna(0.0)

    up_vol = df["volume"].where(df["dir"]>0,0.0)
    dn_vol = df["volume"].where(df["dir"]<0,0.0)
    up_sum = up_vol.rolling(20,min_periods=1).sum()
    dn_sum = dn_vol.rolling(20,min_periods=1).sum()
    df["vol_profile_asym"] = ((up_sum-dn_sum)/(up_sum+dn_sum+EPS)).fillna(0.0)
    df["buy_press_z"]      = _rz(up_sum/(up_sum+dn_sum+EPS),30).fillna(0.0)
    df["delta_at_vwap"]    = _rz((delta_vol*near_vwap).rolling(20,min_periods=1).sum(),30).fillna(0.0)

    df["vol_z30"]     = _rz(df["volume"],30).fillna(0.0)
    df["vol_z100"]    = _rz(df["volume"],min(100,len(df))).fillna(0.0)
    df["vol_trend_z"] = _rz(df["volume"].rolling(5,min_periods=1).mean()/(vol_avg20+EPS),30).fillna(0.0)

    if "hour" in df.columns and hour_vol_avg is not None:
        session_expected = df["hour"].map(hour_vol_avg).fillna(vol_avg20)
        df["session_vol_factor"] = (df["volume"]/(session_expected+EPS)).clip(0,4)
    else:
        df["session_vol_factor"] = df["vol_z30"].clip(-3,3)+1

    price_dir = np.sign(df["close"].diff(3).fillna(0))
    vol_dir   = np.sign(df["volume"].rolling(3,min_periods=1).mean().diff(1).fillna(0))
    df["vol_mom_div"] = (price_dir*vol_dir).fillna(0.0)

    body_ratio = df["abs_body"]/(df["range"]+EPS)
    vol_z20    = _rz(df["volume"],20).fillna(0.0)
    df["vol_efficiency"]  = (df["abs_body"]/(df["atr14"]+EPS)/(df["vol_z30"].abs()+0.5)).clip(0,3)
    df["vol_at_vwap_z"]   = (vol_z20*near_vwap).fillna(0.0)
    df["inst_fp"]         = ((vol_z20>1.5).astype(int)+(df["vwap_dev_atr"].abs()<0.30).astype(int)+(body_ratio<0.25).astype(int)).astype(float)
    df["vwap_band_walk"]  = (df["vwap_sigma"]-1.0).clip(lower=0).rolling(10,min_periods=1).sum()
    df["price_vol_div"]   = (df["abs_body"]/(df["atr14"]+EPS)/(df["vol_z30"].abs()+0.5)).clip(0,3)
    df["absorption_lvl"]  = (vol_z20*(body_ratio<0.25).astype(float)).clip(0,3)
    df["vwap_dist_pct"]   = ((df["close"]-df["vwap"])/(df["vwap"]+EPS)*100).clip(-1,1)

    above_streak, cur = [], 0
    for v in (df["close"]>df["vwap"]).tolist():
        cur = cur+1 if v else 0; above_streak.append(cur)
    df["above_vwap_streak"]    = above_streak
    df["vwap_mr_z"]            = _rz(df["vwap_dev_atr"].abs(),50).fillna(0.0)
    df["cum_delta_vwap_align"] = (np.sign(df["vwap_dev_atr"])*np.sign(df["tick_delta"])).fillna(0.0)
    df["vol_surge_extreme"]    = (vol_z20*(df["vwap_sigma"].abs()>1.5).astype(float)).clip(0,3)
    df["body_ratio"]           = body_ratio
    df["close_grav_f"]         = close_grav
    df["vwap_std_feat"]        = df["vwap_std"].clip(0,5)
    df["atr14_z"]              = _rz(df["atr14"],50).fillna(0.0)
    df["vol_surge_20"]         = vol_z20.clip(-3,3)

    last = df.iloc[-1]
    keys = ["vwap_dev_atr","vwap_sigma","vwap_slope_z","vwap_reclaim_qual","vwap_rejection_qual",
            "vwap_touch_density","tick_delta","cum_delta_20z","vol_profile_asym","buy_press_z",
            "delta_at_vwap","vol_z30","vol_z100","vol_trend_z","session_vol_factor","vol_mom_div",
            "vol_efficiency","vol_at_vwap_z","inst_fp","vwap_band_walk","price_vol_div",
            "absorption_lvl","vwap_dist_pct","above_vwap_streak","vwap_mr_z",
            "cum_delta_vwap_align","vol_surge_extreme","body_ratio","close_grav_f",
            "vwap_std_feat","atr14_z","vol_surge_20"]
    f = {k: float(last[k]) for k in keys}
    f.update({"_atr14": float(last["atr14"]), "_vwap": float(last["vwap"]),
               "_vwap_sigma": float(last["vwap_sigma"]), "_context_close": float(last["close"]),
               "_cum_delta": float(last["cum_delta_20z"]), "_vol_z30": float(last["vol_z30"]),
               "_inst_fp": float(last["inst_fp"]), "_slope_z": float(last["vwap_slope_z"]),
               "_tick_delta": float(last["tick_delta"]), "_vol_efficiency": float(last["vol_efficiency"])})
    return f, df


# ─────────────────────────────────────────────────────────────────────
# Labels — path-aware, 3-bar window
# ─────────────────────────────────────────────────────────────────────

def compute_labels(f: dict, future_candles: list) -> tuple:
    atr  = max(f["_atr14"], EPS)
    vwap = f["_vwap"]
    ref  = f["_context_close"]
    pm   = future_path_metrics(future_candles, atr)
    net3=pm["net3"]; avg_body=pm["avg_body"]; mfe=pm["mfe"]
    pullback=pm["pullback"]; path_range=pm["path_range"]; close_grav=pm["close_grav"]
    c1_above   = future_candles[0]["close"] > vwap
    c3_above   = future_candles[-1]["close"] > vwap
    c1_crossed = (ref > vwap) != c1_above
    sigma=f["_vwap_sigma"]; cum_d=f["_cum_delta"]; slope=f["_slope_z"]
    # HEAD 1: direction — biased by VWAP side + net3 threshold at p33≈0.55
    if   net3 > 0.55 and c3_above:       vwap_dir = 2
    elif net3 < -0.55 and not c3_above:  vwap_dir = 0
    else:                                  vwap_dir = 1
    # HEAD 2: volume character — p33/p75 = 0.46/0.85
    avg_body_c = avg_body
    if   avg_body_c > 0.85:    next_vol_char = 2
    elif avg_body_c < 0.46:    next_vol_char = 0
    else:                       next_vol_char = 1
    # HEAD 3: VWAP interaction — balanced 4-class split
    # reclaim_hold: price crossed & held (pullback small) ~20%
    # walk_extension: stayed above VWAP with momentum (mfe > median) ~25%
    # rejection: closed below + directional move down ~25%
    # test_return: everything else ~30%
    if c1_crossed and pullback < 0.85:
        vwap_int = 2   # reclaim_hold (or reclaim_failed below)
        if not c3_above:   vwap_int = 0   # failed reclaim = rejection
    elif c3_above and mfe > 1.20 and net3 > 0.20:
        vwap_int = 3   # walk_extension
    elif not c3_above and net3 < -0.25:
        vwap_int = 0   # rejection
    else:
        vwap_int = 1   # test_return
    # HEAD 4: institutional signal — simpler: large directional bar = institutional
    # p33(avg_body)=0.46, p50=0.57 → use net direction + body size
    if avg_body_c > 0.62:
        if   net3 > 0.20:    next_inst = 2   # accumulation ~30%
        elif net3 < -0.20:   next_inst = 0   # distribution ~30%
        else:                 next_inst = 1   # neutral ~40%
    else:
        next_inst = 1   # thin bars = neutral
    # HEAD 5: close location — bimodal: 0.25/0.75
    if   close_grav > 0.75:  next_close = 2
    elif close_grav < 0.25:  next_close = 0
    else:                     next_close = 1
    # HEAD 6: follow-through — confirmed~25%, partial~45%, failed~30%
    bullish_ctx = (sigma > 0.30 and cum_d > 0.20 and slope > 0.15)
    bearish_ctx = (sigma < -0.30 and cum_d < -0.20 and slope < -0.15)
    if bullish_ctx:
        if net3 > 0.45 and mfe > pullback * 1.5:    follow = 2
        elif net3 > 0.12:                             follow = 1
        else:                                         follow = 0
    elif bearish_ctx:
        if net3 < -0.45 and mfe > pullback * 1.5:   follow = 2
        elif net3 < -0.12:                            follow = 1
        else:                                         follow = 0
    else:
        if avg_body_c > 0.65 and pullback < 1.0:    follow = 2
        elif avg_body_c > 0.32:                      follow = 1
        else:                                         follow = 0
    return (int(vwap_dir), int(next_vol_char), int(vwap_int),
            int(next_inst), int(next_close), int(follow))


# ─────────────────────────────────────────────────────────────────────
# Scenarios — H1 realistic physics
# (target_bias, base_atr, base_vol, wl, wh, garch_alpha, weight)
# ─────────────────────────────────────────────────────────────────────
SCENARIOS = {
    "institutional_buy":    (0.68, 0.0030, 7000,  0.08, 0.40, 0.18, 0.08),
    "institutional_sell":   (0.32, 0.0030, 7000,  0.08, 0.40, 0.18, 0.08),
    "vwap_reclaim_bull":    (0.63, 0.0026, 5500,  0.12, 0.55, 0.14, 0.08),
    "vwap_reclaim_bear":    (0.37, 0.0026, 5500,  0.12, 0.55, 0.14, 0.08),
    "vwap_rejection":       (0.50, 0.0022, 4500,  0.30, 1.10, 0.16, 0.07),
    "vol_climax_bull":      (0.63, 0.0042, 18000, 0.10, 0.38, 0.25, 0.06),
    "vol_climax_bear":      (0.37, 0.0042, 18000, 0.10, 0.38, 0.25, 0.06),
    "vol_absence":          (0.50, 0.0010, 800,   0.05, 0.28, 0.08, 0.07),
    "absorption":           (0.53, 0.0012, 10000, 0.18, 0.65, 0.10, 0.08),
    "band_walk_bull":       (0.70, 0.0034, 6000,  0.06, 0.32, 0.12, 0.07),
    "band_walk_bear":       (0.30, 0.0034, 6000,  0.06, 0.32, 0.12, 0.07),
    "chop":                 (0.50, 0.0018, 2200,  0.10, 0.50, 0.10, 0.07),
    "london_breakout_bull": (0.67, 0.0038, 9000,  0.06, 0.30, 0.22, 0.05),
    "london_breakout_bear": (0.33, 0.0038, 9000,  0.06, 0.30, 0.22, 0.05),
    "news_spike":           (0.50, 0.0055, 25000, 0.05, 0.20, 0.35, 0.03),
}
SC_KEYS    = list(SCENARIOS.keys())
SC_WEIGHTS = [SCENARIOS[k][6] for k in SC_KEYS]


def generate_sequence(base_price=BASE_PRICE):
    sc_name = random.choices(SC_KEYS, weights=SC_WEIGHTS, k=1)[0]
    tb, base_atr, base_vol, wl, wh, g_alpha, _ = SCENARIOS[sc_name]

    price = base_price + np.random.normal(0, base_atr*1.5)
    vs    = VolState(base_atr, alpha=g_alpha)
    bias  = float(np.clip(0.50 + np.random.normal(0, 0.08), 0.15, 0.85))

    # flip non-named scenarios for symmetry
    if not any(x in sc_name for x in ["bull","bear","sell","buy"]) and random.random()<0.50:
        tb = 1.0 - tb

    start_hour   = random.randint(0, 23)
    vol_climax   = "vol_climax" in sc_name
    band_walk    = "band_walk"  in sc_name
    news_mode    = sc_name == "news_spike"
    london_break = "london_breakout" in sc_name

    candles = []
    for i in range(LOOKBACK + FUTURE_BARS):
        hour  = (start_hour + i) % 24
        ar    = 0.88 if band_walk else (0.72 if news_mode else 0.83)
        bias  = evolve_bias(bias, tb, ar_coef=ar, noise_std=0.05)

        if vol_climax:
            ph = i/(LOOKBACK+FUTURE_BARS)
            vs.base = base_atr * (max(0.4, 2.5 - ph*3.5) if ph>0.6 else (1.0+ph*1.5))
        if news_mode:
            ph = i/(LOOKBACK+FUTURE_BARS)
            vs.base = base_atr*(3.5 if 0.55<ph<0.65 else 0.6)
        if london_break:
            # quiet compression then 08:00 burst
            ph = i/(LOOKBACK+FUTURE_BARS)
            vs.base = base_atr*(0.5 if ph<0.5 else min(2.5, 0.5+(ph-0.5)*8))

        c, price = gen_candle_h1(price, vs, bias, hour, base_vol, wl, wh)
        candles.append(c)

    df_context    = pd.DataFrame(candles[:LOOKBACK])
    future        = candles[LOOKBACK:]
    hour_vol_avg  = df_context.groupby("hour")["volume"].mean().to_dict()
    f, _          = compute_vwap_features(df_context, hour_vol_avg)
    labels        = compute_labels(f, future)

    row = {k: v for k, v in f.items() if not k.startswith("_")}
    (row["label_next_vwap_dir"],       row["label_next_vol_char"],
     row["label_vwap_interaction"],    row["label_next_inst_signal"],
     row["label_next_close_location"], row["label_vwap_follow_through"]) = labels
    return row


if __name__ == "__main__":
    print(f"Generating {N_SAMPLES} VWAP H1 samples (v4) ...")
    rows = [generate_sequence() for _ in trange(N_SAMPLES)]
    df_out = pd.DataFrame(rows)
    lc = [c for c in df_out.columns if c.startswith("label_")]
    fc = [c for c in df_out.columns if not c.startswith("label_")]
    df_out[sorted(fc)+lc].to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved → {OUTPUT_PATH}  shape={df_out.shape}")
    for c in lc:
        print(f"  {c}: {dict(df_out[c].value_counts(normalize=True).sort_index().round(3))}")
