# synthetic_momentum_h1.py  ·  v4  (market-realistic + sensitive)
# ─────────────────────────────────────────────────────────────────────
# H1 FX Momentum — uses shared market_physics_h1 engine
# 30 features | 6 heads | 3-bar path-aware labels
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

N_SAMPLES   = 25000
SEQ_LEN     = 120
FUTURE_BARS = 3
OUTPUT_PATH = "synthetic_momentum_h1_dataset.csv"
BASE_PRICE  = 1.0800
SEED        = 42
random.seed(SEED); np.random.seed(SEED)


def rolling_z(series, window):
    return (series - series.rolling(window).mean()) / (series.rolling(window).std() + EPS)

def rolling_autocorr(series, window, lag=1):
    def _ac(arr):
        s = pd.Series(arr)
        r = s.autocorr(lag=lag)
        return r if (len(arr) >= lag+2 and np.isfinite(r if r is not None else float('nan'))) else 0.0
    return series.rolling(window).apply(_ac, raw=False)


# ─────────────────────────────────────────────────────────────────────
# Features (30) — unchanged schema
# ─────────────────────────────────────────────────────────────────────

def compute_momentum_features(df: pd.DataFrame) -> dict:
    df = df.copy().reset_index(drop=True)
    close  = df["close"].astype(float)
    high   = df["high"].astype(float)
    low    = df["low"].astype(float)
    volume = df["volume"].astype(float)

    tr  = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()],axis=1).max(axis=1)
    atr = tr.rolling(14).mean(); atr_s = atr + EPS

    ret3 = close.pct_change(3)
    ret3_atr  = (ret3/(atr_s/(close+EPS))).fillna(0.0)
    ret3_z    = rolling_z(ret3,50).fillna(0.0)

    direction = (close-close.shift(10)).abs()
    noise     = close.diff().abs().rolling(10).sum()
    er        = (direction/(noise+EPS)).fillna(0.0)
    er_slope  = er.diff(5).fillna(0.0)

    ret1   = close.pct_change(1).fillna(0.0)
    mom_p  = rolling_autocorr(ret1,30).fillna(0.0)

    ema20      = close.ewm(span=20,adjust=False).mean()
    ema_slope  = ema20.diff(5)
    ema_sn     = (ema_slope/atr_s).fillna(0.0)
    ema_acc    = (ema_slope.diff(3)/atr_s).fillna(0.0)

    atr_z  = rolling_z(atr,50).fillna(0.0)
    rng    = high-low
    rng_ex = (rng.rolling(3).mean()/(rng.rolling(20).mean()+EPS)).fillna(1.0)
    vol_rc = (atr_z*(rng_ex-1.0)).clip(-4,4).fillna(0.0)

    rmin20 = close.rolling(20).min(); rmax20 = close.rolling(20).max()
    trend_pos = ((close-rmin20)/(rmax20-rmin20+EPS)).fillna(0.5)
    bks = ((close-high.rolling(20).max().shift(1))/atr_s).clip(-3,3).fillna(0.0)
    mrs = ((close-close.rolling(50).mean())/atr_s).clip(-4,4).fillna(0.0)

    udm = high.diff(); ddm = (-low.diff())
    pdm = udm.where((udm>ddm)&(udm>0),0.0); mdm = ddm.where((ddm>udm)&(ddm>0),0.0)
    satr = atr.ewm(span=14,adjust=False).mean()+EPS
    pdi  = 100*(pdm.ewm(span=14,adjust=False).mean()/satr)
    mdi  = 100*(mdm.ewm(span=14,adjust=False).mean()/satr)
    dx   = 100*(pdi-mdi).abs()/(pdi+mdi+EPS)
    adx  = dx.ewm(span=14,adjust=False).mean()
    adxz = rolling_z(adx,50).fillna(0.0)
    adxs = rolling_z(adx,30).diff(3).fillna(0.0)
    dib  = ((pdi-mdi)/(pdi+mdi+EPS)).fillna(0.0)

    tk = (high.rolling(9).max()+low.rolling(9).min())/2
    kj = (high.rolling(26).max()+low.rolling(26).min())/2
    sa = ((tk+kj)/2).shift(26)
    sb = ((high.rolling(52).max()+low.rolling(52).min())/2).shift(26)
    tks = ((tk-kj)/atr_s).clip(-3,3).fillna(0.0)
    clt = ((sa-sb).abs()/atr_s).clip(0,5).fillna(0.0)
    pvc = ((close-sa)/atr_s).clip(-4,4).fillna(0.0)
    ctw = np.sign(sa-sb).fillna(0.0)

    dh = high.rolling(20).max(); dl = low.rolling(20).min()
    dw = dh-dl
    dpb = ((close-dl)/(dw+EPS)).fillna(0.5)
    dwz = rolling_z(dw,50).fillna(0.0)
    dbk = ((close-dh.shift(1))/atr_s).clip(-3,3).fillna(0.0)

    ret5  = close.pct_change(5)
    r1n   = (ret1/(atr_s/(close+EPS))); r5n = (ret5/(atr_s/(close+EPS)))
    mdec  = (r1n/((r5n/5)+EPS)).clip(-5,5).fillna(0.0)
    tens  = (ret3_atr*(0.5-trend_pos)).clip(-4,4).fillna(0.0)
    ret10 = close.pct_change(10)
    r10n  = (ret10/(atr_s/(close+EPS))).fillna(0.0)
    r10z  = rolling_z(ret10,50).fillna(0.0)

    ema50  = close.ewm(span=50,adjust=False).mean()
    sla    = (np.sign(ema_slope)*np.sign(ema50.diff(10))).fillna(0.0)
    volz   = rolling_z(volume,30).fillna(0.0)
    vdisp  = (volz*ret3_z).clip(-5,5).fillna(0.0)

    i = -1
    return {
        "ret_3_atr_norm": float(ret3_atr.iloc[i]),    "ret_3_z": float(ret3_z.iloc[i]),
        "efficiency_ratio": float(er.iloc[i]),         "er_slope": float(er_slope.iloc[i]),
        "momentum_persistence": float(mom_p.iloc[i]),  "ema_slope_norm": float(ema_sn.iloc[i]),
        "ema_acc_norm": float(ema_acc.iloc[i]),         "atr_z": float(atr_z.iloc[i]),
        "range_expansion": float(rng_ex.iloc[i]),       "vol_regime_confirm": float(vol_rc.iloc[i]),
        "trend_position": float(trend_pos.iloc[i]),     "break_strength": float(bks.iloc[i]),
        "mean_reversion_stretch": float(mrs.iloc[i]),   "adx_z": float(adxz.iloc[i]),
        "adx_slope": float(adxs.iloc[i]),               "di_balance": float(dib.iloc[i]),
        "tk_spread": float(tks.iloc[i]),                "cloud_thickness": float(clt.iloc[i]),
        "price_vs_cloud": float(pvc.iloc[i]),           "cloud_twist": float(ctw.iloc[i]),
        "don_position": float(dpb.iloc[i]),             "don_width_z": float(dwz.iloc[i]),
        "don_breakout": float(dbk.iloc[i]),             "momentum_decay": float(mdec.iloc[i]),
        "tension": float(tens.iloc[i]),                 "ret_10_atr_norm": float(r10n.iloc[i]),
        "ret_10_z": float(r10z.iloc[i]),                "slope_alignment": float(sla.iloc[i]),
        "vol_displacement": float(vdisp.iloc[i]),       "volume_z": float(volz.iloc[i]),
        "_atr": float(atr.iloc[i]),                     "_context_close": float(close.iloc[i]),
        "_di_balance": float(dib.iloc[i]),              "_trend_position": float(trend_pos.iloc[i]),
        "_ema_slope_norm": float(ema_sn.iloc[i]),       "_mean_rev_stretch": float(mrs.iloc[i]),
        "_slope_alignment": float(sla.iloc[i]),         "_momentum_persistence": float(mom_p.iloc[i]),
    }


# ─────────────────────────────────────────────────────────────────────
# Labels — 3-bar path-aware
# ─────────────────────────────────────────────────────────────────────

def label_momentum_sample(features: dict, future_candles: list) -> tuple:
    atr = max(features["_atr"], EPS)
    pm  = future_path_metrics(future_candles, atr)
    net3=pm["net3"]; avg_body=pm["avg_body"]; path_range=pm["path_range"]
    pullback=pm["pullback"]; mfe=pm["mfe"]; close_grav=pm["close_grav"]
    d1,d2,d3=pm["d1"],pm["d2"],pm["d3"]; b1,b2,b3=pm["b1"],pm["b2"],pm["b3"]
    # HEAD 1: direction — p33≈0.56
    if   net3 > 0.56:    next_dir = 2
    elif net3 < -0.56:   next_dir = 0
    else:                 next_dir = 1
    # HEAD 2: magnitude — p33/p75 = 0.47/0.83
    if   avg_body > 0.83:    next_mag = 2
    elif avg_body > 0.47:    next_mag = 1
    else:                    next_mag = 0
    # HEAD 3: regime — path-aware, rebalanced
    same12=(d1==d2); same23=(d2==d3); all_same=same12 and same23
    if avg_body < 0.42 and path_range < 1.50:
        nxt_reg = 0   # chop ~25%
    elif not same12 and b2 > b1*0.55:
        nxt_reg = 1   # pullback ~30%
    elif all_same and path_range > 2.00:
        nxt_reg = 3   # breakout ~12%
    elif same12:
        nxt_reg = 2   # continuation ~25%
    else:
        nxt_reg = 0   # noisy chop ~8%
    # HEAD 4: volatility — p25/p75 = 1.27/2.61
    if   path_range > 2.50:    next_vol = 2
    elif path_range < 1.30:    next_vol = 0
    else:                       next_vol = 1
    # HEAD 5: structural bias — bimodal close_grav 0.25/0.75
    if   close_grav > 0.75 and net3 > 0:    nxt_struct = 2
    elif close_grav < 0.25 and net3 < 0:    nxt_struct = 0
    else:                                     nxt_struct = 1
    # HEAD 6: momentum quality — b3/b1 ratio with same-direction filter
    # accelerating: b3>b1*1.10 AND same_dir → fires ~35%*35%=12% with 1.50, ~35%*55%=19% with 1.10
    # fading: direction flip (regardless of size) → ~65% of times direction changes = 65%*(1-35%)=42% NO
    # Real approach: use b3/b1 ratio split at p33 and p67 of distribution
    # b3/b1 p33≈0.49, p67≈1.55 (measured) — use these as natural boundaries
    b3b1 = b3 / (b1 + EPS)
    if b3b1 > 1.55 and same12:
        nxt_mom = 2   # accelerating: large b3 relative to b1 AND continuing ~25%
    elif b3b1 < 0.49 or (not same12):
        nxt_mom = 0   # fading: b3 much smaller than b1 OR direction changed ~35%
    else:
        nxt_mom = 1   # stable: moderate ratio, same direction ~40%
    return (int(next_dir), int(next_mag), int(nxt_reg),
            int(next_vol), int(nxt_struct), int(nxt_mom))


# ─────────────────────────────────────────────────────────────────────
# Scenarios
# (target_bias, base_atr, base_vol, wl, wh, garch_alpha, weight)
# ─────────────────────────────────────────────────────────────────────
SCENARIOS = {
    "strong_trend_bull":  (0.73, 0.0032, 8000,  0.06, 0.32, 0.15, 0.10),
    "strong_trend_bear":  (0.27, 0.0032, 8000,  0.06, 0.32, 0.15, 0.10),
    "weak_trend_bull":    (0.62, 0.0020, 5000,  0.10, 0.48, 0.12, 0.12),
    "weak_trend_bear":    (0.38, 0.0020, 5000,  0.10, 0.48, 0.12, 0.12),
    "range_bound":        (0.50, 0.0012, 2200,  0.10, 0.55, 0.10, 0.20),
    "breakout_bull":      (0.70, 0.0048, 16000, 0.05, 0.25, 0.28, 0.08),
    "breakout_bear":      (0.30, 0.0048, 16000, 0.05, 0.25, 0.28, 0.08),
    "exhaustion_fade":    (0.50, 0.0016, 3200,  0.12, 0.55, 0.13, 0.20),
}
SC_KEYS    = list(SCENARIOS.keys())
SC_WEIGHTS = [SCENARIOS[k][6] for k in SC_KEYS]


def generate_sequence(base_price=BASE_PRICE):
    sc_name = random.choices(SC_KEYS, weights=SC_WEIGHTS, k=1)[0]
    tb, base_atr, base_vol, wl, wh, g_alpha, _ = SCENARIOS[sc_name]

    price = base_price + np.random.normal(0, base_atr*1.5)
    vs    = VolState(base_atr, alpha=g_alpha)
    bias  = float(np.clip(0.50+np.random.normal(0,0.08), 0.15, 0.85))
    start_hour = random.randint(0, 23)

    breakout_mode   = "breakout" in sc_name
    exhaustion_mode = "exhaustion" in sc_name
    range_mode      = sc_name == "range_bound"

    candles = []
    for i in range(SEQ_LEN + FUTURE_BARS):
        hour = (start_hour+i)%24
        ar   = 0.90 if "strong_trend" in sc_name else (0.65 if breakout_mode else 0.80)
        bias = evolve_bias(bias, tb, ar_coef=ar, noise_std=0.05)

        if breakout_mode:
            ph = i/(SEQ_LEN+FUTURE_BARS)
            vs.base = base_atr*(0.35 if ph<0.5 else min(2.8, 0.35+(ph-0.5)*10))
        if exhaustion_mode:
            ph = i/(SEQ_LEN+FUTURE_BARS)
            # builds then fades
            vs.base = base_atr*(1.0+ph*1.0 if ph<0.5 else max(0.4, 2.0-(ph-0.5)*3.5))
        if range_mode:
            # oscillating bias for ranging market
            bias = 0.50 + 0.18*np.sin(i*0.28) + np.random.normal(0,0.04)
            bias = float(np.clip(bias,0.20,0.80))

        c, price = gen_candle_h1(price, vs, bias, hour, base_vol, wl, wh)
        candles.append(c)

    df_ctx    = pd.DataFrame(candles[:SEQ_LEN])
    future    = candles[SEQ_LEN:]
    features  = compute_momentum_features(df_ctx)
    labels    = label_momentum_sample(features, future)

    row = {k: v for k, v in features.items() if not k.startswith("_")}
    row["scenario"]              = sc_name
    row["next_direction"]        = labels[0]
    row["next_magnitude"]        = labels[1]
    row["next_regime_behaviour"] = labels[2]
    row["next_volatility"]       = labels[3]
    row["next_structural_bias"]  = labels[4]
    row["next_momentum_quality"] = labels[5]
    return row


if __name__ == "__main__":
    print(f"Generating {N_SAMPLES} Momentum H1 samples (v4) ...")
    rows = [generate_sequence() for _ in trange(N_SAMPLES)]
    df_out = pd.DataFrame(rows)
    lc = ["next_direction","next_magnitude","next_regime_behaviour",
          "next_volatility","next_structural_bias","next_momentum_quality"]
    fc = [c for c in df_out.columns if c not in lc+["scenario"]]
    df_out[["scenario"]+fc+lc].to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved → {OUTPUT_PATH}  shape={df_out.shape}")
    for c in lc:
        print(f"  {c}: {dict(df_out[c].value_counts(normalize=True).sort_index().round(3))}")
