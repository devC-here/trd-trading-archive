# synthetic_momentum_h1.py  v5
# New features targeting mag + mom heads:
#   n_vol_regime   — recent vol vs background vol (helps magnitude scale)
#   n_atr_scaled_net — net move of last 4 bars scaled to recent ATR window
#   n_mom_delta_bar  — bar-by-bar delta trend in last 4 bars
#   n_accel_slope    — slope of per-bar deltas (acceleration signal)
import numpy as np, pandas as pd, random, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _market import generate_sample
from _labels import compute_labels_from_bars, LABEL_COLS
try:
    from tqdm import trange
except ImportError:
    def trange(n, **kw): return range(n)

N_SAMPLES = 30_000
OUTPUT    = "synthetic_momentum_h1.csv"
EPS       = 1e-10
random.seed(42); np.random.seed(42)

def _atr(bars):
    trs = []
    for i, b in enumerate(bars):
        hl = b["high"]-b["low"]
        trs.append(hl if i==0 else max(hl, abs(b["high"]-bars[i-1]["close"]), abs(b["low"]-bars[i-1]["close"])))
    return float(np.mean(trs))+EPS

def _delta(bar):
    return float((bar["close"]-bar["low"])/(bar["high"]-bar["low"]+EPS)*2-1)

def _win_feats(bars, atr, pfx):
    n=len(bars)
    closes=[b["close"] for b in bars]; opens=[b["open"] for b in bars]
    highs=[b["high"] for b in bars];   lows=[b["low"] for b in bars]
    vols=[b["volume"] for b in bars];  bodies=[abs(b["close"]-b["open"]) for b in bars]
    dirs=[1 if b["close"]>=b["open"] else -1 for b in bars]
    ranges=[b["high"]-b["low"] for b in bars]
    mid=n//2
    net_ret   = float(np.clip((closes[-1]-opens[0])/atr,-6,6))
    bull_pct  = float(sum(1 for d in dirs if d>0)/n)
    accel     = float(np.clip((closes[-1]-closes[mid])/(atr+EPS)-(closes[mid]-closes[0])/(atr+EPS),-5,5))
    path      = sum(abs(closes[i]-closes[i-1]) for i in range(1,n))+EPS
    eff       = float(np.clip(abs(closes[-1]-closes[0])/path,0,1))
    wh=max(highs); wl=min(lows)
    tpos      = float((closes[-1]-wl)/(wh-wl+EPS))
    up=sum(max(highs[i]-highs[i-1],0) for i in range(1,n))
    dn=sum(max(lows[i-1]-lows[i],0)   for i in range(1,n))+EPS
    di        = float(np.clip((up-dn)/(up+dn+EPS),-1,1))
    atr_ex    = float(np.clip((float(np.mean(ranges[-3:]))+EPS)/(float(np.mean(ranges))+EPS),0.2,4.0))
    v1=float(np.mean(vols[:mid])+EPS); v2=float(np.mean(vols[mid:])+EPS)
    vtrd      = float(np.clip(v2/v1,0.2,5.0))
    bq        = float(np.clip(np.mean([b/(r+EPS) for b,r in zip(bodies,ranges)]),0,1))
    return {f"{pfx}_ret":net_ret,f"{pfx}_bull":bull_pct,f"{pfx}_accel":accel,
            f"{pfx}_eff":eff,f"{pfx}_tpos":tpos,f"{pfx}_di":di,
            f"{pfx}_atrex":atr_ex,f"{pfx}_vtrd":vtrd,f"{pfx}_bq":bq}

def compute_features(ctx):
    atr_bg=_atr(ctx)
    f1=_win_feats(ctx[-4:], atr_bg,"w1")
    f2=_win_feats(ctx[-12:],atr_bg,"w2")
    f3=_win_feats(ctx,      atr_bg,"w3")
    n1=float(np.clip(f2["w2_ret"]  -f3["w3_ret"],  -4,4))
    n2=float(np.clip(f2["w2_eff"]  -f3["w3_eff"],  -1,1))
    n3=float(np.clip(f2["w2_di"]   -f3["w3_di"],   -2,2))
    n4=float(np.clip(f2["w2_atrex"]-f3["w3_atrex"],-2,2))
    n5=float(np.clip(f1["w1_ret"]  -f2["w2_ret"],  -3,3))
    n6=float(np.clip(f1["w1_accel"]-f2["w2_accel"],-3,3))

    # ── NEW: magnitude helpers ──────────────────────────────────────────
    # vol_regime: short-window ATR vs background ATR
    atr_w1 = _atr(ctx[-4:])
    atr_w2 = _atr(ctx[-12:])
    n_vol_regime   = float(np.clip(atr_w1/(atr_bg+EPS), 0.2, 5.0))
    # net move of last 4 bars normalised by the short-window ATR (not bg ATR)
    # this captures whether recent price moved a lot *relative to current vol*
    net_w1 = ctx[-1]["close"] - ctx[-4]["open"]
    n_atr_scaled_net = float(np.clip(net_w1/(atr_w1+EPS), -5.0, 5.0))

    # ── NEW: momentum head helpers ──────────────────────────────────────
    # bar-by-bar delta values for last 4 bars
    deltas_w1 = [_delta(b) for b in ctx[-4:]]
    # linear slope of those 4 delta values (positive = accelerating)
    xs = np.arange(4, dtype=float)
    slope = float(np.polyfit(xs, deltas_w1, 1)[0])          # raw slope
    n_accel_slope  = float(np.clip(slope / (1.0+EPS), -2.0, 2.0))
    # difference between last-2 and first-2 bar deltas
    n_mom_delta_bar = float(np.clip(
        float(np.mean(deltas_w1[2:])) - float(np.mean(deltas_w1[:2])),
        -2.0, 2.0))

    return {**f1,**f2,**f3,
            "n_mom_shift":n1,"n_eff_shift":n2,"n_di_shift":n3,
            "n_atr_shift":n4,"n_1h_vs_3h":n5,"n_accel_rec":n6,
            # new
            "n_vol_regime":n_vol_regime,"n_atr_scaled_net":n_atr_scaled_net,
            "n_accel_slope":n_accel_slope,"n_mom_delta_bar":n_mom_delta_bar}

def main():
    rows=[]
    for _ in trange(N_SAMPLES, desc="Momentum"):
        ctx,fut,_ = generate_sample(model_name="momentum")
        atr       = _atr(ctx)
        rows.append({**compute_features(ctx), **compute_labels_from_bars(fut, atr)})
    df=pd.DataFrame(rows).fillna(0.0)
    df.to_csv(OUTPUT, index=False)
    print(f"\n✅  {OUTPUT}  {df.shape}")
    for lc in LABEL_COLS:
        print(f"  {lc}: {df[lc].value_counts().sort_index().to_dict()}")

if __name__=="__main__": main()
