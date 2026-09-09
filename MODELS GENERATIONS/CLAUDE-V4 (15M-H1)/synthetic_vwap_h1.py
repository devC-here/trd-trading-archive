# synthetic_vwap_h1.py  v5
# New features targeting mag + mom heads:
#   n_vol_regime     — recent ATR vs background ATR
#   n_atr_scaled_net — net move of last 4 bars / short-window ATR
#   n_mom_delta_bar  — last-2 vs first-2 bar deltas in last 4 bars
#   n_accel_slope    — linear slope of per-bar deltas (last 4)
import numpy as np, pandas as pd, random, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _market import generate_sample
from _labels import compute_labels_from_bars, LABEL_COLS
try:
    from tqdm import trange
except ImportError:
    def trange(n, **kw): return range(n)

N_SAMPLES=30_000; OUTPUT="synthetic_vwap_h1.csv"; EPS=1e-10
random.seed(42); np.random.seed(42)

def _atr(bars):
    trs=[]
    for i,b in enumerate(bars):
        hl=b["high"]-b["low"]
        trs.append(hl if i==0 else max(hl,abs(b["high"]-bars[i-1]["close"]),abs(b["low"]-bars[i-1]["close"])))
    return float(np.mean(trs))+EPS

def _vwap(bars):
    tv=[(b["high"]+b["low"]+b["close"])/3*b["volume"] for b in bars]
    vs=[b["volume"] for b in bars]
    return float(sum(tv)/(sum(vs)+EPS))

def _delta(bar):
    return float((bar["close"]-bar["low"])/(bar["high"]-bar["low"]+EPS)*2-1)

def _win_feats(bars, atr, pfx):
    n=len(bars)
    closes=[b["close"] for b in bars]; vols=[b["volume"] for b in bars]
    bodies=[abs(b["close"]-b["open"]) for b in bars]
    ranges=[b["high"]-b["low"] for b in bars]
    dirs=[1 if b["close"]>=b["open"] else -1 for b in bars]
    gravs=[(b["close"]-b["low"])/(b["high"]-b["low"]+EPS) for b in bars]
    cum_tv=0; cum_v=0; vwap_vals=[]
    for b in bars:
        cum_tv+=(b["high"]+b["low"]+b["close"])/3*b["volume"]; cum_v+=b["volume"]
        vwap_vals.append(cum_tv/(cum_v+EPS))
    vdev    = float(np.clip((closes[-1]-vwap_vals[-1])/atr,-5,5))
    deltas  = [_delta(b) for b in bars]
    mid=n//2
    cum_d   = float(np.clip(sum(deltas)/n,-1,1))
    dtrd    = float(np.clip(float(np.mean(deltas[mid:]))-float(np.mean(deltas[:mid])),-2,2))
    upv=sum(vols[i] for i in range(n) if dirs[i]>0)
    dnv=sum(vols[i] for i in range(n) if dirs[i]<0)+EPS
    vasym   = float(np.clip((upv-dnv)/(upv+dnv+EPS),-1,1))
    avg_grav= float(np.mean(gravs))
    crosses = sum(1 for i in range(1,n) if closes[i]>vwap_vals[i] and closes[i-1]<=vwap_vals[i-1])
    avg_br  = float(np.mean([b/(r+EPS) for b,r in zip(bodies,ranges)]))
    reclaim = float(np.clip(crosses*avg_br,0,3))
    va=float(np.mean(vols))+EPS
    vsurge  = float(np.clip(vols[-1]/va,0.2,5.0))
    streak=0
    for i in range(n-1,-1,-1):
        if closes[i]>vwap_vals[i]: streak+=1
        else: break
    above_s = float(np.clip(streak/n,0,1))
    return {f"{pfx}_vdev":vdev,f"{pfx}_cumd":cum_d,f"{pfx}_dtrd":dtrd,
            f"{pfx}_vasym":vasym,f"{pfx}_grav":avg_grav,f"{pfx}_reclaim":reclaim,
            f"{pfx}_vsurge":vsurge,f"{pfx}_above":above_s,f"{pfx}_br":avg_br}

def compute_features(ctx):
    atr_bg=_atr(ctx); vwap_bg=_vwap(ctx)
    f1=_win_feats(ctx[-4:], atr_bg,"w1")
    f2=_win_feats(ctx[-12:],atr_bg,"w2")
    f3=_win_feats(ctx,      atr_bg,"w3")
    last_c=ctx[-1]["close"]
    n1=float(np.clip((last_c-vwap_bg)/atr_bg,-5,5))
    n2=float(np.clip(f2["w2_cumd"] -f3["w3_cumd"], -2,2))
    n3=float(np.clip(f2["w2_vasym"]-f3["w3_vasym"],-2,2))
    n4=float(np.clip(f2["w2_vdev"] -f3["w3_vdev"],  -3,3))
    n5=float(np.clip(f1["w1_cumd"] -f2["w2_cumd"],  -2,2))
    w1_vol=float(np.mean([b["volume"] for b in ctx[-4:]]))+EPS
    bg_vol=float(np.mean([b["volume"] for b in ctx]))+EPS
    n6=float(np.clip(w1_vol/bg_vol,0.2,5.0))

    # ── NEW: magnitude helpers ──────────────────────────────────────────
    atr_w1 = _atr(ctx[-4:])
    n_vol_regime     = float(np.clip(atr_w1/(atr_bg+EPS), 0.2, 5.0))
    net_w1 = ctx[-1]["close"] - ctx[-4]["open"]
    n_atr_scaled_net = float(np.clip(net_w1/(atr_w1+EPS), -5.0, 5.0))

    # ── NEW: momentum head helpers ──────────────────────────────────────
    deltas_w1 = [_delta(b) for b in ctx[-4:]]
    xs = np.arange(4, dtype=float)
    slope = float(np.polyfit(xs, deltas_w1, 1)[0])
    n_accel_slope   = float(np.clip(slope, -2.0, 2.0))
    n_mom_delta_bar = float(np.clip(
        float(np.mean(deltas_w1[2:])) - float(np.mean(deltas_w1[:2])),
        -2.0, 2.0))

    return {**f1,**f2,**f3,
            "n_pvwap":n1,"n_daccel":n2,"n_vshift":n3,
            "n_devshift":n4,"n_1h_delta":n5,"n_volctx":n6,
            # new
            "n_vol_regime":n_vol_regime,"n_atr_scaled_net":n_atr_scaled_net,
            "n_accel_slope":n_accel_slope,"n_mom_delta_bar":n_mom_delta_bar}

def main():
    rows=[]
    for _ in trange(N_SAMPLES, desc="VWAP"):
        ctx,fut,_ = generate_sample(model_name="vwap")
        atr       = _atr(ctx)
        rows.append({**compute_features(ctx), **compute_labels_from_bars(fut, atr)})
    df=pd.DataFrame(rows).fillna(0.0)
    df.to_csv(OUTPUT, index=False)
    print(f"\n✅  {OUTPUT}  {df.shape}")
    for lc in LABEL_COLS:
        print(f"  {lc}: {df[lc].value_counts().sort_index().to_dict()}")

if __name__=="__main__": main()
