# synthetic_microstructure_h1.py  v5
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

N_SAMPLES=30_000; OUTPUT="synthetic_microstructure_h1.csv"; EPS=1e-10
random.seed(42); np.random.seed(42)

def _atr(bars):
    trs=[]
    for i,b in enumerate(bars):
        hl=b["high"]-b["low"]
        trs.append(hl if i==0 else max(hl,abs(b["high"]-bars[i-1]["close"]),abs(b["low"]-bars[i-1]["close"])))
    return float(np.mean(trs))+EPS

def _delta(bar):
    return float((bar["close"]-bar["low"])/(bar["high"]-bar["low"]+EPS)*2-1)

def _energy(bar, atr):
    return float(np.clip((abs(bar["close"]-bar["open"])/atr)*np.sqrt(bar["volume"]/1000+EPS),0,6))

def _win_feats(bars, atr, pfx):
    n=len(bars)
    closes=[b["close"] for b in bars]; opens=[b["open"] for b in bars]
    vols=[b["volume"] for b in bars]
    bodies=[abs(b["close"]-b["open"]) for b in bars]
    ranges=[b["high"]-b["low"] for b in bars]
    dirs=[1 if b["close"]>=b["open"] else -1 for b in bars]
    uw=[abs(max(b["open"],b["close"])-b["high"]) for b in bars]
    lw=[abs(min(b["open"],b["close"])-b["low"])  for b in bars]
    mid=n//2
    deltas=[_delta(b) for b in bars]
    cum_d   = float(np.clip(sum(deltas)/n,-1,1))
    dtrd    = float(np.clip(float(np.mean(deltas[mid:]))-float(np.mean(deltas[:mid])),-2,2))
    avg_br  = float(np.mean([b/(r+EPS) for b,r in zip(bodies,ranges)]))
    tail_a  = float(np.clip((np.mean(lw)-np.mean(uw))/(atr+EPS),-3,3))
    avg_e   = float(np.mean([_energy(b,atr) for b in bars]))
    rej     = float(np.clip((uw[-1]+lw[-1])/(bodies[-1]/atr+EPS),0,5))
    va=float(np.mean(vols))+EPS; vz=float((vols[-1]-va)/(np.std(vols)+EPS))
    absorb  = float(np.clip(vz*(1 if bodies[-1]/(atr+EPS)<0.25 else 0),0,3))
    path    = sum(abs(closes[i]-closes[i-1]) for i in range(1,n))+EPS
    eff     = float(np.clip(abs(closes[-1]-closes[0])/path,0,1))
    gaps    = [abs(opens[i]-closes[i-1])/(atr+EPS) for i in range(1,n)]
    gdirs   = [1 if opens[i]>closes[i-1] else -1 for i in range(1,n)]
    gap_p   = float(np.clip(sum(g*d for g,d in zip(gaps,gdirs)),-3,3))
    return {f"{pfx}_cumd":cum_d,f"{pfx}_dtrd":dtrd,f"{pfx}_br":avg_br,
            f"{pfx}_tail":tail_a,f"{pfx}_eng":avg_e,f"{pfx}_rej":rej,
            f"{pfx}_abs":absorb,f"{pfx}_eff":eff,f"{pfx}_gap":gap_p}

def compute_features(ctx):
    atr_bg=_atr(ctx)
    f1=_win_feats(ctx[-4:], atr_bg,"w1")
    f2=_win_feats(ctx[-12:],atr_bg,"w2")
    f3=_win_feats(ctx,      atr_bg,"w3")
    n1=float(np.clip(f2["w2_cumd"]-f3["w3_cumd"],-2,2))
    bg_e=float(np.mean([_energy(b,atr_bg) for b in ctx]))+EPS
    w1_e=float(np.mean([_energy(b,atr_bg) for b in ctx[-4:]]))+EPS
    n2=float(np.clip(w1_e/bg_e,0.2,5.0))
    bg_br=float(np.mean([abs(b["close"]-b["open"])/(b["high"]-b["low"]+EPS) for b in ctx]))+EPS
    n3=float(np.clip(f1["w1_br"]/bg_br,0.2,4.0))
    n4=float(np.clip(f1["w1_tail"]-f2["w2_tail"],-2,2))
    deltas_bg=[_delta(b) for b in ctx]
    cp=0
    for d in reversed(deltas_bg):
        if d>0: cp+=1
        else: break
    cn=0
    for d in reversed(deltas_bg):
        if d<0: cn+=1
        else: break
    n5=float(np.clip((cp-cn)/12,-2,2))
    bg_vol=float(np.mean([b["volume"] for b in ctx]))+EPS
    w1_vol=float(np.mean([b["volume"] for b in ctx[-4:]]))+EPS
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
            "n_daccel":n1,"n_esurge":n2,"n_brvsbg":n3,
            "n_tailsh":n4,"n_dstreak":n5,"n_volctx":n6,
            # new
            "n_vol_regime":n_vol_regime,"n_atr_scaled_net":n_atr_scaled_net,
            "n_accel_slope":n_accel_slope,"n_mom_delta_bar":n_mom_delta_bar}

def main():
    rows=[]
    for _ in trange(N_SAMPLES, desc="Microstructure"):
        ctx,fut,_ = generate_sample(model_name="microstructure")
        atr       = _atr(ctx)
        rows.append({**compute_features(ctx), **compute_labels_from_bars(fut, atr)})
    df=pd.DataFrame(rows).fillna(0.0)
    df.to_csv(OUTPUT, index=False)
    print(f"\n✅  {OUTPUT}  {df.shape}")
    for lc in LABEL_COLS:
        print(f"  {lc}: {df[lc].value_counts().sort_index().to_dict()}")

if __name__=="__main__": main()
