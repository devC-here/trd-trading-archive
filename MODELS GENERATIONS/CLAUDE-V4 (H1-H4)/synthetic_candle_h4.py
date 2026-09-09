# synthetic_candle_h4.py  v5
# New features targeting mag + mom heads:
#   n_vol_regime     — recent ATR vs background ATR
#   n_atr_scaled_net — net move of last 4 bars / short-window ATR
#   n_mom_delta_bar  — last-2 vs first-2 bar deltas in last 4 bars
#   n_accel_slope    — linear slope of per-bar deltas (last 4)
import numpy as np, pandas as pd, random, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _market_h4 import generate_sample
from _labels_h4 import compute_labels_from_bars, LABEL_COLS
try:
    from tqdm import trange
except ImportError:
    def trange(n, **kw): return range(n)

N_SAMPLES=30_000; OUTPUT="synthetic_candle_h4.csv"; EPS=1e-10
random.seed(42); np.random.seed(42)

def _atr(bars):
    trs=[]
    for i,b in enumerate(bars):
        hl=b["high"]-b["low"]
        trs.append(hl if i==0 else max(hl,abs(b["high"]-bars[i-1]["close"]),abs(b["low"]-bars[i-1]["close"])))
    return float(np.mean(trs))+EPS

def _delta(bar):
    return float((bar["close"]-bar["low"])/(bar["high"]-bar["low"]+EPS)*2-1)

def _ha_body_last(bars, atr):
    ha_c=[(b["open"]+b["high"]+b["low"]+b["close"])/4 for b in bars]
    ha_o=[ha_c[0]]
    for i in range(1,len(bars)): ha_o.append((ha_o[-1]+ha_c[i-1])/2)
    return float((ha_c[-1]-ha_o[-1])/(atr+EPS))

def _win_feats(bars, atr, pfx):
    n=len(bars)
    closes=[b["close"] for b in bars]; opens=[b["open"] for b in bars]
    highs=[b["high"] for b in bars];   lows=[b["low"] for b in bars]
    vols=[b["volume"] for b in bars]
    bodies=[abs(b["close"]-b["open"]) for b in bars]
    ranges=[b["high"]-b["low"] for b in bars]
    dirs=[1 if b["close"]>=b["open"] else -1 for b in bars]
    gravs=[(b["close"]-b["low"])/(b["high"]-b["low"]+EPS) for b in bars]
    brs=[bodies[i]/(ranges[i]+EPS) for i in range(n)]
    uw=[abs(max(b["open"],b["close"])-b["high"]) for b in bars]
    lw=[abs(min(b["open"],b["close"])-b["low"])  for b in bars]
    mid=n//2
    avg_br   = float(np.mean(brs))
    avg_grav = float(np.mean(gravs))
    sd       = float(np.clip((np.mean(lw)-np.mean(uw))/(atr+EPS),-3,3))
    br1=float(np.mean(brs[:mid]))+EPS; br2=float(np.mean(brs[mid:]))
    seq_acc  = float(np.clip((br2-br1)/br1,-3,3))
    dir_rat  = float(sum(1 for d in dirs if d>0)/n)
    ha_last  = float(np.clip(_ha_body_last(bars,atr),-4,4))
    engulf   = float(np.clip((bodies[-1]-bodies[-2])/(atr+EPS)*dirs[-1],-4,4)) if n>=2 else 0.0
    pin_bull = float(np.clip((1-brs[-1])*gravs[-1],0,1))
    pin_bear = float(np.clip((1-brs[-1])*(1-gravs[-1]),0,1))
    pin_sig  = pin_bull-pin_bear
    upv=sum(vols[i] for i in range(n) if dirs[i]>0)
    dnv=sum(vols[i] for i in range(n) if dirs[i]<0)+EPS
    vol_dir  = float(np.clip((upv-dnv)/(upv+dnv+EPS),-1,1))
    return {f"{pfx}_br":avg_br,f"{pfx}_grav":avg_grav,f"{pfx}_sd":sd,
            f"{pfx}_acc":seq_acc,f"{pfx}_drat":dir_rat,f"{pfx}_ha":ha_last,
            f"{pfx}_eng":engulf,f"{pfx}_pin":pin_sig,f"{pfx}_vdir":vol_dir}

def compute_features(ctx):
    atr_bg=_atr(ctx)
    f1=_win_feats(ctx[-4:], atr_bg,"w1")
    f2=_win_feats(ctx[-12:],atr_bg,"w2")
    f3=_win_feats(ctx,      atr_bg,"w3")
    bg_brs=[abs(b["close"]-b["open"])/(b["high"]-b["low"]+EPS) for b in ctx]
    bg_avg_br=float(np.mean(bg_brs))+EPS
    n1=float(np.clip(f2["w2_br"]  -f3["w3_br"],  -2,2))
    n2=float(np.clip(f2["w2_grav"]-f3["w3_grav"],-1,1))
    n3=float(np.clip(f2["w2_acc"] -f3["w3_acc"], -3,3))
    n4=float(np.clip(f2["w2_sd"]  -f3["w3_sd"],  -2,2))
    dirs_all=[1 if b["close"]>=b["open"] else -1 for b in ctx]
    streak=0; last_d=dirs_all[-1]
    for d in reversed(dirs_all):
        if d==last_d: streak+=1
        else: break
    n5=float(np.clip(streak*last_d/12,-1,1))
    bg_rng=float(np.mean([b["high"]-b["low"] for b in ctx]))+EPS
    n6=float(np.clip((ctx[-1]["high"]-ctx[-1]["low"])/bg_rng,0.2,4.0))

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
            "n_br_shift":n1,"n_grav_shift":n2,"n_acc_shift":n3,
            "n_sd_shift":n4,"n_streak":n5,"n_rng_exp":n6,
            # new
            "n_vol_regime":n_vol_regime,"n_atr_scaled_net":n_atr_scaled_net,
            "n_accel_slope":n_accel_slope,"n_mom_delta_bar":n_mom_delta_bar}

def main():
    rows=[]
    for _ in trange(N_SAMPLES, desc="Candle"):
        ctx,fut,_ = generate_sample(model_name="candle")
        atr       = _atr(ctx)
        rows.append({**compute_features(ctx), **compute_labels_from_bars(fut, atr)})
    df=pd.DataFrame(rows).fillna(0.0)
    df.to_csv(OUTPUT, index=False)
    print(f"\n✅  {OUTPUT}  {df.shape}")
    for lc in LABEL_COLS:
        print(f"  {lc}: {df[lc].value_counts().sort_index().to_dict()}")

if __name__=="__main__": main()
