# synthetic_breakout_h4.py  v5
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

N_SAMPLES=30_000; OUTPUT="synthetic_breakout_h4.csv"; EPS=1e-10
random.seed(42); np.random.seed(42)

def _atr(bars):
    trs=[]
    for i,b in enumerate(bars):
        hl=b["high"]-b["low"]
        trs.append(hl if i==0 else max(hl,abs(b["high"]-bars[i-1]["close"]),abs(b["low"]-bars[i-1]["close"])))
    return float(np.mean(trs))+EPS

def _delta(bar):
    return float((bar["close"]-bar["low"])/(bar["high"]-bar["low"]+EPS)*2-1)

def _bb_width(closes, w):
    c=np.array(closes[-w:]) if len(closes)>=w else np.array(closes)
    return float(4*np.std(c)) if len(c)>1 else 0.0

def _kc_width(bars, w):
    n=min(w,len(bars))
    return float(3*np.mean([b["high"]-b["low"] for b in bars[-n:]]))

def _win_feats(bars, atr, bb_bg, pfx):
    n=len(bars)
    closes=[b["close"] for b in bars]; highs=[b["high"] for b in bars]
    lows=[b["low"] for b in bars];     vols=[b["volume"] for b in bars]
    ranges=[b["high"]-b["low"] for b in bars]
    dirs=[1 if b["close"]>=b["open"] else -1 for b in bars]
    bodies=[abs(b["close"]-b["open"]) for b in bars]
    bb_w=_bb_width(closes,n); kc_w=_kc_width(bars,n)
    kb      = float(np.clip(bb_w/(kc_w+EPS),0.1,3.0))
    sq      = float(1 if bb_w<kc_w else 0)
    mid_c=float(np.mean(closes)); std_c=float(np.std(closes))+EPS
    bb_pos  = float(np.clip((closes[-1]-(mid_c-2*std_c))/(4*std_c+EPS),0,1))
    coil    = float(np.clip(1-(max(highs)-min(lows))/(atr*n+EPS),-1,1))
    mid=n//2
    bb1=_bb_width(closes[:mid+1],mid+1); bb2=_bb_width(closes[mid:],n-mid)
    exp_v   = float(np.clip((bb2-bb1)/(bb_bg+EPS),-3,3))
    ub=mid_c+2*std_c; lb=mid_c-2*std_c
    bw      = float(np.clip((sum(1 for c in closes if c>ub)-sum(1 for c in closes if c<lb))/n,-1,1))
    atr_r   = float(np.clip((float(np.mean(ranges[-3:]))+EPS)/(float(np.mean(ranges))+EPS),0.2,4.0))
    bk_v    = float(np.clip(sum(vols[i]*dirs[i] for i in range(n) if closes[i]>ub or closes[i]<lb)/(atr*1000+EPS),-3,3))
    don_b   = float((closes[-1]-min(lows))/(max(highs)-min(lows)+EPS))
    return {f"{pfx}_kb":kb,f"{pfx}_sq":sq,f"{pfx}_bbpos":bb_pos,
            f"{pfx}_coil":coil,f"{pfx}_expv":exp_v,f"{pfx}_bw":bw,
            f"{pfx}_atrr":atr_r,f"{pfx}_bkv":bk_v,f"{pfx}_don":don_b}

def compute_features(ctx):
    atr_bg=_atr(ctx)
    closes_bg=[b["close"] for b in ctx]
    bb_bg=_bb_width(closes_bg,len(ctx))
    f1=_win_feats(ctx[-4:], atr_bg,bb_bg,"w1")
    f2=_win_feats(ctx[-12:],atr_bg,bb_bg,"w2")
    f3=_win_feats(ctx,      atr_bg,bb_bg,"w3")
    sq_flags=[1 if _bb_width([b["close"] for b in ctx[max(0,i-8):i+1]],8)<
                    _kc_width(ctx[max(0,i-8):i+1],8) else 0 for i in range(len(ctx))]
    cons=0
    for s in reversed(sq_flags):
        if s==1: cons+=1
        else: break
    avg_sq=max(float(np.mean(sq_flags)*len(sq_flags)),1.0)
    n1=float(np.clip(cons/avg_sq,0,4))
    bb_w2=_bb_width([b["close"] for b in ctx[-12:]],12)
    n2=float(np.clip(bb_w2/(bb_bg+EPS),0.1,3.0))
    n3=float(np.clip(f1["w1_kb"]  -f2["w2_kb"],  -2,2))
    n4=float(np.clip(f1["w1_coil"]-f2["w2_coil"],-2,2))
    bg_rng=float(np.mean([b["high"]-b["low"] for b in ctx]))+EPS
    w1_rng=float(np.mean([b["high"]-b["low"] for b in ctx[-4:]]))
    n5=float(np.clip(w1_rng/bg_rng,0.2,4.0))
    n6=float(np.clip(f1["w1_bw"]-f2["w2_bw"],-2,2))

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
            "n_sqmat":n1,"n_bbvsbg":n2,"n_kbshift":n3,
            "n_coilsh":n4,"n_rngvsbg":n5,"n_bwshift":n6,
            # new
            "n_vol_regime":n_vol_regime,"n_atr_scaled_net":n_atr_scaled_net,
            "n_accel_slope":n_accel_slope,"n_mom_delta_bar":n_mom_delta_bar}

def main():
    rows=[]
    for _ in trange(N_SAMPLES, desc="Breakout"):
        ctx,fut,_ = generate_sample(model_name="breakout")
        atr       = _atr(ctx)
        rows.append({**compute_features(ctx), **compute_labels_from_bars(fut, atr)})
    df=pd.DataFrame(rows).fillna(0.0)
    df.to_csv(OUTPUT, index=False)
    print(f"\n✅  {OUTPUT}  {df.shape}")
    for lc in LABEL_COLS:
        print(f"  {lc}: {df[lc].value_counts().sort_index().to_dict()}")

if __name__=="__main__": main()
