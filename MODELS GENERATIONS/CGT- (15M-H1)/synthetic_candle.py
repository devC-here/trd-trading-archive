# synthetic_candle.py · v5.1 (latent-state engine)
# SPECIALIST: Candle Sequence & Auction Narrative — balanced distribution
# Architecture: Condition → Latent State → State-Biased Transition
#               → State-Modulated M15 → Features → Labels (unchanged)
# Run: python synthetic_candle.py  →  synthetic_candle_dataset.csv
import pandas as pd, numpy as np, random
try:    from tqdm import trange
except: trange=lambda n,**k:range(n)

N_SAMPLES=10000; OUTPUT_PATH="synthetic_candle_dataset.csv"
BASE_PRICE=1.5000; SEED=42
random.seed(SEED); np.random.seed(SEED); EPS=1e-12
EXPORT_METADATA = True

CONDITIONS={
    "STRONG_BULL":{"w":.04,"vm":1.3,"bs":.0010,"seq":["bull","bull","bull","bull"]},
    "STRONG_BEAR":{"w":.04,"vm":1.3,"bs":.0010,"seq":["bear","bear","bear","bear"]},
    "ACCELERATING_BULL":{"w":.04,"vm":1.6,"bs":.0012,"seq":["bull","bull_s","bull_s","bull_x"]},
    "ACCELERATING_BEAR":{"w":.04,"vm":1.6,"bs":.0012,"seq":["bear","bear_s","bear_s","bear_x"]},
    "MATURE_BULL":{"w":.04,"vm":1.0,"bs":.0008,"seq":["bull","bull","doji","bull_w"]},
    "MATURE_BEAR":{"w":.04,"vm":1.0,"bs":.0008,"seq":["bear","bear","doji","bear_w"]},
    "BULL_PULLBACK":{"w":.05,"vm":.9,"bs":.0007,"seq":["bull","bull","bear","bear"]},
    "BEAR_PULLBACK":{"w":.05,"vm":.9,"bs":.0007,"seq":["bear","bear","bull","bull"]},
    "BULL_PULLBACK_ENDING":{"w":.04,"vm":.8,"bs":.0006,"seq":["bull","bear","bear_w","doji"]},
    "BEAR_PULLBACK_ENDING":{"w":.04,"vm":.8,"bs":.0006,"seq":["bear","bull","bull_w","doji"]},
    "RANGE":{"w":.05,"vm":.7,"bs":.0005,"seq":["bull","bear","bull","bear"]},
    "RANGE_HIGH":{"w":.03,"vm":.7,"bs":.0005,"seq":["bull","bull_w","doji","bear_w"]},
    "RANGE_LOW":{"w":.03,"vm":.7,"bs":.0005,"seq":["bear","bear_w","doji","bull_w"]},
    "RANGE_COMPRESSION":{"w":.05,"vm":.45,"bs":.0002,"seq":["doji","doji","doji","doji"]},
    "BREAKOUT_SETUP_BULL":{"w":.04,"vm":.8,"bs":.0006,"seq":["doji","doji","bull_w","bull"]},
    "BREAKOUT_SETUP_BEAR":{"w":.04,"vm":.8,"bs":.0006,"seq":["doji","doji","bear_w","bear"]},
    "BREAKOUT_CONFIRMED_BULL":{"w":.04,"vm":2.0,"bs":.0014,"seq":["doji","bull","bull_s","bull_x"]},
    "BREAKOUT_CONFIRMED_BEAR":{"w":.04,"vm":2.0,"bs":.0014,"seq":["doji","bear","bear_s","bear_x"]},
    "FALSE_BREAKOUT_BULL":{"w":.03,"vm":1.8,"bs":.0011,"seq":["bull_s","spike_up","rejection","bear"]},
    "FALSE_BREAKOUT_BEAR":{"w":.03,"vm":1.8,"bs":.0011,"seq":["bear_s","spike_dn","rejection","bull"]},
    "EARLY_BULL_REVERSAL":{"w":.04,"vm":1.2,"bs":.0009,"seq":["bear","bear","doji","bull_s"]},
    "EARLY_BEAR_REVERSAL":{"w":.04,"vm":1.2,"bs":.0009,"seq":["bull","bull","doji","bear_s"]},
    "LIQUIDITY_SWEEP_HIGH":{"w":.03,"vm":2.0,"bs":.0013,"seq":["bull","bull","spike_up","bear_s"]},
    "LIQUIDITY_SWEEP_LOW":{"w":.03,"vm":2.0,"bs":.0013,"seq":["bear","bear","spike_dn","bull_s"]},
}
TRANSITIONS={
    "STRONG_BULL":{"STRONG_BULL":.25,"ACCELERATING_BULL":.15,"MATURE_BULL":.25,"EARLY_BEAR_REVERSAL":.10,"BULL_PULLBACK":.25},
    "STRONG_BEAR":{"STRONG_BEAR":.25,"ACCELERATING_BEAR":.15,"MATURE_BEAR":.25,"EARLY_BULL_REVERSAL":.10,"BEAR_PULLBACK":.25},
    "ACCELERATING_BULL":{"STRONG_BULL":.15,"ACCELERATING_BULL":.15,"LIQUIDITY_SWEEP_HIGH":.25,"EARLY_BEAR_REVERSAL":.25,"FALSE_BREAKOUT_BULL":.20},
    "ACCELERATING_BEAR":{"STRONG_BEAR":.15,"ACCELERATING_BEAR":.15,"LIQUIDITY_SWEEP_LOW":.25,"EARLY_BULL_REVERSAL":.25,"FALSE_BREAKOUT_BEAR":.20},
    "MATURE_BULL":{"BULL_PULLBACK":.30,"EARLY_BEAR_REVERSAL":.20,"RANGE":.20,"STRONG_BULL":.15,"RANGE_HIGH":.15},
    "MATURE_BEAR":{"BEAR_PULLBACK":.30,"EARLY_BULL_REVERSAL":.20,"RANGE":.20,"STRONG_BEAR":.15,"RANGE_LOW":.15},
    "BULL_PULLBACK":{"BULL_PULLBACK_ENDING":.35,"STRONG_BULL":.25,"RANGE":.20,"EARLY_BEAR_REVERSAL":.20},
    "BEAR_PULLBACK":{"BEAR_PULLBACK_ENDING":.35,"STRONG_BEAR":.25,"RANGE":.20,"EARLY_BULL_REVERSAL":.20},
    "BULL_PULLBACK_ENDING":{"STRONG_BULL":.35,"ACCELERATING_BULL":.20,"RANGE":.25,"BEAR_PULLBACK":.20},
    "BEAR_PULLBACK_ENDING":{"STRONG_BEAR":.35,"ACCELERATING_BEAR":.20,"RANGE":.25,"BULL_PULLBACK":.20},
    "RANGE":{"RANGE":.25,"RANGE_HIGH":.10,"RANGE_LOW":.10,"RANGE_COMPRESSION":.25,"BREAKOUT_SETUP_BULL":.15,"BREAKOUT_SETUP_BEAR":.15},
    "RANGE_HIGH":{"RANGE":.15,"FALSE_BREAKOUT_BULL":.25,"BREAKOUT_CONFIRMED_BULL":.25,"EARLY_BEAR_REVERSAL":.20,"LIQUIDITY_SWEEP_HIGH":.15},
    "RANGE_LOW":{"RANGE":.15,"FALSE_BREAKOUT_BEAR":.25,"BREAKOUT_CONFIRMED_BEAR":.25,"EARLY_BULL_REVERSAL":.20,"LIQUIDITY_SWEEP_LOW":.15},
    "RANGE_COMPRESSION":{"BREAKOUT_SETUP_BULL":.28,"BREAKOUT_SETUP_BEAR":.28,"RANGE":.25,"RANGE_COMPRESSION":.19},
    "BREAKOUT_SETUP_BULL":{"BREAKOUT_CONFIRMED_BULL":.40,"FALSE_BREAKOUT_BULL":.22,"RANGE":.20,"STRONG_BULL":.18},
    "BREAKOUT_SETUP_BEAR":{"BREAKOUT_CONFIRMED_BEAR":.40,"FALSE_BREAKOUT_BEAR":.22,"RANGE":.20,"STRONG_BEAR":.18},
    "BREAKOUT_CONFIRMED_BULL":{"STRONG_BULL":.40,"ACCELERATING_BULL":.30,"BULL_PULLBACK":.20,"MATURE_BULL":.10},
    "BREAKOUT_CONFIRMED_BEAR":{"STRONG_BEAR":.40,"ACCELERATING_BEAR":.30,"BEAR_PULLBACK":.20,"MATURE_BEAR":.10},
    "FALSE_BREAKOUT_BULL":{"RANGE":.30,"STRONG_BEAR":.30,"BEAR_PULLBACK":.20,"EARLY_BEAR_REVERSAL":.20},
    "FALSE_BREAKOUT_BEAR":{"RANGE":.30,"STRONG_BULL":.30,"BULL_PULLBACK":.20,"EARLY_BULL_REVERSAL":.20},
    "EARLY_BULL_REVERSAL":{"STRONG_BULL":.30,"BREAKOUT_CONFIRMED_BULL":.20,"RANGE":.30,"BULL_PULLBACK_ENDING":.20},
    "EARLY_BEAR_REVERSAL":{"STRONG_BEAR":.30,"BREAKOUT_CONFIRMED_BEAR":.20,"RANGE":.30,"BEAR_PULLBACK_ENDING":.20},
    "LIQUIDITY_SWEEP_HIGH":{"STRONG_BEAR":.35,"EARLY_BEAR_REVERSAL":.30,"RANGE":.20,"FALSE_BREAKOUT_BULL":.15},
    "LIQUIDITY_SWEEP_LOW":{"STRONG_BULL":.35,"EARLY_BULL_REVERSAL":.30,"RANGE":.20,"FALSE_BREAKOUT_BEAR":.15},
}

# ── LATENT STATE (candle specialist: tracks pattern_age, sequence_harmony) ──
class _LS:
    __slots__=("ts","age","vr","cs","cd","pd","lp","bp","rp","pat_age","seq_harm","_vn")
    def __init__(self):
        self.ts=float(np.random.uniform(.3,.7)); self.age=int(np.random.randint(1,6))
        self.vr="normal"; self.cs=float(np.random.uniform(0,.25)); self.cd=0
        self.pd=0.; self.lp=float(np.random.uniform(0,.25)); self.bp=0.; self.rp=0.
        self.pat_age=int(np.random.randint(0,5))   # bars since last strong pattern
        self.seq_harm=float(np.random.uniform(.3,.7))  # 0=chaotic 1=harmonious
        self._vn=1

def _init_ls(cond):
    ls=_LS()
    if any(k in cond for k in ("STRONG","ACCELERATING")):
        ls.ts=np.random.uniform(.65,1.); ls.age=np.random.randint(2,8)
        ls.seq_harm=np.random.uniform(.65,.95); ls._vn=2
    elif "MATURE" in cond:
        ls.ts=np.random.uniform(.35,.65); ls.age=np.random.randint(10,22)
        ls.rp=np.random.uniform(.25,.55); ls.seq_harm=np.random.uniform(.45,.75); ls._vn=1
    elif "PULLBACK" in cond and "ENDING" not in cond:
        ls.ts=np.random.uniform(.2,.5); ls.pd=np.random.uniform(.15,.5)
        ls.seq_harm=np.random.uniform(.2,.5); ls._vn=1
    elif "COMPRESSION" in cond:
        ls.cs=np.random.uniform(.55,.90); ls.cd=np.random.randint(5,20)
        ls.bp=min(ls.cd*.04,.80); ls.ts=np.random.uniform(0,.18)
        ls.seq_harm=np.random.uniform(.55,.85); ls._vn=0
    elif "RANGE" in cond:
        ls.ts=np.random.uniform(0,.30); ls.cs=np.random.uniform(.20,.50)
        ls.seq_harm=np.random.uniform(.35,.65); ls._vn=0
    elif "SWEEP" in cond or "FALSE_BREAKOUT" in cond:
        ls.lp=np.random.uniform(.60,1.); ls.pat_age=0; ls.seq_harm=np.random.uniform(.1,.4); ls._vn=2
    ls.vr=["low","normal","high"][ls._vn]; return ls

def _step_ls(ls,cond):
    if any(k in cond for k in ("STRONG","ACCELERATING","MATURE")): ls.age=min(ls.age+1,40)
    elif any(k in cond for k in ("REVERSAL","BREAKOUT_CONFIRMED")): ls.age=1
    if "ACCELERATING" in cond: ls.ts=min(1.,ls.ts+np.random.uniform(.05,.14))
    elif "MATURE" in cond: ls.ts=max(.05,ls.ts-np.random.uniform(.02,.07))
    elif "PULLBACK" in cond and "ENDING" not in cond: ls.ts=max(.05,ls.ts-np.random.uniform(.05,.12))
    elif "PULLBACK_ENDING" in cond: ls.ts=min(1.,ls.ts+np.random.uniform(.06,.13))
    elif "BREAKOUT_CONFIRMED" in cond: ls.ts=np.random.uniform(.65,.95)
    if "COMPRESSION" in cond or "RANGE" in cond:
        ls.cs=min(1.,ls.cs+np.random.uniform(.04,.09)); ls.cd+=1; ls.bp=min(1.,ls.bp+.03*ls.cd)
    elif any(k in cond for k in ("BREAKOUT_CONFIRMED","STRONG","ACCELERATING")):
        ls.cs=max(0.,ls.cs-np.random.uniform(.2,.5)); ls.cd=0; ls.bp=max(0.,ls.bp-np.random.uniform(.4,.8))
    if "PULLBACK" in cond and "ENDING" not in cond: ls.pd=min(1.,ls.pd+np.random.uniform(.08,.20))
    elif "PULLBACK_ENDING" in cond: ls.pd=max(0.,ls.pd-np.random.uniform(.10,.25))
    elif any(k in cond for k in ("STRONG","ACCELERATING","BREAKOUT_CONFIRMED")): ls.pd=max(0.,ls.pd-.25)
    if any(k in cond for k in ("MATURE","RANGE_HIGH","RANGE_LOW")): ls.lp=min(1.,ls.lp+np.random.uniform(.06,.14))
    elif any(k in cond for k in ("SWEEP","FALSE_BREAKOUT")): ls.lp=max(0.,ls.lp-np.random.uniform(.3,.6))
    age_f=min(ls.age/20.,1.)
    if any(k in cond for k in ("MATURE","SWEEP","FALSE_BREAKOUT")): ls.rp=min(1.,ls.rp+np.random.uniform(.05,.12)+age_f*.04)
    elif any(k in cond for k in ("REVERSAL","BREAKOUT_CONFIRMED","PULLBACK_ENDING")): ls.rp=max(0.,ls.rp-np.random.uniform(.2,.5))
    # sequence harmony: rises in trending, falls in choppy/reversal
    if any(k in cond for k in ("STRONG","ACCELERATING","BULL","BEAR")) and "PULLBACK" not in cond:
        ls.seq_harm=min(1.,ls.seq_harm+np.random.uniform(.03,.08))
    elif any(k in cond for k in ("FALSE_BREAKOUT","SWEEP","REVERSAL","PULLBACK")):
        ls.seq_harm=max(0.,ls.seq_harm-np.random.uniform(.05,.15))
    elif any(k in cond for k in ("RANGE","COMPRESSION")):
        ls.seq_harm=float(np.clip(ls.seq_harm+np.random.uniform(-.05,.05),0.,1.))
    ls.pat_age=ls.pat_age+1  # increments every step; reset externally on strong pattern
    if any(k in cond for k in ("SWEEP","FALSE_BREAKOUT","BREAKOUT_CONFIRMED","REVERSAL")): ls.pat_age=0
    if any(k in cond for k in ("ACCELERATING","BREAKOUT_CONFIRMED","SWEEP")): ls._vn=min(2,ls._vn+1)
    elif "COMPRESSION" in cond: ls._vn=max(0,ls._vn-1)
    elif "RANGE" in cond and np.random.random()<.35: ls._vn=max(0,ls._vn-1)
    ls.vr=["low","normal","high"][ls._vn]
    for a in ("ts","cs","bp","pd","lp","rp","seq_harm"): setattr(ls,a,float(np.clip(getattr(ls,a),0.,1.)))
    return ls

def _next_cond(cond,ls):
    base=TRANSITIONS.get(cond,{})
    if not base:
        ks=list(CONDITIONS.keys()); return random.choices(ks,weights=[CONDITIONS[k]["w"] for k in ks],k=1)[0]
    w={}
    for c,p in base.items():
        b=1.
        if ls.age>14:
            if any(k in c for k in ("MATURE","REVERSAL","PULLBACK")): b*=1.+ls.age*.025
            if any(k in c for k in ("STRONG","ACCELERATING")): b*=max(.1,1.-ls.age*.035)
        if ls.rp>.55:
            if any(k in c for k in ("REVERSAL","SWEEP","FALSE_BREAKOUT")): b*=1.+ls.rp*2.
            if any(k in c for k in ("STRONG","ACCELERATING")): b*=max(.1,1.-ls.rp*.9)
        if ls.bp>.55:
            if any(k in c for k in ("BREAKOUT_CONFIRMED","BREAKOUT_SETUP","FALSE_BREAKOUT")): b*=1.+ls.bp*2.5
            if "COMPRESSION" in c: b*=max(.04,1.-ls.bp*.9)
        if ls.pd>.55:
            if "PULLBACK_ENDING" in c: b*=1.+ls.pd*2.
            if "PULLBACK" in c and "ENDING" not in c: b*=.35
        if ls.lp>.65:
            if any(k in c for k in ("SWEEP","FALSE_BREAKOUT")): b*=1.+ls.lp*2.
        # high seq_harmony → favour continuation over reversal
        if ls.seq_harm>.70:
            if any(k in c for k in ("STRONG","ACCELERATING","PULLBACK_ENDING")): b*=1.+ls.seq_harm*.5
            if any(k in c for k in ("REVERSAL","FALSE_BREAKOUT")): b*=max(.2,1.-ls.seq_harm*.4)
        w[c]=max(1e-6,p*b)
    t=sum(w.values()); ks=list(w.keys())
    return random.choices(ks,weights=[w[k]/t for k in ks],k=1)[0]

def _mod(cond,ls,bs,vm):
    vs={"low":.60,"normal":1.,"high":1.40}[ls.vr]; bs*=vs; vm*=vs
    bt=float(np.clip(.45+.55*ls.ts,.25,1.55))*float(np.clip(1.-ls.age/40.,.4,1.2))
    wx=float(np.clip(1.+ls.age/30.,1.,1.6))*float(np.clip(1.+ls.lp*.8,1.,2.2))*float(np.clip(1.+ls.rp*.6,1.,1.8))
    if "COMPRESSION" in cond or "RANGE" in cond:
        bs*=float(np.clip(1.-ls.cs*.65,.2,1.)); vm*=max(.3,1.-ls.cs*.5)
    if "BREAKOUT_CONFIRMED" in cond: bs*=1.+ls.bp*.8; vm*=1.+ls.bp*.6
    if "PULLBACK" in cond and "ENDING" not in cond:
        bt*=max(.35,1.-ls.pd*.5); wx*=1.+ls.pd*.45
    return float(np.clip(bs,.0001,.004)),float(np.clip(vm,.1,6.)),float(np.clip(wx,.5,3.5)),float(np.clip(bt,.2,1.6))

# ── COMPACT M15 EXPANSION ────────────────────────────────────────────
def _expand(arch,p,bs,vm,wx=1.):
    def c(o,cl,wu,wl,vlo,vhi):
        h=max(o,cl)+abs(float(np.random.normal(0,wu*wx))); l=min(o,cl)-abs(float(np.random.normal(0,wl*wx)))
        return {"open":float(o),"high":max(h,o,cl),"low":min(l,o,cl),"close":float(cl),"volume":int(np.random.uniform(vlo,vhi))}
    r=lambda a,b:float(np.random.uniform(a,b))
    def chain(p0,ds,wus,wls,vlos,vhis):
        cs2=[]; cp=p0
        for d,wu,wl,vlo,vhi in zip(ds,wus,wls,vlos,vhis): n=cp+d*bs; cs2.append(c(cp,n,wu,wl,vlo*vm,vhi*vm)); cp=n
        return cs2
    T={"bull":([r(.15,.45),r(.20,.50),r(.25,.60)],[bs*.10,bs*.10,bs*.06],[bs*.15,bs*.12,bs*.10],[300,400,500],[800,1000,1200]),
       "bull_s":([r(.40,.70),r(.50,.90),r(.60,1.)],[bs*.05,bs*.04,bs*.03],[bs*.10,bs*.09,bs*.07],[600,700,800],[1500,1800,2100]),
       "bull_x":([r(.80,1.4),r(1.,1.8),r(1.2,2.)],[bs*.03,bs*.03,bs*.02],[bs*.07,bs*.06,bs*.05],[1200,1500,2000],[2800,3500,5000]),
       "bull_w":([r(-.08,.18),r(.05,.18),r(0.,.14)],[bs*.25,bs*.22,bs*.25],[bs*.35,bs*.32,bs*.35],[110,120,100],[340,360,310]),
       "bear":([-r(.15,.45),-r(.20,.50),-r(.25,.60)],[bs*.15,bs*.12,bs*.10],[bs*.10,bs*.10,bs*.06],[300,400,500],[800,1000,1200]),
       "bear_s":([-r(.40,.70),-r(.50,.90),-r(.60,1.)],[bs*.10,bs*.09,bs*.07],[bs*.05,bs*.04,bs*.03],[600,700,800],[1500,1800,2100]),
       "bear_x":([-r(.80,1.4),-r(1.,1.8),-r(1.2,2.)],[bs*.07,bs*.06,bs*.05],[bs*.03,bs*.03,bs*.02],[1200,1500,2000],[2800,3500,5000]),
       "bear_w":([-r(-.08,.18),-r(.05,.18),-r(0.,.14)],[bs*.35,bs*.32,bs*.35],[bs*.25,bs*.22,bs*.25],[110,120,100],[340,360,310]),
       "doji":([r(-.12,.12),r(-.10,.10),r(-.08,.08)],[bs*.40,bs*.38,bs*.35],[bs*.40,bs*.38,bs*.35],[110,100,95],[420,400,380])}
    FIRST={"bull":(r(.10,.35),bs*.12,bs*.20,200,600),"bull_s":(r(.30,.60),bs*.06,bs*.15,500,1200),
           "bull_x":(r(.50,1.),bs*.04,bs*.08,800,2000),"bull_w":(r(.05,.18),bs*.20,bs*.30,130,380),
           "bear":(-r(.10,.35),bs*.20,bs*.12,200,600),"bear_s":(-r(.30,.60),bs*.15,bs*.06,500,1200),
           "bear_x":(-r(.50,1.),bs*.08,bs*.04,800,2000),"bear_w":(-r(.05,.18),bs*.30,bs*.20,130,380),
           "doji":(r(-.12,.12),bs*.40,bs*.40,120,450)}
    if arch in FIRST:
        d0,wu0,wl0,vlo0,vhi0=FIRST[arch]; c1=c(p,p+d0*bs,wu0,wl0,vlo0*vm,vhi0*vm)
        ds,wus,wls,vlos,vhis=T[arch]; cs=[c1]+chain(c1["close"],ds,wus,wls,vlos,vhis)
    elif arch=="spike_up":
        cs=[c(p,p+r(.4,.8)*bs,bs*.10,bs*.05,600*vm,1500*vm),
            c(p+r(.4,.8)*bs,p+r(1.1,2.1)*bs,bs*.20,bs*.05,1200*vm,3200*vm),
            c(p+r(1.1,2.1)*bs,p+r(1.3,2.7)*bs,bs*.45,bs*.10,800*vm,2000*vm),
            c(p+r(1.3,2.7)*bs,p+r(1.3,2.7)*bs+r(-.3,.3)*bs,bs*.55,bs*.55,500*vm,1200*vm)]
    elif arch=="spike_dn":
        cs=[c(p,p-r(.4,.8)*bs,bs*.05,bs*.10,600*vm,1500*vm),
            c(p-r(.4,.8)*bs,p-r(1.1,2.1)*bs,bs*.05,bs*.20,1200*vm,3200*vm),
            c(p-r(1.1,2.1)*bs,p-r(1.3,2.7)*bs,bs*.10,bs*.45,800*vm,2000*vm),
            c(p-r(1.3,2.7)*bs,p-r(1.3,2.7)*bs+r(-.3,.3)*bs,bs*.55,bs*.55,500*vm,1200*vm)]
    elif arch=="rejection":
        cs=[c(p,p+r(.3,.6)*bs,bs*.10,bs*.10,400*vm,1000*vm),
            c(p+r(.3,.6)*bs,p+r(.8,1.6)*bs,bs*.55,bs*.06,1000*vm,2800*vm),
            c(p+r(.8,1.6)*bs,p+r(.8,1.6)*bs-r(.5,1.1)*bs,bs*.10,bs*.20,900*vm,2500*vm),
            c(p+r(.3,.5)*bs,p+r(.3,.5)*bs-r(.3,.7)*bs,bs*.12,bs*.15,600*vm,1500*vm)]
    else:
        cs=[]; cp=p
        for _ in range(4): nc=cp+r(-.1,.1)*bs; cs.append(c(cp,nc,bs*.3,bs*.3,200*vm,600*vm)); cp=nc
    while len(cs)<4: cs.append(cs[-1])
    return cs[:4],cs[-1]["close"]

# ── MARKET HISTORY GENERATOR ─────────────────────────────────────────
def _pick():
    ks=list(CONDITIONS.keys()); return random.choices(ks,weights=[CONDITIONS[k]["w"] for k in ks],k=1)[0]

def build_history(base_price=BASE_PRICE):
    price=base_price+np.random.normal(0,.002); cond=_pick(); ls=_init_ls(cond); hist=[]
    for _ in range(random.randint(22,34)):
        cfg=CONDITIONS[cond]; bs,vm,wx,bt=_mod(cond,ls,cfg["bs"],cfg["vm"])
        for a in cfg["seq"]: m,price=_expand(a,price,bs*bt,vm,wx); hist.extend(m)
        ls=_step_ls(ls,cond); cond=_next_cond(cond,ls)
    active=cond; cfg=CONDITIONS[active]; bs,vm,wx,bt=_mod(active,ls,cfg["bs"],cfg["vm"])
    for a in cfg["seq"]: m,price=_expand(a,price,bs*bt,vm,wx); hist.extend(m)
    ls=_step_ls(ls,active); fc=_next_cond(active,ls)
    fcfg=CONDITIONS[fc]; bs,vm,wx,bt=_mod(fc,ls,fcfg["bs"],fcfg["vm"])
    fut=[]
    for a in fcfg["seq"][:1]:
        m,price=_expand(a,price,bs*bt,vm,wx); fut.extend(m)
        if len(fut)>=4: break
    while len(fut)<4: fut.append(fut[-1])
    return hist,active,fc,fut[:4],ls

# ── FEATURES (unchanged) ─────────────────────────────────────────────
def _ols(arr):
    n=len(arr)
    if n<2 or np.std(arr)<EPS: return 0.
    x=np.arange(n,dtype=float); slope,*_=np.polyfit(x,arr,1)
    return float(slope/(np.std(arr)+EPS))
def _corr(a,b):
    if np.std(a)<EPS or np.std(b)<EPS: return 0.
    r=float(np.corrcoef(a,b)[0,1]); return r if np.isfinite(r) else 0.

def compute_features(df):
    df=df.copy().reset_index(drop=True)
    df["rng"]=df["high"]-df["low"]; df["body"]=df["close"]-df["open"]
    df["abs_body"]=df["body"].abs(); df["dir"]=np.sign(df["body"]).fillna(0)
    df["uw"]=df["high"]-df[["open","close"]].max(axis=1); df["lw"]=df[["open","close"]].min(axis=1)-df["low"]
    df["atr14"]=df["rng"].rolling(14,min_periods=1).mean()
    df["br"]=df["abs_body"]/(df["rng"]+EPS); df["cg"]=(df["close"]-df["low"])/(df["rng"]+EPS)
    df["sd"]=(df["lw"]-df["uw"])/(df["rng"]+EPS)
    bull_m=df["dir"]>=0; df["deff"]=np.where(bull_m,df["br"]*df["cg"],df["br"]*(1-df["cg"]))
    vol_avg20=df["volume"].rolling(20,min_periods=1).mean(); df["vw"]=(df["volume"]/(vol_avg20+EPS)).clip(0,4)
    ha_cl=(df["open"]+df["high"]+df["low"]+df["close"])/4; ha_op=ha_cl.ewm(span=3,adjust=False).mean()
    df["ha_body"]=(ha_cl-ha_op)/(df["atr14"]+EPS)
    df["br_avg30"]=df["br"].rolling(30,min_periods=1).mean(); df["br_avg5"]=df["br"].rolling(5,min_periods=1).mean()
    cons,cur,prev=[],0,0
    for d in df["dir"].fillna(0).tolist():
        if d==prev and d!=0: cur+=1
        elif d!=0: cur=1
        else: cur=0
        cons.append(cur); prev=d
    df["consec"]=cons
    WIN=16; w=df.iloc[-WIN:].reset_index(drop=True); atr=float(df["atr14"].iloc[-1]); f={}
    f["lb_cg"]=float(w["cg"].iloc[-1]); f["lb_br"]=float(w["br"].iloc[-1])
    f["lb_sd"]=float(w["sd"].iloc[-1]); f["lb_deff"]=float(w["deff"].iloc[-1]); f["lb_vw"]=float(min(w["vw"].iloc[-1],4.))
    f["cg_slope"]=_ols(w["cg"].values); f["ha_slope"]=_ols(w["ha_body"].values); f["sd_slope"]=_ols(w["sd"].values)
    bar_conv=w["dir"]*w["br"]*w["vw"]; f["vol_narrative"]=_ols(bar_conv.values)
    f["seq_harmony"]=_corr(w["cg"].values,w["rng"].values)
    bm=float(w["br"].mean()); bs_=float(w["br"].std()); f["body_entropy"]=float(bs_/(bm+EPS))
    early_dir=float(np.sign(w["body"].iloc[:8].sum())); late_dir=float(np.sign(w["body"].iloc[8:].sum()))
    f["handoff"]=float(early_dir!=late_dir)
    early_b=float(w["br"].iloc[:8].mean()); late_b=float(w["br"].iloc[8:].mean())
    f["seq_accel"]=float((late_b-early_b)/(bm+EPS))
    ha_arr=w["ha_body"].values; f["ha_body_last"]=float(ha_arr[-1]); f["ha_trend_str"]=float(ha_arr.mean())
    f["ha_flip_count"]=float(sum(1 for i in range(1,len(ha_arr)) if np.sign(ha_arr[i])!=np.sign(ha_arr[i-1])))
    uw=float(w["uw"].iloc[-1]); lw=float(w["lw"].iloc[-1]); rng=float(w["rng"].iloc[-1])
    br=float(w["br"].iloc[-1]); cg=float(w["cg"].iloc[-1]); small=1. if br<.35 else .5
    f["pin_bull"]=float((lw/(rng+EPS))*small*cg); f["pin_bear"]=float((uw/(rng+EPS))*small*(1-cg))
    doji_score=float(max(0,.15-br)/.15); closes=w["close"].values
    seq_pos=(closes[-1]-closes.min())/(closes.max()-closes.min()+EPS); f["doji_context"]=float(doji_score*(seq_pos-.5))
    curr_b=float(w["abs_body"].iloc[-1]); prev_b=float(w["abs_body"].iloc[-2]) if len(w)>=2 else curr_b
    curr_dir=float(w["dir"].iloc[-1]); f["engulf_qual"]=float((curr_b-prev_b)/(atr+EPS)*(curr_dir or 1))
    inside_count=0
    for i in range(1,len(w)):
        if w["high"].iloc[i]<w["high"].iloc[i-1] and w["low"].iloc[i]>w["low"].iloc[i-1]: inside_count+=1
        else: break
    mother_rng=float(w["rng"].iloc[0]) if inside_count>0 else 0.
    f["inside_energy"]=float(inside_count*mother_rng/(atr+EPS))
    f["body_vs_avg"]=float(min(float(df["abs_body"].iloc[-1])/(float(df["br_avg30"].iloc[-1])*float(df["atr14"].iloc[-1])+EPS),4.))
    f["body_accel_s5"]=float(br-float(df["br_avg5"].iloc[-1]))
    br1=br; br3=float(w["br"].iloc[-3:].mean()) if len(w)>=3 else br1; br16=float(w["br"].mean())
    price_up=closes[-1]>closes[0]
    f["fractal_bull"]=float((br1>br3>br16) and price_up); f["fractal_bear"]=float((br1>br3>br16) and not price_up)
    bull_bars=float((w["dir"]>0).sum()); bear_bars=float((w["dir"]<0).sum())
    f["seq_dir_ratio"]=bull_bars/(bull_bars+bear_bars+EPS)
    f["_atr"]=atr if atr>EPS else .0001; f["_ctx_close"]=float(df["close"].iloc[-1])
    f["_ha_trend"]=f["ha_trend_str"]; f["_seq_dir_ratio"]=f["seq_dir_ratio"]
    return f

# ── LABELS (unchanged) ───────────────────────────────────────────────
def compute_labels(f,future):
    atr=max(f["_atr"],EPS); ref=f["_ctx_close"]
    c1,c2=future[0],future[1]; c3=future[2] if len(future)>=3 else c2; c4=future[3] if len(future)>=4 else c3
    h1_close=c4["close"]; h1_high=max(x["high"] for x in future); h1_low=min(x["low"] for x in future)
    h1_rng=(h1_high-h1_low)/atr; net_atr=(h1_close-ref)/atr
    bodies=[abs(x["close"]-x["open"])/atr for x in future]; avg_body=float(np.mean(bodies))
    close_grav=(h1_close-h1_low)/(h1_rng*atr+EPS)
    c1_br=bodies[0]; c1_uw=(c1["high"]-max(c1["open"],c1["close"]))/(c1["high"]-c1["low"]+EPS)
    c1_lw=(min(c1["open"],c1["close"])-c1["low"])/(c1["high"]-c1["low"]+EPS)
    nxt_dir=2 if net_atr>.45 else (0 if net_atr<-.45 else 1)
    body_sz=2 if avg_body>1.2 else (0 if avg_body<.35 else 1)
    if c1_br<.12: nxt_pat=2
    elif (c1_lw>.40 and c1_uw<.25) or (c1_uw>.40 and c1_lw<.25): nxt_pat=1
    elif c1_br>bodies[1]*1.3 and c1_br>.8: nxt_pat=3
    else: nxt_pat=0
    ha=f["_ha_trend"]; sdr=f["_seq_dir_ratio"]
    if ha>.05 and sdr>.55: seq_cont=2 if nxt_dir==2 else (0 if nxt_dir==0 else 1)
    elif ha<-.05 and sdr<.45: seq_cont=2 if nxt_dir==0 else (0 if nxt_dir==2 else 1)
    else: seq_cont=1
    nxt_cg=2 if close_grav>.67 else (0 if close_grav<.33 else 1)
    nxt_vol=2 if h1_rng>3. else (0 if h1_rng<1. else 1)
    return int(nxt_dir),int(body_sz),int(nxt_pat),int(seq_cont),int(nxt_cg),int(nxt_vol)


def context_features_candle(df: pd.DataFrame, atr: float) -> dict:
    """Candle-specialist context features from last 16 M15."""
    w = df.iloc[-16:].reset_index(drop=True)
    body = (w["close"]-w["open"]).astype(float)
    rng  = (w["high"]-w["low"]).astype(float)
    uw   = w["high"] - w[["open","close"]].max(axis=1)
    lw   = w[["open","close"]].min(axis=1) - w["low"]
    dirs = np.sign(body)
    br   = body.abs()/(rng+EPS)
    # pattern_density_16: fraction of bars with a recognisable pattern (pin/doji/engulf)
    pin   = ((lw/(rng+EPS)>0.35)|(uw/(rng+EPS)>0.35)).astype(int)
    doji  = (br<0.15).astype(int)
    engulf= (body.abs()>body.abs().shift(1).fillna(0)*1.2).astype(int)
    pattern_density_16 = float((pin|doji|engulf).mean())
    # sequence_strength_16: signed directional persistence
    bull_frac = float((dirs>0).sum()/16)
    bear_frac = float((dirs<0).sum()/16)
    sequence_strength_16 = float(bull_frac - bear_frac)  # +1=all bull, -1=all bear
    # harmony_score_16: does body size increase in direction of trend?
    trend_dir = 1 if sequence_strength_16 > 0 else -1
    dir_bars  = w[dirs == trend_dir]
    if len(dir_bars) >= 4:
        br_dir = (dir_bars["close"]-dir_bars["open"]).abs()/(dir_bars["high"]-dir_bars["low"]+EPS)
        harmony_score_16 = float(np.clip(br_dir.iloc[-2:].mean() - br_dir.iloc[:2].mean(), -1, 1))
    else:
        harmony_score_16 = 0.0
    # close_gravity_mean_16: avg close position within bar range
    cg_mean_16 = float(((w["close"]-w["low"])/(rng+EPS)).mean())
    return {"pattern_density_16": pattern_density_16, "sequence_strength_16": sequence_strength_16,
            "harmony_score_16": harmony_score_16, "close_gravity_mean_16": cg_mean_16}

# ── SEQUENCE GENERATOR & MAIN ─────────────────────────────────────────
def generate_sequence(base_price=BASE_PRICE):
    hist,active,fc,fut,ls=build_history(base_price)
    df=pd.DataFrame(hist); f=compute_features(df)
    atr=max(f["_atr"],EPS); ctx=context_features_candle(df,atr)
    labels=compute_labels(f,fut)
    row={k:v for k,v in f.items() if not k.startswith("_")}
    row.update(ctx); row["condition"]=active; row["future_condition"]=fc
    if EXPORT_METADATA:
        row["meta_trend_strength"]=round(ls.ts,3); row["meta_trend_age"]=ls.age
        row["meta_vol_regime"]=ls.vr; row["meta_compression_score"]=round(ls.cs,3)
        row["meta_compression_duration"]=ls.cd; row["meta_pullback_depth"]=round(ls.pd,3)
        row["meta_liquidity_pressure"]=round(ls.lp,3); row["meta_breakout_pressure"]=round(ls.bp,3)
        row["meta_reversal_pressure"]=round(ls.rp,3); row["meta_pattern_age"]=ls.pat_age
        row["meta_seq_harmony"]=round(ls.seq_harm,3)
    row["label_next_direction"],row["label_next_body_size"],row["label_next_pattern"],\
    row["label_seq_continuation"],row["label_next_close_grav"],row["label_next_volatility"]=labels
    return row

if __name__=="__main__":
    print(f"Generating {N_SAMPLES} candle samples | v5 latent-state engine | candle specialist")
    rows=[generate_sequence() for _ in trange(N_SAMPLES,desc="candle")]
    df_out=pd.DataFrame(rows)
    lc=[c for c in df_out.columns if c.startswith("label_")]; mc=["condition","future_condition"]
    fc2=[c for c in df_out.columns if c not in lc+mc]
    df_out=df_out[mc+sorted(fc2)+lc]; df_out.to_csv(OUTPUT_PATH,index=False)
    maps={"label_next_direction":{0:"bearish",1:"doji",2:"bullish"},"label_next_body_size":{0:"small",1:"normal",2:"large"},
          "label_next_pattern":{0:"impulse",1:"pin",2:"inside_doji",3:"engulf"},
          "label_seq_continuation":{0:"reversal",1:"indecision",2:"continuation"},
          "label_next_close_grav":{0:"lower_third",1:"mid",2:"upper_third"},
          "label_next_volatility":{0:"compression",1:"normal",2:"expansion"}}
    print(f"\n✅  Saved → {OUTPUT_PATH}  shape={df_out.shape}  features={len(fc2)}")
    for l in lc:
        counts=df_out[l].value_counts().sort_index()
        print(f"    {l}: { {maps[l].get(k,k):f'{100*v/len(df_out):.1f}%' for k,v in counts.items()} }")
