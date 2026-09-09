# synthetic_vwap.py · v5.1 (latent-state engine + VWAP-specialist conditions)
# SPECIALIST: VWAP / Volume — overweights MATURE, RANGE, SWEEP, PULLBACK
# NEW v5.1: 7 VWAP-only conditions; volume-state → price generation;
#           context features; latent metadata export.
# Run: python synthetic_vwap.py  →  synthetic_vwap_dataset.csv
import pandas as pd, numpy as np, random
try:    from tqdm import trange
except: trange=lambda n,**k:range(n)

N_SAMPLES=10000; OUTPUT_PATH="synthetic_vwap_dataset.csv"
BASE_PRICE=1.5000; SEED=42
random.seed(SEED); np.random.seed(SEED); EPS=1e-12
EXPORT_METADATA = True

# ── CONDITIONS (VWAP-specialist; includes 7 VWAP-only conditions) ────
CONDITIONS = {
    # Standard conditions (reduced weights to make room for VWAP-specific)
    "STRONG_BULL":             {"w":0.02,"vm":1.3, "bs":0.0010,"seq":["bull","bull","bull","bull"]},
    "STRONG_BEAR":             {"w":0.02,"vm":1.3, "bs":0.0010,"seq":["bear","bear","bear","bear"]},
    "ACCELERATING_BULL":       {"w":0.01,"vm":1.6, "bs":0.0012,"seq":["bull","bull_s","bull_s","bull_x"]},
    "ACCELERATING_BEAR":       {"w":0.01,"vm":1.6, "bs":0.0012,"seq":["bear","bear_s","bear_s","bear_x"]},
    "MATURE_BULL":             {"w":0.07,"vm":1.0, "bs":0.0008,"seq":["bull","bull","doji","bull_w"]},
    "MATURE_BEAR":             {"w":0.07,"vm":1.0, "bs":0.0008,"seq":["bear","bear","doji","bear_w"]},
    "BULL_PULLBACK":           {"w":0.05,"vm":0.9, "bs":0.0007,"seq":["bull","bull","bear","bear"]},
    "BEAR_PULLBACK":           {"w":0.05,"vm":0.9, "bs":0.0007,"seq":["bear","bear","bull","bull"]},
    "BULL_PULLBACK_ENDING":    {"w":0.03,"vm":0.8, "bs":0.0006,"seq":["bull","bear","bear_w","doji"]},
    "BEAR_PULLBACK_ENDING":    {"w":0.03,"vm":0.8, "bs":0.0006,"seq":["bear","bull","bull_w","doji"]},
    "RANGE":                   {"w":0.06,"vm":0.7, "bs":0.0005,"seq":["bull","bear","bull","bear"]},
    "RANGE_HIGH":              {"w":0.04,"vm":0.7, "bs":0.0005,"seq":["bull","bull_w","doji","bear_w"]},
    "RANGE_LOW":               {"w":0.04,"vm":0.7, "bs":0.0005,"seq":["bear","bear_w","doji","bull_w"]},
    "RANGE_COMPRESSION":       {"w":0.02,"vm":0.45,"bs":0.0002,"seq":["doji","doji","doji","doji"]},
    "BREAKOUT_SETUP_BULL":     {"w":0.01,"vm":0.8, "bs":0.0006,"seq":["doji","doji","bull_w","bull"]},
    "BREAKOUT_SETUP_BEAR":     {"w":0.01,"vm":0.8, "bs":0.0006,"seq":["doji","doji","bear_w","bear"]},
    "BREAKOUT_CONFIRMED_BULL": {"w":0.01,"vm":2.2, "bs":0.0015,"seq":["doji","bull","bull_s","bull_x"]},
    "BREAKOUT_CONFIRMED_BEAR": {"w":0.01,"vm":2.2, "bs":0.0015,"seq":["doji","bear","bear_s","bear_x"]},
    "FALSE_BREAKOUT_BULL":     {"w":0.01,"vm":1.8, "bs":0.0011,"seq":["bull_s","spike_up","rejection","bear"]},
    "FALSE_BREAKOUT_BEAR":     {"w":0.01,"vm":1.8, "bs":0.0011,"seq":["bear_s","spike_dn","rejection","bull"]},
    "EARLY_BULL_REVERSAL":     {"w":0.02,"vm":1.2, "bs":0.0009,"seq":["bear","bear","doji","bull_s"]},
    "EARLY_BEAR_REVERSAL":     {"w":0.02,"vm":1.2, "bs":0.0009,"seq":["bull","bull","doji","bear_s"]},
    "LIQUIDITY_SWEEP_HIGH":    {"w":0.06,"vm":2.0, "bs":0.0013,"seq":["bull","bull","spike_up","bear_s"]},
    "LIQUIDITY_SWEEP_LOW":     {"w":0.06,"vm":2.0, "bs":0.0013,"seq":["bear","bear","spike_dn","bull_s"]},
    # ── VWAP-ONLY conditions ─────────────────────────────────────────
    "VWAP_RECLAIM":            {"w":0.05,"vm":1.4, "bs":0.0009,"seq":["bear_w","doji","bull","bull"]},
    "VWAP_REJECTION":          {"w":0.05,"vm":1.4, "bs":0.0009,"seq":["bull","spike_up","rejection","bear_w"]},
    "ACCUMULATION":            {"w":0.05,"vm":0.8, "bs":0.0005,"seq":["bear_w","doji","doji","bull_w"]},
    "DISTRIBUTION":            {"w":0.05,"vm":0.8, "bs":0.0005,"seq":["bull_w","doji","doji","bear_w"]},
    "HIGH_VOLUME_ACCEPTANCE":  {"w":0.04,"vm":2.0, "bs":0.0008,"seq":["bull","bull","bull_w","doji"]},
    "LOW_VOLUME_DRIFT":        {"w":0.03,"vm":0.35,"bs":0.0004,"seq":["bull_w","bull_w","doji","bear_w"]},
    "VOLUME_CLIMAX":           {"w":0.04,"vm":2.8, "bs":0.0014,"seq":["bull_x","spike_up","rejection","bear_s"]},
}
TRANSITIONS = {
    "STRONG_BULL":{"STRONG_BULL":0.20,"MATURE_BULL":0.25,"EARLY_BEAR_REVERSAL":0.10,"BULL_PULLBACK":0.25,"VWAP_REJECTION":0.10,"DISTRIBUTION":0.10},
    "STRONG_BEAR":{"STRONG_BEAR":0.20,"MATURE_BEAR":0.25,"EARLY_BULL_REVERSAL":0.10,"BEAR_PULLBACK":0.25,"VWAP_RECLAIM":0.10,"ACCUMULATION":0.10},
    "ACCELERATING_BULL":{"STRONG_BULL":0.15,"LIQUIDITY_SWEEP_HIGH":0.20,"EARLY_BEAR_REVERSAL":0.20,"VOLUME_CLIMAX":0.20,"DISTRIBUTION":0.15,"VWAP_REJECTION":0.10},
    "ACCELERATING_BEAR":{"STRONG_BEAR":0.15,"LIQUIDITY_SWEEP_LOW":0.20,"EARLY_BULL_REVERSAL":0.20,"VOLUME_CLIMAX":0.20,"ACCUMULATION":0.15,"VWAP_RECLAIM":0.10},
    "MATURE_BULL":{"BULL_PULLBACK":0.25,"EARLY_BEAR_REVERSAL":0.15,"RANGE":0.15,"VWAP_REJECTION":0.20,"DISTRIBUTION":0.15,"HIGH_VOLUME_ACCEPTANCE":0.10},
    "MATURE_BEAR":{"BEAR_PULLBACK":0.25,"EARLY_BULL_REVERSAL":0.15,"RANGE":0.15,"VWAP_RECLAIM":0.20,"ACCUMULATION":0.15,"HIGH_VOLUME_ACCEPTANCE":0.10},
    "BULL_PULLBACK":{"BULL_PULLBACK_ENDING":0.30,"STRONG_BULL":0.20,"VWAP_RECLAIM":0.25,"ACCUMULATION":0.15,"RANGE":0.10},
    "BEAR_PULLBACK":{"BEAR_PULLBACK_ENDING":0.30,"STRONG_BEAR":0.20,"VWAP_REJECTION":0.25,"DISTRIBUTION":0.15,"RANGE":0.10},
    "BULL_PULLBACK_ENDING":{"STRONG_BULL":0.30,"VWAP_RECLAIM":0.25,"HIGH_VOLUME_ACCEPTANCE":0.20,"RANGE":0.15,"BEAR_PULLBACK":0.10},
    "BEAR_PULLBACK_ENDING":{"STRONG_BEAR":0.30,"VWAP_REJECTION":0.25,"HIGH_VOLUME_ACCEPTANCE":0.20,"RANGE":0.15,"BULL_PULLBACK":0.10},
    "RANGE":{"RANGE":0.20,"RANGE_HIGH":0.10,"RANGE_LOW":0.10,"RANGE_COMPRESSION":0.15,"LOW_VOLUME_DRIFT":0.20,"ACCUMULATION":0.13,"DISTRIBUTION":0.12},
    "RANGE_HIGH":{"RANGE":0.15,"FALSE_BREAKOUT_BULL":0.20,"VWAP_REJECTION":0.30,"EARLY_BEAR_REVERSAL":0.20,"LIQUIDITY_SWEEP_HIGH":0.15},
    "RANGE_LOW":{"RANGE":0.15,"FALSE_BREAKOUT_BEAR":0.20,"VWAP_RECLAIM":0.30,"EARLY_BULL_REVERSAL":0.20,"LIQUIDITY_SWEEP_LOW":0.15},
    "RANGE_COMPRESSION":{"BREAKOUT_SETUP_BULL":0.20,"BREAKOUT_SETUP_BEAR":0.20,"LOW_VOLUME_DRIFT":0.30,"RANGE":0.15,"ACCUMULATION":0.15},
    "BREAKOUT_SETUP_BULL":{"BREAKOUT_CONFIRMED_BULL":0.35,"FALSE_BREAKOUT_BULL":0.20,"HIGH_VOLUME_ACCEPTANCE":0.25,"VWAP_REJECTION":0.20},
    "BREAKOUT_SETUP_BEAR":{"BREAKOUT_CONFIRMED_BEAR":0.35,"FALSE_BREAKOUT_BEAR":0.20,"HIGH_VOLUME_ACCEPTANCE":0.25,"VWAP_RECLAIM":0.20},
    "BREAKOUT_CONFIRMED_BULL":{"STRONG_BULL":0.35,"HIGH_VOLUME_ACCEPTANCE":0.30,"BULL_PULLBACK":0.20,"DISTRIBUTION":0.15},
    "BREAKOUT_CONFIRMED_BEAR":{"STRONG_BEAR":0.35,"HIGH_VOLUME_ACCEPTANCE":0.30,"BEAR_PULLBACK":0.20,"ACCUMULATION":0.15},
    "FALSE_BREAKOUT_BULL":{"RANGE":0.25,"STRONG_BEAR":0.20,"VWAP_REJECTION":0.25,"DISTRIBUTION":0.20,"BEAR_PULLBACK":0.10},
    "FALSE_BREAKOUT_BEAR":{"RANGE":0.25,"STRONG_BULL":0.20,"VWAP_RECLAIM":0.25,"ACCUMULATION":0.20,"BULL_PULLBACK":0.10},
    "EARLY_BULL_REVERSAL":{"STRONG_BULL":0.25,"VWAP_RECLAIM":0.30,"HIGH_VOLUME_ACCEPTANCE":0.20,"RANGE":0.15,"BULL_PULLBACK_ENDING":0.10},
    "EARLY_BEAR_REVERSAL":{"STRONG_BEAR":0.25,"VWAP_REJECTION":0.30,"HIGH_VOLUME_ACCEPTANCE":0.20,"RANGE":0.15,"BEAR_PULLBACK_ENDING":0.10},
    "LIQUIDITY_SWEEP_HIGH":{"STRONG_BEAR":0.25,"EARLY_BEAR_REVERSAL":0.20,"VWAP_REJECTION":0.30,"DISTRIBUTION":0.15,"FALSE_BREAKOUT_BULL":0.10},
    "LIQUIDITY_SWEEP_LOW":{"STRONG_BULL":0.25,"EARLY_BULL_REVERSAL":0.20,"VWAP_RECLAIM":0.30,"ACCUMULATION":0.15,"FALSE_BREAKOUT_BEAR":0.10},
    # VWAP-only transitions
    "VWAP_RECLAIM":{"STRONG_BULL":0.25,"HIGH_VOLUME_ACCEPTANCE":0.25,"BULL_PULLBACK":0.20,"MATURE_BULL":0.15,"DISTRIBUTION":0.15},
    "VWAP_REJECTION":{"STRONG_BEAR":0.25,"HIGH_VOLUME_ACCEPTANCE":0.25,"BEAR_PULLBACK":0.20,"MATURE_BEAR":0.15,"ACCUMULATION":0.15},
    "ACCUMULATION":{"VWAP_RECLAIM":0.30,"BULL_PULLBACK_ENDING":0.25,"HIGH_VOLUME_ACCEPTANCE":0.20,"RANGE":0.15,"LOW_VOLUME_DRIFT":0.10},
    "DISTRIBUTION":{"VWAP_REJECTION":0.30,"BEAR_PULLBACK_ENDING":0.25,"HIGH_VOLUME_ACCEPTANCE":0.20,"RANGE":0.15,"LOW_VOLUME_DRIFT":0.10},
    "HIGH_VOLUME_ACCEPTANCE":{"STRONG_BULL":0.20,"STRONG_BEAR":0.20,"MATURE_BULL":0.15,"MATURE_BEAR":0.15,"VOLUME_CLIMAX":0.15,"DISTRIBUTION":0.15},
    "LOW_VOLUME_DRIFT":{"RANGE":0.30,"ACCUMULATION":0.20,"DISTRIBUTION":0.20,"RANGE_COMPRESSION":0.20,"LOW_VOLUME_DRIFT":0.10},
    "VOLUME_CLIMAX":{"STRONG_BEAR":0.25,"STRONG_BULL":0.15,"EARLY_BEAR_REVERSAL":0.20,"FALSE_BREAKOUT_BULL":0.20,"VWAP_REJECTION":0.20},
}

# ── LATENT STATE (VWAP specialist: volume-state drives price) ────────
class _LS:
    __slots__=("ts","age","vr","cs","cd","pd","lp","bp","rp",
               "vol_pressure","accept_str","distrib_str","accum_str","_vn")
    def __init__(self):
        self.ts=float(np.random.uniform(.3,.7)); self.age=int(np.random.randint(1,6))
        self.vr="normal"; self.cs=float(np.random.uniform(0,.2)); self.cd=0
        self.pd=0.; self.lp=float(np.random.uniform(0,.2)); self.bp=0.; self.rp=0.
        self.vol_pressure  = float(np.random.uniform(.3,.7))  # overall volume urgency
        self.accept_str    = float(np.random.uniform(0,.3))   # market accepting price (high vol at price)
        self.distrib_str   = float(np.random.uniform(0,.3))   # distribution (vol at highs)
        self.accum_str     = float(np.random.uniform(0,.3))   # accumulation (vol at lows)
        self._vn=1

def _init_ls(cond):
    ls=_LS()
    if "VOLUME_CLIMAX" in cond:
        ls.vol_pressure=np.random.uniform(.80,1.); ls.distrib_str=np.random.uniform(.6,.9); ls._vn=2
    elif "HIGH_VOLUME_ACCEPTANCE" in cond:
        ls.vol_pressure=np.random.uniform(.65,.90); ls.accept_str=np.random.uniform(.5,.85); ls._vn=2
    elif "ACCUMULATION" in cond:
        ls.accum_str=np.random.uniform(.5,.85); ls.vol_pressure=np.random.uniform(.4,.7); ls._vn=1
    elif "DISTRIBUTION" in cond:
        ls.distrib_str=np.random.uniform(.5,.85); ls.vol_pressure=np.random.uniform(.4,.7); ls._vn=1
    elif "VWAP_RECLAIM" in cond:
        ls.accum_str=np.random.uniform(.4,.7); ls.ts=np.random.uniform(.4,.7); ls._vn=1
    elif "VWAP_REJECTION" in cond:
        ls.distrib_str=np.random.uniform(.4,.7); ls.lp=np.random.uniform(.4,.7); ls._vn=1
    elif "LOW_VOLUME_DRIFT" in cond:
        ls.vol_pressure=np.random.uniform(.05,.25); ls.ts=np.random.uniform(.1,.35); ls._vn=0
    elif any(k in cond for k in ("MATURE","RANGE_HIGH","RANGE_LOW")):
        ls.ts=np.random.uniform(.35,.65); ls.age=np.random.randint(10,22)
        ls.lp=np.random.uniform(.3,.6); ls._vn=1
    elif "SWEEP" in cond:
        ls.lp=np.random.uniform(.70,1.); ls.vol_pressure=np.random.uniform(.7,.95); ls._vn=2
    elif any(k in cond for k in ("STRONG","ACCELERATING")):
        ls.ts=np.random.uniform(.65,1.); ls.age=np.random.randint(2,8); ls._vn=2
    elif "RANGE" in cond:
        ls.ts=np.random.uniform(0,.30); ls.cs=np.random.uniform(.20,.50); ls._vn=0
    ls.vr=["low","normal","high"][ls._vn]; return ls

def _step_ls(ls,cond):
    # standard state updates
    if any(k in cond for k in ("STRONG","ACCELERATING","MATURE")): ls.age=min(ls.age+1,40)
    elif any(k in cond for k in ("REVERSAL","BREAKOUT_CONFIRMED","VWAP_RECLAIM","VWAP_REJECTION")): ls.age=1
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
    elif "PULLBACK_ENDING" in cond or any(k in cond for k in ("VWAP_RECLAIM","ACCUMULATION")): ls.pd=max(0.,ls.pd-.20)
    if any(k in cond for k in ("MATURE","RANGE_HIGH","RANGE_LOW","SWEEP")): ls.lp=min(1.,ls.lp+np.random.uniform(.06,.14))
    elif any(k in cond for k in ("SWEEP","FALSE_BREAKOUT","VOLUME_CLIMAX")): ls.lp=max(0.,ls.lp-np.random.uniform(.3,.6))
    # VWAP-specialist volume state updates
    if "VOLUME_CLIMAX" in cond:
        ls.vol_pressure=min(1.,ls.vol_pressure+np.random.uniform(.15,.30))
        ls.distrib_str=min(1.,ls.distrib_str+np.random.uniform(.10,.20))
    elif "HIGH_VOLUME_ACCEPTANCE" in cond:
        ls.vol_pressure=min(1.,ls.vol_pressure+np.random.uniform(.08,.18))
        ls.accept_str=min(1.,ls.accept_str+np.random.uniform(.08,.18))
    elif "ACCUMULATION" in cond:
        ls.accum_str=min(1.,ls.accum_str+np.random.uniform(.08,.16)); ls.distrib_str=max(0.,ls.distrib_str-.08)
    elif "DISTRIBUTION" in cond:
        ls.distrib_str=min(1.,ls.distrib_str+np.random.uniform(.08,.16)); ls.accum_str=max(0.,ls.accum_str-.08)
    elif "VWAP_RECLAIM" in cond: ls.accum_str=min(1.,ls.accum_str+.10); ls.distrib_str=max(0.,ls.distrib_str-.10)
    elif "VWAP_REJECTION" in cond: ls.distrib_str=min(1.,ls.distrib_str+.10); ls.accum_str=max(0.,ls.accum_str-.10)
    elif "LOW_VOLUME_DRIFT" in cond:
        ls.vol_pressure=max(0.,ls.vol_pressure-np.random.uniform(.08,.15))
        ls.accept_str=max(0.,ls.accept_str-.05); ls.accum_str=max(0.,ls.accum_str-.05); ls.distrib_str=max(0.,ls.distrib_str-.05)
    elif any(k in cond for k in ("STRONG","ACCELERATING")):
        ls.vol_pressure=min(1.,ls.vol_pressure+.08); ls.accept_str=min(1.,ls.accept_str+.05)
    else: ls.vol_pressure=max(0.,ls.vol_pressure-np.random.uniform(0,.05))
    # vol regime
    if any(k in cond for k in ("ACCELERATING","BREAKOUT_CONFIRMED","SWEEP","VOLUME_CLIMAX","HIGH_VOLUME_ACCEPTANCE")): ls._vn=min(2,ls._vn+1)
    elif any(k in cond for k in ("COMPRESSION","LOW_VOLUME_DRIFT")): ls._vn=max(0,ls._vn-1)
    elif "RANGE" in cond and np.random.random()<.35: ls._vn=max(0,ls._vn-1)
    ls.vr=["low","normal","high"][ls._vn]
    for a in ("ts","cs","bp","pd","lp","rp","vol_pressure","accept_str","distrib_str","accum_str"):
        setattr(ls,a,float(np.clip(getattr(ls,a),0.,1.))); return ls

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
        if ls.lp>.60:
            if any(k in c for k in ("SWEEP","FALSE_BREAKOUT","VWAP_REJECTION","VWAP_RECLAIM")): b*=1.+ls.lp*2.
        if ls.rp>.55:
            if any(k in c for k in ("REVERSAL","SWEEP","FALSE_BREAKOUT","VWAP_REJECTION")): b*=1.+ls.rp*1.8
            if any(k in c for k in ("STRONG","ACCELERATING")): b*=max(.1,1.-ls.rp*.9)
        # volume state biases
        if ls.distrib_str>.55:
            if any(k in c for k in ("DISTRIBUTION","VWAP_REJECTION","VOLUME_CLIMAX")): b*=1.+ls.distrib_str*2.
        if ls.accum_str>.55:
            if any(k in c for k in ("ACCUMULATION","VWAP_RECLAIM","HIGH_VOLUME_ACCEPTANCE")): b*=1.+ls.accum_str*2.
        if ls.vol_pressure>.75:
            if any(k in c for k in ("VOLUME_CLIMAX","HIGH_VOLUME_ACCEPTANCE")): b*=1.+ls.vol_pressure*1.5
        if ls.vol_pressure<.25:
            if "LOW_VOLUME_DRIFT" in c: b*=2.5
        w[c]=max(1e-6,p*b)
    t=sum(w.values()); ks=list(w.keys())
    return random.choices(ks,weights=[w[k]/t for k in ks],k=1)[0]

def _mod(cond,ls,bs,vm):
    # Volume-first: vol_pressure drives vm before trend_strength drives bs
    vp_scale = 0.5 + ls.vol_pressure  # 0.5-1.5 range
    vs={"low":.60,"normal":1.,"high":1.40}[ls.vr]
    bs*=vs; vm*=vs*vp_scale
    bt=float(np.clip(.45+.55*ls.ts,.25,1.55))*float(np.clip(1.-ls.age/40.,.4,1.2))
    wx=float(np.clip(1.+ls.age/30.,1.,1.6))*float(np.clip(1.+ls.lp*.8,1.,2.2))
    # VWAP-condition specific modulation
    if "VOLUME_CLIMAX" in cond: vm*=1.+ls.vol_pressure; bs*=1.2; wx*=1.3
    elif "HIGH_VOLUME_ACCEPTANCE" in cond: vm*=1.+ls.accept_str*.8; bt*=0.85
    elif "ACCUMULATION" in cond: vm*=1.+ls.accum_str*.5; bt*=0.70; wx*=1.2
    elif "DISTRIBUTION" in cond: vm*=1.+ls.distrib_str*.5; bt*=0.70; wx*=1.2
    elif "LOW_VOLUME_DRIFT" in cond: vm*=max(.25,1.-ls.vol_pressure*.7); bs*=.6; bt*=0.5
    elif "VWAP_RECLAIM" in cond: vm*=1.+ls.accum_str*.4; bt*=0.9
    elif "VWAP_REJECTION" in cond: vm*=1.+ls.distrib_str*.4; bt*=0.9; wx*=1.15
    if "COMPRESSION" in cond or "RANGE" in cond:
        bs*=float(np.clip(1.-ls.cs*.65,.2,1.)); vm*=max(.3,1.-ls.cs*.5)
    if "BREAKOUT_CONFIRMED" in cond: bs*=1.+ls.bp*.8; vm*=1.+ls.bp*.6
    if "PULLBACK" in cond and "ENDING" not in cond: bt*=max(.35,1.-ls.pd*.5); wx*=1.+ls.pd*.45
    return float(np.clip(bs,.0001,.004)),float(np.clip(vm,.1,7.)),float(np.clip(wx,.5,3.5)),float(np.clip(bt,.2,1.6))

# ── COMPACT M15 EXPANSION (same as other generators) ─────────────────
def _expand(arch,p,bs,vm,wx=1.):
    def c(o,cl,wu,wl,vlo,vhi):
        h=max(o,cl)+abs(float(np.random.normal(0,wu*wx))); l=min(o,cl)-abs(float(np.random.normal(0,wl*wx)))
        return {"open":float(o),"high":max(h,o,cl),"low":min(l,o,cl),"close":float(cl),"volume":int(np.random.uniform(vlo,vhi))}
    r=lambda a,b:float(np.random.uniform(a,b))
    if arch=="bull":
        cs=[c(p,p+r(.10,.35)*bs,bs*.12,bs*.20,200*vm,600*vm)]
        for d,wu,wl in [(.25,.10,.15),(.30,.10,.12),(.35,.06,.10)]: cs.append(c(cs[-1]["close"],cs[-1]["close"]+r(.15,d+.20)*bs,bs*wu,bs*wl,300*vm,900*vm))
    elif arch=="bull_s":
        cs=[c(p,p+r(.30,.60)*bs,bs*.06,bs*.15,500*vm,1200*vm)]
        for d,wu,wl in [(.45,.05,.10),(.55,.04,.09),(.65,.03,.07)]: cs.append(c(cs[-1]["close"],cs[-1]["close"]+r(.35,d+.20)*bs,bs*wu,bs*wl,600*vm,1500*vm))
    elif arch=="bull_x":
        cs=[c(p,p+r(.50,1.0)*bs,bs*.04,bs*.08,800*vm,2000*vm)]
        for d,wu,wl in [(.85,.03,.07),(1.1,.03,.06),(1.4,.02,.05)]: cs.append(c(cs[-1]["close"],cs[-1]["close"]+r(.70,d+.30)*bs,bs*wu,bs*wl,1200*vm,3500*vm))
    elif arch=="bull_w":
        cs=[c(p,p+r(.05,.18)*bs,bs*.20,bs*.30,130*vm,380*vm)]
        for d in [r(-.08,.18),r(.05,.18),r(0.,.14)]: cs.append(c(cs[-1]["close"],cs[-1]["close"]+d*bs,bs*.25,bs*.35,110*vm,350*vm))
    elif arch=="bear":
        cs=[c(p,p-r(.10,.35)*bs,bs*.20,bs*.12,200*vm,600*vm)]
        for d,wu,wl in [(.25,.15,.10),(.30,.12,.10),(.35,.10,.06)]: cs.append(c(cs[-1]["close"],cs[-1]["close"]-r(.15,d+.20)*bs,bs*wu,bs*wl,300*vm,900*vm))
    elif arch=="bear_s":
        cs=[c(p,p-r(.30,.60)*bs,bs*.15,bs*.06,500*vm,1200*vm)]
        for d,wu,wl in [(.45,.10,.05),(.55,.09,.04),(.65,.07,.03)]: cs.append(c(cs[-1]["close"],cs[-1]["close"]-r(.35,d+.20)*bs,bs*wu,bs*wl,600*vm,1500*vm))
    elif arch=="bear_x":
        cs=[c(p,p-r(.50,1.0)*bs,bs*.08,bs*.04,800*vm,2000*vm)]
        for d,wu,wl in [(.85,.07,.03),(1.1,.06,.03),(1.4,.05,.02)]: cs.append(c(cs[-1]["close"],cs[-1]["close"]-r(.70,d+.30)*bs,bs*wu,bs*wl,1200*vm,3500*vm))
    elif arch=="bear_w":
        cs=[c(p,p-r(.05,.18)*bs,bs*.30,bs*.20,130*vm,380*vm)]
        for d in [r(-.08,.18),r(.05,.18),r(0.,.14)]: cs.append(c(cs[-1]["close"],cs[-1]["close"]-d*bs,bs*.35,bs*.25,110*vm,350*vm))
    elif arch=="doji":
        cs=[c(p,p+r(-.12,.12)*bs,bs*.40,bs*.40,120*vm,450*vm)]
        for d in [r(-.12,.12),r(-.10,.10),r(-.08,.08)]: cs.append(c(cs[-1]["close"],cs[-1]["close"]+d*bs,bs*.38,bs*.38,100*vm,420*vm))
    elif arch=="spike_up":
        cs=[c(p,p+r(.4,.8)*bs,bs*.10,bs*.05,600*vm,1500*vm),c(p+r(.4,.8)*bs,p+r(1.1,2.1)*bs,bs*.20,bs*.05,1200*vm,3200*vm),
            c(p+r(1.1,2.1)*bs,p+r(1.3,2.7)*bs,bs*.45,bs*.10,800*vm,2000*vm),c(p+r(1.3,2.7)*bs,p+r(1.3,2.7)*bs+r(-.3,.3)*bs,bs*.55,bs*.55,500*vm,1200*vm)]
    elif arch=="spike_dn":
        cs=[c(p,p-r(.4,.8)*bs,bs*.05,bs*.10,600*vm,1500*vm),c(p-r(.4,.8)*bs,p-r(1.1,2.1)*bs,bs*.05,bs*.20,1200*vm,3200*vm),
            c(p-r(1.1,2.1)*bs,p-r(1.3,2.7)*bs,bs*.10,bs*.45,800*vm,2000*vm),c(p-r(1.3,2.7)*bs,p-r(1.3,2.7)*bs+r(-.3,.3)*bs,bs*.55,bs*.55,500*vm,1200*vm)]
    elif arch=="rejection":
        cs=[c(p,p+r(.3,.6)*bs,bs*.10,bs*.10,400*vm,1000*vm),c(p+r(.3,.6)*bs,p+r(.8,1.6)*bs,bs*.55,bs*.06,1000*vm,2800*vm),
            c(p+r(.8,1.6)*bs,p+r(.8,1.6)*bs-r(.5,1.1)*bs,bs*.10,bs*.20,900*vm,2500*vm),c(p+r(.3,.5)*bs,p+r(.3,.5)*bs-r(.3,.7)*bs,bs*.12,bs*.15,600*vm,1500*vm)]
    else:
        cs=[]; cp=p
        for _ in range(4): nc=cp+r(-.1,.1)*bs; cs.append(c(cp,nc,bs*.3,bs*.3,200*vm,600*vm)); cp=nc
    while len(cs)<4: cs.append(cs[-1])
    return cs[:4],cs[-1]["close"]

# ── MARKET HISTORY GENERATOR ─────────────────────────────────────────
def _pick():
    ks=list(CONDITIONS.keys()); return random.choices(ks,weights=[CONDITIONS[k]["w"] for k in ks],k=1)[0]

def generate_market_history(base_price=BASE_PRICE):
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

# ── FEATURES (unchanged from v5) ─────────────────────────────────────
def _rz(s,w): return (s-s.rolling(w).mean())/(s.rolling(w).std()+EPS)

def compute_vwap_features(df,hour_vol_avg=None):
    df=df.copy().reset_index(drop=True)
    df["range"]=df["high"]-df["low"]; df["body"]=df["close"]-df["open"]
    df["abs_body"]=df["body"].abs(); df["dir"]=np.sign(df["body"]).fillna(0)
    df["upper_wick"]=df["high"]-df[["open","close"]].max(axis=1)
    df["lower_wick"]=df[["open","close"]].min(axis=1)-df["low"]
    hl=df["range"]; hc=(df["high"]-df["close"].shift(1)).abs().fillna(0); lc=(df["low"]-df["close"].shift(1)).abs().fillna(0)
    df["atr14"]=pd.concat([hl,hc,lc],axis=1).max(axis=1).rolling(14,min_periods=1).mean()
    tp=(df["high"]+df["low"]+df["close"])/3; tp_vol=tp*df["volume"]
    vwap50=tp_vol.rolling(50,min_periods=1).sum()/(df["volume"].rolling(50,min_periods=1).sum()+EPS)
    df["vwap"]=vwap50
    vwap_sq_dev=(tp-vwap50)**2*df["volume"]
    vwap_var=vwap_sq_dev.rolling(50,min_periods=1).sum()/(df["volume"].rolling(50,min_periods=1).sum()+EPS)
    df["vwap_std"]=np.sqrt(vwap_var.clip(lower=0))/(df["atr14"]+EPS)
    df["vwap_dev_atr"]=(df["close"]-df["vwap"])/(df["atr14"]+EPS)
    df["vwap_sigma"]=df["vwap_dev_atr"]/(df["vwap_std"]+EPS)
    df["vwap_slope_z"]=_rz((df["vwap"].diff(5)/(df["atr14"]+EPS)).fillna(0),30).fillna(0.)
    vol_avg20=df["volume"].rolling(20,min_periods=1).mean()
    vol_z_cross=(_rz(df["volume"],20)+1).clip(0,3)
    near_vwap=(df["vwap_dev_atr"].abs()<.50).astype(int)
    cross_bull=((df["close"]>df["vwap"])&(df["close"].shift(1)<=df["vwap"].shift(1))).astype(float)
    cross_bear=((df["close"]<df["vwap"])&(df["close"].shift(1)>=df["vwap"].shift(1))).astype(float)
    df["vwap_reclaim_qual"]=(df["abs_body"]/(df["atr14"]+EPS)*vol_z_cross*cross_bull).fillna(0).clip(0,3)
    df["vwap_rejection_qual"]=(df["abs_body"]/(df["atr14"]+EPS)*vol_z_cross*cross_bear).fillna(0).clip(0,3)
    df["vwap_touch_density"]=near_vwap.rolling(20,min_periods=1).sum()/20.
    close_grav=(df["close"]-df["low"])/(df["range"]+EPS); df["tick_delta"]=close_grav*2-1
    delta_vol=df["tick_delta"]*df["volume"]
    df["cum_delta_20z"]=_rz(delta_vol.rolling(20,min_periods=1).sum(),50).fillna(0.)
    up_vol=df["volume"].where(df["dir"]>0,0.); dn_vol=df["volume"].where(df["dir"]<0,0.)
    up_sum=up_vol.rolling(20,min_periods=1).sum(); dn_sum=dn_vol.rolling(20,min_periods=1).sum()
    df["vol_profile_asym"]=((up_sum-dn_sum)/(up_sum+dn_sum+EPS)).fillna(0.)
    df["buy_press_z"]=_rz(up_sum/(up_sum+dn_sum+EPS),30).fillna(0.)
    df["delta_at_vwap"]=_rz((delta_vol*near_vwap).rolling(20,min_periods=1).sum(),30).fillna(0.)
    df["vol_z30"]=_rz(df["volume"],30).fillna(0.); df["vol_z100"]=_rz(df["volume"],min(100,len(df))).fillna(0.)
    df["vol_trend_z"]=_rz(df["volume"].rolling(5,min_periods=1).mean()/(vol_avg20+EPS),30).fillna(0.)
    if "hour" in df.columns and hour_vol_avg:
        df["session_vol_factor"]=(df["volume"]/df["hour"].map(hour_vol_avg).fillna(vol_avg20)).clip(0,4)
    else: df["session_vol_factor"]=df["vol_z30"].clip(-3,3)+1
    price_dir=np.sign(df["close"].diff(3).fillna(0)); vol_dir=np.sign(df["volume"].rolling(3,min_periods=1).mean().diff(1).fillna(0))
    df["vol_mom_div"]=(price_dir*vol_dir).fillna(0.)
    df["vol_efficiency"]=(df["abs_body"]/(df["atr14"]+EPS)/(df["vol_z30"].abs()+.5)).clip(0,3)
    vol_z20=_rz(df["volume"],20).fillna(0.)
    df["vol_at_vwap_z"]=(vol_z20*near_vwap).fillna(0.)
    df["inst_fp"]=((vol_z20>1.5).astype(int)+(df["vwap_dev_atr"].abs()<.30).astype(int)+((df["abs_body"]/(df["range"]+EPS))<.25).astype(int)).astype(float)
    df["vwap_band_walk"]=(df["vwap_sigma"]-1.).clip(lower=0).rolling(10,min_periods=1).sum()
    df["price_vol_div"]=(df["abs_body"]/(df["atr14"]+EPS)/(vol_z20.abs()+.5)).clip(0,3)
    df["absorption_lvl"]=(vol_z20*((df["abs_body"]/(df["range"]+EPS))<.25).astype(float)).clip(0,3)
    df["vwap_dist_pct"]=((df["close"]-df["vwap"])/(df["vwap"]+EPS)*100).clip(-1,1)
    above_streak=[]; cur=0
    for v in (df["close"]>df["vwap"]).tolist(): cur=cur+1 if v else 0; above_streak.append(cur)
    df["above_vwap_streak"]=above_streak
    df["vwap_mr_z"]=_rz(df["vwap_dev_atr"].abs(),50).fillna(0.)
    df["cum_delta_vwap_align"]=(np.sign(df["vwap_dev_atr"])*np.sign(df["tick_delta"])).fillna(0.)
    df["vol_surge_extreme"]=(vol_z20*(df["vwap_sigma"].abs()>1.5).astype(float)).clip(0,3)
    df["body_ratio"]=df["abs_body"]/(df["range"]+EPS); df["close_grav_f"]=close_grav
    df["vwap_std_feat"]=df["vwap_std"].clip(0,5); df["atr14_z"]=_rz(df["atr14"],50).fillna(0.); df["vol_surge_20"]=vol_z20.clip(-3,3)
    last=df.iloc[-1]
    f={k:float(last[k]) for k in ["vwap_dev_atr","vwap_sigma","vwap_slope_z","vwap_reclaim_qual","vwap_rejection_qual",
       "vwap_touch_density","tick_delta","cum_delta_20z","vol_profile_asym","buy_press_z","delta_at_vwap",
       "vol_z30","vol_z100","vol_trend_z","session_vol_factor","vol_mom_div","vol_efficiency","vol_at_vwap_z",
       "inst_fp","vwap_band_walk","price_vol_div","absorption_lvl","vwap_dist_pct","above_vwap_streak",
       "vwap_mr_z","cum_delta_vwap_align","vol_surge_extreme","body_ratio","close_grav_f","vwap_std_feat","atr14_z","vol_surge_20"]}
    f["_atr"]=float(last["atr14"]) if float(last["atr14"])>EPS else .0001
    f["_context_close"]=float(last["close"]); f["_vwap"]=float(last["vwap"])
    f["_cum_delta"]=float(last["cum_delta_20z"]); f["_vwap_sigma"]=float(last["vwap_sigma"])
    f["_slope_z"]=float(last["vwap_slope_z"]); f["_tick_delta"]=float(last["tick_delta"])
    return f,df

# ── CONTEXT FEATURES ─────────────────────────────────────────────────
def context_features_vwap(df,atr):
    w=df.iloc[-16:].reset_index(drop=True)
    tp=(w["high"]+w["low"]+w["close"])/3
    tp_vol=tp*w["volume"]
    vwap_ctx=float(tp_vol.sum()/(w["volume"].sum()+EPS))
    close=w["close"].astype(float)
    vwap_dist=(close-vwap_ctx)/(atr+EPS)
    above=(close>vwap_ctx).astype(int)
    vwap_acceptance_16=float(above.mean())           # fraction of context bars above VWAP
    cross_up=((above==1)&(above.shift(1)==0)).sum(); cross_dn=((above==0)&(above.shift(1)==1)).sum()
    vwap_rejection_16=float((cross_dn-cross_up)/max(cross_dn+cross_up,1))  # net rejections
    vwap_distance_16=float(np.clip(vwap_dist.iloc[-1],-3,3))               # last-bar distance
    vol_avg=float(w["volume"].mean())
    vol_surge_ctx=float(np.clip(float(w["volume"].iloc[-4:].mean())/(vol_avg+EPS)-1,-1,2))  # late surge
    return {"vwap_acceptance_16":vwap_acceptance_16,"vwap_rejection_16":vwap_rejection_16,
            "vwap_distance_16":vwap_distance_16,"vol_surge_ctx_16":vol_surge_ctx}

# ── LABELS (unchanged) ───────────────────────────────────────────────
def compute_labels(f,future_candles):
    atr=max(f["_atr"],EPS); ref=f["_context_close"]; vwap=f["_vwap"]
    c1=future_candles[0]; c2=future_candles[1] if len(future_candles)>=2 else c1
    c3=future_candles[2] if len(future_candles)>=3 else c2; c4=future_candles[3] if len(future_candles)>=4 else c3
    h1_close=c4["close"]; h1_high=max(x["high"] for x in future_candles); h1_low=min(x["low"] for x in future_candles)
    h1_range=(h1_high-h1_low)/atr; net_atr=(h1_close-ref)/atr
    bodies_atr=[abs(x["close"]-x["open"])/atr for x in future_candles]; avg_body=float(np.mean(bodies_atr))
    close_grav=(h1_close-h1_low)/(h1_range*atr+EPS)
    c4_above_vwap=c4["close"]>vwap; c1_above_vwap=c1["close"]>vwap; c1_crossed=(ref>vwap)!=c1_above_vwap
    vwap_dir=2 if net_atr>.40 and c4_above_vwap else (0 if net_atr<-.40 and not c4_above_vwap else 1)
    next_vol_char=2 if avg_body>1.2 else (0 if avg_body<.3 else 1)
    if c1_crossed and c4_above_vwap: vwap_int=2 if bodies_atr[0]>.5 and avg_body>.7 else 1
    elif c4_above_vwap and net_atr>.55: vwap_int=3
    elif not c1_above_vwap and not c4_above_vwap and net_atr<-.35: vwap_int=0
    else: vwap_int=1
    next_inst=(2 if net_atr>.3 else 0) if avg_body>.8 and h1_range>1.5 else 1
    next_close=2 if close_grav>.67 else (0 if close_grav<.33 else 1)
    sigma=f["_vwap_sigma"]; cum_d=f["_cum_delta"]; slope=f["_slope_z"]
    bull_ctx=sigma>.3 and cum_d>.3 and slope>.2; bear_ctx=sigma<-.3 and cum_d<-.3 and slope<-.2
    if bull_ctx:   follow=2 if net_atr>.70 else (1 if net_atr>.15 else 0)
    elif bear_ctx: follow=2 if net_atr<-.70 else (1 if net_atr<-.15 else 0)
    else:          follow=2 if avg_body>.8 else (1 if avg_body>.3 else 0)
    return int(vwap_dir),int(next_vol_char),int(vwap_int),int(next_inst),int(next_close),int(follow)

# ── SEQUENCE GENERATOR & MAIN ─────────────────────────────────────────
def generate_sequence(base_price=BASE_PRICE):
    hist,active,fc,fut,ls=generate_market_history(base_price)
    df=pd.DataFrame(hist); df["hour"]=[(i*3)%24 for i in range(len(df))]
    hva=df.groupby("hour")["volume"].mean().to_dict()
    f,df2=compute_vwap_features(df,hour_vol_avg=hva)
    atr=max(f["_atr"],EPS); ctx=context_features_vwap(df2,atr)
    labels=compute_labels(f,fut)
    row={k:v for k,v in f.items() if not k.startswith("_")}
    row.update(ctx); row["condition"]=active; row["future_condition"]=fc
    if EXPORT_METADATA:
        row["meta_trend_strength"]=round(ls.ts,3); row["meta_trend_age"]=ls.age
        row["meta_vol_regime"]=ls.vr; row["meta_compression_score"]=round(ls.cs,3)
        row["meta_compression_duration"]=ls.cd; row["meta_pullback_depth"]=round(ls.pd,3)
        row["meta_liquidity_pressure"]=round(ls.lp,3); row["meta_breakout_pressure"]=round(ls.bp,3)
        row["meta_reversal_pressure"]=round(ls.rp,3)
        row["meta_vol_pressure"]=round(ls.vol_pressure,3); row["meta_accept_str"]=round(ls.accept_str,3)
        row["meta_distrib_str"]=round(ls.distrib_str,3); row["meta_accum_str"]=round(ls.accum_str,3)
    (row["label_next_vwap_dir"],row["label_next_vol_char"],row["label_vwap_interaction"],
     row["label_next_inst_signal"],row["label_next_close_location"],row["label_vwap_follow_through"])=labels
    return row

if __name__=="__main__":
    print(f"Generating {N_SAMPLES} VWAP samples | v5.1 | VWAP-specialist + volume-first + 7 new conditions")
    rows=[generate_sequence() for _ in trange(N_SAMPLES,desc="vwap")]
    df_out=pd.DataFrame(rows)
    lc=[c for c in df_out.columns if c.startswith("label_")]; mc=["condition","future_condition"]
    meta=[c for c in df_out.columns if c.startswith("meta_")]
    fc2=[c for c in df_out.columns if c not in lc+mc+meta]
    df_out=df_out[mc+sorted(fc2)+meta+lc]; df_out.to_csv(OUTPUT_PATH,index=False)
    maps={"label_next_vwap_dir":{0:"below_away",1:"oscillate_near",2:"above_away"},
          "label_next_vol_char":{0:"declining",1:"stable",2:"surge"},
          "label_vwap_interaction":{0:"rejection",1:"test_return",2:"reclaim_hold",3:"walk_extension"},
          "label_next_inst_signal":{0:"distribution",1:"neutral",2:"accumulation"},
          "label_next_close_location":{0:"lower_third",1:"mid",2:"upper_third"},
          "label_vwap_follow_through":{0:"no",1:"partial",2:"confirmed"}}
    print(f"\n✅  Saved → {OUTPUT_PATH}  shape={df_out.shape}  features={len(fc2)}  meta={len(meta)}")
    for l in lc:
        counts=df_out[l].value_counts().sort_index()
        print(f"    {l}: { {maps[l].get(k,k):f'{100*v/len(df_out):.1f}%' for k,v in counts.items()} }")
