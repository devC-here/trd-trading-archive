# synthetic_microstructure.py · v5.1 (latent-state engine)
# SPECIALIST: Orderflow & Tick Physics — overweights SWEEP/FALSE_BREAKOUT/REVERSAL
# Latent extras: buy_pressure, sell_pressure, absorption_strength, delta_imbalance
# Orderflow drives volume which drives price (not reverse)
# Run: python synthetic_microstructure.py  →  synthetic_microstructure_dataset.csv
import pandas as pd, numpy as np, random
try:    from tqdm import trange
except: trange=lambda n,**k:range(n)

N_SAMPLES=100000; OUTPUT_PATH="synthetic_microstructure_dataset.csv"
BASE_PRICE=1.5000; SEED=42
random.seed(SEED); np.random.seed(SEED); EPS=1e-12
EXPORT_METADATA = True

CONDITIONS={
    "STRONG_BULL":{"w":.02,"vm":1.3,"bs":.0010,"seq":["bull","bull","bull","bull"]},
    "STRONG_BEAR":{"w":.02,"vm":1.3,"bs":.0010,"seq":["bear","bear","bear","bear"]},
    "ACCELERATING_BULL":{"w":.02,"vm":1.6,"bs":.0012,"seq":["bull","bull_s","bull_s","bull_x"]},
    "ACCELERATING_BEAR":{"w":.02,"vm":1.6,"bs":.0012,"seq":["bear","bear_s","bear_s","bear_x"]},
    "MATURE_BULL":{"w":.03,"vm":1.0,"bs":.0008,"seq":["bull","bull","doji","bull_w"]},
    "MATURE_BEAR":{"w":.03,"vm":1.0,"bs":.0008,"seq":["bear","bear","doji","bear_w"]},
    "BULL_PULLBACK":{"w":.04,"vm":.9,"bs":.0007,"seq":["bull","bull","bear","bear"]},
    "BEAR_PULLBACK":{"w":.04,"vm":.9,"bs":.0007,"seq":["bear","bear","bull","bull"]},
    "BULL_PULLBACK_ENDING":{"w":.03,"vm":.8,"bs":.0006,"seq":["bull","bear","bear_w","doji"]},
    "BEAR_PULLBACK_ENDING":{"w":.03,"vm":.8,"bs":.0006,"seq":["bear","bull","bull_w","doji"]},
    "RANGE":{"w":.04,"vm":.7,"bs":.0005,"seq":["bull","bear","bull","bear"]},
    "RANGE_HIGH":{"w":.03,"vm":.7,"bs":.0005,"seq":["bull","bull_w","doji","bear_w"]},
    "RANGE_LOW":{"w":.03,"vm":.7,"bs":.0005,"seq":["bear","bear_w","doji","bull_w"]},
    "RANGE_COMPRESSION":{"w":.03,"vm":.45,"bs":.0002,"seq":["doji","doji","doji","doji"]},
    "BREAKOUT_SETUP_BULL":{"w":.02,"vm":.8,"bs":.0006,"seq":["doji","doji","bull_w","bull"]},
    "BREAKOUT_SETUP_BEAR":{"w":.02,"vm":.8,"bs":.0006,"seq":["doji","doji","bear_w","bear"]},
    "BREAKOUT_CONFIRMED_BULL":{"w":.06,"vm":2.0,"bs":.0014,"seq":["doji","bull","bull_s","bull_x"]},
    "BREAKOUT_CONFIRMED_BEAR":{"w":.06,"vm":2.0,"bs":.0014,"seq":["doji","bear","bear_s","bear_x"]},
    "FALSE_BREAKOUT_BULL":{"w":.08,"vm":1.8,"bs":.0011,"seq":["bull_s","spike_up","rejection","bear"]},
    "FALSE_BREAKOUT_BEAR":{"w":.08,"vm":1.8,"bs":.0011,"seq":["bear_s","spike_dn","rejection","bull"]},
    "EARLY_BULL_REVERSAL":{"w":.08,"vm":1.2,"bs":.0009,"seq":["bear","bear","doji","bull_s"]},
    "EARLY_BEAR_REVERSAL":{"w":.08,"vm":1.2,"bs":.0009,"seq":["bull","bull","doji","bear_s"]},
    "LIQUIDITY_SWEEP_HIGH":{"w":.09,"vm":2.0,"bs":.0013,"seq":["bull","bull","spike_up","bear_s"]},
    "LIQUIDITY_SWEEP_LOW":{"w":.09,"vm":2.0,"bs":.0013,"seq":["bear","bear","spike_dn","bull_s"]},
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

# ── LATENT STATE (microstructure specialist: orderflow primitives) ──
class _LS:
    __slots__=("ts","age","vr","cs","cd","pd","lp","bp","rp",
               "buy_p","sell_p","abs_str","delta_imb","_vn")
    def __init__(self):
        self.ts=float(np.random.uniform(.3,.7)); self.age=int(np.random.randint(1,6))
        self.vr="normal"; self.cs=float(np.random.uniform(0,.25)); self.cd=0
        self.pd=0.; self.lp=float(np.random.uniform(0,.25)); self.bp=0.; self.rp=0.
        # orderflow latent vars (0-1)
        self.buy_p=float(np.random.uniform(.3,.7))   # buy pressure
        self.sell_p=float(np.random.uniform(.3,.7))  # sell pressure
        self.abs_str=float(np.random.uniform(0,.3))  # absorption strength
        self.delta_imb=float(np.random.uniform(-.3,.3))  # signed delta imbalance
        self._vn=1

def _init_ls(cond):
    ls=_LS()
    if any(k in cond for k in ("SWEEP","FALSE_BREAKOUT")):
        ls.lp=np.random.uniform(.65,1.); ls.buy_p=np.random.uniform(.6,.9)
        ls.sell_p=np.random.uniform(.6,.9); ls.abs_str=np.random.uniform(.4,.8)
        ls.delta_imb=np.random.uniform(-.8,.8); ls._vn=2
    elif any(k in cond for k in ("REVERSAL","EARLY")):
        ls.rp=np.random.uniform(.55,.85); ls.abs_str=np.random.uniform(.3,.7)
        ls.delta_imb=np.random.uniform(-.5,.5); ls._vn=1
    elif any(k in cond for k in ("STRONG","ACCELERATING")):
        ls.ts=np.random.uniform(.65,1.); ls.age=np.random.randint(2,8)
        ls.buy_p=np.random.uniform(.65,1.) if "BULL" in cond else np.random.uniform(0.,.35)
        ls.sell_p=1.-ls.buy_p; ls.delta_imb=ls.buy_p-.5; ls._vn=2
    elif "BREAKOUT_CONFIRMED" in cond:
        ls.ts=np.random.uniform(.65,.95); ls.bp=np.random.uniform(.65,1.)
        ls.buy_p=np.random.uniform(.6,.9) if "BULL" in cond else np.random.uniform(0.,.4)
        ls.delta_imb=(ls.buy_p-.5)*2; ls._vn=2
    elif "MATURE" in cond:
        ls.ts=np.random.uniform(.35,.65); ls.age=np.random.randint(10,22)
        ls.rp=np.random.uniform(.25,.55); ls.abs_str=np.random.uniform(.2,.5); ls._vn=1
    elif "RANGE" in cond or "COMPRESSION" in cond:
        ls.ts=np.random.uniform(0,.30); ls.cs=np.random.uniform(.20,.55)
        ls.buy_p=np.random.uniform(.4,.6); ls.sell_p=np.random.uniform(.4,.6)
        ls.abs_str=np.random.uniform(.3,.7); ls.delta_imb=np.random.uniform(-.1,.1); ls._vn=0
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
    if any(k in cond for k in ("MATURE","RANGE_HIGH","RANGE_LOW","SWEEP")): ls.lp=min(1.,ls.lp+np.random.uniform(.06,.14))
    elif any(k in cond for k in ("SWEEP","FALSE_BREAKOUT")): ls.lp=max(0.,ls.lp-np.random.uniform(.3,.6))
    age_f=min(ls.age/20.,1.)
    if any(k in cond for k in ("MATURE","SWEEP","FALSE_BREAKOUT")): ls.rp=min(1.,ls.rp+np.random.uniform(.05,.12)+age_f*.04)
    elif any(k in cond for k in ("REVERSAL","BREAKOUT_CONFIRMED","PULLBACK_ENDING")): ls.rp=max(0.,ls.rp-np.random.uniform(.2,.5))
    # orderflow primitives update
    bull_cond=any(k in cond for k in ("BULL","STRONG_BULL","ACCELERATING_BULL","BREAKOUT_CONFIRMED_BULL","EARLY_BULL","SWEEP_LOW"))
    bear_cond=any(k in cond for k in ("BEAR","STRONG_BEAR","ACCELERATING_BEAR","BREAKOUT_CONFIRMED_BEAR","EARLY_BEAR","SWEEP_HIGH"))
    if bull_cond: ls.buy_p=min(1.,ls.buy_p+np.random.uniform(.05,.15)); ls.sell_p=max(0.,ls.sell_p-np.random.uniform(.03,.10))
    elif bear_cond: ls.sell_p=min(1.,ls.sell_p+np.random.uniform(.05,.15)); ls.buy_p=max(0.,ls.buy_p-np.random.uniform(.03,.10))
    else: ls.buy_p=float(np.clip(ls.buy_p+np.random.normal(0,.05),.1,.9)); ls.sell_p=float(np.clip(ls.sell_p+np.random.normal(0,.05),.1,.9))
    if any(k in cond for k in ("SWEEP","FALSE_BREAKOUT","REVERSAL")):
        ls.abs_str=min(1.,ls.abs_str+np.random.uniform(.1,.3))
        ls.delta_imb*=np.random.uniform(.3,.7)  # delta flips/neutralises on sweep
    elif any(k in cond for k in ("RANGE","COMPRESSION")):
        ls.abs_str=min(1.,ls.abs_str+np.random.uniform(.03,.08))
        ls.delta_imb=float(np.clip(ls.delta_imb+np.random.normal(0,.05),-.5,.5))
    else: ls.abs_str=max(0.,ls.abs_str-np.random.uniform(.02,.06))
    ls.delta_imb=float(np.clip((ls.buy_p-ls.sell_p)+np.random.normal(0,.05),-1.,1.))
    if any(k in cond for k in ("ACCELERATING","BREAKOUT_CONFIRMED","SWEEP")): ls._vn=min(2,ls._vn+1)
    elif "COMPRESSION" in cond: ls._vn=max(0,ls._vn-1)
    elif "RANGE" in cond and np.random.random()<.35: ls._vn=max(0,ls._vn-1)
    ls.vr=["low","normal","high"][ls._vn]
    for a in ("ts","cs","bp","pd","lp","rp","buy_p","sell_p","abs_str"):
        setattr(ls,a,float(np.clip(getattr(ls,a),0.,1.)))
    ls.delta_imb=float(np.clip(ls.delta_imb,-1.,1.)); return ls

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
        # delta_imbalance gates directional continuation
        if ls.delta_imb>.4:
            if "BULL" in c: b*=1.+ls.delta_imb*.8
            if "BEAR" in c and "PULL" not in c: b*=max(.2,1.-ls.delta_imb*.5)
        elif ls.delta_imb<-.4:
            if "BEAR" in c: b*=1.+abs(ls.delta_imb)*.8
            if "BULL" in c and "PULL" not in c: b*=max(.2,1.-abs(ls.delta_imb)*.5)
        # high absorption → sweep/reversal more likely
        if ls.abs_str>.6:
            if any(k in c for k in ("SWEEP","REVERSAL","FALSE_BREAKOUT")): b*=1.+ls.abs_str*1.5
        w[c]=max(1e-6,p*b)
    t=sum(w.values()); ks=list(w.keys())
    return random.choices(ks,weights=[w[k]/t for k in ks],k=1)[0]

def _mod(cond,ls,bs,vm):
    vs={"low":.60,"normal":1.,"high":1.40}[ls.vr]; bs*=vs; vm*=vs
    bt=float(np.clip(.45+.55*ls.ts,.25,1.55))*float(np.clip(1.-ls.age/40.,.4,1.2))
    # orderflow drives volume (not just vol_mult): delta_imbalance skews volume up on directional bars
    of_vm=1.+abs(ls.delta_imb)*.5; vm*=of_vm
    wx=float(np.clip(1.+ls.age/30.,1.,1.6))*float(np.clip(1.+ls.lp*.8,1.,2.2))*float(np.clip(1.+ls.rp*.6,1.,1.8))
    # absorption: large volume + small bodies — increase wicks, reduce body
    if ls.abs_str>.4: bt*=max(.3,1.-ls.abs_str*.5); wx*=1.+ls.abs_str*.6
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
def _rz(s,w): return (s-s.rolling(w).mean())/(s.rolling(w).std()+EPS)

def compute_features(df):
    df=df.copy().reset_index(drop=True)
    df["rng"]=df["high"]-df["low"]; df["body"]=df["close"]-df["open"]
    df["abs_body"]=df["body"].abs(); df["dir"]=np.sign(df["body"]).fillna(0).astype(int)
    df["uw"]=df["high"]-df[["open","close"]].max(axis=1); df["lw"]=df[["open","close"]].min(axis=1)-df["low"]
    df["br"]=df["abs_body"]/(df["rng"]+EPS); df["ur"]=df["uw"]/(df["rng"]+EPS); df["lr"]=df["lw"]/(df["rng"]+EPS)
    df["cg"]=(df["close"]-df["low"])/(df["rng"]+EPS)
    df["atr5"]=df["rng"].rolling(5,min_periods=1).mean(); df["atr10"]=df["rng"].rolling(10,min_periods=1).mean()
    df["body_z"]=_rz(df["abs_body"],40).fillna(0); df["rng_z"]=_rz(df["rng"],40).fillna(0)
    df["tick_d"]=df["cg"]*2-1; delta_vol=df["tick_d"]*df["volume"]
    df["of5"]=(df["dir"]*df["volume"]).rolling(5,min_periods=1).sum()
    df["of10"]=(df["dir"]*df["volume"]).rolling(10,min_periods=1).sum()
    df["cum_d5z"]=_rz(delta_vol.rolling(5,min_periods=1).sum(),20).fillna(0)
    df["cum_d10z"]=_rz(delta_vol.rolling(10,min_periods=1).sum(),30).fillna(0)
    price_dir=np.sign(df["close"].diff(3).fillna(0)); delta_dir=np.sign(delta_vol.rolling(3,min_periods=1).sum())
    df["delta_div"]=(price_dir*delta_dir).fillna(0)
    vol_avg10=df["volume"].rolling(10,min_periods=1).mean()
    df["abs_fp"]=((df["br"]<.25).astype(int)+(df["rng"]/(df["atr10"]+EPS)>.80).astype(int)+(df["volume"]/(vol_avg10+EPS)>1.20).astype(int)).astype(float)
    df["opp_wick"]=np.where(df["dir"]>0,df["lw"],df["uw"])
    df["abs_raw"]=(df["abs_body"]<df["abs_body"].rolling(5,min_periods=1).mean()*.6).astype(int)*df["opp_wick"]
    df["abs_raw"]=df["abs_raw"].rolling(5,min_periods=1).sum()
    vol_avg5=df["volume"].rolling(5,min_periods=1).mean()
    df["bar_e"]=(df["rng"]/(df["atr5"]+EPS)*df["volume"]/(vol_avg5+EPS)).clip(0,6)
    dom_wick=df[["uw","lw"]].max(axis=1); rej_dir=np.where(df["lw"]>df["uw"],1,-1)
    df["rej_qual"]=(dom_wick/(df["rng"]+EPS)*df["volume"]/(vol_avg10+EPS)*rej_dir).clip(-4,4)
    df["gap_p"]=((df["open"]-df["close"].shift(1).fillna(df["open"]))/(df["atr5"]+EPS)).clip(-3,3)
    df["pace"]=(df["volume"]/(df["rng"]+EPS)).clip(0,1e6); df["pace_z"]=_rz(df["pace"],20).fillna(0)
    df["liq_vac"]=(df["rng"]/(df["atr10"]+EPS))/(df["volume"]/(vol_avg10+EPS)+EPS); df["liq_vac"]=df["liq_vac"].clip(0,5)
    df["exhaust"]=((df["uw"]+df["lw"])/(df["abs_body"]+EPS)).clip(0,6)
    df["tail_asym"]=(df["uw"]-df["lw"])/(df["rng"]+EPS)
    df["mid_disp"]=((df["open"]+df["close"])/2-(df["high"]+df["low"])/2)/(df["rng"]+EPS)
    df["body_e"]=(df["abs_body"]/(df["atr5"]+EPS)*df["volume"]/(vol_avg5+EPS)).clip(0,6)
    bull_m=df["dir"]>=0; df["deff"]=np.where(bull_m,df["br"]*df["cg"],df["br"]*(1-df["cg"]))
    cons,cur,prev=[],0,0
    for d in df["dir"].tolist():
        if d==prev and d!=0: cur+=1
        elif d!=0: cur=1
        else: cur=0
        cons.append(cur); prev=d
    df["consec"]=cons
    last=df.iloc[-1]
    f={k:float(last[k]) for k in ["abs_body","uw","lw","rng","br","ur","lr","body_z","rng_z",
       "tick_d","cum_d5z","cum_d10z","delta_div","of5","of10","abs_fp","abs_raw","bar_e","body_e",
       "pace_z","rej_qual","gap_p","liq_vac","exhaust","tail_asym","mid_disp","cg","deff","consec"]}
    f["pullback_z"]=float(_rz(df["rng"]/df["atr5"],20).fillna(0).iloc[-1])
    f["vol_z10"]=float(_rz(df["volume"],10).fillna(0).iloc[-1]); f["vol_z30"]=float(_rz(df["volume"],30).fillna(0).iloc[-1])
    f["_atr"]=float(last["atr5"]) if float(last["atr5"])>EPS else .0001
    f["_ctx_close"]=float(last["close"]); f["_tick_d"]=float(last["tick_d"])
    f["_cum_d5z"]=float(last["cum_d5z"]); f["_bar_e"]=float(last["bar_e"])
    f["_rej_qual"]=float(last["rej_qual"]); f["_abs_fp"]=float(last["abs_fp"])
    return f

# ── LABELS (unchanged) ───────────────────────────────────────────────
def compute_labels(f,future):
    atr=max(f["_atr"],EPS); ref=f["_ctx_close"]
    c1=future[0]; c2=future[1] if len(future)>=2 else c1
    c3=future[2] if len(future)>=3 else c2; c4=future[3] if len(future)>=4 else c3
    h1_close=c4["close"]; h1_high=max(x["high"] for x in future); h1_low=min(x["low"] for x in future)
    h1_rng=(h1_high-h1_low)/atr; net_atr=(h1_close-ref)/atr
    bodies=[abs(x["close"]-x["open"])/atr for x in future]; avg_body=float(np.mean(bodies))
    c1_bull=c1["close"]>c1["open"]; c1_br=bodies[0]
    c1_uw=(c1["high"]-max(c1["open"],c1["close"]))/(c1["high"]-c1["low"]+EPS)
    c1_lw=(min(c1["open"],c1["close"])-c1["low"])/(c1["high"]-c1["low"]+EPS)
    c1_cg=(c1["close"]-c1["low"])/(c1["high"]-c1["low"]+EPS)
    flow_dir=2 if net_atr>.35 else (0 if net_atr<-.35 else 1)
    if c1_br<.15 and (c1_uw+c1_lw)>.5: nxt_intent=0
    elif (c1_lw>.40 and c1_uw<.20) or (c1_uw>.40 and c1_lw<.20): nxt_intent=1
    elif c1_br>.8 and c1_cg>.65: nxt_intent=2
    else: nxt_intent=3 if bodies[1]<c1_br*.60 else 2
    nxt_mag=2 if avg_body>1.5 else (0 if avg_body<.40 else 1)
    rej_score=c1_uw if c1_bull else c1_lw; nxt_rej=2 if rej_score>.45 else (1 if rej_score>.20 else 0)
    tick_d=f["_tick_d"]; cum_d=f["_cum_d5z"]
    bull_flow=tick_d>.15 and cum_d>.2; bear_flow=tick_d<-.15 and cum_d<-.2
    if bull_flow:   follow=2 if net_atr>.60 else (1 if net_atr>.10 else 0)
    elif bear_flow: follow=2 if net_atr<-.60 else (1 if net_atr<-.10 else 0)
    else:           follow=1
    last_rng=float(f["rng"])/atr; rng_type=0 if h1_rng<last_rng*.7 else (2 if h1_rng>last_rng*1.5 else 1)
    return int(flow_dir),int(nxt_intent),int(nxt_mag),int(nxt_rej),int(follow),int(rng_type)


def context_features_micro(df: pd.DataFrame, atr: float) -> dict:
    """Microstructure-specialist context features from last 16 M15."""
    w  = df.iloc[-16:].reset_index(drop=True)
    rng = (w["high"]-w["low"]).astype(float)
    body= (w["close"]-w["open"]).astype(float)
    br  = body.abs()/(rng+EPS)
    cg  = (w["close"]-w["low"])/(rng+EPS)
    vol = w["volume"].astype(float)
    tick_d = cg*2-1  # signed tick delta proxy
    delta_vol = tick_d * vol
    # absorption_density_16: fraction of bars showing absorption signature
    abs_mask = (br < 0.25) & (rng > rng.mean()*0.75) & (vol > vol.mean()*1.10)
    absorption_density_16 = float(abs_mask.mean())
    # delta_imbalance_16: cumulative signed delta normalised by total volume
    total_vol = float(vol.sum()); cum_delta = float(delta_vol.sum())
    delta_imbalance_16 = float(np.clip(cum_delta / (total_vol + EPS), -1, 1))
    # pressure_shift_16: sign change in delta between first 8 and last 8 bars
    early_d = float(delta_vol.iloc[:8].sum()); late_d = float(delta_vol.iloc[8:].sum())
    pressure_shift_16 = float(np.clip((late_d - early_d) / (atr * 500 + EPS), -2, 2))
    # wick_dominance_16: avg ratio of total wick to body (exhaustion proxy)
    uw = w["high"] - w[["open","close"]].max(axis=1)
    lw = w[["open","close"]].min(axis=1) - w["low"]
    wick_dominance_16 = float(((uw+lw)/(body.abs()+EPS)).clip(0,5).mean())
    return {"absorption_density_16": absorption_density_16,
            "delta_imbalance_16": delta_imbalance_16,
            "pressure_shift_16": pressure_shift_16,
            "wick_dominance_16": wick_dominance_16}

# ── SEQUENCE GENERATOR & MAIN ─────────────────────────────────────────
def generate_sequence(base_price=BASE_PRICE):
    hist,active,fc,fut,ls=build_history(base_price)
    df=pd.DataFrame(hist); f=compute_features(df)
    atr=max(f["_atr"],EPS); ctx=context_features_micro(df,atr)
    labels=compute_labels(f,fut)
    row={k:v for k,v in f.items() if not k.startswith("_")}
    row.update(ctx); row["condition"]=active; row["future_condition"]=fc
    if EXPORT_METADATA:
        row["meta_trend_strength"]=round(ls.ts,3); row["meta_trend_age"]=ls.age
        row["meta_vol_regime"]=ls.vr; row["meta_compression_score"]=round(ls.cs,3)
        row["meta_compression_duration"]=ls.cd; row["meta_pullback_depth"]=round(ls.pd,3)
        row["meta_liquidity_pressure"]=round(ls.lp,3); row["meta_breakout_pressure"]=round(ls.bp,3)
        row["meta_reversal_pressure"]=round(ls.rp,3)
        row["meta_buy_pressure"]=round(ls.buy_p,3); row["meta_sell_pressure"]=round(ls.sell_p,3)
        row["meta_absorption_strength"]=round(ls.abs_str,3); row["meta_delta_imbalance"]=round(ls.delta_imb,3)
    row["label_next_flow_dir"],row["label_next_bar_intent"],row["label_next_magnitude"],\
    row["label_next_rejection"],row["label_momentum_follow"],row["label_next_range_type"]=labels
    return row

if __name__=="__main__":
    print(f"Generating {N_SAMPLES} microstructure samples | v5 latent-state engine | orderflow specialist")
    rows=[generate_sequence() for _ in trange(N_SAMPLES,desc="microstructure")]
    df_out=pd.DataFrame(rows)
    lc=[c for c in df_out.columns if c.startswith("label_")]; mc=["condition","future_condition"]
    fc2=[c for c in df_out.columns if c not in lc+mc]
    df_out=df_out[mc+sorted(fc2)+lc]; df_out.to_csv(OUTPUT_PATH,index=False)
    maps={"label_next_flow_dir":{0:"bearish",1:"contested",2:"bullish"},
          "label_next_bar_intent":{0:"absorption",1:"rejection",2:"impulse",3:"fade"},
          "label_next_magnitude":{0:"thin",1:"normal",2:"strong"},
          "label_next_rejection":{0:"none",1:"partial",2:"strong"},
          "label_momentum_follow":{0:"failed",1:"partial",2:"confirmed"},
          "label_next_range_type":{0:"inside_compress",1:"normal",2:"outside_expand"}}
    print(f"\n✅  Saved → {OUTPUT_PATH}  shape={df_out.shape}  features={len(fc2)}")
    for l in lc:
        counts=df_out[l].value_counts().sort_index()
        print(f"    {l}: { {maps[l].get(k,k):f'{100*v/len(df_out):.1f}%' for k,v in counts.items()} }")
