# train_candle_h1.py  v4
# Changes from v3:
#   • patience 20→35, pct_start 0.10→0.05  (fixes premature early-stop)
#   • mom HEAD_WEIGHT 1.2→1.7              (forces model to prioritise momentum head)
#   • n_in auto-scales with n_feat=37 correctly (was already dynamic, confirmed)
import os, random
import numpy as np, pandas as pd
from collections import Counter
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import RobustScaler
import joblib

DATA_PATH   = "synthetic_candle_h1.csv"
SCALER_PATH = "candle_h1_scaler.pkl"
BEST_PATH   = "candle_h1_best.pth"
FINAL_PATH  = "candle_h1_final.pth"

SEED=42; EPOCHS=80; BATCH=512; LR=3e-4; WD=1e-4
VAL_RATIO=0.12; TRUNK=256; DROP=0.30

LABEL_COLS=["label_h1_direction","label_h1_magnitude","label_h1_range_type",
            "label_h1_close_loc","label_h1_body_ratio","label_h1_momentum"]
HEAD_SIZES=[3,3,3,3,3,3]
HEAD_NAMES=["dir","mag","range","loc","body","mom"]
HEAD_WEIGHTS=[1.7, 1.0, 0.8, 1.0, 1.5, 1.7]   # mom: 1.2→1.7
CLASS_WEIGHTS=[
    torch.tensor([1.4,0.5,1.4],dtype=torch.float32),
    torch.tensor([1.3,1.0,0.9],dtype=torch.float32),
    torch.tensor([1.1,0.9,1.1],dtype=torch.float32),
    torch.tensor([1.2,0.5,1.2],dtype=torch.float32),
    torch.tensor([1.1,1.0,0.9],dtype=torch.float32),
    torch.tensor([1.3,0.5,1.3],dtype=torch.float32),
]
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

class ResBlock(nn.Module):
    def __init__(self,d,drop):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,d),nn.BatchNorm1d(d),nn.GELU(),nn.Dropout(drop),
                               nn.Linear(d,d),nn.BatchNorm1d(d))
        self.act=nn.GELU()
    def forward(self,x): return self.act(self.net(x)+x)

class H1Net(nn.Module):
    # Feature layout: W1 cols 0-8, W2 cols 9-17, W3 cols 18-26, norm 27-32, new 33-36
    # Input = full_x(37) + W1_extra(0:9) x2 + W2_extra(9:18) x1 = 37+18+9 = 64
    N_W1=9; N_W2=9
    def __init__(self,n_feat,trunk=TRUNK,drop=DROP):
        super().__init__()
        n_in = n_feat + self.N_W1*2 + self.N_W2
        half = trunk//2
        self.stem=nn.Sequential(nn.Linear(n_in,trunk),nn.BatchNorm1d(trunk),nn.GELU(),nn.Dropout(drop))
        self.res=nn.Sequential(ResBlock(trunk,drop),ResBlock(trunk,drop))
        self.mid=nn.Sequential(nn.Linear(trunk,half),nn.BatchNorm1d(half),nn.GELU(),nn.Dropout(drop*0.7))
        self.heads=nn.ModuleList([nn.Linear(half,s) for s in HEAD_SIZES])
    def forward(self,x):
        x=torch.cat([x,
                     x[:, :self.N_W1],
                     x[:, :self.N_W1],
                     x[:, self.N_W1:self.N_W1+self.N_W2]],
                    dim=1)
        h=self.mid(self.res(self.stem(x)))
        return tuple(hd(h) for hd in self.heads)

class H1DS(Dataset):
    def __init__(self,X,Ys):
        self.X=torch.tensor(X,dtype=torch.float32)
        self.Ys=[torch.tensor(y,dtype=torch.long) for y in Ys]
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return (self.X[i],)+tuple(y[i] for y in self.Ys)

def run_epoch(model,loader,loss_fns,opt=None):
    tr=opt is not None; model.train() if tr else model.eval()
    tot=0.0; n=0; correct=[0]*6; totals=[0]*6
    with torch.set_grad_enabled(tr):
        for batch in loader:
            xb=batch[0].to(DEVICE); ys=[b.to(DEVICE) for b in batch[1:]]
            if tr: opt.zero_grad()
            outs=model(xb)
            losses=[loss_fns[i](outs[i],ys[i]) for i in range(6)]
            loss=sum(HEAD_WEIGHTS[i]*losses[i] for i in range(6))
            if tr:
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            bs=xb.size(0); tot+=loss.item()*bs; n+=bs
            for i in range(6):
                correct[i]+=(outs[i].argmax(1)==ys[i]).sum().item(); totals[i]+=ys[i].size(0)
    return tot/n,[correct[i]/max(totals[i],1) for i in range(6)]

def main():
    print(f"[Candle H1 v4]  device={DEVICE}")
    df=pd.read_csv(DATA_PATH); print(f"Loaded {DATA_PATH} {df.shape}")
    Ys=[df[c].astype(int).values for c in LABEL_COLS]
    feat_cols=[c for c in df.columns if c not in LABEL_COLS]
    X=df[feat_cols].fillna(0.0).values.astype(np.float32)
    print(f"Features:{len(feat_cols)}  Samples:{len(df)}")

    n=len(df); nv=int(n*VAL_RATIO)
    perm=np.random.RandomState(SEED).permutation(n)
    tri,vai=perm[:-nv],perm[-nv:]
    Xtr,Xva=X[tri],X[vai]; Ytr=[y[tri] for y in Ys]; Yva=[y[vai] for y in Ys]

    sc=RobustScaler(); Xtr_s=sc.fit_transform(Xtr).astype(np.float32); Xva_s=sc.transform(Xva).astype(np.float32)
    joblib.dump({"scaler":sc,"feature_names":feat_cols},SCALER_PATH)
    print(f"Scaler → {SCALER_PATH}")

    dir_w={0:1.5,1:0.3,2:1.5}; body_w={0:1.3,1:0.4,2:1.3}
    sw=np.array([dir_w[Ytr[0][i]]*body_w[Ytr[4][i]] for i in range(len(Ytr[0]))],dtype=np.float32)
    sampler=WeightedRandomSampler(sw,num_samples=len(sw),replacement=True)
    tr_l=DataLoader(H1DS(Xtr_s,Ytr),BATCH,sampler=sampler,num_workers=0)
    va_l=DataLoader(H1DS(Xva_s,Yva),BATCH,shuffle=False,num_workers=0)

    model=H1Net(len(feat_cols)).to(DEVICE)
    print(f"Params:{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    loss_fns=[nn.CrossEntropyLoss(weight=CLASS_WEIGHTS[i].to(DEVICE),label_smoothing=0.08) for i in range(6)]

    print("\nLabel distributions (train):")
    for i,lc in enumerate(LABEL_COLS): print(f"  [{HEAD_NAMES[i]:6}] {dict(Counter(Ytr[i].tolist()))}")

    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=LR*5,steps_per_epoch=len(tr_l),
                                               epochs=EPOCHS,pct_start=0.05)  # 0.10→0.05

    best=float("inf"); bep=0; patience=35; no_imp=0   # patience 20→35
    print(f"\nTraining {EPOCHS} epochs | batch={BATCH} | lr={LR}\n{'─'*80}")
    for ep in range(1,EPOCHS+1):
        tl,_=run_epoch(model,tr_l,loss_fns,opt)
        vl,vacc=run_epoch(model,va_l,loss_fns)
        sched.step()
        sv=""
        if vl<best: best=vl; bep=ep; no_imp=0; torch.save(model.state_dict(),BEST_PATH); sv="  ← best"
        else: no_imp+=1
        if ep%10==0 or ep==1:
            ac="  ".join(f"{HEAD_NAMES[i]}={vacc[i]:.3f}" for i in range(6))
            print(f"Ep {ep:03d} | TrL={tl:.4f} VaL={vl:.4f} | {ac}{sv}")
        if no_imp>=patience: print(f"Early stop ep {ep}"); break

    torch.save(model.state_dict(),FINAL_PATH)
    print(f"\n✅  Best  → {BEST_PATH}  (ep {bep} vaL={best:.4f})")
    print(f"✅  Final → {FINAL_PATH}")
    print(f"✅  Scaler→ {SCALER_PATH}")

if __name__=="__main__": main()
