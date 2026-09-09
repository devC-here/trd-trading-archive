# train_vwap.py  v3 — rebuilt from dataset
# Dataset: synthetic_vwap_dataset.csv (20000x38) | Features:32 | Labels:6
# Trade bias: vwap_dir + follow_through weighted toward actionable classes
import os, random
import numpy as np
import pandas as pd
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import RobustScaler
import joblib

DATA_PATH        = "synthetic_vwap_dataset.csv"
SCALER_PATH      = "vwap_scaler.pkl"
BEST_MODEL_PATH  = "vwap_best.pth"
FINAL_MODEL_PATH = "vwap_final.pth"

BATCH_SIZE=256; EPOCHS=60; LR=1e-3; WD=1e-4
VAL_RATIO=0.12; TRUNK_DIM=256; DROP=0.25; SEED=42

# Labels confirmed from dataset:
# label_next_vwap_dir       : 0=below_away 1=oscillate  2=above_away  -> 3
# label_next_vol_char       : 0=declining  1=stable     2=surge       -> 3
# label_vwap_interaction    : 0=rejection  1=test_return 2=reclaim 3=walk -> 4
# label_next_inst_signal    : 0=distribution 1=neutral  2=accumulation -> 3
# label_next_close_location : 0=lower      1=mid        2=upper       -> 3
# label_vwap_follow_through : 0=no         1=partial    2=confirmed   -> 3
LABEL_COLS = [
    "label_next_vwap_dir", "label_next_vol_char", "label_vwap_interaction",
    "label_next_inst_signal", "label_next_close_location", "label_vwap_follow_through",
]
HEAD_SIZES   = [3, 3, 4, 3, 3, 3]
HEAD_WEIGHTS = [1.5, 1.0, 1.1, 1.1, 0.9, 1.5]

CLASS_WEIGHTS = [
    torch.tensor([1.6, 0.5, 1.6],       dtype=torch.float32),
    torch.tensor([0.8, 0.9, 2.2],       dtype=torch.float32),
    torch.tensor([1.3, 0.8, 2.0, 2.0],  dtype=torch.float32),
    torch.tensor([1.7, 0.5, 1.7],       dtype=torch.float32),
    torch.tensor([1.3, 0.6, 1.3],       dtype=torch.float32),
    torch.tensor([0.7, 1.0, 1.9],       dtype=torch.float32),
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

class ResidualBlock(nn.Module):
    def __init__(self, dim, drop=0.25):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(dim,dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(drop),
            nn.Linear(dim,dim), nn.BatchNorm1d(dim))
        self.act = nn.GELU()
    def forward(self, x): return self.act(self.seq(x) + x)

class VwapNet(nn.Module):
    def __init__(self, input_dim, trunk=TRUNK_DIM, drop=DROP):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(input_dim,trunk), nn.BatchNorm1d(trunk), nn.GELU(), nn.Dropout(drop))
        self.res  = nn.Sequential(ResidualBlock(trunk,drop), ResidualBlock(trunk,drop))
        self.mid  = nn.Sequential(nn.Linear(trunk,trunk//2), nn.BatchNorm1d(trunk//2), nn.GELU(), nn.Dropout(drop*0.7))
        self.heads = nn.ModuleList([nn.Linear(trunk//2, s) for s in HEAD_SIZES])
    def forward(self, x):
        h = self.mid(self.res(self.stem(x)))
        return tuple(head(h) for head in self.heads)

class VwapDS(Dataset):
    def __init__(self, X, Y):
        self.X  = torch.tensor(X.astype(np.float32))
        self.Ys = [torch.tensor(y.astype(np.int64)) for y in Y]
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return (self.X[i],) + tuple(y[i] for y in self.Ys)

def run_epoch(model, loader, loss_fns, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    tot_loss=0.0; n=0; correct=[0]*6; totals=[0]*6
    with torch.set_grad_enabled(train):
        for batch in loader:
            xb = batch[0].to(DEVICE)
            ys = [b.to(DEVICE) for b in batch[1:]]
            outs   = model(xb)
            losses = [loss_fns[i](outs[i], ys[i]) for i in range(6)]
            loss   = sum(HEAD_WEIGHTS[i]*losses[i] for i in range(6))
            if train:
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot_loss += loss.item()*xb.size(0); n += xb.size(0)
            for i in range(6):
                correct[i] += (outs[i].argmax(1)==ys[i]).sum().item()
                totals[i]  += ys[i].size(0)
    return tot_loss/n, [correct[i]/totals[i] for i in range(6)]

def main():
    print(f"Device: {DEVICE}")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {DATA_PATH}  shape={df.shape}")
    for lc in LABEL_COLS:
        if lc not in df.columns:
            raise RuntimeError(f"Missing '{lc}'. Cols: {list(df.columns)}")

    Ys        = [df[lc].astype(int).values for lc in LABEL_COLS]
    feat_cols = [c for c in df.columns if c not in LABEL_COLS]
    X         = df[feat_cols].fillna(0.0)
    print(f"Features:{len(feat_cols)}  Samples:{len(df)}")

    n=len(df); n_val=int(n*VAL_RATIO)
    idx=np.random.RandomState(SEED).permutation(n)
    tr_idx,va_idx = idx[:-n_val], idx[-n_val:]

    Xtr,Xva = X.iloc[tr_idx].values, X.iloc[va_idx].values
    Ytr=[y[tr_idx] for y in Ys]; Yva=[y[va_idx] for y in Ys]

    scaler=RobustScaler()
    Xtr_s=scaler.fit_transform(Xtr); Xva_s=scaler.transform(Xva)
    joblib.dump({"scaler":scaler,"feature_names":list(feat_cols)}, SCALER_PATH)
    print(f"Scaler -> {SCALER_PATH}")

    vdir_w={0:1.6,1:0.5,2:1.6}; follow_w={0:0.7,1:1.0,2:1.8}
    sw=np.array([vdir_w[Ytr[0][i]]*follow_w[Ytr[5][i]] for i in range(len(Ytr[0]))],dtype=np.float32)
    sampler=WeightedRandomSampler(sw,num_samples=len(sw),replacement=True)

    tr_loader=DataLoader(VwapDS(Xtr_s,Ytr),BATCH_SIZE,sampler=sampler,num_workers=0)
    va_loader=DataLoader(VwapDS(Xva_s,Yva),BATCH_SIZE,shuffle=False,num_workers=0)

    model    = VwapNet(Xtr_s.shape[1]).to(DEVICE)
    loss_fns = [nn.CrossEntropyLoss(weight=CLASS_WEIGHTS[i].to(DEVICE)) for i in range(6)]

    hs=["vdir","vol","vint","inst","loc","follow"]
    print("\nLabel distributions (train):")
    for i,lc in enumerate(LABEL_COLS):
        print(f"  [{hs[i]:6}] {lc}: {dict(Counter(Ytr[i].tolist()))}")

    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, steps_per_epoch=len(tr_loader), epochs=EPOCHS, pct_start=0.1)

    best_val=float("inf")
    print(f"\nTraining {EPOCHS} epochs...\n{'─'*80}")
    for ep in range(1, EPOCHS+1):
        tr_loss,_     = run_epoch(model,tr_loader,loss_fns,opt)
        va_loss,va_acc= run_epoch(model,va_loader,loss_fns)
        sched.step()
        saved=""
        if va_loss < best_val:
            best_val=va_loss; torch.save(model.state_dict(),BEST_MODEL_PATH); saved="  <- best"
        if ep%5==0 or ep==1:
            acc_str="  ".join(f"{hs[i]}={va_acc[i]:.3f}" for i in range(6))
            print(f"Ep {ep:03d} | trL={tr_loss:.4f} vaL={va_loss:.4f} | {acc_str}{saved}")

    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"\nBest  -> {BEST_MODEL_PATH}  (vaL={best_val:.4f})")
    print(f"Final -> {FINAL_MODEL_PATH}")
    print(f"Scaler-> {SCALER_PATH}")

if __name__ == "__main__":
    main()
