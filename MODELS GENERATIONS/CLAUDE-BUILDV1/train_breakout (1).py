# train_breakout.py
# Trains the Breakout / Squeeze multi-head model
# 27 features → 6 heads | ResidualBlock × 2 | Stem(256) + Mid(128)
import os, random
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
import joblib

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw): return x

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────
DATA_PATH        = "synthetic_breakout_dataset.csv"
SCALER_PATH      = "breakout_scaler.pkl"
BEST_MODEL_PATH  = "breakout_best.pth"
FINAL_MODEL_PATH = "breakout_final.pth"

BATCH_SIZE  = 256
EPOCHS      = 50
LR          = 1e-3
WD          = 1e-4
VAL_RATIO   = 0.12
TRUNK_DIM   = 256
DROP        = 0.25
SEED        = 42

# sq_state(4) bk_dir(3) bk_qual(4) vol_phase(4) band_str(3) entry_timing(3)
HEAD_SIZES   = [4, 3, 4, 4, 3, 3]
HEAD_WEIGHTS = [0.9, 0.9, 1.0, 0.9, 0.7, 1.2]   # entry_timing highest
LABEL_COLS   = ["label_sq_state","label_bk_dir","label_bk_qual",
                "label_vol_phase","label_band_str","label_entry_timing"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────
# Architecture
# ─────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, dim, drop=0.25):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(drop),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.seq(x) + x)


class BreakoutNet(nn.Module):
    def __init__(self, input_dim, trunk=TRUNK_DIM, drop=DROP):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, trunk),
            nn.BatchNorm1d(trunk), nn.GELU(), nn.Dropout(drop)
        )
        self.res = nn.Sequential(
            ResidualBlock(trunk, drop),
            ResidualBlock(trunk, drop),
        )
        self.mid = nn.Sequential(
            nn.Linear(trunk, trunk // 2),
            nn.BatchNorm1d(trunk // 2), nn.GELU(), nn.Dropout(drop * 0.7)
        )
        h = trunk // 2
        self.h0 = nn.Linear(h, HEAD_SIZES[0])
        self.h1 = nn.Linear(h, HEAD_SIZES[1])
        self.h2 = nn.Linear(h, HEAD_SIZES[2])
        self.h3 = nn.Linear(h, HEAD_SIZES[3])
        self.h4 = nn.Linear(h, HEAD_SIZES[4])
        self.h5 = nn.Linear(h, HEAD_SIZES[5])

    def forward(self, x):
        h = self.mid(self.res(self.stem(x)))
        return self.h0(h), self.h1(h), self.h2(h), self.h3(h), self.h4(h), self.h5(h)


# ─────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────

class BreakoutDS(Dataset):
    def __init__(self, X, Y_list):
        self.X  = torch.tensor(X.astype(np.float32))
        self.Ys = [torch.tensor(y.astype(np.int64)) for y in Y_list]

    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return (self.X[i],) + tuple(y[i] for y in self.Ys)


def class_weights(arr, n):
    cnt = Counter(arr.astype(int).tolist())
    tot = sum(cnt.values())
    return torch.tensor(
        [tot / (cnt.get(k,0)*n) if cnt.get(k,0)>0 else 1.0 for k in range(n)],
        dtype=torch.float32)


def run_epoch(model, loader, loss_fns, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    total_loss, total_n = 0.0, 0
    correct = [0]*6; totals = [0]*6

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            xb = batch[0].to(DEVICE)
            ys = [b.to(DEVICE) for b in batch[1:]]
            outs = model(xb)
            losses = [loss_fns[i](outs[i], ys[i]) for i in range(6)]
            loss   = sum(HEAD_WEIGHTS[i] * losses[i] for i in range(6))

            if train:
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            total_loss += loss.item() * xb.size(0)
            total_n    += xb.size(0)
            for i in range(6):
                pred = outs[i].argmax(1)
                correct[i] += (pred == ys[i]).sum().item()
                totals[i]  += ys[i].size(0)

    return total_loss / total_n, [correct[i]/totals[i] for i in range(6)]


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    df        = pd.read_csv(DATA_PATH)
    Ys        = [df[lc].astype(int).values for lc in LABEL_COLS]
    feat_cols = [c for c in df.columns if c not in LABEL_COLS]
    X         = df[feat_cols].fillna(0.0)
    print(f"  Samples={len(df)}  Features={len(feat_cols)}")

    n     = len(df)
    n_val = int(n * VAL_RATIO)
    idx   = np.random.RandomState(SEED).permutation(n)
    tr_idx, va_idx = idx[:-n_val], idx[-n_val:]

    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    Ytr  = [y[tr_idx] for y in Ys]
    Yva  = [y[va_idx] for y in Ys]

    scaler = RobustScaler()
    Xtr_s  = scaler.fit_transform(Xtr.values)
    Xva_s  = scaler.transform(Xva.values)
    joblib.dump({"scaler": scaler, "feature_names": list(feat_cols)}, SCALER_PATH)
    print(f"  Scaler → {SCALER_PATH}")

    tr_loader = DataLoader(BreakoutDS(Xtr_s, Ytr), BATCH_SIZE, shuffle=True,  num_workers=0)
    va_loader = DataLoader(BreakoutDS(Xva_s, Yva), BATCH_SIZE, shuffle=False, num_workers=0)

    model    = BreakoutNet(Xtr_s.shape[1]).to(DEVICE)
    loss_fns = []
    for i, (lc, nc) in enumerate(zip(LABEL_COLS, HEAD_SIZES)):
        w = class_weights(Ytr[i], nc).to(DEVICE)
        loss_fns.append(nn.CrossEntropyLoss(weight=w))
        print(f"  Head {i} ({lc}): classes={nc}  w={w.cpu().numpy().round(3)}")

    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR*0.05)

    best_val = float("inf")
    print(f"\nTraining {EPOCHS} epochs ...\n")
    for ep in range(1, EPOCHS + 1):
        tr_loss, _ = run_epoch(model, tr_loader, loss_fns, opt)
        va_loss, va_acc = run_epoch(model, va_loader, loss_fns)
        sched.step()
        saved = ""
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH); saved = "  ← best"
        if ep % 5 == 0 or ep == 1:
            print(f"Ep {ep:03d} | trL={tr_loss:.4f} | vaL={va_loss:.4f} | "
                  f"vaAcc=[{' '.join(f'{a:.2f}' for a in va_acc)}]{saved}")

    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"\n✅  Best  → {BEST_MODEL_PATH}  (vaL={best_val:.4f})")
    print(f"✅  Final → {FINAL_MODEL_PATH}")

if __name__ == "__main__":
    main()
