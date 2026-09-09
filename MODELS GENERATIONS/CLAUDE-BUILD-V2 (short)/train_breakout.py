# train_breakout.py  ·  v3  — rebuilt from dataset
# Dataset : synthetic_breakout_dataset.csv  (20000 × 33)
# Features: 27  |  Labels: 6 next-candle heads
# Trade bias: directional classes (bear/bull) upweighted vs flat/neutral
# ─────────────────────────────────────────────────────────────────────
import os, random
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import RobustScaler
import joblib

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────
DATA_PATH        = "synthetic_breakout_dataset.csv"
SCALER_PATH      = "breakout_scaler.pkl"
BEST_MODEL_PATH  = "breakout_best.pth"
FINAL_MODEL_PATH = "breakout_final.pth"

BATCH_SIZE = 256
EPOCHS     = 60
LR         = 1e-3
WD         = 1e-4
VAL_RATIO  = 0.12
TRUNK_DIM  = 256
DROP       = 0.25
SEED       = 42

# ── Labels confirmed from dataset columns ───────────────────────────
# label_next_direction    : 0=bearish  1=flat      2=bullish    → 3 classes
# label_next_magnitude    : 0=small    1=medium    2=large      → 3 classes
# label_next_pattern      : 0=cont     1=reversal  2=indecision → 3 classes
# label_next_volatility   : 0=contract 1=normal    2=expand     → 3 classes
# label_next_close_loc    : 0=lower    1=mid       2=upper      → 3 classes
# label_bk_follow_through : 0=no       1=partial   2=yes        → 3 classes
LABEL_COLS = [
    "label_next_direction",
    "label_next_magnitude",
    "label_next_pattern",
    "label_next_volatility",
    "label_next_close_loc",
    "label_bk_follow_through",
]
HEAD_SIZES = [3, 3, 3, 3, 3, 3]

# Head loss weights — direction and follow-through matter most for trading
HEAD_WEIGHTS = [1.4, 1.1, 1.1, 0.9, 0.9, 1.6]

# Per-class cost weights — push model to predict directional outcomes more
# direction : downweight flat(1), upweight bear(0)/bull(2)
# magnitude : upweight large(2)
# pattern   : upweight cont(0)/reversal(1), downweight indecision(2)
# volatility: upweight expansion(2)
# close_loc : downweight mid(1), upweight extremes
# follow    : upweight confirmed(2), downweight no(0)
CLASS_WEIGHTS = [
    torch.tensor([1.6, 0.6, 1.6], dtype=torch.float32),
    torch.tensor([0.9, 1.0, 2.2], dtype=torch.float32),
    torch.tensor([1.4, 1.4, 0.7], dtype=torch.float32),
    torch.tensor([1.0, 0.9, 1.8], dtype=torch.float32),
    torch.tensor([1.3, 0.6, 1.3], dtype=torch.float32),
    torch.tensor([0.7, 1.0, 1.9], dtype=torch.float32),
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────
# MODEL
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
        self.heads = nn.ModuleList([nn.Linear(h, s) for s in HEAD_SIZES])

    def forward(self, x):
        h = self.mid(self.res(self.stem(x)))
        return tuple(head(h) for head in self.heads)


# ─────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────

class BreakoutDS(Dataset):
    def __init__(self, X, Y_list):
        self.X  = torch.tensor(X.astype(np.float32))
        self.Ys = [torch.tensor(y.astype(np.int64)) for y in Y_list]

    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return (self.X[i],) + tuple(y[i] for y in self.Ys)


def run_epoch(model, loader, loss_fns, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    total_loss, total_n = 0.0, 0
    correct = [0] * 6; totals = [0] * 6

    with torch.set_grad_enabled(train):
        for batch in loader:
            xb = batch[0].to(DEVICE)
            ys = [b.to(DEVICE) for b in batch[1:]]
            outs   = model(xb)
            losses = [loss_fns[i](outs[i], ys[i]) for i in range(6)]
            loss   = sum(HEAD_WEIGHTS[i] * losses[i] for i in range(6))
            if train:
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total_loss += loss.item() * xb.size(0); total_n += xb.size(0)
            for i in range(6):
                pred = outs[i].argmax(1)
                correct[i] += (pred == ys[i]).sum().item()
                totals[i]  += ys[i].size(0)

    return total_loss / total_n, [correct[i] / totals[i] for i in range(6)]


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Device : {DEVICE}")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded : {DATA_PATH}  shape={df.shape}")

    for lc in LABEL_COLS:
        if lc not in df.columns:
            raise RuntimeError(f"Missing column '{lc}'. Columns in file: {list(df.columns)}")

    Ys        = [df[lc].astype(int).values for lc in LABEL_COLS]
    feat_cols = [c for c in df.columns if c not in LABEL_COLS]
    X         = df[feat_cols].fillna(0.0)
    print(f"Features: {len(feat_cols)}  Samples: {len(df)}")

    n     = len(df)
    n_val = int(n * VAL_RATIO)
    idx   = np.random.RandomState(SEED).permutation(n)
    tr_idx, va_idx = idx[:-n_val], idx[-n_val:]

    Xtr, Xva = X.iloc[tr_idx].values, X.iloc[va_idx].values
    Ytr = [y[tr_idx] for y in Ys]
    Yva = [y[va_idx] for y in Ys]

    scaler = RobustScaler()
    Xtr_s  = scaler.fit_transform(Xtr)
    Xva_s  = scaler.transform(Xva)
    joblib.dump({"scaler": scaler, "feature_names": list(feat_cols)}, SCALER_PATH)
    print(f"Scaler  → {SCALER_PATH}")

    # ── Weighted sampler: oversample directional + confirmed rows ────
    # direction bear(0)/bull(2) = 1.6x,  flat(1) = 0.6x
    # follow_through confirmed(2) = 1.8x, partial(1) = 1.0x, no(0) = 0.7x
    dir_w    = {0: 1.6, 1: 0.6, 2: 1.6}
    follow_w = {0: 0.7, 1: 1.0, 2: 1.8}
    sw = np.array([dir_w[Ytr[0][i]] * follow_w[Ytr[5][i]]
                   for i in range(len(Ytr[0]))], dtype=np.float32)
    sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)

    tr_loader = DataLoader(BreakoutDS(Xtr_s, Ytr), BATCH_SIZE, sampler=sampler, num_workers=0)
    va_loader = DataLoader(BreakoutDS(Xva_s, Yva), BATCH_SIZE, shuffle=False, num_workers=0)

    model    = BreakoutNet(Xtr_s.shape[1]).to(DEVICE)
    loss_fns = [nn.CrossEntropyLoss(weight=CLASS_WEIGHTS[i].to(DEVICE)) for i in range(6)]

    heads_short = ["dir", "mag", "pat", "vol", "loc", "follow"]
    print("\nLabel distributions (train):")
    for i, lc in enumerate(LABEL_COLS):
        print(f"  [{heads_short[i]:6}] {lc}: {dict(Counter(Ytr[i].tolist()))}")

    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, steps_per_epoch=len(tr_loader), epochs=EPOCHS, pct_start=0.1
    )

    best_val = float("inf")
    print(f"\nTraining {EPOCHS} epochs ...\n{'─'*80}")

    for ep in range(1, EPOCHS + 1):
        tr_loss, _      = run_epoch(model, tr_loader, loss_fns, opt)
        va_loss, va_acc = run_epoch(model, va_loader, loss_fns)
        sched.step()
        saved = ""
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH); saved = "  ← best"
        if ep % 5 == 0 or ep == 1:
            acc_str = "  ".join(f"{heads_short[i]}={va_acc[i]:.3f}" for i in range(6))
            print(f"Ep {ep:03d} | trL={tr_loss:.4f} vaL={va_loss:.4f} | {acc_str}{saved}")

    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"\n✅  Best  → {BEST_MODEL_PATH}  (vaL={best_val:.4f})")
    print(f"✅  Final → {FINAL_MODEL_PATH}")
    print(f"✅  Scaler→ {SCALER_PATH}")


if __name__ == "__main__":
    main()
