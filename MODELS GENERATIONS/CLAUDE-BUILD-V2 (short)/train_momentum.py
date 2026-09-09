# train_momentum.py  ·  v3  — rebuilt from dataset
# Dataset : synthetic_momentum_dataset.csv  (25000 × 37)
# Features: 30  |  Labels: 6 next-candle heads  (NO label_ prefix in this dataset)
# Trade bias: direction + next_regime_behaviour weighted toward actionable classes
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
DATA_PATH        = "synthetic_momentum_dataset.csv"
SCALER_PATH      = "momentum_scaler.pkl"
BEST_MODEL_PATH  = "momentum_best.pth"
FINAL_MODEL_PATH = "momentum_final.pth"

BATCH_SIZE = 512
EPOCHS     = 60
LR         = 3e-4
WD         = 1e-4
VAL_RATIO  = 0.10
TRUNK_DIM  = 256
DROP       = 0.25
SEED       = 42

# ── Labels confirmed from dataset columns (NO label_ prefix here!) ──
# next_direction          : 0=bearish  1=flat      2=bullish      → 3 classes
# next_magnitude          : 0=small    1=large                    → 2 classes
# next_regime_behaviour   : 0=chop     1=pullback  2=continuation 3=breakout → 4 classes
# next_volatility         : 0=contract 1=normal    2=expand       → 3 classes
# next_structural_bias    : 0=bearish  1=neutral   2=bullish      → 3 classes
# next_momentum_quality   : 0=fading   1=stable    2=accelerating → 3 classes
LABEL_COLS = [
    "next_direction",
    "next_magnitude",
    "next_regime_behaviour",
    "next_volatility",
    "next_structural_bias",
    "next_momentum_quality",
]
HEAD_SIZES = [3, 2, 4, 3, 3, 3]

# Non-label column to exclude from features
EXCLUDE_COLS = LABEL_COLS + ["scenario"]

# Head weights — direction + regime are primary trade signals
HEAD_WEIGHTS = [1.5, 1.0, 1.3, 0.9, 1.1, 1.2]

# Per-class cost weights
# direction       : upweight bear(0)/bull(2), downweight flat(1)
# magnitude       : upweight large(1) — small moves aren't worth trading
# regime_behaviour: upweight continuation(2)/breakout(3), moderate pullback(1)
# volatility      : upweight expand(2)
# structural_bias : upweight bear(0)/bull(2), downweight neutral(1)
# momentum_quality: upweight accelerating(2), downweight fading(0)
CLASS_WEIGHTS = [
    torch.tensor([1.6, 0.5, 1.6], dtype=torch.float32),
    torch.tensor([0.8, 1.5], dtype=torch.float32),
    torch.tensor([0.7, 1.2, 1.5, 2.2], dtype=torch.float32),
    torch.tensor([1.0, 0.9, 1.8], dtype=torch.float32),
    torch.tensor([1.4, 0.5, 1.4], dtype=torch.float32),
    torch.tensor([0.7, 1.0, 1.7], dtype=torch.float32),
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


class MomentumNet(nn.Module):
    def __init__(self, input_dim, trunk=TRUNK_DIM, drop=DROP):
        super().__init__()
        half = trunk // 2
        self.stem = nn.Sequential(
            nn.Linear(input_dim, trunk),
            nn.BatchNorm1d(trunk), nn.GELU(), nn.Dropout(drop),
        )
        self.res = nn.Sequential(
            ResidualBlock(trunk, drop),
            ResidualBlock(trunk, drop),
        )
        self.mid = nn.Sequential(
            nn.Linear(trunk, half),
            nn.BatchNorm1d(half), nn.GELU(), nn.Dropout(drop * 0.7),
        )
        self.heads = nn.ModuleList([nn.Linear(half, s) for s in HEAD_SIZES])

    def forward(self, x):
        h = self.mid(self.res(self.stem(x)))
        return tuple(head(h) for head in self.heads)


# ─────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────

class MomentumDS(Dataset):
    def __init__(self, X, y_list):
        self.x  = torch.tensor(X.astype(np.float32))
        self.ys = [torch.tensor(y.astype(np.int64)) for y in y_list]

    def __len__(self): return self.x.shape[0]
    def __getitem__(self, idx):
        return (self.x[idx],) + tuple(y[idx] for y in self.ys)


def run_epoch(model, loader, loss_fns, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    tot_loss = 0.0; n_total = 0
    correct = [0] * 6; totals = [0] * 6

    with torch.set_grad_enabled(train):
        for batch in loader:
            xb = batch[0].to(DEVICE)
            ys = [b.to(DEVICE) for b in batch[1:]]
            if train: opt.zero_grad()
            outs   = model(xb)
            losses = [loss_fns[i](outs[i], ys[i]) for i in range(6)]
            loss   = sum(HEAD_WEIGHTS[i] * losses[i] for i in range(6))
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            bs = xb.size(0)
            tot_loss += loss.item() * bs; n_total += bs
            for i in range(6):
                pred = outs[i].argmax(dim=1)
                correct[i] += (pred == ys[i]).sum().item()
                totals[i]  += ys[i].size(0)

    return tot_loss / n_total, [correct[i] / max(totals[i], 1) for i in range(6)]


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Device : {DEVICE}")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded : {DATA_PATH}  shape={df.shape}")

    for c in LABEL_COLS:
        if c not in df.columns:
            raise RuntimeError(f"Missing column '{c}'. Columns: {list(df.columns)}")

    Ys        = [df[c].astype(int).values for c in LABEL_COLS]
    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X         = df[feat_cols].fillna(0.0)
    print(f"Features: {len(feat_cols)}  Samples: {len(df)}")
    print(f"Excluded non-feature cols: {EXCLUDE_COLS}")

    n     = len(df)
    n_val = int(n * VAL_RATIO)
    perm  = np.random.RandomState(SEED).permutation(n)
    tr_idx, va_idx = perm[:-n_val], perm[-n_val:]

    Xtr = X.iloc[tr_idx].values; Xva = X.iloc[va_idx].values
    Ytr = [y[tr_idx] for y in Ys]; Yva = [y[va_idx] for y in Ys]

    scaler = RobustScaler()
    Xtr_sc = scaler.fit_transform(Xtr)
    Xva_sc = scaler.transform(Xva)
    joblib.dump({"scaler": scaler, "feature_names": list(feat_cols)}, SCALER_PATH)
    print(f"Scaler  → {SCALER_PATH}")

    # ── Weighted sampler: oversample directional + regime-active rows ─
    # direction bear(0)/bull(2) = 1.6x, flat(1) = 0.5x
    # regime_behaviour cont(2)/breakout(3) = 1.8x, pullback(1) = 1.1x, chop(0) = 0.6x
    dir_w    = {0: 1.6, 1: 0.5, 2: 1.6}
    regime_w = {0: 0.6, 1: 1.1, 2: 1.8, 3: 1.8}
    sw = np.array([dir_w[Ytr[0][i]] * regime_w[Ytr[2][i]]
                   for i in range(len(Ytr[0]))], dtype=np.float32)
    sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)

    tr_loader = DataLoader(MomentumDS(Xtr_sc, Ytr), BATCH_SIZE, sampler=sampler,
                           num_workers=0, pin_memory=True)
    va_loader = DataLoader(MomentumDS(Xva_sc, Yva), BATCH_SIZE, shuffle=False,
                           num_workers=0, pin_memory=True)

    model = MomentumNet(input_dim=len(feat_cols)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model   : {n_params:,} trainable params")

    loss_fns = [nn.CrossEntropyLoss(weight=CLASS_WEIGHTS[i].to(DEVICE)) for i in range(6)]

    heads_short = ["dir", "mag", "reg", "vol", "struct", "qual"]
    print("\nLabel distributions (train):")
    for i, lc in enumerate(LABEL_COLS):
        print(f"  [{heads_short[i]:7}] {lc}: {dict(Counter(Ytr[i].tolist()))}")

    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR * 3, steps_per_epoch=len(tr_loader), epochs=EPOCHS, pct_start=0.1
    )

    best_val = float("inf"); best_ep = 0
    print(f"\nTraining {EPOCHS} epochs  |  batch={BATCH_SIZE}  |  lr={LR}\n{'─'*80}")

    for ep in range(1, EPOCHS + 1):
        tr_loss, _      = run_epoch(model, tr_loader, loss_fns, opt)
        va_loss, va_acc = run_epoch(model, va_loader, loss_fns)
        sched.step()
        saved = ""
        if va_loss < best_val:
            best_val = va_loss; best_ep = ep
            torch.save(model.state_dict(), BEST_MODEL_PATH); saved = "  ← best"
        acc_str = "  ".join(f"{heads_short[i]}={va_acc[i]:.3f}" for i in range(6))
        print(f"Ep {ep:03d} | TrL={tr_loss:.4f}  VaL={va_loss:.4f} | {acc_str}{saved}")

    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"\n{'─'*80}")
    print(f"✅  Best  → {BEST_MODEL_PATH}  (epoch {best_ep}, vaL={best_val:.4f})")
    print(f"✅  Final → {FINAL_MODEL_PATH}")
    print(f"✅  Scaler→ {SCALER_PATH}")


if __name__ == "__main__":
    main()
