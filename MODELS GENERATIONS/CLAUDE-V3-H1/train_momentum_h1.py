# train_momentum_h1.py  ·  v4  — H1 recalibrated
# Dataset : synthetic_momentum_h1_dataset.csv  (25000 × 37)
# Features: 30  |  Labels: 6 heads (3-bar path-aware labels)
#
# H1 changes vs v3  *** CRITICAL ***:
#   • HEAD_SIZES[1]: 2 → 3  (next_magnitude now has small/medium/large, not just small/large)
#   • CLASS_WEIGHTS[1]: 2-element → 3-element tensor
#   • DATA_PATH → h1 dataset
#   • mom_quality: fading(0) at 58% → heavily downweight, upweight accelerating(2)
#   • mom_regime: rebalanced across 4 classes
#   • Sampler: updated proportions for H1 distributions
#   • EPOCHS 60 → 80
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
DATA_PATH        = "synthetic_momentum_h1_dataset.csv"
SCALER_PATH      = "momentum_h1_scaler.pkl"
BEST_MODEL_PATH  = "momentum_h1_best.pth"
FINAL_MODEL_PATH = "momentum_h1_final.pth"

BATCH_SIZE = 512
EPOCHS     = 80
LR         = 3e-4
WD         = 1e-4
VAL_RATIO  = 0.10
TRUNK_DIM  = 256
DROP       = 0.25
SEED       = 42

# ── Labels — H1 3-bar path-aware ────────────────────────────────────
# next_direction        : 0=bearish  1=flat       2=bullish           → 3 cls
# next_magnitude        : 0=small    1=medium     2=large             → 3 cls  ← NOW 3 (was 2)
# next_regime_behaviour : 0=chop     1=pullback   2=continuation 3=breakout → 4 cls
# next_volatility       : 0=contract 1=normal     2=expanding         → 3 cls
# next_structural_bias  : 0=bearish  1=neutral    2=bullish           → 3 cls
# next_momentum_quality : 0=fading   1=stable     2=accelerating      → 3 cls
#
# H1 measured distributions:
#   direction:   33% / 31% / 36%   — near equal
#   magnitude:   35% / 39% / 26%   — large underrepresented
#   regime:      35% / 21% / 25% / 18%  — pullback+breakout sparse
#   volatility:  28% / 45% / 27%   — near equal
#   structural:  30% / 40% / 31%   — near equal
#   quality:     58% / 29% / 12%   — fading hugely dominant
LABEL_COLS = [
    "next_direction",
    "next_magnitude",
    "next_regime_behaviour",
    "next_volatility",
    "next_structural_bias",
    "next_momentum_quality",
]
HEAD_SIZES = [3, 3, 4, 3, 3, 3]   # ← [1] was 2 in M15 version, NOW 3

EXCLUDE_COLS = LABEL_COLS + ["scenario"]

# Head loss weights — direction + quality + regime primary
HEAD_WEIGHTS = [1.5, 1.0, 1.3, 0.9, 1.1, 1.4]

# Class weights — inverse-frequency calibrated
# direction:   near-equal → light adjustment
# magnitude:   large(2) at 26% → upweight; small/med near equal
# regime:      pullback(1) 21%, breakout(3) 18% → upweight; chop(0) 35% → downweight
# volatility:  near-equal → light
# structural:  near-equal → light
# quality:     fading(0) 58% → heavy downweight; accelerating(2) 12% → strong upweight
CLASS_WEIGHTS = [
    torch.tensor([1.00, 1.08, 0.92], dtype=torch.float32),           # direction
    torch.tensor([0.91, 0.84, 1.25], dtype=torch.float32),           # magnitude (3 cls)
    torch.tensor([0.67, 1.10, 0.95, 1.28], dtype=torch.float32),    # regime
    torch.tensor([1.14, 0.70, 1.17], dtype=torch.float32),           # volatility
    torch.tensor([1.10, 0.83, 1.07], dtype=torch.float32),           # structural
    torch.tensor([0.40, 0.78, 1.84], dtype=torch.float32),           # quality (fading >> upweight accel)
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
                correct[i] += (outs[i].argmax(1) == ys[i]).sum().item()
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

    # Validate magnitude has 3 classes (critical check)
    mag_classes = sorted(df["next_magnitude"].unique())
    if len(mag_classes) != 3:
        raise RuntimeError(
            f"next_magnitude has {len(mag_classes)} unique classes {mag_classes}. "
            f"Expected 3 (small/medium/large). Check dataset generation."
        )

    Ys        = [df[c].astype(int).values for c in LABEL_COLS]
    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X         = df[feat_cols].fillna(0.0)
    print(f"Features: {len(feat_cols)}  Samples: {len(df)}")

    n     = len(df)
    n_val = int(n * VAL_RATIO)
    perm  = np.random.RandomState(SEED).permutation(n)
    tr_idx, va_idx = perm[:-n_val], perm[-n_val:]

    Xtr, Xva = X.iloc[tr_idx].values, X.iloc[va_idx].values
    Ytr = [y[tr_idx] for y in Ys]
    Yva = [y[va_idx] for y in Ys]

    scaler = RobustScaler()
    Xtr_sc = scaler.fit_transform(Xtr)
    Xva_sc = scaler.transform(Xva)
    joblib.dump({"scaler": scaler, "feature_names": list(feat_cols)}, SCALER_PATH)
    print(f"Scaler  → {SCALER_PATH}")

    # Sampler: oversample directional + accelerating quality rows
    # H1: direction near-equal so moderate weights; quality: accel(2) only 12% → 2.0x
    dir_w     = {0: 1.4, 1: 0.55, 2: 1.4}    # upweight directional bars
    quality_w = {0: 0.45, 1: 0.90, 2: 2.0}   # heavily oversample accelerating
    sw = np.array([dir_w[Ytr[0][i]] * quality_w[Ytr[5][i]]
                   for i in range(len(Ytr[0]))], dtype=np.float32)
    sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)

    tr_loader = DataLoader(MomentumDS(Xtr_sc, Ytr), BATCH_SIZE, sampler=sampler,
                           num_workers=0, pin_memory=True)
    va_loader = DataLoader(MomentumDS(Xva_sc, Yva), BATCH_SIZE, shuffle=False,
                           num_workers=0, pin_memory=True)

    model    = MomentumNet(input_dim=len(feat_cols)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    loss_fns = [nn.CrossEntropyLoss(weight=CLASS_WEIGHTS[i].to(DEVICE)) for i in range(6)]

    hs = ["dir", "mag", "reg", "vol", "struct", "qual"]
    print(f"\nModel   : {n_params:,} params")
    print("\nLabel distributions (train):")
    for i, lc in enumerate(LABEL_COLS):
        cnt = dict(Counter(Ytr[i].tolist()))
        pct = {k: f"{100*v/len(Ytr[i]):.1f}%" for k, v in cnt.items()}
        print(f"  [{hs[i]:6}] {lc}: {pct}")

    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR * 3, steps_per_epoch=len(tr_loader),
        epochs=EPOCHS, pct_start=0.1,
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
        if ep % 5 == 0 or ep == 1:
            acc_str = "  ".join(f"{hs[i]}={va_acc[i]:.3f}" for i in range(6))
            print(f"Ep {ep:03d} | TrL={tr_loss:.4f}  VaL={va_loss:.4f} | {acc_str}{saved}")

    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"\n{'─'*80}")
    print(f"✅  Best  → {BEST_MODEL_PATH}  (epoch {best_ep}, vaL={best_val:.4f})")
    print(f"✅  Final → {FINAL_MODEL_PATH}")
    print(f"✅  Scaler→ {SCALER_PATH}")


if __name__ == "__main__":
    main()
