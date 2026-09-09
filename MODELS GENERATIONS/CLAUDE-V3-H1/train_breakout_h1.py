# train_breakout_h1.py  ·  v4  — H1 recalibrated
# Dataset : synthetic_breakout_h1_dataset.csv  (20000 × 33)
# Features: 27-32  |  Labels: 6 heads (3-bar path-aware labels)
#
# H1 changes vs v3:
#   • DATA_PATH → h1 dataset
#   • label_next_pattern: reversal(1) now dominant at ~51% — downweight
#     continuation(0) 26% and indecision(2) 23% — upweight both
#   • label_next_close_loc: mid(1) at 19% → upweight; extremes near equal
#   • label_bk_follow_through: partial(1) dominant 55% → downweight; confirmed(2) 19% sparse
#   • Sampler: direction near-equal so focus on follow-through quality
#   • CLASS_WEIGHTS fully recalibrated from H1 distributions
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
DATA_PATH        = "synthetic_breakout_h1_dataset.csv"
SCALER_PATH      = "breakout_h1_scaler.pkl"
BEST_MODEL_PATH  = "breakout_h1_best.pth"
FINAL_MODEL_PATH = "breakout_h1_final.pth"

BATCH_SIZE = 256
EPOCHS     = 80
LR         = 1e-3
WD         = 1e-4
VAL_RATIO  = 0.12
TRUNK_DIM  = 256
DROP       = 0.25
SEED       = 42

# ── Labels — H1 3-bar path-aware ────────────────────────────────────
# label_next_direction    : 0=bearish  1=flat        2=bullish        → 3 cls
# label_next_magnitude    : 0=small    1=medium      2=large          → 3 cls
# label_next_pattern      : 0=cont     1=reversal    2=indecision     → 3 cls
# label_next_volatility   : 0=contract 1=normal      2=expand         → 3 cls
# label_next_close_loc    : 0=lower    1=mid         2=upper          → 3 cls
# label_bk_follow_through : 0=no       1=partial     2=yes/confirmed  → 3 cls
#
# H1 measured distributions:
#   direction:   32% / 35% / 33%   — near equal ✓ (best-balanced of all 5)
#   magnitude:   31% / 42% / 29%   — near equal
#   pattern:     26% / 51% / 23%   — reversal dominant
#   volatility:  24% / 43% / 31%   — expand slight surplus
#   close_loc:   41% / 19% / 40%   — mid sparse
#   follow:      28% / 55% / 19%   — partial dominant, confirmed sparse
LABEL_COLS = [
    "label_next_direction",
    "label_next_magnitude",
    "label_next_pattern",
    "label_next_volatility",
    "label_next_close_loc",
    "label_bk_follow_through",
]
HEAD_SIZES = [3, 3, 3, 3, 3, 3]

# Head loss weights — direction + follow-through most actionable
HEAD_WEIGHTS = [1.4, 1.1, 1.1, 0.9, 0.9, 1.6]

# Class weights — inverse-frequency calibrated
# direction:   near-equal → very light
# magnitude:   medium(1) 42% → moderate downweight; large(2) 29% → upweight
# pattern:     reversal(1) 51% → downweight; cont(0)/indecision(2) ~25% → upweight
# volatility:  contracting(0) 24% → upweight; expand slight surplus → moderate
# close_loc:   mid(1) 19% → strong upweight; extremes near equal
# follow:      partial(1) 55% → downweight; confirmed(2) 19% → upweight; no(0) 28% → moderate
CLASS_WEIGHTS = [
    torch.tensor([1.05, 0.95, 1.01], dtype=torch.float32),           # direction
    torch.tensor([1.05, 0.77, 1.18], dtype=torch.float32),           # magnitude
    torch.tensor([1.11, 0.59, 1.31], dtype=torch.float32),           # pattern
    torch.tensor([1.34, 0.73, 0.93], dtype=torch.float32),           # volatility
    torch.tensor([0.72, 1.55, 0.73], dtype=torch.float32),           # close_loc
    torch.tensor([1.00, 0.52, 1.48], dtype=torch.float32),           # follow_through
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
            nn.BatchNorm1d(trunk), nn.GELU(), nn.Dropout(drop),
        )
        self.res = nn.Sequential(
            ResidualBlock(trunk, drop),
            ResidualBlock(trunk, drop),
        )
        self.mid = nn.Sequential(
            nn.Linear(trunk, trunk // 2),
            nn.BatchNorm1d(trunk // 2), nn.GELU(), nn.Dropout(drop * 0.7),
        )
        self.heads = nn.ModuleList([nn.Linear(trunk // 2, s) for s in HEAD_SIZES])

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
                correct[i] += (outs[i].argmax(1) == ys[i]).sum().item()
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
            raise RuntimeError(f"Missing column '{lc}'. Columns: {list(df.columns)}")

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

    # Sampler: direction near-equal so light; focus on follow-through quality
    # confirmed(2) only 19% — strongly oversample; partial(1) 55% — undersample
    dir_w    = {0: 1.3, 1: 0.65, 2: 1.3}
    follow_w = {0: 0.90, 1: 0.55, 2: 2.00}
    sw = np.array([dir_w[Ytr[0][i]] * follow_w[Ytr[5][i]]
                   for i in range(len(Ytr[0]))], dtype=np.float32)
    sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)

    tr_loader = DataLoader(BreakoutDS(Xtr_s, Ytr), BATCH_SIZE, sampler=sampler, num_workers=0)
    va_loader = DataLoader(BreakoutDS(Xva_s, Yva), BATCH_SIZE, shuffle=False, num_workers=0)

    model    = BreakoutNet(input_dim=Xtr_s.shape[1]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    loss_fns = [nn.CrossEntropyLoss(weight=CLASS_WEIGHTS[i].to(DEVICE)) for i in range(6)]

    hs = ["dir", "mag", "pat", "vol", "loc", "follow"]
    print(f"\nModel   : {n_params:,} params")
    print("\nLabel distributions (train):")
    for i, lc in enumerate(LABEL_COLS):
        cnt = dict(Counter(Ytr[i].tolist()))
        pct = {k: f"{100*v/len(Ytr[i]):.1f}%" for k, v in cnt.items()}
        print(f"  [{hs[i]:6}] {lc}: {pct}")

    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, steps_per_epoch=len(tr_loader),
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
            print(f"Ep {ep:03d} | trL={tr_loss:.4f} vaL={va_loss:.4f} | {acc_str}{saved}")

    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"\n{'─'*80}")
    print(f"✅  Best  → {BEST_MODEL_PATH}  (epoch {best_ep}, vaL={best_val:.4f})")
    print(f"✅  Final → {FINAL_MODEL_PATH}")
    print(f"✅  Scaler→ {SCALER_PATH}")


if __name__ == "__main__":
    main()
