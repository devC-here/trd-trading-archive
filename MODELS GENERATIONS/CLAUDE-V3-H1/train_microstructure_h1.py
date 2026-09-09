# train_microstructure_h1.py  ·  v4  — H1 recalibrated
# Dataset : synthetic_microstructure_h1_dataset.csv  (20000 × 38)
# Features: 32  |  Labels: 6 heads (3-bar path-aware labels)
#
# H1 changes vs v3:
#   • DATA_PATH → h1 dataset
#   • bar_intent: fade(3) now dominant at 64% → heavily downweight fade,
#     upweight absorption(0)/rejection(1)/impulse(2)
#   • momentum_follow: partial(1) dominant 61% → downweight, upweight confirmed(2)
#   • Sampler: flow_dir near-equal, so moderate; focus on confirmed follow
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
DATA_PATH        = "synthetic_microstructure_h1_dataset.csv"
SCALER_PATH      = "microstructure_h1_scaler.pkl"
BEST_MODEL_PATH  = "microstructure_h1_best.pth"
FINAL_MODEL_PATH = "microstructure_h1_final.pth"

BATCH_SIZE = 256
EPOCHS     = 80
LR         = 1e-3
WD         = 1e-4
VAL_RATIO  = 0.12
TRUNK_DIM  = 256
DROP       = 0.25
SEED       = 42

# ── Labels — H1 3-bar path-aware ────────────────────────────────────
# label_next_flow_dir   : 0=bearish    1=contested   2=bullish        → 3 cls
# label_next_bar_intent : 0=absorption 1=rejection   2=impulse 3=fade → 4 cls
# label_next_magnitude  : 0=thin       1=normal      2=strong         → 3 cls
# label_next_rejection  : 0=none       1=partial     2=strong         → 3 cls
# label_momentum_follow : 0=failed     1=partial     2=confirmed      → 3 cls
# label_next_range_type : 0=compress   1=normal      2=expand         → 3 cls
#
# H1 measured distributions:
#   flow_dir:     33% / 33% / 35%   — near equal ✓
#   bar_intent:   10% /  7% / 19% / 64%  — fade dominant!
#   magnitude:    31% / 38% / 31%   — near equal ✓
#   rejection:    27% / 56% / 17%   — partial dominant
#   follow:       26% / 61% / 14%   — partial dominant, confirmed sparse
#   range_type:   24% / 44% / 33%   — normal dominant, compress sparse
LABEL_COLS = [
    "label_next_flow_dir",
    "label_next_bar_intent",
    "label_next_magnitude",
    "label_next_rejection",
    "label_momentum_follow",
    "label_next_range_type",
]
HEAD_SIZES = [3, 4, 3, 3, 3, 3]

# Head loss weights — flow_dir + momentum_follow primary trade signals
HEAD_WEIGHTS = [1.5, 1.2, 1.0, 0.9, 1.5, 1.0]

# Class weights — inverse-frequency calibrated
# flow_dir:     near-equal → light adjustment
# bar_intent:   absorption(0)10%→upweight, rejection(1)7%→upweight,
#               impulse(2)19%→moderate, fade(3)64%→strong downweight
# magnitude:    near-equal → light
# rejection:    strong(2) 17% → upweight; partial(1) 56% → downweight
# follow:       confirmed(2) 14% → strong upweight; partial(1) 61% → downweight
# range_type:   compress(0) 24% → upweight; normal fine; expand moderate
CLASS_WEIGHTS = [
    torch.tensor([1.02, 1.02, 0.96], dtype=torch.float32),           # flow_dir
    torch.tensor([1.90, 2.50, 1.10, 0.30], dtype=torch.float32),    # bar_intent (fade capped at 0.30)
    torch.tensor([1.07, 0.86, 1.07], dtype=torch.float32),           # magnitude
    torch.tensor([0.96, 0.47, 1.57], dtype=torch.float32),           # rejection
    torch.tensor([0.88, 0.37, 1.74], dtype=torch.float32),           # momentum_follow
    torch.tensor([1.33, 0.71, 0.96], dtype=torch.float32),           # range_type
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


class MicroNet(nn.Module):
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

class MicroDS(Dataset):
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

    # Sampler: flow_dir near-equal so light weights; focus on confirmed follow
    # bar_intent: oversample non-fade classes (absorption/rejection/impulse)
    # momentum_follow: heavily oversample confirmed(2)
    flow_w   = {0: 1.4, 1: 0.65, 2: 1.4}    # upweight directional flow
    follow_w = {0: 0.90, 1: 0.55, 2: 2.20}  # confirmed follow is rare and high-signal
    # Also weight by bar_intent: don't oversample fade rows
    intent_w = {0: 2.0, 1: 2.5, 2: 1.5, 3: 0.40}  # mirror class weights
    sw = np.array([flow_w[Ytr[0][i]] * follow_w[Ytr[4][i]] * intent_w[Ytr[1][i]]
                   for i in range(len(Ytr[0]))], dtype=np.float32)
    sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)

    tr_loader = DataLoader(MicroDS(Xtr_s, Ytr), BATCH_SIZE, sampler=sampler, num_workers=0)
    va_loader = DataLoader(MicroDS(Xva_s, Yva), BATCH_SIZE, shuffle=False, num_workers=0)

    model    = MicroNet(input_dim=Xtr_s.shape[1]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    loss_fns = [nn.CrossEntropyLoss(weight=CLASS_WEIGHTS[i].to(DEVICE)) for i in range(6)]

    hs = ["flow", "intent", "mag", "rej", "follow", "range"]
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
