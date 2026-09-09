# train_momentum.py  ·  v2
# ─────────────────────────────────────────────────────────────────────────────
# Trains MomentumMultiHeadNet on synthetic_momentum_dataset.csv
# Architecture: stem → 2× ResidualBlock → mid → 6 task heads
# Per-head class-weighted CrossEntropy handles label imbalance.
#
# Outputs:
#   momentum_best.pth      ← use this for deployment
#   momentum_final.pth
#   momentum_scaler.pkl    ← RobustScaler bundle {scaler, feature_names}
# ─────────────────────────────────────────────────────────────────────────────

import os, random
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
import joblib

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_PATH       = "synthetic_momentum_dataset.csv"
SCALER_PATH     = "momentum_scaler.pkl"
MODEL_PATH      = "momentum_final.pth"
BEST_MODEL_PATH = "momentum_best.pth"

BATCH_SIZE  = 512
EPOCHS      = 60
LR          = 3e-4
WD          = 1e-4
VAL_RATIO   = 0.10
RNG         = 42
TRUNK_DIM   = 256
DROP        = 0.25

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Head definitions — order must match label_cols in synthetic_momentum.py
HEAD_NAMES   = ["momentum_state", "momentum_slope", "regime",
                "volatility_state", "structure_bias", "entry_quality"]
HEAD_CLASSES = [3, 3, 4, 3, 3, 3]
HEAD_WEIGHTS = [1.0, 0.9, 1.1, 0.8, 0.9, 1.2]
# entry_quality and regime get highest weight — most actionable for context engine

LABEL_COLS = HEAD_NAMES


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class MomentumDS(Dataset):
    def __init__(self, X: np.ndarray, y_list):
        self.x  = torch.tensor(X.astype(np.float32))
        self.ys = [torch.tensor(y.astype(np.int64)) for y in y_list]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return (self.x[idx],) + tuple(y[idx] for y in self.ys)


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class ResidualBlock(nn.Module):
    def __init__(self, dim: int, drop: float = 0.25):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.seq(x) + x)


class MomentumMultiHeadNet(nn.Module):
    """
    Shared residual trunk → 6 independent classification heads.

    Architecture:
        Input (30) → Stem (256) → ResBlock × 2 → Mid (128) → 6 heads
    """
    def __init__(self, input_dim: int, trunk_dim: int = TRUNK_DIM, drop: float = DROP):
        super().__init__()
        half = trunk_dim // 2

        self.stem = nn.Sequential(
            nn.Linear(input_dim, trunk_dim),
            nn.BatchNorm1d(trunk_dim),
            nn.GELU(),
            nn.Dropout(drop),
        )
        self.res = nn.Sequential(
            ResidualBlock(trunk_dim, drop),
            ResidualBlock(trunk_dim, drop),
        )
        self.mid = nn.Sequential(
            nn.Linear(trunk_dim, half),
            nn.BatchNorm1d(half),
            nn.GELU(),
            nn.Dropout(drop * 0.7),
        )

        self.head_momentum_state   = nn.Linear(half, 3)
        self.head_momentum_slope   = nn.Linear(half, 3)
        self.head_regime           = nn.Linear(half, 4)
        self.head_volatility_state = nn.Linear(half, 3)
        self.head_structure_bias   = nn.Linear(half, 3)
        self.head_entry_quality    = nn.Linear(half, 3)

    def forward(self, x):
        h = self.mid(self.res(self.stem(x)))
        return (
            self.head_momentum_state(h),
            self.head_momentum_slope(h),
            self.head_regime(h),
            self.head_volatility_state(h),
            self.head_structure_bias(h),
            self.head_entry_quality(h),
        )


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def compute_class_weights(arr: np.ndarray, n_classes: int) -> torch.Tensor:
    cnt   = Counter(arr.astype(int).tolist())
    total = sum(cnt.values())
    return torch.tensor(
        [total / (cnt.get(k, 1) * n_classes) for k in range(n_classes)],
        dtype=torch.float32
    )


def run_epoch(model, loader, loss_fns, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    tot_loss = 0.0
    n_total  = 0
    correct  = [0] * 6
    totals   = [0] * 6

    with torch.set_grad_enabled(training):
        for batch in loader:
            xb = batch[0].to(DEVICE)
            ys = [b.to(DEVICE) for b in batch[1:]]

            if training:
                optimizer.zero_grad()

            outs   = model(xb)
            losses = [loss_fns[i](outs[i], ys[i]) for i in range(6)]
            loss   = sum(HEAD_WEIGHTS[i] * losses[i] for i in range(6))

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            bs        = xb.size(0)
            tot_loss += loss.item() * bs
            n_total  += bs

            for i in range(6):
                pred       = outs[i].argmax(dim=1)
                correct[i] += (pred == ys[i]).sum().item()
                totals[i]  += ys[i].size(0)

    avg_loss = tot_loss / n_total
    accs     = [correct[i] / max(totals[i], 1) for i in range(6)]
    return avg_loss, accs


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    torch.manual_seed(RNG)
    np.random.seed(RNG)
    random.seed(RNG)

    print(f"Device : {DEVICE}")
    print(f"Loading: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    for c in LABEL_COLS:
        if c not in df.columns:
            raise RuntimeError(f"Missing label column: {c}")

    Ys        = [df[c].astype(int).values for c in LABEL_COLS]
    feat_cols = [c for c in df.columns if c not in LABEL_COLS + ["scenario"]]
    X         = df[feat_cols].fillna(0.0)

    print(f"Samples : {len(df)}")
    print(f"Features: {len(feat_cols)} → {feat_cols}")

    # Train / val split
    n     = len(df)
    n_val = int(n * VAL_RATIO)
    perm  = np.random.RandomState(RNG).permutation(n)
    tr_idx, va_idx = perm[:-n_val], perm[-n_val:]

    Xtr = X.iloc[tr_idx].values
    Xva = X.iloc[va_idx].values
    Ytr = [y[tr_idx] for y in Ys]
    Yva = [y[va_idx] for y in Ys]

    # Scale — RobustScaler is resistant to the fat tails in ATR/displacement features
    scaler   = RobustScaler()
    Xtr_sc   = scaler.fit_transform(Xtr)
    Xva_sc   = scaler.transform(Xva)
    joblib.dump({"scaler": scaler, "feature_names": feat_cols}, SCALER_PATH)
    print(f"Scaler  : saved → {SCALER_PATH}")

    # DataLoaders
    tr_loader = DataLoader(MomentumDS(Xtr_sc, Ytr), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    va_loader = DataLoader(MomentumDS(Xva_sc, Yva), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # Model
    model = MomentumMultiHeadNet(input_dim=len(feat_cols)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model   : {n_params:,} trainable parameters")

    # Per-head class-weighted loss — handles label imbalance automatically
    loss_fns = []
    print("\nHead class weights:")
    for i, ncls in enumerate(HEAD_CLASSES):
        w = compute_class_weights(Ytr[i], ncls).to(DEVICE)
        loss_fns.append(nn.CrossEntropyLoss(weight=w))
        print(f"  [{HEAD_NAMES[i]:<20}] classes={ncls}  weights={w.cpu().numpy().round(3)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.05)

    best_val_loss = float("inf")
    best_ep       = 0

    print(f"\nTraining {EPOCHS} epochs  |  batch={BATCH_SIZE}  |  lr={LR}\n{'─'*80}")

    for ep in range(1, EPOCHS + 1):
        tr_loss, tr_accs = run_epoch(model, tr_loader, loss_fns, optimizer)
        va_loss, va_accs = run_epoch(model, va_loader, loss_fns)
        scheduler.step()

        saved = ""
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_ep       = ep
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            saved = "  ← best"

        # Compact per-epoch log
        acc_str = "  ".join(f"{HEAD_NAMES[i][:7]}={va_accs[i]:.3f}" for i in range(6))
        print(f"Ep {ep:03d} | TrL={tr_loss:.4f}  VaL={va_loss:.4f} | {acc_str}{saved}")

    torch.save(model.state_dict(), MODEL_PATH)

    print(f"\n{'─'*80}")
    print(f"Final   → {MODEL_PATH}")
    print(f"Best    → {BEST_MODEL_PATH}  (epoch {best_ep}, val_loss={best_val_loss:.4f})")
    print(f"Scaler  → {SCALER_PATH}")


if __name__ == "__main__":
    main()
