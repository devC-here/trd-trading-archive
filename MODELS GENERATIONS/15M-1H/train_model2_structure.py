# train_model2_structure.py  ·  v1
# ─────────────────────────────────────────────────────────────────────
# MODEL 2 — STRUCTURAL PRICE LOCATION SPECIALIST
#
# Dataset    : model2_structure_dataset.csv
# Features   : 5  (43_fib_382_dist, 49_s1_dist, 48_r1_dist,
#                   35_lin_reg_intercept, 69_high_low_ratio)
# Target     : E_s ∈ [-1.0, +1.0]  (continuous scalar regression)
#              -1.0 = exhaustion at structural floor (Support 1 / 38.2% Fib)
#              +1.0 = overextension at structural ceiling (Resistance 1 / Fib top)
# Preprocess : StandardScaler REQUIRED — 35_lin_reg_intercept is a large
#              nominal price (~1.2000) while fib distances are micro-decimals
#              (~0.0002). Raw training would collapse micro-distance weights.
# Loss       : HuberLoss (robust against breakout tail anomalies)
# ─────────────────────────────────────────────────────────────────────

import os, random
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────
DATA_PATH        = "model2_structure_dataset.csv"
SCALER_PATH      = "model2_scaler.pkl"
BEST_MODEL_PATH  = "model2_best.pth"
FINAL_MODEL_PATH = "model2_final.pth"

BATCH_SIZE = 256
EPOCHS     = 60
LR         = 1e-3
WD         = 1e-4
VAL_RATIO  = 0.12
TRUNK_DIM  = 128
DROP       = 0.20
SEED       = 42

FEATURE_COLS = [
    "43_fib_382_dist",
    "49_s1_dist",
    "48_r1_dist",
    "35_lin_reg_intercept",
    "69_high_low_ratio",
]

TARGET_COL = "label_structure"   # float: -1 (BUY/support), 0 (HOLD), +1 (SELL/resistance)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, dim, drop=0.20):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(drop),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.seq(x) + x)


class StructureNet(nn.Module):
    """
    Single-head regression network for structural price positioning.
    Output is a scalar in [-1, +1] via Tanh activation — no CrossEntropy,
    no class head. HuberLoss drives training.
    """
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
        self.head = nn.Sequential(
            nn.Linear(trunk // 2, 1),
            nn.Tanh()               # bounds output to [-1, +1]
        )

    def forward(self, x):
        h = self.mid(self.res(self.stem(x)))
        return self.head(h).squeeze(-1)   # shape: (batch,)


# ─────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────

class StructureDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.astype(np.float32))
        self.y = torch.tensor(y.astype(np.float32))

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def run_epoch(model, loader, loss_fn, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    total_loss, total_n = 0.0, 0

    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out  = model(xb)
            loss = loss_fn(out, yb)
            if train:
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total_loss += loss.item() * xb.size(0)
            total_n    += xb.size(0)

    # Sign-accuracy: predicted sign matches target sign (exclude ~0 targets)
    return total_loss / total_n


def sign_accuracy(model, loader):
    """Fraction of rows where sign(pred) == sign(target) for |target| > 0.1."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            mask = yb.abs() > 0.1
            if mask.sum() == 0: continue
            correct += (pred[mask].sign() == yb[mask].sign()).sum().item()
            total   += mask.sum().item()
    return correct / max(total, 1)


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Device : {DEVICE}")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded : {DATA_PATH}  shape={df.shape}")

    for c in FEATURE_COLS + [TARGET_COL]:
        if c not in df.columns:
            raise RuntimeError(f"Missing column '{c}'. Columns: {list(df.columns)}")

    X = df[FEATURE_COLS].fillna(0.0).values
    y = df[TARGET_COL].astype(float).values
    print(f"Features : {len(FEATURE_COLS)}  |  Samples : {len(df)}")
    print(f"Target   : min={y.min():.3f}  max={y.max():.3f}  mean={y.mean():.3f}")

    # ── StandardScaler (mandatory — intercept is ~1.2, distances are ~0.0002) ──
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    joblib.dump({"feature_names": FEATURE_COLS, "scaler": scaler}, SCALER_PATH)
    print(f"Scaler  → {SCALER_PATH}  (StandardScaler fitted on {len(df):,} rows)")

    # ── Train / val split ────────────────────────────────────────────────
    n     = len(df)
    n_val = int(n * VAL_RATIO)
    idx   = np.random.RandomState(SEED).permutation(n)
    tr_idx, va_idx = idx[:-n_val], idx[-n_val:]

    Xtr, Xva = X[tr_idx], X[va_idx]
    ytr, yva = y[tr_idx], y[va_idx]

    tr_loader = DataLoader(StructureDS(Xtr, ytr), BATCH_SIZE, shuffle=True,  num_workers=0)
    va_loader = DataLoader(StructureDS(Xva, yva), BATCH_SIZE, shuffle=False, num_workers=0)

    model   = StructureNet(Xtr.shape[1]).to(DEVICE)
    loss_fn = nn.HuberLoss(delta=0.5)   # robust to extreme breakout tails
    opt     = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched   = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, steps_per_epoch=len(tr_loader), epochs=EPOCHS, pct_start=0.1
    )

    print(f"\nTarget distribution — BUY(-1): {(y==-1).sum():,}  "
          f"HOLD(0): {(y==0).sum():,}  SELL(+1): {(y==1).sum():,}")

    best_val = float("inf")
    print(f"\nTraining {EPOCHS} epochs ...\n{'─'*75}")

    for ep in range(1, EPOCHS + 1):
        tr_loss = run_epoch(model, tr_loader, loss_fn, opt)
        va_loss = run_epoch(model, va_loader, loss_fn)
        sched.step()
        saved = ""
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH); saved = "  ← best"
        if ep % 5 == 0 or ep == 1:
            va_sacc = sign_accuracy(model, va_loader)
            print(f"Ep {ep:03d} | trL={tr_loss:.5f} | vaL={va_loss:.5f}  vaSignAcc={va_sacc:.4f}{saved}")

    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"\n✅  Best   → {BEST_MODEL_PATH}  (vaL={best_val:.5f})")
    print(f"✅  Final  → {FINAL_MODEL_PATH}")
    print(f"✅  Scaler → {SCALER_PATH}")


if __name__ == "__main__":
    main()
