# train_model1_momentum.py  ·  v1
# ─────────────────────────────────────────────────────────────────────
# MODEL 1 — DIRECTIONAL MOMENTUM SPECIALIST
#
# Dataset    : model1_momentum_dataset.csv
# Features   : 6  (13_roc_4, 58_skewness_returns,
#                   101_consecutive_direction_counter, 103_final_candle_color,
#                   15_roc_acceleration, 36_r_squared)
# Target     : T_m = [P(Bullish), P(Bearish)]   → single binary head
# Preprocess : NONE — features are naturally scale-invariant / bounded.
#              (Train Raw. A global scaler would wash out subtle streak
#               differences that tree/NN splits rely on.)
# HOLD rows (label_momentum == -1) are EXCLUDED — Model 1 only answers
# the bullish-vs-bearish momentum question, not the no-trade question.
# ─────────────────────────────────────────────────────────────────────

import os, random
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import joblib

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────
DATA_PATH        = "model1_momentum_dataset.csv"
META_PATH        = "model1_meta.pkl"          # stores feature list (no scaler needed)
BEST_MODEL_PATH  = "model1_best.pth"
FINAL_MODEL_PATH = "model1_final.pth"

BATCH_SIZE = 256
EPOCHS     = 60
LR         = 1e-3
WD         = 1e-4
VAL_RATIO  = 0.12
TRUNK_DIM  = 128       # smaller trunk — only 6 input features, avoid overkill
DROP       = 0.20
SEED       = 42

# ── Feature columns (raw — no scaling applied) ─────────────────────
FEATURE_COLS = [
    "13_roc_4",
    "58_skewness_returns",
    "101_consecutive_direction_counter",
    "103_final_candle_color",
    "15_roc_acceleration",
    "36_r_squared",
]

# ── Target column ───────────────────────────────────────────────────
# label_momentum:  1 = Bullish,  0 = Bearish,  -1 = Hold (excluded below)
TARGET_COL  = "label_momentum"
HEAD_SIZE   = 2          # binary: [P(Bearish), P(Bullish)]

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


class MomentumNet(nn.Module):
    """
    Single-head binary classifier for directional momentum.
    Smaller trunk than the multi-head breakout net since Model 1 only
    has 6 raw input features and one output head.
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
        self.head = nn.Linear(trunk // 2, HEAD_SIZE)

    def forward(self, x):
        h = self.mid(self.res(self.stem(x)))
        return self.head(h)


# ─────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────

class MomentumDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.astype(np.float32))
        self.y = torch.tensor(y.astype(np.int64))

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def run_epoch(model, loader, loss_fn, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    total_loss, total_n, correct = 0.0, 0, 0

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
            correct    += (out.argmax(1) == yb).sum().item()

    return total_loss / total_n, correct / total_n


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

    # ── Drop HOLD rows — Model 1 is strictly bullish-vs-bearish ──────────
    before = len(df)
    df = df[df[TARGET_COL].isin([0, 1])].reset_index(drop=True)
    print(f"Dropped HOLD rows : {before - len(df):,}  →  remaining {len(df):,}")

    X = df[FEATURE_COLS].fillna(0.0).values
    y = df[TARGET_COL].astype(int).values
    print(f"Features: {len(FEATURE_COLS)}  Samples: {len(df)}")
    print(f"Class balance: {dict(Counter(y.tolist()))}")

    # ── Train / val split ────────────────────────────────────────────────
    n     = len(df)
    n_val = int(n * VAL_RATIO)
    idx   = np.random.RandomState(SEED).permutation(n)
    tr_idx, va_idx = idx[:-n_val], idx[-n_val:]

    Xtr, Xva = X[tr_idx], X[va_idx]
    ytr, yva = y[tr_idx], y[va_idx]

    # ── NO SCALER ─────────────────────────────────────────────────────────
    # Per the training profile: these 6 features are naturally bounded
    # (percentages, signed integers, R² in [0,1]) and scale-invariant.
    # We still save feature names for inference-time column alignment.
    joblib.dump({"feature_names": FEATURE_COLS, "scaler": None}, META_PATH)
    print(f"Meta    → {META_PATH}  (no scaler used — raw features)")

    # ── Weighted sampler — balance bull/bear if dataset is skewed ──────────
    cnt        = Counter(ytr.tolist())
    class_w    = {cls: len(ytr) / (len(cnt) * n) for cls, n in cnt.items()}
    sample_w   = np.array([class_w[label] for label in ytr], dtype=np.float32)
    sampler    = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

    tr_loader = DataLoader(MomentumDS(Xtr, ytr), BATCH_SIZE, sampler=sampler, num_workers=0)
    va_loader = DataLoader(MomentumDS(Xva, yva), BATCH_SIZE, shuffle=False, num_workers=0)

    model = MomentumNet(Xtr.shape[1]).to(DEVICE)

    # Class weights also passed to loss for extra robustness on top of sampler
    class_weight_tensor = torch.tensor(
        [class_w.get(0, 1.0), class_w.get(1, 1.0)], dtype=torch.float32
    ).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=class_weight_tensor)

    print(f"\nLabel distribution (train): {dict(Counter(ytr.tolist()))}")
    print(f"Label distribution (val)  : {dict(Counter(yva.tolist()))}")

    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, steps_per_epoch=len(tr_loader), epochs=EPOCHS, pct_start=0.1
    )

    best_val = float("inf")
    print(f"\nTraining {EPOCHS} epochs ...\n{'─'*70}")

    for ep in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, tr_loader, loss_fn, opt)
        va_loss, va_acc = run_epoch(model, va_loader, loss_fn)
        sched.step()
        saved = ""
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH); saved = "  ← best"
        if ep % 5 == 0 or ep == 1:
            print(f"Ep {ep:03d} | trL={tr_loss:.4f} trA={tr_acc:.4f} | "
                  f"vaL={va_loss:.4f} vaA={va_acc:.4f}{saved}")

    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"\n✅  Best  → {BEST_MODEL_PATH}  (vaL={best_val:.4f})")
    print(f"✅  Final → {FINAL_MODEL_PATH}")
    print(f"✅  Meta  → {META_PATH}")


if __name__ == "__main__":
    main()
