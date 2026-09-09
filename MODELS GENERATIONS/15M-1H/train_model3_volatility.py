# train_model3_volatility.py  ·  v1
# ─────────────────────────────────────────────────────────────────────
# MODEL 3 — VOLATILITY SQUEEZE SPECIALIST
#
# Dataset    : model3_volatility_dataset.csv
# Features   : 4  (62_bollinger_bandwidth_z_score, 63_keltner_channel_squeeze,
#                   72_true_range_z_score, 65_fractal_dimension_index)
# Target     : A_v = binary regime label
#              Class 0 = Chop / Sideways  (original label == HOLD)
#              Class 1 = Breakout / Expansion (original label == BUY or SELL)
# Preprocess : NONE — all features are Z-scores or bounded indices already.
#              Scaling would destroy the calibrated Z-score semantics.
# Loss       : CrossEntropyLoss  (binary classification, 2-class head)
#
# ⚠️  CRITICAL DESIGN NOTE:
#   Model 3 is NON-DIRECTIONAL. It does NOT predict up or down.
#   Feeding it bullish-vs-bearish labels collapses accuracy to ~50%.
#   BUY and SELL are both mapped to Class 1 (Expansion). Direction is
#   delegated entirely to Model 1 (Momentum) and Model 2 (Structure).
# ─────────────────────────────────────────────────────────────────────

import os, random
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import joblib

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────
DATA_PATH        = "model3_volatility_dataset.csv"
META_PATH        = "model3_meta.pkl"
BEST_MODEL_PATH  = "model3_best.pth"
FINAL_MODEL_PATH = "model3_final.pth"

BATCH_SIZE = 256
EPOCHS     = 60
LR         = 1e-3
WD         = 1e-4
VAL_RATIO  = 0.12
TRUNK_DIM  = 64        # only 4 input features — keep trunk lean
DROP       = 0.20
SEED       = 42

FEATURE_COLS = [
    "62_bollinger_bandwidth_z_score",
    "63_keltner_channel_squeeze",
    "72_true_range_z_score",
    "65_fractal_dimension_index",
]

# label_volatility: 0 = Chop/HOLD, 1 = Breakout/BUY+SELL
TARGET_COL = "label_volatility"
HEAD_SIZE  = 2   # [P(Chop), P(Breakout)]

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


class VolatilityNet(nn.Module):
    """
    Binary regime classifier: Chop (0) vs Breakout (1).
    Lean trunk — only 4 features, all pre-normalised Z-scores/indices.
    No direction inference. Pure volatility texture detection.
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

class VolatilityDS(Dataset):
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


def get_val_probs(model, loader):
    """Return (y_true, p_breakout) arrays for threshold tuning / AUC."""
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            out = model(xb)
            prob = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            ps.append(prob); ys.append(yb.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def find_best_threshold(y_true, probs):
    """
    Scan thresholds and return the one maximising weighted F1.

    WHY THIS MATTERS FOR MODEL 3:
    The training sampler oversamples the minority Chop class to ~50/50 so
    the network learns balanced gradients. But the natural class ratio is
    ~80% Breakout / 20% Chop, so the default 0.5 decision boundary is
    miscalibrated at inference time (it over-predicts Chop). Tuning the
    threshold on validation data — AFTER training — restores a realistic
    operating point without touching the learned representation.
    """
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_true, preds, average="weighted")
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)


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
    y = df[TARGET_COL].astype(int).values
    print(f"Features : {len(FEATURE_COLS)}  |  Samples : {len(df)}")
    print(f"Regime distribution : {dict(Counter(y.tolist()))}")
    print(f"  Class 0 = Chop/HOLD  |  Class 1 = Breakout (BUY+SELL merged)")

    # ── NO SCALER — Z-scores and FDI are self-normalised ─────────────────
    joblib.dump({"feature_names": FEATURE_COLS, "scaler": None}, META_PATH)
    print(f"Meta    → {META_PATH}  (no scaler — raw Z-score features)")

    # ── Train / val split ────────────────────────────────────────────────
    n     = len(df)
    n_val = int(n * VAL_RATIO)
    idx   = np.random.RandomState(SEED).permutation(n)
    tr_idx, va_idx = idx[:-n_val], idx[-n_val:]

    Xtr, Xva = X[tr_idx], X[va_idx]
    ytr, yva = y[tr_idx], y[va_idx]

    # ── Weighted sampler — balance Chop vs Breakout ──────────────────────
    cnt       = Counter(ytr.tolist())
    class_w   = {cls: len(ytr) / (len(cnt) * n) for cls, n in cnt.items()}
    sample_w  = np.array([class_w[label] for label in ytr], dtype=np.float32)
    sampler   = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

    tr_loader = DataLoader(VolatilityDS(Xtr, ytr), BATCH_SIZE, sampler=sampler, num_workers=0)
    va_loader = DataLoader(VolatilityDS(Xva, yva), BATCH_SIZE, shuffle=False, num_workers=0)

    model = VolatilityNet(Xtr.shape[1]).to(DEVICE)

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

    # ── Threshold calibration (CRITICAL — see find_best_threshold docstring) ──
    y_true, probs = get_val_probs(model, va_loader)
    auc           = roc_auc_score(y_true, probs)
    best_t, best_f1 = find_best_threshold(y_true, probs)
    calibrated_preds = (probs >= best_t).astype(int)

    print(f"\n{'─'*70}")
    print(f"  THRESHOLD CALIBRATION  (default 0.5 is miscalibrated on imbalanced val set)")
    print(f"{'─'*70}")
    print(f"  Validation AUC          : {auc:.4f}")
    print(f"  Best threshold          : {best_t:.2f}  (weighted-F1={best_f1:.4f})")
    print(f"  Accuracy @ 0.50 (naive) : {((probs>=0.5).astype(int)==y_true).mean():.4f}")
    print(f"  Accuracy @ {best_t:.2f} (tuned) : {(calibrated_preds==y_true).mean():.4f}")
    print(f"\n{classification_report(y_true, calibrated_preds, target_names=['Chop','Breakout'])}")

    # Persist the tuned threshold alongside the model — inference must use it
    joblib.dump(
        {"feature_names": FEATURE_COLS, "scaler": None,
         "decision_threshold": best_t, "val_auc": auc},
        META_PATH
    )

    print(f"\n✅  Best  → {BEST_MODEL_PATH}  (vaL={best_val:.4f})")
    print(f"✅  Final → {FINAL_MODEL_PATH}")
    print(f"✅  Meta  → {META_PATH}  (includes tuned decision_threshold={best_t:.2f})")
    print(f"\n⚠️  Reminder: Model 3 outputs P(Chop) vs P(Breakout) — NOT direction.")
    print(f"   Use Model 1 + Model 2 to determine bullish vs bearish after breakout.")
    print(f"⚠️  At inference, classify Breakout when P(Breakout) >= {best_t:.2f}, NOT >= 0.5.")


if __name__ == "__main__":
    main()
