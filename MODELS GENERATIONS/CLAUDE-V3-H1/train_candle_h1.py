# train_candle_h1.py  ·  v4  — H1 recalibrated
# Dataset : synthetic_candle_h1_dataset.csv  (20000 × 36)
# Features: 30  |  Labels: 6 heads (3-bar path-aware labels)
#
# H1 changes vs v3:
#   • DATA_PATH → h1 dataset
#   • label_next_pattern: pin(1) now only ~2% on H1 → extreme upweight (cap at 3.5)
#   • label_seq_continuation: reversal(0) at 12% → upweight; indecision(1) 54% → downweight
#   • label_next_close_grav: bimodal distribution — mid(1) now ~28% (was 4%)
#     close_loc mid class no longer needs extreme upweight
#   • Sampler: direction near-equal, focus on reversal+continuation seq_cont
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
DATA_PATH        = "synthetic_candle_h1_dataset.csv"
SCALER_PATH      = "candle_h1_scaler.pkl"
BEST_MODEL_PATH  = "candle_h1_best.pth"
FINAL_MODEL_PATH = "candle_h1_final.pth"

BATCH_SIZE = 256
EPOCHS     = 80
LR         = 1e-3
WD         = 1e-4
VAL_RATIO  = 0.12
TRUNK_DIM  = 256
DROP       = 0.25
SEED       = 42

# ── Labels — H1 3-bar path-aware ────────────────────────────────────
# label_next_direction   : 0=bearish   1=doji/flat   2=bullish        → 3 cls
# label_next_body_size   : 0=small     1=normal      2=large          → 3 cls
# label_next_pattern     : 0=impulse   1=pin         2=inside_doji 3=engulf → 4 cls
# label_seq_continuation : 0=reversal  1=indecision  2=continuation   → 3 cls
# label_next_close_grav  : 0=lower     1=mid         2=upper          → 3 cls
# label_next_volatility  : 0=compress  1=normal      2=expansion      → 3 cls
#
# H1 measured distributions:
#   direction:   30% / 34% / 36%   — near equal
#   body_size:   34% / 42% / 24%   — large sparse
#   pattern:     46% /  2% / 24% / 29%  — pin VERY sparse!
#   seq_cont:    12% / 54% / 32%   — reversal sparse
#   close_grav:  34% / 28% / 38%   — more balanced than M15
#   volatility:  31% / 43% / 27%   — near equal
LABEL_COLS = [
    "label_next_direction",
    "label_next_body_size",
    "label_next_pattern",
    "label_seq_continuation",
    "label_next_close_grav",
    "label_next_volatility",
]
HEAD_SIZES = [3, 3, 4, 3, 3, 3]

# Head loss weights — direction + seq_continuation primary
HEAD_WEIGHTS = [1.5, 1.0, 1.2, 1.4, 0.9, 1.0]

# Class weights — inverse-frequency calibrated
# direction:   near-equal → light
# body_size:   large(2) 24% → upweight
# pattern:     pin(1) 2% → cap at 3.5; impulse(0) 46% → downweight;
#              inside_doji(2) 24%, engulf(3) 29% → moderate upweight
# seq_cont:    reversal(0) 12% → strong upweight; indecision(1) 54% → downweight
# close_grav:  lower(0) 34%, mid(1) 28%, upper(2) 38% — near balanced now
# volatility:  near-equal → light
CLASS_WEIGHTS = [
    torch.tensor([1.10, 0.97, 0.92], dtype=torch.float32),           # direction
    torch.tensor([0.93, 0.77, 1.30], dtype=torch.float32),           # body_size
    torch.tensor([0.55, 3.50, 0.98, 0.87], dtype=torch.float32),    # pattern (pin capped at 3.5)
    torch.tensor([1.86, 0.44, 0.70], dtype=torch.float32),           # seq_continuation
    torch.tensor([0.96, 1.19, 0.85], dtype=torch.float32),           # close_grav (balanced)
    torch.tensor([1.05, 0.75, 1.20], dtype=torch.float32),           # volatility
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


class CandleNet(nn.Module):
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

class CandleDS(Dataset):
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

    # Sampler: direction near-equal, moderate weights
    # seq_continuation: oversample reversal(0) heavily (12% in data)
    # pattern: oversample pin(1) rows (only 2%!) — otherwise model ignores pins entirely
    dir_w  = {0: 1.4, 1: 0.55, 2: 1.4}
    cont_w = {0: 2.5, 1: 0.45, 2: 1.2}    # reversal rare → aggressively oversample
    pin_w  = {0: 0.7, 1: 4.0, 2: 1.2, 3: 1.1}  # pin rows extremely rare → 4x
    sw = np.array([dir_w[Ytr[0][i]] * cont_w[Ytr[3][i]] * pin_w[Ytr[2][i]]
                   for i in range(len(Ytr[0]))], dtype=np.float32)
    sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)

    tr_loader = DataLoader(CandleDS(Xtr_s, Ytr), BATCH_SIZE, sampler=sampler, num_workers=0)
    va_loader = DataLoader(CandleDS(Xva_s, Yva), BATCH_SIZE, shuffle=False, num_workers=0)

    model    = CandleNet(input_dim=Xtr_s.shape[1]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    loss_fns = [nn.CrossEntropyLoss(weight=CLASS_WEIGHTS[i].to(DEVICE)) for i in range(6)]

    hs = ["dir", "body", "pat", "cont", "cg", "vol"]
    print(f"\nModel   : {n_params:,} params")
    print("\nLabel distributions (train):")
    for i, lc in enumerate(LABEL_COLS):
        cnt = dict(Counter(Ytr[i].tolist()))
        pct = {k: f"{100*v/len(Ytr[i]):.1f}%" for k, v in cnt.items()}
        print(f"  [{hs[i]:5}] {lc}: {pct}")

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
