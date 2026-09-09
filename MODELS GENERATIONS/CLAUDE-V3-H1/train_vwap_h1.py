# train_vwap_h1.py  ·  v4  — H1 recalibrated
# Dataset : synthetic_vwap_h1_dataset.csv  (20000 × 38)
# Features: 32  |  Labels: 6 heads (3-bar path-aware labels)
#
# H1 changes vs v3:
#   • DATA_PATH → h1 dataset
#   • CLASS_WEIGHTS recalibrated from measured H1 distributions
#   • vwap_interaction class2 (reclaim_hold ~1%) capped weight at 2.5
#   • Sampler rebalanced: vwap_dir + follow_through updated proportions
#   • EPOCHS 60 → 80 (H1 patterns more complex, GARCH vol needs more signal)
#   • Label close_location: mid class (class1) now ~14% — upweighted
#   • HEAD_SIZES unchanged [3,3,4,3,3,3]
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
DATA_PATH        = "synthetic_vwap_h1_dataset.csv"
SCALER_PATH      = "vwap_h1_scaler.pkl"
BEST_MODEL_PATH  = "vwap_h1_best.pth"
FINAL_MODEL_PATH = "vwap_h1_final.pth"

BATCH_SIZE = 256
EPOCHS     = 80
LR         = 1e-3
WD         = 1e-4
VAL_RATIO  = 0.12
TRUNK_DIM  = 256
DROP       = 0.25
SEED       = 42

# ── Labels — H1 3-bar path-aware ────────────────────────────────────
# label_next_vwap_dir       : 0=below_away  1=oscillate   2=above_away  → 3 cls
# label_next_vol_char       : 0=declining   1=stable      2=surge       → 3 cls
# label_vwap_interaction    : 0=rejection   1=test_return 2=reclaim 3=walk → 4 cls
# label_next_inst_signal    : 0=distribution 1=neutral    2=accumulation → 3 cls
# label_next_close_location : 0=lower_third  1=mid        2=upper_third  → 3 cls
# label_vwap_follow_through : 0=no           1=partial    2=confirmed    → 3 cls
#
# H1 measured distributions (2000 samples):
#   vwap_dir:    25% / 51% / 24%
#   vol_char:    33% / 43% / 24%
#   interaction: 30% / 51% /  1% / 19%   ← reclaim_hold rare but high-signal
#   inst_signal: 20% / 59% / 21%
#   close_loc:   43% / 14% / 43%         ← mid class thin, upweight
#   follow:      21% / 54% / 25%
LABEL_COLS = [
    "label_next_vwap_dir",
    "label_next_vol_char",
    "label_vwap_interaction",
    "label_next_inst_signal",
    "label_next_close_location",
    "label_vwap_follow_through",
]
HEAD_SIZES   = [3, 3, 4, 3, 3, 3]

# Head loss weights — vwap_dir + follow_through are primary trade signals
HEAD_WEIGHTS = [1.5, 1.0, 1.2, 1.1, 0.9, 1.5]

# Class weights — inverse-frequency calibrated, rare classes capped at 2.5
# vwap_dir:       near-equal → moderate adjustment
# vol_char:       surge(2) underrepresented → upweight
# interaction:    reclaim_hold(2) ~1% → cap at 2.5 (would be 3.7 uncapped)
# inst_signal:    neutral(1) dominant → downweight
# close_loc:      mid(1) ~14% → upweight heavily
# follow_through: no(0) slightly sparse → upweight
CLASS_WEIGHTS = [
    torch.tensor([1.20, 0.60, 1.25], dtype=torch.float32),          # vwap_dir
    torch.tensor([1.00, 0.75, 1.35], dtype=torch.float32),          # vol_char
    torch.tensor([1.00, 0.55, 2.50, 1.55], dtype=torch.float32),   # interaction (reclaim capped)
    torch.tensor([1.30, 0.45, 1.25], dtype=torch.float32),          # inst_signal
    torch.tensor([0.62, 1.80, 0.63], dtype=torch.float32),          # close_loc (mid upweighted)
    torch.tensor([1.35, 0.52, 1.15], dtype=torch.float32),          # follow_through
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────
# MODEL — unchanged architecture, H1 feature count same (32)
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


class VwapNet(nn.Module):
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

class VwapDS(Dataset):
    def __init__(self, X, Y):
        self.X  = torch.tensor(X.astype(np.float32))
        self.Ys = [torch.tensor(y.astype(np.int64)) for y in Y]

    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        return (self.X[i],) + tuple(y[i] for y in self.Ys)


def run_epoch(model, loader, loss_fns, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    tot_loss = 0.0; n = 0
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
            tot_loss += loss.item() * xb.size(0); n += xb.size(0)
            for i in range(6):
                correct[i] += (outs[i].argmax(1) == ys[i]).sum().item()
                totals[i]  += ys[i].size(0)

    return tot_loss / n, [correct[i] / totals[i] for i in range(6)]


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

    # Sampler: oversample directional vwap_dir (0/2) + confirmed follow (2)
    # H1 distribution is near-equal for vwap_dir — moderate weights
    # follow_through: confirmed(2)=25%, partial(1)=54%, no(0)=21%
    vdir_w   = {0: 1.5, 1: 0.5, 2: 1.5}   # upweight directional
    follow_w = {0: 1.2, 1: 0.7, 2: 1.8}   # upweight confirmed
    sw = np.array([vdir_w[Ytr[0][i]] * follow_w[Ytr[5][i]]
                   for i in range(len(Ytr[0]))], dtype=np.float32)
    sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)

    tr_loader = DataLoader(VwapDS(Xtr_s, Ytr), BATCH_SIZE, sampler=sampler, num_workers=0)
    va_loader = DataLoader(VwapDS(Xva_s, Yva), BATCH_SIZE, shuffle=False, num_workers=0)

    model    = VwapNet(input_dim=Xtr_s.shape[1]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    loss_fns = [nn.CrossEntropyLoss(weight=CLASS_WEIGHTS[i].to(DEVICE)) for i in range(6)]

    hs = ["vdir", "vol", "vint", "inst", "loc", "follow"]
    print(f"\nModel   : {n_params:,} params")
    print("\nLabel distributions (train):")
    for i, lc in enumerate(LABEL_COLS):
        cnt = dict(Counter(Ytr[i].tolist()))
        pct = {k: f"{100*v/len(Ytr[i]):.1f}%" for k,v in cnt.items()}
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
