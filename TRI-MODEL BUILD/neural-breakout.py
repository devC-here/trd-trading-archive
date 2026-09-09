# train_breakout_multihead.py
import os
import random
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import RobustScaler
import joblib
from tqdm import tqdm

# -------------------------
# Config
# -------------------------
DATA_PATH = "synthetic_breakout_dataset.csv"
SCALER_PATH = "breakout_scaler.pkl"
MODEL_PATH = "breakout_multihead_final.pth"
BEST_MODEL_PATH = "breakout_multihead_best.pth"

BATCH_SIZE = 256
EPOCHS = 40
LR = 1e-4
WD = 1e-4
VAL_RATIO = 0.1
RNG = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loss head weights (tuneable)
HEAD_LOSS_WEIGHTS = [1.0, 1.0, 0.9]  # vol_regime, breakout_bias, breakout_strength

# -------------------------
# Dataset wrapper
# -------------------------
class BreakoutDataset(Dataset):
    def __init__(self, X, y1, y2, y3):
        # X: numpy array or DataFrame
        if isinstance(X, pd.DataFrame):
            self.x = torch.tensor(X.values.astype(np.float32))
        else:
            self.x = torch.tensor(X.astype(np.float32))
        # targets must be integer class labels
        self.y1 = torch.tensor(y1.astype(np.int64))
        self.y2 = torch.tensor(y2.astype(np.int64))
        self.y3 = torch.tensor(y3.astype(np.int64))

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y1[idx], self.y2[idx], self.y3[idx]

# -------------------------
# Model
# -------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim, drop=0.25):
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

class BreakoutMultiHeadNet(nn.Module):
    def __init__(self, input_dim, trunk_dim=256, drop=0.25):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, trunk_dim),
            nn.BatchNorm1d(trunk_dim),
            nn.GELU(),
            nn.Dropout(drop)
        )
        self.res = nn.Sequential(
            ResidualBlock(trunk_dim, drop),
            ResidualBlock(trunk_dim, drop),
        )
        self.mid = nn.Sequential(
            nn.Linear(trunk_dim, trunk_dim//2),
            nn.BatchNorm1d(trunk_dim//2),
            nn.GELU(),
            nn.Dropout(drop)
        )
        # Heads
        self.head_vol = nn.Linear(trunk_dim//2, 3)    # vol_regime (3 classes)
        self.head_bias = nn.Linear(trunk_dim//2, 3)   # breakout_bias (3 classes)
        self.head_strength = nn.Linear(trunk_dim//2, 2) # breakout_strength (2 classes)

    def forward(self, x):
        h = self.stem(x)
        h = self.res(h)
        h = self.mid(h)
        return self.head_vol(h), self.head_bias(h), self.head_strength(h)

# -------------------------
# Helpers
# -------------------------
def compute_class_weights(arr, n_classes):
    cnt = Counter(arr.astype(int).tolist())
    total = sum(cnt.values())
    weights = []
    for k in range(n_classes):
        if cnt.get(k, 0) > 0:
            weights.append(total / (cnt.get(k, 0) * n_classes))
        else:
            weights.append(1.0)
    return torch.tensor(weights, dtype=torch.float32)

def epoch_eval(model, loader, loss_fns, device):
    model.eval()
    tot_loss = 0.0
    tot = 0
    correct = [0,0,0]
    totals = [0,0,0]
    with torch.no_grad():
        for xb, y1, y2, y3 in loader:
            xb = xb.to(device)
            y1 = y1.to(device)
            y2 = y2.to(device)
            y3 = y3.to(device)

            o1, o2, o3 = model(xb)
            l1 = loss_fns[0](o1, y1)
            l2 = loss_fns[1](o2, y2)
            l3 = loss_fns[2](o3, y3)

            loss = HEAD_LOSS_WEIGHTS[0]*l1 + HEAD_LOSS_WEIGHTS[1]*l2 + HEAD_LOSS_WEIGHTS[2]*l3
            tot_loss += loss.item() * xb.size(0)
            tot += xb.size(0)

            pred1 = o1.argmax(dim=1)
            pred2 = o2.argmax(dim=1)
            pred3 = o3.argmax(dim=1)

            correct[0] += (pred1 == y1).sum().item()
            correct[1] += (pred2 == y2).sum().item()
            correct[2] += (pred3 == y3).sum().item()

            totals[0] += y1.size(0)
            totals[1] += y2.size(0)
            totals[2] += y3.size(0)

    avg_loss = tot_loss / tot if tot>0 else 0.0
    accs = [correct[i] / totals[i] if totals[i]>0 else 0.0 for i in range(3)]
    return avg_loss, accs

# -------------------------
# Main
# -------------------------
def main():
    # reproducibility
    random_state = np.random.RandomState(RNG)
    torch.manual_seed(RNG)
    np.random.seed(RNG)
    random.seed(RNG)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RNG)

    print("Loading dataset:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    # Ensure expected label columns exist
    expected_labels = ["vol_regime", "breakout_bias", "breakout_strength"]
    for c in expected_labels:
        if c not in df.columns:
            raise RuntimeError(f"Missing label column {c} in dataset")

    # Separate labels
    Y1 = df["vol_regime"].astype(int).values
    Y2 = df["breakout_bias"].astype(int).values
    Y3 = df["breakout_strength"].astype(int).values

    # Features = all columns except the labels
    feat_cols = [c for c in df.columns if c not in expected_labels]
    X = df[feat_cols].fillna(0.0)
    print(f"Features shape: {X.shape}, Labels: {Y1.shape}")

    # Split train/val
    n = len(df)
    n_val = int(n * VAL_RATIO)
    idx = random_state.permutation(n)
    tr_idx, va_idx = idx[:-n_val], idx[-n_val:]
    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    Y1tr, Y1va = Y1[tr_idx], Y1[va_idx]
    Y2tr, Y2va = Y2[tr_idx], Y2[va_idx]
    Y3tr, Y3va = Y3[tr_idx], Y3[va_idx]

    # Scale
    scaler = RobustScaler()
    Xtr_scaled = scaler.fit_transform(Xtr.values)
    Xva_scaled = scaler.transform(Xva.values)
    joblib.dump({"scaler": scaler, "feature_names": feat_cols}, SCALER_PATH)
    print("Scaler saved ->", SCALER_PATH)

    # Datasets & loaders
    dtr = BreakoutDataset(Xtr_scaled, Y1tr, Y2tr, Y3tr)
    dva = BreakoutDataset(Xva_scaled, Y1va, Y2va, Y3va)
    tr_loader = DataLoader(dtr, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    va_loader = DataLoader(dva, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    # Model
    model = BreakoutMultiHeadNet(input_dim=Xtr_scaled.shape[1]).to(DEVICE)

    # Per-head weights & loss functions
    w1 = compute_class_weights(Y1tr, 3).to(DEVICE)
    w2 = compute_class_weights(Y2tr, 3).to(DEVICE)
    w3 = compute_class_weights(Y3tr, 2).to(DEVICE)
    loss_fns = [
        nn.CrossEntropyLoss(weight=w1),
        nn.CrossEntropyLoss(weight=w2),
        nn.CrossEntropyLoss(weight=w3)
    ]
    print("Class weights head1:", w1.cpu().numpy(), "head2:", w2.cpu().numpy(), "head3:", w3.cpu().numpy())

    # Optimizer & scheduler
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.6, patience=3)

    best_val = float("inf")

    # Training loop
    for ep in range(1, EPOCHS+1):
        model.train()
        running_loss = 0.0
        running_n = 0
        running_corr = [0,0,0]
        running_tot = [0,0,0]

        for xb, y1, y2, y3 in tqdm(tr_loader, desc=f"Train {ep}/{EPOCHS}", leave=False):
            xb = xb.to(DEVICE)
            y1 = y1.to(DEVICE)
            y2 = y2.to(DEVICE)
            y3 = y3.to(DEVICE)

            opt.zero_grad()
            o1, o2, o3 = model(xb)

            l1 = loss_fns[0](o1, y1)
            l2 = loss_fns[1](o2, y2)
            l3 = loss_fns[2](o3, y3)

            loss = HEAD_LOSS_WEIGHTS[0]*l1 + HEAD_LOSS_WEIGHTS[1]*l2 + HEAD_LOSS_WEIGHTS[2]*l3
            loss.backward()
            opt.step()

            running_loss += loss.item() * xb.size(0)
            running_n += xb.size(0)

            p1 = o1.argmax(dim=1)
            p2 = o2.argmax(dim=1)
            p3 = o3.argmax(dim=1)

            running_corr[0] += (p1 == y1).sum().item()
            running_corr[1] += (p2 == y2).sum().item()
            running_corr[2] += (p3 == y3).sum().item()

            running_tot[0] += y1.size(0)
            running_tot[1] += y2.size(0)
            running_tot[2] += y3.size(0)

        train_loss = running_loss / running_n
        train_accs = [running_corr[i] / running_tot[i] for i in range(3)]

        # Validation
        val_loss, val_accs = epoch_eval(model, va_loader, loss_fns, DEVICE)
        sched.step(val_loss)

        print(f"Epoch {ep:02d} | Train Loss {train_loss:.4f} | Train Accs {train_accs} | Val Loss {val_loss:.4f} | Val Accs {val_accs}")

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print("Best model saved ->", BEST_MODEL_PATH)

    # Final save
    torch.save(model.state_dict(), MODEL_PATH)
    print("Final model saved ->", MODEL_PATH)
    print("Training complete.")

if __name__ == "__main__":
    main()
