# train_momentum_multihead.py
import os
import random
import numpy as np
import pandas as pd
from collections import Counter
import joblib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

# Config
DATA_PATH = "synthetic_momentum_dataset.csv"
SCALER_PATH = "momentum_scaler.pkl"
MODEL_PATH = "momentum_multihead_final.pth"
BEST_MODEL_PATH = "momentum_multihead_best.pth"

BATCH_SIZE = 256
EPOCHS = 40
LR = 1e-4
WD = 1e-4
VAL_RATIO = 0.1
RNG = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HEAD_LOSS_WEIGHTS = [1.0, 0.9, 1.0]  # momentum_state, slope_zone, entry_bias

# ---------------- Dataset ----------------
class MomentumDataset(Dataset):
    def __init__(self, X, y1, y2, y3):
        self.x = torch.tensor(X.astype(np.float32))
        self.y1 = torch.tensor(y1.astype(np.int64))
        self.y2 = torch.tensor(y2.astype(np.int64))
        self.y3 = torch.tensor(y3.astype(np.int64))

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y1[idx], self.y2[idx], self.y3[idx]

# ---------------- Model ----------------
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

class MomentumMultiHeadNet(nn.Module):
    def __init__(self, input_dim, trunk_dim=192, drop=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, trunk_dim),
            nn.BatchNorm1d(trunk_dim),
            nn.GELU(),
            nn.Dropout(drop)
        )
        self.res = nn.Sequential(ResidualBlock(trunk_dim, drop), ResidualBlock(trunk_dim, drop))
        self.mid = nn.Sequential(
            nn.Linear(trunk_dim, trunk_dim//2),
            nn.BatchNorm1d(trunk_dim//2),
            nn.GELU(),
            nn.Dropout(drop)
        )
        self.head_momentum = nn.Linear(trunk_dim//2, 3)   # momentum_state
        self.head_slope = nn.Linear(trunk_dim//2, 3)      # slope_zone
        self.head_bias = nn.Linear(trunk_dim//2, 3)       # entry_bias

    def forward(self, x):
        h = self.stem(x)
        h = self.res(h)
        h = self.mid(h)
        return self.head_momentum(h), self.head_slope(h), self.head_bias(h)

# ---------------- Helpers ----------------
def compute_class_weights(arr, n_classes):
    cnt = Counter(arr.astype(int).tolist())
    total = sum(cnt.values())
    weights = []
    for k in range(n_classes):
        if cnt.get(k,0) > 0:
            weights.append(total / (cnt.get(k,0) * n_classes))
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
            y1 = y1.to(device); y2 = y2.to(device); y3 = y3.to(device)
            o1, o2, o3 = model(xb)
            l1 = loss_fns[0](o1, y1); l2 = loss_fns[1](o2, y2); l3 = loss_fns[2](o3, y3)
            loss = HEAD_LOSS_WEIGHTS[0]*l1 + HEAD_LOSS_WEIGHTS[1]*l2 + HEAD_LOSS_WEIGHTS[2]*l3
            tot_loss += loss.item() * xb.size(0); tot += xb.size(0)
            correct[0] += (o1.argmax(dim=1) == y1).sum().item()
            correct[1] += (o2.argmax(dim=1) == y2).sum().item()
            correct[2] += (o3.argmax(dim=1) == y3).sum().item()
            totals[0] += y1.size(0); totals[1] += y2.size(0); totals[2] += y3.size(0)
    avg_loss = tot_loss / tot if tot>0 else 0.0
    accs = [correct[i] / totals[i] if totals[i]>0 else 0.0 for i in range(3)]
    return avg_loss, accs

# ---------------- Main training ----------------
def main():
    torch.manual_seed(RNG); np.random.seed(RNG); random.seed(RNG)
    print("Loading", DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    expected = ["momentum_state","slope_zone","entry_bias"]
    for c in expected:
        if c not in df.columns:
            raise RuntimeError(f"Missing label {c}")

    Y1 = df["momentum_state"].astype(int).values
    Y2 = df["slope_zone"].astype(int).values
    Y3 = df["entry_bias"].astype(int).values

    feat_cols = [c for c in df.columns if c not in expected]
    X = df[feat_cols].fillna(0.0)

    # train/val split
    n = len(df)
    n_val = int(n * VAL_RATIO)
    idx = np.random.RandomState(RNG).permutation(n)
    tr_idx, va_idx = idx[:-n_val], idx[-n_val:]
    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    Y1tr, Y1va = Y1[tr_idx], Y1[va_idx]
    Y2tr, Y2va = Y2[tr_idx], Y2[va_idx]
    Y3tr, Y3va = Y3[tr_idx], Y3[va_idx]

    # scale
    scaler = RobustScaler()
    Xtr_scaled = scaler.fit_transform(Xtr.values)
    Xva_scaled = scaler.transform(Xva.values)
    joblib.dump({"scaler": scaler, "feature_names": feat_cols}, SCALER_PATH)
    print("Scaler saved ->", SCALER_PATH)

    # datasets
    dtr = MomentumDataset(Xtr_scaled, Y1tr, Y2tr, Y3tr)
    dva = MomentumDataset(Xva_scaled, Y1va, Y2va, Y3va)
    tr_loader = DataLoader(dtr, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    va_loader = DataLoader(dva, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # model
    model = MomentumMultiHeadNet(input_dim=Xtr_scaled.shape[1]).to(DEVICE)

    # losses & class weights
    w1 = compute_class_weights(Y1tr, 3).to(DEVICE)
    w2 = compute_class_weights(Y2tr, 3).to(DEVICE)
    w3 = compute_class_weights(Y3tr, 3).to(DEVICE)
    loss_fns = [nn.CrossEntropyLoss(weight=w1), nn.CrossEntropyLoss(weight=w2), nn.CrossEntropyLoss(weight=w3)]

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.6, patience=3)

    best_val = float("inf")
    for ep in range(1, EPOCHS+1):
        model.train()
        run_loss=0.0; run_n=0
        run_corr=[0,0,0]; run_tot=[0,0,0]
        for xb, y1, y2, y3 in tqdm(tr_loader, desc=f"Train {ep}/{EPOCHS}", leave=False):
            xb = xb.to(DEVICE); y1 = y1.to(DEVICE); y2 = y2.to(DEVICE); y3 = y3.to(DEVICE)
            opt.zero_grad()
            o1,o2,o3 = model(xb)
            l1 = loss_fns[0](o1,y1); l2 = loss_fns[1](o2,y2); l3 = loss_fns[2](o3,y3)
            loss = HEAD_LOSS_WEIGHTS[0]*l1 + HEAD_LOSS_WEIGHTS[1]*l2 + HEAD_LOSS_WEIGHTS[2]*l3
            loss.backward(); opt.step()
            run_loss += loss.item()*xb.size(0); run_n += xb.size(0)
            run_corr[0] += (o1.argmax(dim=1)==y1).sum().item()
            run_corr[1] += (o2.argmax(dim=1)==y2).sum().item()
            run_corr[2] += (o3.argmax(dim=1)==y3).sum().item()
            run_tot[0] += y1.size(0); run_tot[1] += y2.size(0); run_tot[2] += y3.size(0)
        train_loss = run_loss/run_n
        train_accs = [run_corr[i]/run_tot[i] for i in range(3)]

        val_loss, val_accs = epoch_eval(model, va_loader, loss_fns, DEVICE)
        sched.step(val_loss)

        print(f"Epoch {ep:02d} | TrainL {train_loss:.4f} | TrainAcc {train_accs} | ValL {val_loss:.4f} | ValAcc {val_accs}")

        if val_loss < best_val:
            best_val=val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print("Best model saved ->", BEST_MODEL_PATH)

    torch.save(model.state_dict(), MODEL_PATH)
    print("Final model saved ->", MODEL_PATH)

if __name__=="__main__":
    main()
