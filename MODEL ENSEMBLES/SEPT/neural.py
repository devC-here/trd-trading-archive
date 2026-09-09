import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
import joblib
from tqdm import tqdm
from collections import Counter

# Paths
DATA_PATH = "synthetic_candle_ratios.csv"  # Updated path for new dataset
SCALER_PATH = "candle_ratios_scaler.pkl"
MODEL_PATH = "candle_ratios_model.pth"
BEST_MODEL_PATH = "candle_ratios_model_best.pth"

# Hyperparameters
BATCH_SIZE = 256
EPOCHS = 40
LR = 1e-4
WD = 1e-4
VAL_RATIO = 0.1
RNG = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Target
TARGETS = {
    "target": 3  # Target values: -1 (below EMA), 0 (neutral), 1 (above EMA)
}

class CandleRatiosDS(Dataset):
    def __init__(self, X, Y):
        self.x = torch.tensor(X.values.astype(np.float32))
        self.y = torch.tensor(Y.values + 1, dtype=torch.long)  # Shift targets from [-1, 0, 1] to [0, 1, 2]
        
    def __len__(self): 
        return len(self.x)
    
    def __getitem__(self, i): 
        return self.x[i], self.y[i]

class ResidualBlock(nn.Module):
    def __init__(self, dim, drop=0.35):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(drop),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim)
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.seq(x) + x)

class CandleRatiosNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, 512), 
            nn.BatchNorm1d(512), 
            nn.GELU(), 
            nn.Dropout(0.35)
        )
        self.blocks = nn.Sequential(
            ResidualBlock(512, 0.35), 
            ResidualBlock(512, 0.35), 
            ResidualBlock(512, 0.35)
        )
        self.mid = nn.Sequential(
            nn.Linear(512, 256), 
            nn.BatchNorm1d(256), 
            nn.GELU(), 
            nn.Dropout(0.30)
        )
        self.head = nn.Linear(256, 3)  # For the target classes (0, 1, -1 shifted to 0, 1, 2)

    def forward(self, x):
        h = self.stem(x)
        h = self.blocks(h)
        h = self.mid(h)
        return self.head(h)

def class_weights(series, n):
    cnt = Counter(series.astype(int).tolist())
    tot = sum(cnt.values())
    w = []
    for k in range(n):
        w.append(tot / (cnt.get(k, 0) * n) if cnt.get(k, 0) > 0 else 1.0)
    return torch.tensor(w, dtype=torch.float32)

# Loading data
print(f"Loading {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
Y = df["target"]
X = df.drop(columns=["target"])
feat_cols = sorted(X.columns.tolist())
X = X[feat_cols]

print(f"Features {X.shape}, Target {Y.shape}")

# Scaling
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X.values)
joblib.dump({"scaler": scaler, "feature_names": feat_cols}, SCALER_PATH)
print(f"Scaler saved to {SCALER_PATH}")

# Splitting data into train and validation sets
n = len(df)
n_val = int(n * VAL_RATIO)
perm = np.random.RandomState(RNG).permutation(n)
tr_idx, va_idx = perm[:-n_val], perm[-n_val:]
Xtr, Xva = X_scaled[tr_idx], X_scaled[va_idx]
Ytr, Yva = Y.iloc[tr_idx], Y.iloc[va_idx]

dtr = CandleRatiosDS(pd.DataFrame(Xtr, columns=feat_cols), Ytr)
dva = CandleRatiosDS(pd.DataFrame(Xva, columns=feat_cols), Yva)
tr_loader = DataLoader(dtr, batch_size=BATCH_SIZE, shuffle=True)
va_loader = DataLoader(dva, batch_size=BATCH_SIZE, shuffle=False)

# Model setup
model = CandleRatiosNet(input_dim=X_scaled.shape[1]).to(DEVICE)

# Loss function
w = class_weights(Y + 1, 3).to(DEVICE)  # Adjust class weights for shifted targets
loss_fn = nn.CrossEntropyLoss(weight=w)

# Optimizer and scheduler
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.6, patience=3)

best_val_loss = float("inf")

# Training loop
for ep in range(1, EPOCHS + 1):
    model.train()
    tr_loss = 0
    tr_corr = 0
    tr_tot = 0
    
    for xb, ys in tqdm(tr_loader, desc=f"Train {ep}/{EPOCHS}", leave=False):
        xb = xb.to(DEVICE)
        ys = ys.to(DEVICE)

        opt.zero_grad()
        outs = model(xb)
        
        loss = loss_fn(outs, ys)
        loss.backward()
        opt.step()

        tr_loss += loss.item() * xb.size(0)
        tr_corr += (outs.argmax(1) == ys).sum().item()
        tr_tot += ys.size(0)

    # Train metrics
    trL = tr_loss / len(dtr)
    trA = tr_corr / tr_tot

    # Validation loop
    model.eval()
    va_loss = 0
    va_corr = 0
    va_tot = 0
    with torch.no_grad():
        for xb, ys in tqdm(va_loader, desc=f"Val {ep}/{EPOCHS}", leave=False):
            xb = xb.to(DEVICE)
            ys = ys.to(DEVICE)

            outs = model(xb)
            loss = loss_fn(outs, ys)
            va_loss += loss.item() * xb.size(0)
            va_corr += (outs.argmax(1) == ys).sum().item()
            va_tot += ys.size(0)

    # Validation metrics
    vaL = va_loss / len(dva)
    vaA = va_corr / va_tot

    sched.step(vaL)

    print(f"Epoch {ep:02d} | Train Loss {trL:.4f} | Train Acc {trA:.4f} | Val Loss {vaL:.4f} | Val Acc {vaA:.4f}")

    if vaL < best_val_loss:
        best_val_loss = vaL
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"Best model updated -> {BEST_MODEL_PATH}")

# Save the final model
torch.save(model.state_dict(), MODEL_PATH)
print(f"Final model saved -> {MODEL_PATH}")
