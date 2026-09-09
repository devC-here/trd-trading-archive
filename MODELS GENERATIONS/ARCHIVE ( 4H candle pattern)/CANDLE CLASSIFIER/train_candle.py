# ─────────────────────────────────────────────
# Train Candle Classifier (5-feature version)
# ─────────────────────────────────────────────

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib

# CONFIG
DATA_PATH   = "synthetic_candle_clf_dataset.csv"
MODEL_PATH  = "candle_model.pth"
SCALER_PATH = "candle_scaler.pkl"

BATCH_SIZE = 256
EPOCHS     = 30
LR         = 1e-3
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURES = [
    "wick_ratio",
    "body_ratio",
    "tail_ratio",
    "wick_vs_tail",
    "body_vs_wick"
]

LABEL_COL = "label_candle_type"


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class CandleDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class CandleNet(nn.Module):
    def __init__(self, input_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
def main():
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES].values
    Y = df[LABEL_COL].values

    # SCALE
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    joblib.dump({
        "scaler": scaler,
        "features": FEATURES
    }, SCALER_PATH)

    # SPLIT
    n = len(X)
    split = int(n * 0.85)

    Xtr, Xva = X[:split], X[split:]
    Ytr, Yva = Y[:split], Y[split:]

    train_loader = DataLoader(CandleDataset(Xtr, Ytr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(CandleDataset(Xva, Yva), batch_size=BATCH_SIZE)

    model = CandleNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    print("\nTraining...\n")

    for ep in range(EPOCHS):
        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            logits = model(xb)
            loss = loss_fn(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # VALIDATION
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                preds = model(xb).argmax(1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        acc = correct / total
        print(f"Epoch {ep+1:02d} | Loss={total_loss:.4f} | Val Acc={acc:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)

    print("\n✅ Model saved:", MODEL_PATH)
    print("✅ Scaler saved:", SCALER_PATH)


if __name__ == "__main__":
    main()