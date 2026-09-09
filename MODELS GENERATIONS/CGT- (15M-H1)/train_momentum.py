# train_momentum.py · v4
# Dataset : synthetic_momentum_dataset.csv
# Labels  : next_direction, next_magnitude, next_regime_behaviour,
#           next_volatility, next_structural_bias, next_momentum_quality
# New     : TRAIN_ON_METADATA flag, balanced_accuracy + F1 validation,
#           feature importance (permutation), confidence export,
#           early stopping, full checkpoint + metadata JSON
# ─────────────────────────────────────────────────────────────────────
import os, json, random, time
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import balanced_accuracy_score, f1_score
import joblib

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────
DATA_PATH        = "synthetic_momentum_dataset.csv"
OUT_DIR          = "momentum_model"
SCALER_PATH      = f"{OUT_DIR}/scaler.pkl"
BEST_MODEL_PATH  = f"{OUT_DIR}/best.pth"
FINAL_MODEL_PATH = f"{OUT_DIR}/final.pth"
META_PATH        = f"{OUT_DIR}/metadata.json"
REPORT_PATH      = f"{OUT_DIR}/report.txt"
IMPORTANCE_PATH  = f"{OUT_DIR}/feature_importance.csv"

BATCH_SIZE       = 512
EPOCHS           = 80
LR               = 3e-4
WD               = 1e-4
VAL_RATIO        = 0.10
TRUNK_DIM        = 256
DROP             = 0.25
SEED             = 42
PATIENCE         = 10          # early stopping
TRAIN_ON_METADATA = False      # set True to include meta_* columns in training

os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Labels (no label_ prefix in momentum dataset) ───────────────────
LABEL_COLS  = ["next_direction","next_magnitude","next_regime_behaviour",
               "next_volatility","next_structural_bias","next_momentum_quality"]
HEAD_SIZES  = [3, 3, 4, 3, 3, 3]
HEAD_NAMES  = ["direction","magnitude","regime","volatility","struct_bias","mom_quality"]
HEAD_WEIGHTS= [1.5, 1.0, 1.3, 0.9, 1.1, 1.2]
# direction: upweight bear/bull; magnitude: upweight large; regime: upweight cont/breakout
CLASS_WEIGHTS = [
    torch.tensor([1.6, 0.5, 1.6]),
    torch.tensor([0.9, 1.0, 2.0]),
    torch.tensor([0.7, 1.2, 1.5, 2.2]),
    torch.tensor([1.0, 0.9, 1.8]),
    torch.tensor([1.4, 0.5, 1.4]),
    torch.tensor([0.7, 1.0, 1.7]),
]
LABEL_MAPS = {
    "direction":    {0:"bearish",1:"flat",2:"bullish"},
    "magnitude":    {0:"small",1:"medium",2:"large"},
    "regime":       {0:"chop",1:"pullback",2:"continuation",3:"breakout"},
    "volatility":   {0:"contracting",1:"normal",2:"expanding"},
    "struct_bias":  {0:"bearish",1:"neutral",2:"bullish"},
    "mom_quality":  {0:"fading",1:"stable",2:"accelerating"},
}

# ─────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, d, drop):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,d), nn.BatchNorm1d(d), nn.GELU(),
                                 nn.Dropout(drop), nn.Linear(d,d), nn.BatchNorm1d(d))
        self.act = nn.GELU()
    def forward(self, x): return self.act(self.net(x) + x)

class MomentumNet(nn.Module):
    def __init__(self, input_dim, trunk=TRUNK_DIM, drop=DROP):
        super().__init__()
        h = trunk // 2
        self.stem = nn.Sequential(nn.Linear(input_dim,trunk), nn.BatchNorm1d(trunk),
                                  nn.GELU(), nn.Dropout(drop))
        self.res  = nn.Sequential(ResBlock(trunk,drop), ResBlock(trunk,drop))
        self.mid  = nn.Sequential(nn.Linear(trunk,h), nn.BatchNorm1d(h),
                                  nn.GELU(), nn.Dropout(drop*0.7))
        self.heads = nn.ModuleList([nn.Linear(h, s) for s in HEAD_SIZES])
    def forward(self, x):
        h = self.mid(self.res(self.stem(x)))
        return tuple(head(h) for head in self.heads)

# ─────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────

class MomentumDS(Dataset):
    def __init__(self, X, Ys):
        self.X  = torch.tensor(X, dtype=torch.float32)
        self.Ys = [torch.tensor(y, dtype=torch.long) for y in Ys]
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return (self.X[i],) + tuple(y[i] for y in self.Ys)

# ─────────────────────────────────────────────────────────────────────
# TRAIN / EVAL
# ─────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, loss_fns, opt=None, sched=None):
    train = opt is not None
    model.train() if train else model.eval()
    tot_loss = 0.0; n = 0
    all_preds = [[] for _ in HEAD_SIZES]
    all_true  = [[] for _ in HEAD_SIZES]
    with torch.set_grad_enabled(train):
        for batch in loader:
            xb = batch[0].to(DEVICE)
            ys = [b.to(DEVICE) for b in batch[1:]]
            outs   = model(xb)
            losses = [loss_fns[i](outs[i], ys[i]) for i in range(len(HEAD_SIZES))]
            loss   = sum(HEAD_WEIGHTS[i] * losses[i] for i in range(len(HEAD_SIZES)))
            if train:
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if sched: sched.step()
            tot_loss += loss.item() * xb.size(0); n += xb.size(0)
            for i in range(len(HEAD_SIZES)):
                all_preds[i].extend(outs[i].argmax(1).cpu().tolist())
                all_true[i].extend(ys[i].cpu().tolist())
    # per-head metrics
    metrics = []
    for i in range(len(HEAD_SIZES)):
        acc  = sum(p==t for p,t in zip(all_preds[i],all_true[i])) / len(all_true[i])
        bacc = balanced_accuracy_score(all_true[i], all_preds[i])
        mf1  = f1_score(all_true[i], all_preds[i], average="macro",  zero_division=0)
        wf1  = f1_score(all_true[i], all_preds[i], average="weighted",zero_division=0)
        metrics.append({"acc":acc,"bacc":bacc,"macro_f1":mf1,"weighted_f1":wf1,
                         "preds":all_preds[i],"true":all_true[i]})
    return tot_loss / n, metrics

# ─────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE  (permutation on validation set)
# ─────────────────────────────────────────────────────────────────────

def permutation_importance(model, Xva, Yva, feat_names, loss_fns, n_repeats=3):
    model.eval()
    Xt = torch.tensor(Xva, dtype=torch.float32).to(DEVICE)
    Yt = [torch.tensor(y, dtype=torch.long).to(DEVICE) for y in Yva]
    with torch.no_grad():
        outs = model(Xt)
        base_loss = sum(HEAD_WEIGHTS[i]*loss_fns[i](outs[i],Yt[i])
                        for i in range(len(HEAD_SIZES))).item()
    scores = {}
    for j, name in enumerate(feat_names):
        deltas = []
        for _ in range(n_repeats):
            Xp = Xt.clone()
            idx = torch.randperm(Xp.size(0))
            Xp[:, j] = Xp[idx, j]
            with torch.no_grad():
                outs_p = model(Xp)
                pl = sum(HEAD_WEIGHTS[i]*loss_fns[i](outs_p[i],Yt[i])
                         for i in range(len(HEAD_SIZES))).item()
            deltas.append(pl - base_loss)
        scores[name] = float(np.mean(deltas))
    return sorted(scores.items(), key=lambda x: -x[1])

# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print(f"Device: {DEVICE}\n")

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded: {DATA_PATH}  shape={df.shape}")
    for c in LABEL_COLS:
        if c not in df.columns:
            raise RuntimeError(f"Missing label column: '{c}'")

    # ── column partitioning ──────────────────────────────────────────
    meta_cols = [c for c in df.columns if c.startswith("meta_")]
    id_cols   = ["condition","future_condition"] + meta_cols
    exclude   = set(LABEL_COLS + id_cols)
    feat_cols = [c for c in df.columns if c not in exclude]
    if TRAIN_ON_METADATA:
        numeric_meta = [c for c in meta_cols
                        if pd.api.types.is_numeric_dtype(df[c])]
        feat_cols = feat_cols + numeric_meta
        print(f"TRAIN_ON_METADATA=True  adding {len(numeric_meta)} meta cols")

    X  = df[feat_cols].fillna(0.0).values.astype(np.float32)
    Ys = [df[c].astype(int).values for c in LABEL_COLS]
    print(f"Features: {len(feat_cols)}  Samples: {len(df)}")
    print(f"Meta cols available: {len(meta_cols)}  "
          f"({'included' if TRAIN_ON_METADATA else 'excluded'})")

    # ── split ────────────────────────────────────────────────────────
    n     = len(df); n_val = int(n * VAL_RATIO)
    perm  = np.random.RandomState(SEED).permutation(n)
    tr_i, va_i = perm[:-n_val], perm[-n_val:]
    Xtr, Xva   = X[tr_i], X[va_i]
    Ytr = [y[tr_i] for y in Ys]; Yva = [y[va_i] for y in Ys]

    scaler = RobustScaler()
    Xtr_s  = scaler.fit_transform(Xtr); Xva_s = scaler.transform(Xva)
    joblib.dump({"scaler":scaler,"feature_names":feat_cols,
                 "train_on_metadata":TRAIN_ON_METADATA}, SCALER_PATH)

    # ── sampler: oversample directional + active-regime rows ─────────
    dir_w    = {0:1.6, 1:0.5, 2:1.6}
    regime_w = {0:0.6, 1:1.1, 2:1.8, 3:1.8}
    sw = np.array([dir_w[Ytr[0][i]] * regime_w[Ytr[2][i]]
                   for i in range(len(Ytr[0]))], dtype=np.float32)
    sampler   = WeightedRandomSampler(sw, len(sw), replacement=True)
    tr_loader = DataLoader(MomentumDS(Xtr_s,Ytr), BATCH_SIZE, sampler=sampler, num_workers=0)
    va_loader = DataLoader(MomentumDS(Xva_s,Yva), BATCH_SIZE, shuffle=False, num_workers=0)

    model    = MomentumNet(len(feat_cols)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model : {n_params:,} params\n")
    loss_fns = [nn.CrossEntropyLoss(weight=CLASS_WEIGHTS[i].to(DEVICE))
                for i in range(len(HEAD_SIZES))]

    print("Label distributions (train):")
    for i,lc in enumerate(LABEL_COLS):
        print(f"  {HEAD_NAMES[i]:15}: {dict(sorted(Counter(Ytr[i].tolist()).items()))}")

    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR*3, steps_per_epoch=len(tr_loader), epochs=EPOCHS, pct_start=0.1)

    best_val = float("inf"); patience_ctr = 0; best_ep = 0
    best_va_metrics = None
    print(f"\nTraining  epochs={EPOCHS}  patience={PATIENCE}\n{'─'*90}")
    header = f"{'Ep':>4} | {'TrL':>7} {'VaL':>7} | " + \
             "  ".join(f"{n[:5]:>5}acc/{n[:5]:>5}bac" for n in HEAD_NAMES)
    print(header)

    for ep in range(1, EPOCHS+1):
        tr_loss, _       = run_epoch(model, tr_loader, loss_fns, opt, sched)
        va_loss, va_mets = run_epoch(model, va_loader, loss_fns)
        saved = ""
        if va_loss < best_val:
            best_val = va_loss; best_ep = ep; patience_ctr = 0
            best_va_metrics = va_mets
            torch.save(model.state_dict(), BEST_MODEL_PATH); saved = " ←best"
        else:
            patience_ctr += 1
        if ep % 5 == 0 or ep == 1 or saved:
            mstr = "  ".join(f"{m['acc']:.3f}/{m['bacc']:.3f}" for m in va_mets)
            print(f"{ep:4d} | {tr_loss:7.4f} {va_loss:7.4f} | {mstr}{saved}")
        if patience_ctr >= PATIENCE:
            print(f"\nEarly stop at epoch {ep}  (no improvement for {PATIENCE} epochs)")
            break

    torch.save(model.state_dict(), FINAL_MODEL_PATH)

    # ── feature importance ───────────────────────────────────────────
    print("\nComputing permutation importance...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    imp = permutation_importance(model, Xva_s, Yva, feat_cols, loss_fns)
    imp_df = pd.DataFrame(imp[:30], columns=["feature","importance_delta_loss"])
    imp_df.to_csv(IMPORTANCE_PATH, index=False)
    print(f"Top-10 features:")
    for name, score in imp[:10]:
        print(f"  {name:<40} {score:+.4f}")

    # ── training report ──────────────────────────────────────────────
    dur = time.time() - t0
    lines = [
        f"Momentum Model Training Report",
        f"{'='*60}",
        f"Date        : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Dataset     : {DATA_PATH}  shape={df.shape}",
        f"Features    : {len(feat_cols)}  (metadata={'included' if TRAIN_ON_METADATA else 'excluded'})",
        f"Samples     : train={len(tr_i)}  val={len(va_i)}",
        f"Params      : {n_params:,}",
        f"Best epoch  : {best_ep}  val_loss={best_val:.4f}",
        f"Duration    : {dur/60:.1f} min",
        f"",
        f"Per-head validation metrics (best epoch):",
        f"{'Head':<18} {'Acc':>7} {'BalAcc':>8} {'MacroF1':>9} {'WtdF1':>8}",
        f"{'─'*50}",
    ]
    if best_va_metrics:
        for i, m in enumerate(best_va_metrics):
            lines.append(f"{HEAD_NAMES[i]:<18} {m['acc']:7.3f} {m['bacc']:8.3f} "
                         f"{m['macro_f1']:9.3f} {m['weighted_f1']:8.3f}")
    lines += ["", "Top-30 Feature Importances (permutation):"]
    for name, score in imp[:30]:
        lines.append(f"  {name:<40} {score:+.4f}")
    report_text = "\n".join(lines)
    with open(REPORT_PATH, "w") as f: f.write(report_text)
    print(f"\n{report_text}")

    # ── metadata JSON ────────────────────────────────────────────────
    meta_json = {
        "model": "momentum", "version": "v4",
        "date": datetime.now().isoformat(),
        "dataset": DATA_PATH, "dataset_shape": list(df.shape),
        "feature_count": len(feat_cols), "feature_names": feat_cols,
        "label_count": len(LABEL_COLS), "label_cols": LABEL_COLS,
        "head_names": HEAD_NAMES, "head_sizes": HEAD_SIZES,
        "label_maps": LABEL_MAPS,
        "train_on_metadata": TRAIN_ON_METADATA,
        "best_epoch": best_ep, "best_val_loss": round(best_val,5),
        "n_params": n_params, "trunk_dim": TRUNK_DIM,
        "class_counts": {LABEL_COLS[i]: dict(sorted(Counter(Ys[i].tolist()).items()))
                         for i in range(len(LABEL_COLS))},
    }
    with open(META_PATH,"w") as f: json.dump(meta_json, f, indent=2)
    print(f"\n✅  {BEST_MODEL_PATH}\n✅  {FINAL_MODEL_PATH}\n✅  {SCALER_PATH}")
    print(f"✅  {META_PATH}\n✅  {REPORT_PATH}\n✅  {IMPORTANCE_PATH}")

# ─────────────────────────────────────────────────────────────────────
# INFERENCE INTERFACE
# ─────────────────────────────────────────────────────────────────────

def predict(model, scaler_bundle, raw_features: dict) -> list:
    """
    Returns list of head predictions in standard ensemble format:
    [{"head_name":..., "prediction":..., "confidence":..., "probabilities":[...]}, ...]
    """
    scaler = scaler_bundle["scaler"]
    fnames = scaler_bundle["feature_names"]
    x = np.array([[raw_features.get(f, 0.0) for f in fnames]], dtype=np.float32)
    x = scaler.transform(x)
    xt = torch.tensor(x).to(DEVICE)
    model.eval()
    with torch.no_grad():
        outs = model(xt)
    results = []
    for i, out in enumerate(outs):
        probs = torch.softmax(out, dim=1).squeeze().cpu().tolist()
        pred  = int(np.argmax(probs))
        results.append({
            "head_name":    HEAD_NAMES[i],
            "prediction":   pred,
            "confidence":   round(float(max(probs)), 4),
            "probabilities": [round(p, 4) for p in probs],
        })
    return results


if __name__ == "__main__":
    main()
