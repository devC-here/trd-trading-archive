
# train_vwap.py · v5
# Dataset : synthetic_vwap_dataset.csv
# Labels  : label_next_vwap_dir, label_next_vol_char, label_vwap_interaction,
#           label_next_inst_signal, label_next_close_location, label_vwap_follow_through
# Mirrors train_momentum/train_breakout v5:
#   - Dedicated deeper branch for vwap_dir (primary signal)
#   - Inverse-frequency class weights from actual data
#   - Sampler amplifies vwap_dir extremes + follow_through=confirmed
#   - Label smoothing 0.05, trunk 384, 3 res blocks, patience 15
#   - No subfolders; ASCII-only console; report written utf-8
# -----------------------------------------------------------------------
import json, random, time
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

# ── CONFIG ---------------------------------------------------------------
DATA_PATH        = "synthetic_vwap_dataset.csv"
SCALER_PATH      = "vwap_scaler.pkl"
BEST_MODEL_PATH  = "vwap_best.pth"
FINAL_MODEL_PATH = "vwap_final.pth"
META_PATH        = "vwap_metadata.json"
REPORT_PATH      = "vwap_report.txt"
IMPORTANCE_PATH  = "vwap_importance.csv"

BATCH_SIZE        = 256
EPOCHS            = 120
LR                = 2e-4
WD                = 1e-4
VAL_RATIO         = 0.12
TRUNK_DIM         = 384
DIR_BRANCH_DIM    = 192
DROP              = 0.20
SEED              = 42
PATIENCE          = 15
LABEL_SMOOTH      = 0.05
TRAIN_ON_METADATA = False

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── LABELS (label_ prefix) ----------------------------------------------
LABEL_COLS  = ["label_next_vwap_dir","label_next_vol_char","label_vwap_interaction",
               "label_next_inst_signal","label_next_close_location","label_vwap_follow_through"]
HEAD_SIZES  = [3, 3, 4, 3, 3, 3]
HEAD_NAMES  = ["vwap_dir","vol_char","vwap_interaction","inst_signal","close_loc","follow_through"]
# vwap_dir is the primary directional signal; follow_through confirms it
HEAD_WEIGHTS = [2.5, 0.8, 1.1, 0.9, 0.8, 1.5]

LABEL_MAPS = {
    "vwap_dir":         {0:"below_away",   1:"oscillate_near", 2:"above_away"},
    "vol_char":         {0:"declining",    1:"stable",         2:"surge"},
    "vwap_interaction": {0:"rejection",    1:"test_return",    2:"reclaim_hold", 3:"walk_extension"},
    "inst_signal":      {0:"distribution", 1:"neutral",        2:"accumulation"},
    "close_loc":        {0:"lower_third",  1:"mid",            2:"upper_third"},
    "follow_through":   {0:"no",           1:"partial",        2:"confirmed"},
}

# ── MODEL ----------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, d, drop):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d,d), nn.BatchNorm1d(d), nn.GELU(), nn.Dropout(drop),
            nn.Linear(d,d), nn.BatchNorm1d(d))
        self.act = nn.GELU()
    def forward(self, x): return self.act(self.net(x) + x)

class VwapNet(nn.Module):
    """
    Shared trunk -> 3 res blocks.
    vwap_dir: dedicated deeper branch (primary directional signal).
    Others: shared mid representation.
    """
    def __init__(self, input_dim, trunk=TRUNK_DIM, drop=DROP):
        super().__init__()
        h  = trunk // 2
        db = DIR_BRANCH_DIM
        self.stem = nn.Sequential(
            nn.Linear(input_dim, trunk), nn.BatchNorm1d(trunk),
            nn.GELU(), nn.Dropout(drop))
        self.res = nn.Sequential(
            ResBlock(trunk, drop), ResBlock(trunk, drop), ResBlock(trunk, drop))
        self.mid = nn.Sequential(
            nn.Linear(trunk, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(drop*0.6))
        self.dir_branch = nn.Sequential(
            nn.Linear(trunk, db), nn.BatchNorm1d(db), nn.GELU(), nn.Dropout(drop*0.6),
            ResBlock(db, drop*0.5),
            nn.Linear(db, db//2), nn.BatchNorm1d(db//2), nn.GELU(), nn.Dropout(drop*0.4))
        self.dir_head    = nn.Linear(db//2, HEAD_SIZES[0])
        self.other_heads = nn.ModuleList([nn.Linear(h, HEAD_SIZES[i])
                                          for i in range(1, len(HEAD_SIZES))])

    def forward(self, x):
        trunk   = self.res(self.stem(x))
        dir_out = self.dir_head(self.dir_branch(trunk))
        shared  = self.mid(trunk)
        return (dir_out,) + tuple(h(shared) for h in self.other_heads)

# ── DATASET -------------------------------------------------------------

class VwapDS(Dataset):
    def __init__(self, X, Ys):
        self.X  = torch.tensor(X, dtype=torch.float32)
        self.Ys = [torch.tensor(y, dtype=torch.long) for y in Ys]
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return (self.X[i],) + tuple(y[i] for y in self.Ys)

# ── INVERSE-FREQUENCY CLASS WEIGHTS ------------------------------------

def inv_freq_weights(y_array, n_classes, smooth=0.1):
    counts  = np.maximum(np.bincount(y_array, minlength=n_classes).astype(float), 1)
    weights = 1.0 / counts
    weights = weights / weights.mean()
    return torch.tensor((1-smooth)*weights + smooth, dtype=torch.float32)

# ── TRAIN / EVAL --------------------------------------------------------

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
    metrics = []
    for i in range(len(HEAD_SIZES)):
        acc  = sum(p==t for p,t in zip(all_preds[i],all_true[i])) / max(len(all_true[i]),1)
        bacc = balanced_accuracy_score(all_true[i], all_preds[i])
        mf1  = f1_score(all_true[i], all_preds[i], average="macro",   zero_division=0)
        wf1  = f1_score(all_true[i], all_preds[i], average="weighted", zero_division=0)
        metrics.append({"acc":acc,"bacc":bacc,"macro_f1":mf1,"weighted_f1":wf1})
    return tot_loss / n, metrics

# ── PERMUTATION IMPORTANCE (vwap_dir head) -------------------------------

def permutation_importance(model, Xva, Yva, feat_names, loss_fns, n_repeats=3):
    model.eval()
    Xt = torch.tensor(Xva, dtype=torch.float32).to(DEVICE)
    Yt = [torch.tensor(y, dtype=torch.long).to(DEVICE) for y in Yva]
    with torch.no_grad():
        outs = model(Xt)
        base = loss_fns[0](outs[0], Yt[0]).item()
    scores = {}
    for j, name in enumerate(feat_names):
        deltas = []
        for _ in range(n_repeats):
            Xp = Xt.clone()
            Xp[:, j] = Xp[torch.randperm(Xp.size(0)), j]
            with torch.no_grad():
                deltas.append(loss_fns[0](model(Xp)[0], Yt[0]).item() - base)
        scores[name] = float(np.mean(deltas))
    return sorted(scores.items(), key=lambda x: -x[1])

# ── MAIN ----------------------------------------------------------------

def main():
    t0 = time.time()
    print(f"Device: {DEVICE}")

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded: {DATA_PATH}  shape={df.shape}")
    for c in LABEL_COLS:
        if c not in df.columns:
            raise RuntimeError(f"Missing label column: '{c}'")

    meta_cols = [c for c in df.columns if c.startswith("meta_")]
    id_cols   = ["condition","future_condition"] + meta_cols
    exclude   = set(LABEL_COLS + id_cols)
    feat_cols = [c for c in df.columns if c not in exclude]
    if TRAIN_ON_METADATA:
        feat_cols += [c for c in meta_cols if pd.api.types.is_numeric_dtype(df[c])]

    X  = df[feat_cols].fillna(0.0).values.astype(np.float32)
    Ys = [df[c].astype(int).values for c in LABEL_COLS]
    print(f"Features: {len(feat_cols)}  Samples: {len(df)}  "
          f"Meta: {len(meta_cols)} ({'in' if TRAIN_ON_METADATA else 'ex'}cluded)")

    n = len(df); n_val = int(n * VAL_RATIO)
    perm = np.random.RandomState(SEED).permutation(n)
    tr_i, va_i = perm[:-n_val], perm[-n_val:]
    Xtr, Xva = X[tr_i], X[va_i]
    Ytr = [y[tr_i] for y in Ys]; Yva = [y[va_i] for y in Ys]

    scaler = RobustScaler()
    Xtr_s  = scaler.fit_transform(Xtr); Xva_s = scaler.transform(Xva)
    joblib.dump({"scaler":scaler,"feature_names":feat_cols,
                 "train_on_metadata":TRAIN_ON_METADATA}, SCALER_PATH)
    print(f"Scaler -> {SCALER_PATH}")

    # ── class weights from actual data ----------------------------------
    class_weights = [inv_freq_weights(Ytr[i], HEAD_SIZES[i]) for i in range(len(HEAD_SIZES))]
    # vwap_dir: amplify above_away/below_away over oscillate_near
    class_weights[0] = class_weights[0] * torch.tensor([1.4, 0.4, 1.4])
    class_weights[0] = class_weights[0] / class_weights[0].mean()
    # follow_through: amplify confirmed(2)
    class_weights[5] = class_weights[5] * torch.tensor([0.7, 1.0, 1.8])
    class_weights[5] = class_weights[5] / class_weights[5].mean()

    # ── sampler: vwap_dir extremes 3x + follow_through confirmed 2.5x ----
    dir_sw    = {0:3.0, 1:0.3, 2:3.0}
    follow_sw = {0:0.6, 1:1.0, 2:2.5}
    sw = np.array([dir_sw[Ytr[0][i]] * follow_sw[Ytr[5][i]]
                   for i in range(len(Ytr[0]))], dtype=np.float32)
    sampler   = WeightedRandomSampler(sw, len(sw), replacement=True)
    tr_loader = DataLoader(VwapDS(Xtr_s,Ytr), BATCH_SIZE, sampler=sampler, num_workers=0)
    va_loader = DataLoader(VwapDS(Xva_s,Yva), BATCH_SIZE, shuffle=False,  num_workers=0)

    model    = VwapNet(len(feat_cols)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model : {n_params:,} params")

    loss_fns = [nn.CrossEntropyLoss(weight=class_weights[i].to(DEVICE),
                                    label_smoothing=LABEL_SMOOTH)
                for i in range(len(HEAD_SIZES))]

    print("\nLabel distributions (train):")
    for i, lc in enumerate(LABEL_COLS):
        dist = dict(sorted(Counter(Ytr[i].tolist()).items()))
        cw   = [round(v.item(),2) for v in class_weights[i]]
        print(f"  {HEAD_NAMES[i]:16}: {dist}  cw={cw}")

    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR*4, steps_per_epoch=len(tr_loader),
        epochs=EPOCHS, pct_start=0.08, anneal_strategy="cos")

    best_val = float("inf"); patience_ctr = 0; best_ep = 0
    best_va_metrics = None

    print(f"\nTraining  epochs={EPOCHS}  patience={PATIENCE}  trunk={TRUNK_DIM}")
    print(f"{'Ep':>4} | {'TrL':>7} {'VaL':>7} | "
          + "  ".join(f"{n[:6]:>9}" for n in HEAD_NAMES))
    print("-" * 100)

    for ep in range(1, EPOCHS+1):
        tr_loss, _       = run_epoch(model, tr_loader, loss_fns, opt, sched)
        va_loss, va_mets = run_epoch(model, va_loader, loss_fns)
        saved = ""
        if va_loss < best_val:
            best_val = va_loss; best_ep = ep; patience_ctr = 0
            best_va_metrics = va_mets
            torch.save(model.state_dict(), BEST_MODEL_PATH); saved = " *best*"
        else:
            patience_ctr += 1
        if ep % 5 == 0 or ep == 1 or saved:
            mstr = "  ".join(f"{m['acc']:.3f}/{m['bacc']:.3f}" for m in va_mets)
            print(f"{ep:4d} | {tr_loss:7.4f} {va_loss:7.4f} | {mstr}{saved}")
        if patience_ctr >= PATIENCE:
            print(f"\nEarly stop ep {ep}")
            break

    torch.save(model.state_dict(), FINAL_MODEL_PATH)

    # ── feature importance (vwap_dir head) ------------------------------
    print("\nPermutation importance (vwap_dir head)...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE,
                                     weights_only=True))
    imp = permutation_importance(model, Xva_s, Yva, feat_cols, loss_fns)
    pd.DataFrame(imp[:30], columns=["feature","delta_vwapdir_loss"]).to_csv(
        IMPORTANCE_PATH, index=False)
    print("Top-10:")
    for name, score in imp[:10]:
        print(f"  {name:<38} {score:+.4f}")

    # ── report ---------------------------------------------------------
    dur = time.time() - t0
    sep = "=" * 60
    lines = [
        "VWAP Model - Training Report", sep,
        f"Date     : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Dataset  : {DATA_PATH}  shape={df.shape}",
        f"Features : {len(feat_cols)}  (meta {'in' if TRAIN_ON_METADATA else 'ex'}cluded)",
        f"Split    : train={len(tr_i)}  val={len(va_i)}",
        f"Params   : {n_params:,}",
        f"Best ep  : {best_ep}  val_loss={best_val:.4f}",
        f"Duration : {dur/60:.1f} min", "",
        f"{'Head':<18} {'Acc':>7} {'BalAcc':>8} {'MacroF1':>9} {'WtdF1':>8}",
        "-" * 54,
    ]
    if best_va_metrics:
        for i, m in enumerate(best_va_metrics):
            lines.append(f"{HEAD_NAMES[i]:<18} {m['acc']:7.3f} {m['bacc']:8.3f} "
                         f"{m['macro_f1']:9.3f} {m['weighted_f1']:8.3f}")
    lines += ["", "Top-30 Feature Importances (vwap_dir head permutation):"]
    for name, score in imp[:30]:
        lines.append(f"  {name:<38} {score:+.4f}")
    report_text = "\n".join(lines)
    with open(REPORT_PATH, "w", encoding="utf-8") as f: f.write(report_text)
    print(f"\n{report_text}")

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "model":"vwap","version":"v5",
            "date":datetime.now().isoformat(),
            "dataset":DATA_PATH,"dataset_shape":list(df.shape),
            "feature_count":len(feat_cols),"feature_names":feat_cols,
            "label_cols":LABEL_COLS,"head_names":HEAD_NAMES,"head_sizes":HEAD_SIZES,
            "label_maps":LABEL_MAPS,"train_on_metadata":TRAIN_ON_METADATA,
            "best_epoch":best_ep,"best_val_loss":round(best_val,5),
            "n_params":n_params,"trunk_dim":TRUNK_DIM,
            "class_counts":{LABEL_COLS[i]:dict(sorted(Counter(Ys[i].tolist()).items()))
                            for i in range(len(LABEL_COLS))},
        }, f, indent=2)

    print(f"\nSaved: {BEST_MODEL_PATH}  {FINAL_MODEL_PATH}  {SCALER_PATH}")
    print(f"       {META_PATH}  {REPORT_PATH}  {IMPORTANCE_PATH}")

# ── INFERENCE INTERFACE -------------------------------------------------

def predict(model, scaler_bundle, raw_features: dict) -> list:
    """Standard ensemble format: [{head_name, prediction, confidence, probabilities}]"""
    scaler = scaler_bundle["scaler"]
    fnames = scaler_bundle["feature_names"]
    x  = np.array([[raw_features.get(f, 0.0) for f in fnames]], dtype=np.float32)
    xt = torch.tensor(scaler.transform(x)).to(DEVICE)
    model.eval()
    with torch.no_grad():
        outs = model(xt)
    return [{"head_name": HEAD_NAMES[i],
             "prediction": int(outs[i].argmax(1).item()),
             "confidence": round(float(torch.softmax(outs[i],1).max().item()), 4),
             "probabilities": [round(p,4) for p in torch.softmax(outs[i],1).squeeze().tolist()]}
            for i in range(len(HEAD_SIZES))]


if __name__ == "__main__":
    main()