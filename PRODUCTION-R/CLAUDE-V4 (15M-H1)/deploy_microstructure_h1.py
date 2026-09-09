# deploy_microstructure_h1.py  v1
# Live inference for the Microstructure model
# Architecture and features match train_microstructure_h1.py + synthetic_microstructure_h1.py exactly
# Features : 37  |  Input dim : 64 (37 + W1×2 + W2×1)  |  Heads : 6  |  HEAD_SIZES : [3,3,3,3,3,3]
# ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

try:
    from utils import resource_path
except ImportError:
    def resource_path(p): return p

EPS      = 1e-10
LOOKBACK = 130
TRUNK    = 256
DROP     = 0.30


# ─────────────────────────────────────────────────────────────────────
# Architecture — must be byte-identical to train_microstructure_h1.py
# ─────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, d, drop=DROP):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d), nn.BatchNorm1d(d), nn.GELU(), nn.Dropout(drop),
            nn.Linear(d, d), nn.BatchNorm1d(d),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.net(x) + x)


class H1Net(nn.Module):
    N_W1 = 9; N_W2 = 9

    def __init__(self, n_feat, trunk=TRUNK, drop=DROP):
        super().__init__()
        n_in = n_feat + self.N_W1 * 2 + self.N_W2
        half = trunk // 2
        self.stem = nn.Sequential(
            nn.Linear(n_in, trunk), nn.BatchNorm1d(trunk), nn.GELU(), nn.Dropout(drop))
        self.res  = nn.Sequential(ResBlock(trunk, drop), ResBlock(trunk, drop))
        self.mid  = nn.Sequential(
            nn.Linear(trunk, half), nn.BatchNorm1d(half), nn.GELU(), nn.Dropout(drop * 0.7))
        self.heads = nn.ModuleList([nn.Linear(half, s) for s in [3, 3, 3, 3, 3, 3]])

    def forward(self, x):
        x = torch.cat([x,
                        x[:, :self.N_W1],
                        x[:, :self.N_W1],
                        x[:, self.N_W1:self.N_W1 + self.N_W2]], dim=1)
        h = self.mid(self.res(self.stem(x)))
        return tuple(hd(h) for hd in self.heads)


# ─────────────────────────────────────────────────────────────────────
# Feature computation — identical to synthetic_microstructure_h1.py v5
# ─────────────────────────────────────────────────────────────────────

def _atr(bars: list) -> float:
    trs = []
    for i, b in enumerate(bars):
        hl = b["high"] - b["low"]
        trs.append(hl if i == 0 else max(
            hl, abs(b["high"] - bars[i-1]["close"]), abs(b["low"] - bars[i-1]["close"])))
    return float(np.mean(trs)) + EPS

def _delta(bar: dict) -> float:
    return float((bar["close"] - bar["low"]) / (bar["high"] - bar["low"] + EPS) * 2 - 1)

def _energy(bar: dict, atr: float) -> float:
    return float(np.clip(
        (abs(bar["close"] - bar["open"]) / atr) * np.sqrt(bar["volume"] / 1000 + EPS), 0, 6))

def _win_feats(bars: list, atr: float, pfx: str) -> dict:
    n = len(bars)
    closes = [b["close"] for b in bars]; opens  = [b["open"]  for b in bars]
    vols   = [b["volume"] for b in bars]
    bodies = [abs(b["close"] - b["open"]) for b in bars]
    ranges = [b["high"] - b["low"] for b in bars]
    dirs   = [1 if b["close"] >= b["open"] else -1 for b in bars]
    uw     = [abs(max(b["open"], b["close"]) - b["high"]) for b in bars]
    lw     = [abs(min(b["open"], b["close"]) - b["low"])  for b in bars]
    mid = n // 2
    deltas  = [_delta(b) for b in bars]
    cum_d   = float(np.clip(sum(deltas) / n, -1, 1))
    dtrd    = float(np.clip(float(np.mean(deltas[mid:])) - float(np.mean(deltas[:mid])), -2, 2))
    avg_br  = float(np.mean([b / (r + EPS) for b, r in zip(bodies, ranges)]))
    tail_a  = float(np.clip((np.mean(lw) - np.mean(uw)) / (atr + EPS), -3, 3))
    avg_e   = float(np.mean([_energy(b, atr) for b in bars]))
    rej     = float(np.clip((uw[-1] + lw[-1]) / (bodies[-1] / atr + EPS), 0, 5))
    va = float(np.mean(vols)) + EPS; vz = float((vols[-1] - va) / (np.std(vols) + EPS))
    absorb  = float(np.clip(vz * (1 if bodies[-1] / (atr + EPS) < 0.25 else 0), 0, 3))
    path    = sum(abs(closes[i] - closes[i-1]) for i in range(1, n)) + EPS
    eff     = float(np.clip(abs(closes[-1] - closes[0]) / path, 0, 1))
    gaps    = [abs(opens[i] - closes[i-1]) / (atr + EPS) for i in range(1, n)]
    gdirs   = [1 if opens[i] > closes[i-1] else -1 for i in range(1, n)]
    gap_p   = float(np.clip(sum(g * d for g, d in zip(gaps, gdirs)), -3, 3))
    return {f"{pfx}_cumd": cum_d, f"{pfx}_dtrd": dtrd, f"{pfx}_br": avg_br,
            f"{pfx}_tail": tail_a, f"{pfx}_eng": avg_e, f"{pfx}_rej": rej,
            f"{pfx}_abs": absorb, f"{pfx}_eff": eff, f"{pfx}_gap": gap_p}

def compute_features(bars: list) -> dict:
    atr_bg = _atr(bars)
    f1 = _win_feats(bars[-4:],  atr_bg, "w1")
    f2 = _win_feats(bars[-12:], atr_bg, "w2")
    f3 = _win_feats(bars,       atr_bg, "w3")
    n1 = float(np.clip(f2["w2_cumd"] - f3["w3_cumd"], -2, 2))
    bg_e = float(np.mean([_energy(b, atr_bg) for b in bars])) + EPS
    w1_e = float(np.mean([_energy(b, atr_bg) for b in bars[-4:]])) + EPS
    n2   = float(np.clip(w1_e / bg_e, 0.2, 5.0))
    bg_br = float(np.mean([abs(b["close"] - b["open"]) / (b["high"] - b["low"] + EPS)
                            for b in bars])) + EPS
    n3   = float(np.clip(f1["w1_br"] / bg_br, 0.2, 4.0))
    n4   = float(np.clip(f1["w1_tail"] - f2["w2_tail"], -2, 2))
    deltas_bg = [_delta(b) for b in bars]
    cp = 0
    for d in reversed(deltas_bg):
        if d > 0: cp += 1
        else: break
    cn = 0
    for d in reversed(deltas_bg):
        if d < 0: cn += 1
        else: break
    n5   = float(np.clip((cp - cn) / 12, -2, 2))
    bg_vol = float(np.mean([b["volume"] for b in bars])) + EPS
    w1_vol = float(np.mean([b["volume"] for b in bars[-4:]])) + EPS
    n6   = float(np.clip(w1_vol / bg_vol, 0.2, 5.0))
    # magnitude + momentum helpers
    atr_w1 = _atr(bars[-4:])
    n_vol_regime     = float(np.clip(atr_w1 / (atr_bg + EPS), 0.2, 5.0))
    net_w1           = bars[-1]["close"] - bars[-4]["open"]
    n_atr_scaled_net = float(np.clip(net_w1 / (atr_w1 + EPS), -5.0, 5.0))
    deltas_w1        = [_delta(b) for b in bars[-4:]]
    xs               = np.arange(4, dtype=float)
    n_accel_slope    = float(np.clip(float(np.polyfit(xs, deltas_w1, 1)[0]), -2.0, 2.0))
    n_mom_delta_bar  = float(np.clip(
        float(np.mean(deltas_w1[2:])) - float(np.mean(deltas_w1[:2])), -2.0, 2.0))
    return {**f1, **f2, **f3,
            "n_daccel": n1, "n_esurge": n2, "n_brvsbg": n3,
            "n_tailsh": n4, "n_dstreak": n5, "n_volctx": n6,
            "n_vol_regime": n_vol_regime, "n_atr_scaled_net": n_atr_scaled_net,
            "n_accel_slope": n_accel_slope, "n_mom_delta_bar": n_mom_delta_bar}


# ─────────────────────────────────────────────────────────────────────
# Label maps — from _labels.py v5
# ─────────────────────────────────────────────────────────────────────

HEAD_NAMES  = ["direction", "magnitude", "range_type", "close_loc", "body_ratio", "momentum"]
HEAD_LABELS = {
    "direction":  {0: "bearish",  1: "flat",     2: "bullish"},
    "magnitude":  {0: "small",    1: "medium",   2: "large"},
    "range_type": {0: "narrow",   1: "normal",   2: "wide"},
    "close_loc":  {0: "lower",    1: "mid",      2: "upper"},
    "body_ratio": {0: "weak",     1: "moderate", 2: "strong"},
    "momentum":   {0: "fading",   1: "stable",   2: "building"},
}


# ─────────────────────────────────────────────────────────────────────
# Predictor
# ─────────────────────────────────────────────────────────────────────

class MicrostructurePredictor:
    def __init__(self,
                 model_path:  str = "microstructure_h1_best.pth",
                 scaler_path: str = "microstructure_h1_scaler.pkl"):
        model_path  = resource_path(model_path)
        scaler_path = resource_path(scaler_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bundle            = joblib.load(scaler_path)
        self.scaler       = bundle["scaler"]
        self.feature_cols = bundle["feature_names"]
        self.model = H1Net(n_feat=len(self.feature_cols)).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        if MT5_AVAILABLE:
            if not mt5.initialize():
                raise RuntimeError("MT5 initialization failed")

    def _bars_from_df(self, df: pd.DataFrame) -> list:
        df = df.copy().reset_index(drop=True)
        return [{"open": r.open, "high": r.high, "low": r.low,
                 "close": r.close, "volume": r.volume}
                for r in df.itertuples()]

    def predict(self, symbol: str, timeframe=None) -> dict:
        if not MT5_AVAILABLE:
            raise RuntimeError("MetaTrader5 not available")
        if timeframe is None:
            timeframe = mt5.TIMEFRAME_H1
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, LOOKBACK)
        if rates is None or len(rates) < 80:
            return {"error": f"Insufficient bars: {0 if rates is None else len(rates)}"}
        df = pd.DataFrame(rates)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        return self.predict_from_df(df)

    def predict_from_df(self, df: pd.DataFrame) -> dict:
        bars = self._bars_from_df(df)[-80:]
        feat_dict = compute_features(bars)
        row    = np.array([[feat_dict.get(c, 0.0) for c in self.feature_cols]], dtype=np.float32)
        scaled = self.scaler.transform(row)
        x      = torch.tensor(scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outs = self.model(x)
        result: dict = {}
        for name, out in zip(HEAD_NAMES, outs):
            probs    = torch.softmax(out, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            lmap     = HEAD_LABELS[name]
            result[name] = {
                "class":      pred_idx,
                "label":      lmap[pred_idx],
                "confidence": float(probs.max()),
                "probs":      {lmap[k]: round(float(probs[k]), 4) for k in range(len(probs))},
            }
        result["signal"]     = result["direction"]["label"]
        result["confidence"] = result["direction"]["confidence"]
        result["trade_active"] = bool(
            result["direction"]["class"] != 1
            and result["momentum"]["class"] != 0
        )
        return result

    def shutdown(self):
        if MT5_AVAILABLE:
            mt5.shutdown()


# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    predictor = MicrostructurePredictor()
    result    = predictor.predict("EURUSD")
    print("\n── Microstructure H1 Prediction ──────────────────────")
    for head in HEAD_NAMES:
        v = result[head]
        print(f"  {head:<12} → {v['label']:<10} conf={v['confidence']:.2%}")
    print(f"\n  signal       : {result['signal'].upper()}")
    print(f"  trade_active : {result['trade_active']}  (conf={result['confidence']:.2%})")
    predictor.shutdown()
