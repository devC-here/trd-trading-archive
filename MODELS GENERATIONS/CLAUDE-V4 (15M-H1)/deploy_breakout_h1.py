# deploy_breakout_h1.py  v1
# Live inference for the Breakout / Squeeze model
# Architecture and features match train_breakout_h1.py + synthetic_breakout_h1.py exactly
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
# Architecture — must be byte-identical to train_breakout_h1.py
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
# Feature computation — identical to synthetic_breakout_h1.py v5
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

def _bb_width(closes: list, w: int) -> float:
    c = np.array(closes[-w:]) if len(closes) >= w else np.array(closes)
    return float(4 * np.std(c)) if len(c) > 1 else 0.0

def _kc_width(bars: list, w: int) -> float:
    n = min(w, len(bars))
    return float(3 * np.mean([b["high"] - b["low"] for b in bars[-n:]]))

def _win_feats(bars: list, atr: float, bb_bg: float, pfx: str) -> dict:
    n = len(bars)
    closes = [b["close"] for b in bars]; highs = [b["high"] for b in bars]
    lows   = [b["low"]   for b in bars]; vols  = [b["volume"] for b in bars]
    ranges = [b["high"] - b["low"] for b in bars]
    dirs   = [1 if b["close"] >= b["open"] else -1 for b in bars]
    bodies = [abs(b["close"] - b["open"]) for b in bars]
    bb_w = _bb_width(closes, n); kc_w = _kc_width(bars, n)
    kb      = float(np.clip(bb_w / (kc_w + EPS), 0.1, 3.0))
    sq      = float(1 if bb_w < kc_w else 0)
    mid_c = float(np.mean(closes)); std_c = float(np.std(closes)) + EPS
    bb_pos  = float(np.clip((closes[-1] - (mid_c - 2 * std_c)) / (4 * std_c + EPS), 0, 1))
    coil    = float(np.clip(1 - (max(highs) - min(lows)) / (atr * n + EPS), -1, 1))
    mid = n // 2
    bb1 = _bb_width(closes[:mid+1], mid+1); bb2 = _bb_width(closes[mid:], n - mid)
    exp_v   = float(np.clip((bb2 - bb1) / (bb_bg + EPS), -3, 3))
    ub = mid_c + 2 * std_c; lb = mid_c - 2 * std_c
    bw      = float(np.clip(
        (sum(1 for c in closes if c > ub) - sum(1 for c in closes if c < lb)) / n, -1, 1))
    atr_r   = float(np.clip(
        (float(np.mean(ranges[-3:])) + EPS) / (float(np.mean(ranges)) + EPS), 0.2, 4.0))
    bk_v    = float(np.clip(
        sum(vols[i] * dirs[i] for i in range(n) if closes[i] > ub or closes[i] < lb) / (atr * 1000 + EPS),
        -3, 3))
    don_b   = float((closes[-1] - min(lows)) / (max(highs) - min(lows) + EPS))
    return {f"{pfx}_kb": kb, f"{pfx}_sq": sq, f"{pfx}_bbpos": bb_pos,
            f"{pfx}_coil": coil, f"{pfx}_expv": exp_v, f"{pfx}_bw": bw,
            f"{pfx}_atrr": atr_r, f"{pfx}_bkv": bk_v, f"{pfx}_don": don_b}

def compute_features(bars: list) -> dict:
    atr_bg    = _atr(bars)
    closes_bg = [b["close"] for b in bars]
    bb_bg     = _bb_width(closes_bg, len(bars))
    f1 = _win_feats(bars[-4:],  atr_bg, bb_bg, "w1")
    f2 = _win_feats(bars[-12:], atr_bg, bb_bg, "w2")
    f3 = _win_feats(bars,       atr_bg, bb_bg, "w3")
    sq_flags = [1 if _bb_width([b["close"] for b in bars[max(0, i-8):i+1]], 8) <
                     _kc_width(bars[max(0, i-8):i+1], 8) else 0
                for i in range(len(bars))]
    cons = 0
    for s in reversed(sq_flags):
        if s == 1: cons += 1
        else: break
    avg_sq = max(float(np.mean(sq_flags) * len(sq_flags)), 1.0)
    n1 = float(np.clip(cons / avg_sq, 0, 4))
    bb_w2 = _bb_width([b["close"] for b in bars[-12:]], 12)
    n2 = float(np.clip(bb_w2 / (bb_bg + EPS), 0.1, 3.0))
    n3 = float(np.clip(f1["w1_kb"]   - f2["w2_kb"],   -2, 2))
    n4 = float(np.clip(f1["w1_coil"] - f2["w2_coil"], -2, 2))
    bg_rng = float(np.mean([b["high"] - b["low"] for b in bars])) + EPS
    w1_rng = float(np.mean([b["high"] - b["low"] for b in bars[-4:]]))
    n5 = float(np.clip(w1_rng / bg_rng, 0.2, 4.0))
    n6 = float(np.clip(f1["w1_bw"] - f2["w2_bw"], -2, 2))
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
            "n_sqmat": n1, "n_bbvsbg": n2, "n_kbshift": n3,
            "n_coilsh": n4, "n_rngvsbg": n5, "n_bwshift": n6,
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

class BreakoutPredictor:
    def __init__(self,
                 model_path:  str = "breakout_h1_best.pth",
                 scaler_path: str = "breakout_h1_scaler.pkl"):
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
    predictor = BreakoutPredictor()
    result    = predictor.predict("EURUSD")
    print("\n── Breakout H1 Prediction ────────────────────────────")
    for head in HEAD_NAMES:
        v = result[head]
        print(f"  {head:<12} → {v['label']:<10} conf={v['confidence']:.2%}")
    print(f"\n  signal       : {result['signal'].upper()}")
    print(f"  trade_active : {result['trade_active']}  (conf={result['confidence']:.2%})")
    predictor.shutdown()
