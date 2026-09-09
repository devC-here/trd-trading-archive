# deploy_candle_h1.py  v1
# Live inference for the Candle model
# Architecture and features match train_candle_h1.py + synthetic_candle_h1.py exactly
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
# Architecture — must be byte-identical to train_candle_h1.py
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
# Feature computation — identical to synthetic_candle_h1.py v5
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

def _ha_body_last(bars: list, atr: float) -> float:
    ha_c = [(b["open"] + b["high"] + b["low"] + b["close"]) / 4 for b in bars]
    ha_o = [ha_c[0]]
    for i in range(1, len(bars)):
        ha_o.append((ha_o[-1] + ha_c[i-1]) / 2)
    return float((ha_c[-1] - ha_o[-1]) / (atr + EPS))

def _win_feats(bars: list, atr: float, pfx: str) -> dict:
    n = len(bars)
    closes = [b["close"] for b in bars]; opens  = [b["open"]  for b in bars]
    highs  = [b["high"]  for b in bars]; lows   = [b["low"]   for b in bars]
    vols   = [b["volume"] for b in bars]
    bodies = [abs(b["close"] - b["open"]) for b in bars]
    ranges = [b["high"] - b["low"] for b in bars]
    dirs   = [1 if b["close"] >= b["open"] else -1 for b in bars]
    gravs  = [(b["close"] - b["low"]) / (b["high"] - b["low"] + EPS) for b in bars]
    brs    = [bodies[i] / (ranges[i] + EPS) for i in range(n)]
    uw     = [abs(max(b["open"], b["close"]) - b["high"]) for b in bars]
    lw     = [abs(min(b["open"], b["close"]) - b["low"])  for b in bars]
    mid = n // 2
    avg_br   = float(np.mean(brs))
    avg_grav = float(np.mean(gravs))
    sd       = float(np.clip((np.mean(lw) - np.mean(uw)) / (atr + EPS), -3, 3))
    br1 = float(np.mean(brs[:mid])) + EPS; br2 = float(np.mean(brs[mid:]))
    seq_acc  = float(np.clip((br2 - br1) / br1, -3, 3))
    dir_rat  = float(sum(1 for d in dirs if d > 0) / n)
    ha_last  = float(np.clip(_ha_body_last(bars, atr), -4, 4))
    engulf   = float(np.clip((bodies[-1] - bodies[-2]) / (atr + EPS) * dirs[-1], -4, 4)) if n >= 2 else 0.0
    pin_bull = float(np.clip((1 - brs[-1]) * gravs[-1], 0, 1))
    pin_bear = float(np.clip((1 - brs[-1]) * (1 - gravs[-1]), 0, 1))
    pin_sig  = pin_bull - pin_bear
    upv = sum(vols[i] for i in range(n) if dirs[i] > 0)
    dnv = sum(vols[i] for i in range(n) if dirs[i] < 0) + EPS
    vol_dir  = float(np.clip((upv - dnv) / (upv + dnv + EPS), -1, 1))
    return {f"{pfx}_br": avg_br, f"{pfx}_grav": avg_grav, f"{pfx}_sd": sd,
            f"{pfx}_acc": seq_acc, f"{pfx}_drat": dir_rat, f"{pfx}_ha": ha_last,
            f"{pfx}_eng": engulf, f"{pfx}_pin": pin_sig, f"{pfx}_vdir": vol_dir}

def compute_features(bars: list) -> dict:
    atr_bg = _atr(bars)
    f1 = _win_feats(bars[-4:],  atr_bg, "w1")
    f2 = _win_feats(bars[-12:], atr_bg, "w2")
    f3 = _win_feats(bars,       atr_bg, "w3")
    bg_brs    = [abs(b["close"] - b["open"]) / (b["high"] - b["low"] + EPS) for b in bars]
    bg_avg_br = float(np.mean(bg_brs)) + EPS
    n1 = float(np.clip(f2["w2_br"]   - f3["w3_br"],   -2, 2))
    n2 = float(np.clip(f2["w2_grav"] - f3["w3_grav"], -1, 1))
    n3 = float(np.clip(f2["w2_acc"]  - f3["w3_acc"],  -3, 3))
    n4 = float(np.clip(f2["w2_sd"]   - f3["w3_sd"],   -2, 2))
    dirs_all = [1 if b["close"] >= b["open"] else -1 for b in bars]
    streak = 0; last_d = dirs_all[-1]
    for d in reversed(dirs_all):
        if d == last_d: streak += 1
        else: break
    n5 = float(np.clip(streak * last_d / 12, -1, 1))
    bg_rng = float(np.mean([b["high"] - b["low"] for b in bars])) + EPS
    n6 = float(np.clip((bars[-1]["high"] - bars[-1]["low"]) / bg_rng, 0.2, 4.0))
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
            "n_br_shift": n1, "n_grav_shift": n2, "n_acc_shift": n3,
            "n_sd_shift": n4, "n_streak": n5, "n_rng_exp": n6,
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

class CandlePredictor:
    def __init__(self,
                 model_path:  str = "candle_h1_best.pth",
                 scaler_path: str = "candle_h1_scaler.pkl"):
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
    predictor = CandlePredictor()
    result    = predictor.predict("EURUSD")
    print("\n── Candle H1 Prediction ──────────────────────────────")
    for head in HEAD_NAMES:
        v = result[head]
        print(f"  {head:<12} → {v['label']:<10} conf={v['confidence']:.2%}")
    print(f"\n  signal       : {result['signal'].upper()}")
    print(f"  trade_active : {result['trade_active']}  (conf={result['confidence']:.2%})")
    predictor.shutdown()
