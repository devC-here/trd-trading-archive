# candle_deploy.py
# Live inference for the Candle Sequence model
# Architecture and features match train_candle.py + synthetic_candle.py exactly
# Features : 30  |  Heads : 6  |  HEAD_SIZES : [3, 3, 4, 3, 3, 3]
#
# Candle numbering convention (MT5 pos=1 baseline — live bar is never touched):
#   offset 0  →  t-1  (last fully closed bar)
#   offset 1  →  t-2
#   offset k  →  t-(k+1)
#   result["bar"] always equals t-(offset+1)
# ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
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

EPS      = 1e-12
WIN      = 10
LOOKBACK = 60
TRUNK    = 256
DROP     = 0.25


# ─────────────────────────────────────────────────────────────────────
# Architecture  ·  must be byte-identical to train_candle.py
# ─────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, dim, drop=DROP):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(drop),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.seq(x) + x)


class CandleNet(nn.Module):
    HEAD_SIZES = [3, 3, 4, 3, 3, 3]

    def __init__(self, input_dim, trunk=TRUNK, drop=DROP):
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
        h = trunk // 2
        self.heads = nn.ModuleList([nn.Linear(h, s) for s in self.HEAD_SIZES])

    def forward(self, x):
        h = self.mid(self.res(self.stem(x)))
        return tuple(head(h) for head in self.heads)


# ─────────────────────────────────────────────────────────────────────
# Feature computation  ·  identical to synthetic_candle.py
# ─────────────────────────────────────────────────────────────────────

def _ols_slope(arr):
    n = len(arr)
    if n < 2: return 0.0
    s = np.std(arr)
    if s < EPS: return 0.0
    return float(np.polyfit(np.arange(n, dtype=float), arr, 1)[0] / (s + EPS))


def _safe_corr(a, b):
    if np.std(a) < EPS or np.std(b) < EPS: return 0.0
    try:
        r, _ = sp_stats.pearsonr(a, b)
        return float(r) if np.isfinite(r) else 0.0
    except Exception:
        return 0.0


def compute_candle_features(df: pd.DataFrame) -> dict:
    df = df.copy().reset_index(drop=True)

    df["range"]          = df["high"] - df["low"]
    df["body"]           = df["close"] - df["open"]
    df["abs_body"]       = df["body"].abs()
    df["dir"]            = np.sign(df["body"]).fillna(0)
    df["upper_wick"]     = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"]     = df[["open", "close"]].min(axis=1) - df["low"]
    df["atr14"]          = df["range"].rolling(14, min_periods=1).mean()
    df["abs_body_avg30"] = df["abs_body"].rolling(30, min_periods=1).mean()
    df["range_avg30"]    = df["range"].rolling(30, min_periods=1).mean()
    df["vol_avg20"]      = df["volume"].rolling(20, min_periods=1).mean()
    df["body_ratio"]     = df["abs_body"] / (df["range"] + EPS)
    df["body_rat_avg5"]  = df["body_ratio"].rolling(5, min_periods=1).mean()
    df["close_grav"]     = (df["close"] - df["low"]) / (df["range"] + EPS)
    df["shadow_dom"]     = (df["lower_wick"] - df["upper_wick"]) / (df["range"] + EPS)
    bull                 = df["dir"] >= 0
    df["dir_eff"]        = np.where(bull,
                                     df["body_ratio"] * df["close_grav"],
                                     df["body_ratio"] * (1 - df["close_grav"]))
    df["vol_weight"]     = (df["volume"] / (df["vol_avg20"] + EPS)).clip(0, 4)

    ha_close      = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_open       = ha_close.ewm(span=3, adjust=False).mean()
    df["ha_body"] = (ha_close - ha_open) / (df["atr14"] + EPS)

    cons, cur, prev = [], 0, 0
    for d in df["dir"].fillna(0).tolist():
        if d == prev and d != 0:  cur += 1
        elif d != 0:               cur  = 1
        else:                      cur  = 0
        cons.append(cur)
        prev = d
    df["consec_len"] = cons

    w       = df.iloc[-WIN:].reset_index(drop=True)
    atr_val = float(df["atr14"].iloc[-1])
    f       = {}

    f["lb_close_grav"] = float(w["close_grav"].iloc[-1])
    f["lb_body_ratio"] = float(w["body_ratio"].iloc[-1])
    f["lb_shadow_dom"] = float(w["shadow_dom"].iloc[-1])
    f["lb_vol_weight"] = float(min(w["vol_weight"].iloc[-1], 4.0))
    f["lb_dir_eff"]    = float(w["dir_eff"].iloc[-1])

    f["seq_cg_slope"]     = _ols_slope(w["close_grav"].values)
    bm = float(w["body_ratio"].mean()); bs = float(w["body_ratio"].std())
    f["body_entropy"]     = float(bs / (bm + EPS))
    f["wick_press_slope"] = _ols_slope(w["shadow_dom"].values)
    bar_conv              = w["dir"] * w["body_ratio"] * w["vol_weight"]
    f["vol_narrative"]    = _ols_slope(bar_conv.values)
    f["seq_harmony"]      = _safe_corr(w["close_grav"].values, w["range"].values)
    early_dir             = np.sign(w["body"].iloc[:5].sum())
    late_dir              = np.sign(w["body"].iloc[5:].sum())
    f["handoff"]          = float(early_dir != late_dir)
    early_body            = float(w["body_ratio"].iloc[:5].mean())
    late_body             = float(w["body_ratio"].iloc[5:].mean())
    f["handoff_str"]      = float((late_body - early_body) / (atr_val + EPS))
    full_mean             = float(w["body_ratio"].mean()) + EPS
    f["seq_accel"]        = float((late_body - early_body) / full_mean)

    curr_body  = float(w["abs_body"].iloc[-1])
    prior_body = float(w["abs_body"].iloc[-2]) if len(w) >= 2 else curr_body
    curr_dir   = float(w["dir"].iloc[-1])
    f["engulf_qual"] = float((curr_body - prior_body) / (atr_val + EPS) * (curr_dir or 1))

    inside_count = 0
    for i in range(1, len(w)):
        if w["high"].iloc[i] < w["high"].iloc[i-1] and w["low"].iloc[i] > w["low"].iloc[i-1]:
            inside_count += 1
        else:
            break
    mother_range       = float(w["range"].iloc[0]) if inside_count > 0 else 0.0
    f["inside_energy"] = float(inside_count * mother_range / (atr_val + EPS))

    f["ha_body_last"]  = float(w["ha_body"].iloc[-1])
    ha_arr             = w["ha_body"].values
    f["ha_body_slope"] = float((ha_arr[-1] - ha_arr[max(0, len(ha_arr) - 4)]) / (atr_val + EPS))

    uw  = float(w["upper_wick"].iloc[-1])
    lw  = float(w["lower_wick"].iloc[-1])
    rng = float(w["range"].iloc[-1])
    br  = float(w["body_ratio"].iloc[-1])
    cg  = float(w["close_grav"].iloc[-1])
    bsm = 1.0 if br < 0.35 else 0.5
    f["pin_bull_qual"] = float((lw / (rng + EPS)) * bsm * cg)
    f["pin_bear_qual"] = float((uw / (rng + EPS)) * bsm * (1 - cg))

    closes            = w["close"].values
    c_min, c_max      = float(closes.min()), float(closes.max())
    seq_pos           = (closes[-1] - c_min) / (c_max - c_min + EPS)
    doji_score        = float(max(0, 0.15 - br) / 0.15)
    f["doji_context"] = float(doji_score * (seq_pos - 0.5))

    f["body_vs_avg"]  = float(min(float(df["abs_body"].iloc[-1]) /
                                   (float(df["abs_body_avg30"].iloc[-1]) + EPS), 4.0))
    f["range_vs_avg"] = float(min(float(df["range"].iloc[-1]) /
                                   (float(df["range_avg30"].iloc[-1]) + EPS), 4.0))
    br5_avg           = float(df["body_rat_avg5"].iloc[-1])
    f["body_accel"]   = float(br - br5_avg)

    br1      = float(w["body_ratio"].iloc[-1])
    br3      = float(w["body_ratio"].iloc[-3:].mean()) if len(w) >= 3 else br1
    br10     = float(w["body_ratio"].mean())
    price_up = closes[-1] > closes[0]
    f["fractal_bull"] = float((br1 > br3 > br10) and price_up)
    f["fractal_bear"] = float((br1 > br3 > br10) and not price_up)

    f["seq_dir_ratio"]  = float((w["dir"] > 0).mean())
    f["consec_len"]     = float(w["consec_len"].iloc[-1])
    vol_surge_w         = float(w["vol_weight"].mean())
    br_mean_w           = float(w["body_ratio"].mean()) + EPS
    f["vol_climax_div"] = float(br_mean_w / (vol_surge_w + EPS))
    f["ha_trend_str"]   = float(w["ha_body"].ewm(span=5, adjust=False).mean().iloc[-1])
    f["seq_body_avg"]   = float(w["body_ratio"].mean())

    return f


# ─────────────────────────────────────────────────────────────────────
# Label maps
# ─────────────────────────────────────────────────────────────────────

HEAD_LABELS = {
    "next_direction":   {0: "bearish",     1: "doji",       2: "bullish"},
    "next_body_size":   {0: "small",       1: "normal",     2: "large"},
    "next_pattern":     {0: "impulse",     1: "pin",        2: "inside_doji", 3: "engulf"},
    "seq_continuation": {0: "reversal",    1: "indecision", 2: "continuation"},
    "next_close_grav":  {0: "lower_third", 1: "mid",        2: "upper_third"},
    "next_volatility":  {0: "compression", 1: "normal",     2: "expansion"},
}


# ─────────────────────────────────────────────────────────────────────
# Inference helper — identical structure for every offset
# ─────────────────────────────────────────────────────────────────────

def _build_result(outs, offset: int) -> dict:
    """
    Build a full result dict. offset=0 → t-1 (last closed bar).
    result["bar"] is the human-readable label t-(offset+1).
    """
    result: dict = {
        "offset": offset,
        "bar":    f"t-{offset + 1}",   # t-1 for offset=0, t-2 for offset=1, ...
    }
    for name, out in zip(HEAD_LABELS.keys(), outs):
        probs    = torch.softmax(out, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        lmap     = HEAD_LABELS[name]
        result[name] = {
            "class":      pred_idx,
            "label":      lmap[pred_idx],
            "confidence": float(probs.max()),
            "probs":      {lmap[k]: round(float(probs[k]), 4) for k in range(len(probs))},
        }
    result["signal"]       = result["next_direction"]["label"]
    result["confidence"]   = result["next_direction"]["confidence"]
    result["continuation"] = result["seq_continuation"]["label"]
    result["trade_active"] = bool(
        result["next_direction"]["class"] != 1
        and result["seq_continuation"]["class"] != 1
    )
    return result


def _print_result(r: dict) -> None:
    """Print full head detail for one result — same layout for every bar."""
    print(f"\n  ── {r['bar']} ────────────────────────────────────────")
    for head in HEAD_LABELS:
        v = r[head]
        print(f"    {head:<20} → {v['label']:<14} conf={v['confidence']:.2%}")
    print(f"    {'signal':<20}   {r['signal'].upper():<14} conf={r['confidence']:.2%}")
    print(f"    {'continuation':<20}   {r['continuation'].upper()}")
    print(f"    {'trade_active':<20}   {r['trade_active']}")


# ─────────────────────────────────────────────────────────────────────
# Predictor
# ─────────────────────────────────────────────────────────────────────

class CandlePredictor:
    def __init__(self,
                 model_path:  str = "candle_best.pth",
                 scaler_path: str = "candle_scaler.pkl"):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        bundle            = joblib.load(resource_path(scaler_path))
        self.scaler       = bundle["scaler"]
        self.feature_cols = bundle["feature_names"]

        self.model = CandleNet(input_dim=len(self.feature_cols)).to(self.device)
        self.model.load_state_dict(
            torch.load(resource_path(model_path), map_location=self.device, weights_only=True))
        self.model.eval()

        if MT5_AVAILABLE:
            if not mt5.initialize():
                raise RuntimeError("MT5 initialization failed")

    # ── convenience wrappers (return single dict, bar=t-1) ──────────

    def predict(self, symbol: str, timeframe=None) -> dict:
        """Predict t-1 (last closed bar). Returns single result dict."""
        return self.predict_multi(symbol, timeframe=timeframe, n=1)[0]

    def predict_from_df(self, df: pd.DataFrame) -> dict:
        """Predict t-1 from a pre-built DataFrame. Returns single result dict."""
        return self.predict_multi_from_df(df, n=1)[0]

    # ── multi-bar prediction ─────────────────────────────────────────

    def predict_multi(self, symbol: str, timeframe=None, n: int = 1) -> list:
        """
        Predict the last n closed bars in one call.

        Candle numbering (MT5 pos=1 — live bar is never included):
            n=1  →  [t-1]
            n=2  →  [t-1, t-2]
            n=4  →  [t-1, t-2, t-3, t-4]

        Returns list of n result dicts, each containing:
            "bar"          – "t-1", "t-2", ... (human label)
            "offset"       – 0, 1, 2, ...      (internal index)
            "signal"       – direction label
            "confidence"   – direction confidence
            "continuation" – seq_continuation label
            "trade_active" – bool
            <head_name>    – full {class, label, confidence, probs} for each head
        """
        if not MT5_AVAILABLE:
            raise RuntimeError("MetaTrader5 not available")
        if timeframe is None:
            timeframe = mt5.TIMEFRAME_H1

        # pos=1 skips the live bar; fetch LOOKBACK + (n-1) closed bars
        fetch = LOOKBACK + (n - 1) + 5
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, fetch)
        if rates is None or len(rates) < LOOKBACK + (n - 1):
            got = 0 if rates is None else len(rates)
            return [{"error": f"Insufficient bars: got {got}, need {LOOKBACK + (n-1)}",
                     "bar": f"t-{i+1}"} for i in range(n)]

        df = pd.DataFrame(rates)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        return self.predict_multi_from_df(df, n=n)

    def predict_multi_from_df(self, df: pd.DataFrame, n: int = 1) -> list:
        """
        Run inference for n closed bars against a pre-built DataFrame.

        The DataFrame is assumed to contain only closed bars (live bar excluded).
        df.iloc[-1] = t-1, df.iloc[-2] = t-2, etc.

        Requires at least LOOKBACK + (n-1) rows.
        """
        min_required = LOOKBACK + (n - 1)
        if len(df) < min_required:
            return [{"error": f"Need {min_required} rows for n={n}, got {len(df)}",
                     "bar": f"t-{i+1}"} for i in range(n)]

        results = []
        for offset in range(n):
            # offset=0 → full df (t-1 is iloc[-1])
            # offset=1 → drop last row (t-2 becomes iloc[-1])
            chunk = df.iloc[:len(df) - offset] if offset > 0 else df

            feat_dict = compute_candle_features(chunk)
            row    = np.array([[feat_dict.get(c, 0.0) for c in self.feature_cols]], dtype=np.float32)
            scaled = self.scaler.transform(row)
            x      = torch.tensor(scaled, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                outs = self.model(x)

            results.append(_build_result(outs, offset))

        return results

    def shutdown(self):
        if MT5_AVAILABLE:
            mt5.shutdown()


# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    predictor = CandlePredictor()

    results = predictor.predict_multi("EURUSD", mt5.TIMEFRAME_H1, n=4)

    for r in results:
        print(f"\n══ Candle Sequence  {r['bar']} ══════════════════════════════")
        for head, v in r.items():
            if isinstance(v, dict) and "label" in v:
                print(f"  {head:<20} → {v['label']:<14} conf={v['confidence']:.2%}")
        print(f"  {'direction':<20} : {r['signal'].upper()}")
        print(f"  {'continuation':<20} : {r['continuation'].upper()}")
        print(f"  {'trade_active':<20} : {r['trade_active']}  (conf={r['confidence']:.2%})")

    predictor.shutdown()
