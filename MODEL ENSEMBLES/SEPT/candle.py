# candles_deploy.py
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

SEQ_LEN = 20

# -------- Model (match training) --------
class ResidualBlock(nn.Module):
    def __init__(self, dim, drop=0.35):
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
    def forward(self, x): return self.act(self.seq(x)+x)

class CandleNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.35),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(512, 0.35),
            ResidualBlock(512, 0.35),
            ResidualBlock(512, 0.35),
        )
        self.mid = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.30),
        )
        self.heads = nn.ModuleList([
            nn.Linear(256, 12),  # pattern_primary
            nn.Linear(256, 4),   # dominance
            nn.Linear(256, 3),   # strength_bin
            nn.Linear(256, 3),   # follow_through_bias
            nn.Linear(256, 3),   # failure_risk
            nn.Linear(256, 2),   # pattern_persistence
            nn.Linear(256, 3),   # volatility_regime
            nn.Linear(256, 3),   # entry_signal
        ])
    def forward(self, x):
        h = self.stem(x); h = self.blocks(h); h = self.mid(h)
        return [head(h) for head in self.heads]

# -------- Pattern helpers (for flags) --------
def base_features(df):
    df = df.copy()
    df["range"] = df["high"] - df["low"]
    df["body"] = (df["close"] - df["open"]).abs()
    df["dir"] = np.sign(df["close"] - df["open"]).fillna(0.0)
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    rng = df["range"] + 1e-12
    df["body_ratio"]  = df["body"] / rng
    df["uw_ratio"]    = df["upper_wick"] / rng
    df["lw_ratio"]    = df["lower_wick"] / rng
    df["co_ret"]      = (df["close"] - df["open"]) / (df["open"] + 1e-12)
    df["hl_vol"]      = df["range"].rolling(10).std().fillna(0.0)
    df["body_z10"]    = (df["body"] - df["body"].rolling(10).mean()) / (df["body"].rolling(10).std() + 1e-12)
    df["range_z10"]   = (df["range"] - df["range"].rolling(10).mean()) / (df["range"].rolling(10).std() + 1e-12)
    df["dir_run"]     = (df["dir"] == df["dir"].shift(1)).rolling(5).sum().fillna(0.0)
    # calm/normal/expanding via range mean buckets (robust to live)
    m20 = df["range"].rolling(20).mean()
    q35 = df["range"].rolling(60).quantile(0.35).fillna(df["range"].quantile(0.35))
    q70 = df["range"].rolling(60).quantile(0.70).fillna(df["range"].quantile(0.70))
    bins = np.where(m20 <= q35, 0, np.where(m20 <= q70, 1, 2))
    df["vol_regime"] = pd.Series(bins, index=df.index).fillna(1).astype(int)
    return df.fillna(0.0)

def is_doji(r, thresh=0.15): return r["body_ratio"] < thresh
def is_pin_bull(r): return (r["lw_ratio"]>0.6) and (r["uw_ratio"]<0.2) and (r["dir"]>=0)
def is_pin_bear(r): return (r["uw_ratio"]>0.6) and (r["lw_ratio"]<0.2) and (r["dir"]<=0)
def engulf(df, i):
    if i<1: return (False, False)
    prev, cur = df.iloc[i-1], df.iloc[i]
    bull = (cur["close"]>cur["open"]) and (prev["close"]<prev["open"]) and \
           (cur["close"]>=prev["open"]) and (cur["open"]<=prev["close"])
    bear = (cur["close"]<cur["open"]) and (prev["close"]>prev["open"]) and \
           (cur["open"]>=prev["close"]) and (cur["close"]<=prev["open"])
    return (bull, bear)
def inside(df, i):
    if i<1: return False
    p, c = df.iloc[i-1], df.iloc[i]
    return (c["high"]<=p["high"]) and (c["low"]>=p["low"])
def outside(df, i):
    if i<1: return False
    p, c = df.iloc[i-1], df.iloc[i]
    return (c["high"]>=p["high"]) and (c["low"]<=p["low"])
def morning_star(df, i):
    if i<2: return False
    a,b,c = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
    cond_a = a["close"]<a["open"]
    cond_b = is_doji(b) or (b["body_ratio"]<0.25)
    cond_c = c["close"]>c["open"] and c["close"] > (a["open"] - 0.25*(a["open"]-a["close"]))
    return cond_a and cond_b and cond_c
def evening_star(df, i):
    if i<2: return False
    a,b,c = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
    cond_a = a["close"]>a["open"]
    cond_b = is_doji(b) or (b["body_ratio"]<0.25)
    cond_c = c["close"]<c["open"] and c["close"] < (a["open"] - 0.75*(a["open"]-a["close"]))
    return cond_a and cond_b and cond_c
def three_soldiers(df, i):
    if i<2: return False
    a,b,c = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
    return (a["close"]>a["open"]) and (b["close"]>b["open"]) and (c["close"]>c["open"]) and \
           (a["body_ratio"]>0.4) and (b["body_ratio"]>0.4) and (c["body_ratio"]>0.4)
def three_crows(df, i):
    if i<2: return False
    a,b,c = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
    return (a["close"]<a["open"]) and (b["close"]<b["open"]) and (c["close"]<c["open"]) and \
           (a["body_ratio"]>0.4) and (b["body_ratio"]>0.4) and (c["body_ratio"]>0.4)

# -------- Predictor --------
class CandlePredictor:
    def __init__(self,
                 model_path="candles_model_best.pth",
                 scaler_path="candles_scaler.pkl"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bundle = joblib.load(scaler_path)
        self.scaler = bundle["scaler"]
        self.feature_cols = bundle["feature_names"]

        self.model = CandleNet(input_dim=len(self.feature_cols)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        if not mt5.initialize():
            raise RuntimeError("MT5 initialization failed")

    def _build_features(self, raw_df):
        df = raw_df.copy()
        df = df.tail(max(SEQ_LEN*3, 80)).reset_index(drop=True)

        df_feat = base_features(df)
        df_feat = df_feat.tail(SEQ_LEN).reset_index(drop=True)

        # Flatten — identical to generator
        per_bar_cols = [
            "body_ratio","uw_ratio","lw_ratio","co_ret","dir","body_z10","range_z10","dir_run"
        ]
        flat = {}
        for i in range(SEQ_LEN):
            for c in per_bar_cols:
                flat[f"{c}_t-{i}"] = float(df_feat.iloc[i][c])

        # Live flags on last bar
        i = len(df_feat)-1
        bull_eng, bear_eng = engulf(df_feat, i)
        flat["flag_doji"] = float(is_doji(df_feat.iloc[i]))
        flat["flag_pin_bull"] = float(is_pin_bull(df_feat.iloc[i]))
        flat["flag_pin_bear"] = float(is_pin_bear(df_feat.iloc[i]))
        flat["flag_engulf_bull"] = float(bull_eng)
        flat["flag_engulf_bear"] = float(bear_eng)
        flat["flag_inside_bar"] = float(inside(df_feat, i))
        flat["flag_outside_bar"] = float(outside(df_feat, i))
        flat["flag_morning_star"] = float(morning_star(df_feat, i))
        flat["flag_evening_star"] = float(evening_star(df_feat, i))
        flat["flag_three_soldiers"] = float(three_soldiers(df_feat, i))
        flat["flag_three_crows"] = float(three_crows(df_feat, i))

        # Context aggregates
        recent = df_feat.tail(10)
        flat["pct_bull_10"] = float((recent["dir"]>0).mean())
        flat["avg_body_10"] = float(recent["body"].mean())
        flat["avg_range_10"] = float(recent["range"].mean())
        flat["vol_regime"] = int(df_feat["vol_regime"].iloc[-1])

        feat_df = pd.DataFrame([flat])
        feat_df = feat_df.reindex(columns=self.feature_cols, fill_value=0.0)
        return feat_df

    def predict(self, symbol="EURUSDm", timeframe=mt5.TIMEFRAME_M15, n_bars=300):
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
        if rates is None or len(rates) < (SEQ_LEN+10):
            return {"error":"Insufficient data"}
        df = pd.DataFrame(rates)[["open","high","low","close"]]

        feat_df = self._build_features(df)
        X = torch.tensor(self.scaler.transform(feat_df.values), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outs = self.model(X)

        preds = [int(o.argmax(1).item()) for o in outs]
        probs = [torch.softmax(o, dim=1).cpu().numpy()[0].tolist() for o in outs]
        confs = [float(max(p)) for p in probs]

        return {
            "pattern_primary": preds[0],
            "dominance": preds[1],
            "strength_bin": preds[2],
            "follow_through_bias": preds[3],
            "failure_risk": preds[4],
            "pattern_persistence": preds[5],
            "volatility_regime": preds[6],
            "entry_signal": preds[7],
            "probabilities": {
                "pattern_primary": probs[0],
                "dominance": probs[1],
                "strength_bin": probs[2],
                "follow_through_bias": probs[3],
                "failure_risk": probs[4],
                "pattern_persistence": probs[5],
                "volatility_regime": probs[6],
                "entry_signal": probs[7],
            },
            "confidences": {
                "pattern_primary": confs[0],
                "dominance": confs[1],
                "strength_bin": confs[2],
                "follow_through_bias": confs[3],
                "failure_risk": confs[4],
                "pattern_persistence": confs[5],
                "volatility_regime": confs[6],
                "entry_signal": confs[7],
            }
        }

    def shutdown(self): mt5.shutdown()

# ----- Example -----
if __name__ == "__main__":
    p = CandlePredictor()
    out = p.predict("EURUSDm", mt5.TIMEFRAME_M15)
    print("📊 Candlestick Pattern Prediction:", out)
    p.shutdown()
