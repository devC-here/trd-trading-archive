# deploy_model2_structure.py
# Live inference for the Structural Price Location Specialist
# Architecture and features match train_model2_structure.py exactly
# Features : 5  |  Output : continuous scalar E_s ∈ [-1.0, +1.0]
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

EPS      = 1e-12
LOOKBACK = 60           # bars fetched from MT5 (16 used in window + warmup)
TRUNK    = 128
DROP     = 0.20


# ─────────────────────────────────────────────────────────────────────
# Architecture  ·  byte-identical to train_model2_structure.py
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


class StructureNet(nn.Module):
    def __init__(self, input_dim, trunk=TRUNK, drop=DROP):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, trunk),
            nn.BatchNorm1d(trunk), nn.GELU(), nn.Dropout(drop)
        )
        self.res = nn.Sequential(
            ResidualBlock(trunk, drop),
            ResidualBlock(trunk, drop),
        )
        self.mid = nn.Sequential(
            nn.Linear(trunk, trunk // 2),
            nn.BatchNorm1d(trunk // 2), nn.GELU(), nn.Dropout(drop * 0.7)
        )
        self.head = nn.Sequential(
            nn.Linear(trunk // 2, 1),
            nn.Tanh()               # bounds output to [-1, +1]
        )

    def forward(self, x):
        h = self.mid(self.res(self.stem(x)))
        return self.head(h).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────
# Feature computation  ·  matches model2_structure_generator.py
# ─────────────────────────────────────────────────────────────────────

def _ols_intercept(y):
    n  = len(y)
    x  = np.arange(n, dtype=float)
    mx, my = x.mean(), y.mean()
    slope  = ((x - mx) * (y - my)).sum() / (((x - mx) ** 2).sum() + EPS)
    return float(my - slope * mx)


def compute_structure_features(df: pd.DataFrame) -> dict:
    """
    df : DataFrame with [open, high, low, close, volume]. Min 16 rows.
    Returns dict of 5 float features (raw — scaler applied externally).
    """
    df = df.copy().reset_index(drop=True)
    window = df.tail(16)

    C     = window["close"].values.astype(float)
    H     = window["high"].values.astype(float)
    L     = window["low"].values.astype(float)

    H_max  = H.max()
    L_min  = L.min()
    pivot  = (H_max + L_min + C[-1]) / 3.0

    # 43_fib_382_dist — distance from 38.2% Fibonacci retracement
    fib_382      = L_min + 0.382 * (H_max - L_min)
    fib_382_dist = float(C[-1] - fib_382)

    # 49_s1_dist — distance from Support 1 pivot
    s1      = 2.0 * pivot - H_max
    s1_dist = float(C[-1] - s1)

    # 48_r1_dist — distance from Resistance 1 pivot
    r1      = 2.0 * pivot - L_min
    r1_dist = float(r1 - C[-1])

    # 35_lin_reg_intercept — OLS y-intercept across 16 close prices
    lin_reg_intercept = _ols_intercept(C)

    # 69_high_low_ratio — envelope elongation ratio
    high_low_ratio = float(H_max / (L_min + EPS))

    return {
        "43_fib_382_dist":      fib_382_dist,
        "49_s1_dist":           s1_dist,
        "48_r1_dist":           r1_dist,
        "35_lin_reg_intercept": lin_reg_intercept,
        "69_high_low_ratio":    high_low_ratio,
    }


# ─────────────────────────────────────────────────────────────────────
# Spatial zone interpretation
# ─────────────────────────────────────────────────────────────────────

def interpret_es(es: float) -> str:
    if es <= -0.60:   return "deep_support_floor"
    elif es <= -0.20: return "near_support"
    elif es <   0.20: return "equilibrium"
    elif es <   0.60: return "near_resistance"
    else:             return "deep_resistance_ceiling"


# ─────────────────────────────────────────────────────────────────────
# Predictor
# ─────────────────────────────────────────────────────────────────────

class StructurePredictor:
    def __init__(self,
                 model_path:  str = "model2_best.pth",
                 scaler_path: str = "model2_scaler.pkl"):

        model_path  = resource_path(model_path)
        scaler_path = resource_path(scaler_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        bundle            = joblib.load(scaler_path)
        self.scaler       = bundle["scaler"]      # StandardScaler — MANDATORY
        self.feature_cols = bundle["feature_names"]

        self.model = StructureNet(input_dim=len(self.feature_cols)).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()

        if MT5_AVAILABLE:
            if not mt5.initialize():
                raise RuntimeError("MT5 initialization failed")

    def predict(self, symbol: str, timeframe=None) -> dict:
        if not MT5_AVAILABLE:
            raise RuntimeError("MetaTrader5 not available")
        if timeframe is None:
            timeframe = mt5.TIMEFRAME_H4
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, LOOKBACK)
        if rates is None or len(rates) < 16:
            return {"error": f"Insufficient bars: {0 if rates is None else len(rates)}"}
        df = pd.DataFrame(rates)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        return self.predict_from_df(df)

    def predict_from_df(self, df: pd.DataFrame) -> dict:
        feat_dict = compute_structure_features(df)
        row = np.array([[feat_dict[c] for c in self.feature_cols]], dtype=np.float32)

        # StandardScaler applied — critical for intercept vs micro-distance scale gap
        scaled = self.scaler.transform(row)
        x      = torch.tensor(scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            es = float(self.model(x).cpu().numpy()[0])

        es = float(np.clip(es, -1.0, 1.0))

        return {
            "structural_position": {
                "E_s":        es,
                "zone":       interpret_es(es),
                "bias":       "support_floor" if es < -0.20 else
                              ("resistance_ceiling" if es > 0.20 else "neutral"),
            },
            "signal":   "buy_zone"  if es < -0.20 else
                        ("sell_zone" if es >  0.20 else "neutral"),
            "E_s":      es,
            "features": feat_dict,
        }

    def shutdown(self):
        if MT5_AVAILABLE:
            mt5.shutdown()


# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    predictor = StructurePredictor()
    result    = predictor.predict("EURUSD")
    print("\n── Model 2 · Structural Price Location ─────────────")
    sp = result["structural_position"]
    print(f"  E_s score   → {sp['E_s']:+.4f}")
    print(f"  zone        → {sp['zone']}")
    print(f"  bias        → {sp['bias'].upper()}")
    print(f"  signal      → {result['signal'].upper()}")
    print(f"\n  Features used (raw, pre-scale):")
    for k, v in result["features"].items():
        print(f"    {k:<35} {v:+.8f}")
    predictor.shutdown()
