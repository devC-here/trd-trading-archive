import os
import pandas as pd
import MetaTrader5 as mt5

# -----------------------------
# Model imports
# -----------------------------
from momentumdeploy import MomentumPredictor
from breakoutdeploy import BreakoutPredictor
from microstructure import MicrostructurePredictor
from VWdeploy import VwapVolumePredictor
from candledeploy import CandlestickSequencePredictor

# -----------------------------
# Encoder
# -----------------------------
from code1 import build_condition_code


# ============================================================
# Initialize predictors
# ============================================================

PREDICTORS = {
    "momentum": MomentumPredictor(),
    "breakout": BreakoutPredictor(),
    "micro": MicrostructurePredictor(),
    "vwap": VwapVolumePredictor(),
    "candle": CandlestickSequencePredictor(),
}


# ============================================================
# Fake MT5 adapter
# Converts CSV candles → MT5 format
# ============================================================

class FakeMT5:

    def __init__(self, df):
        self.df = df

    def copy_rates_from_pos(self, symbol, timeframe, start_pos, count):

        data = self.df.tail(count)

        rates = []

        for _, r in data.iterrows():

            rates.append({
                "time": int(pd.Timestamp(r["timestamp"]).timestamp()),
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
                "tick_volume": int(r["volume"]),   # <-- using CSV volume
                "spread": 0,
                "real_volume": int(r["volume"])
            })

        return rates


# ============================================================
# Generate condition code
# ============================================================

def generate_condition_code(df):

    fake = FakeMT5(df)

    # override MT5 data source
    mt5.copy_rates_from_pos = fake.copy_rates_from_pos

    momentum = PREDICTORS["momentum"].predict("SIM", None)
    breakout = PREDICTORS["breakout"].predict("SIM", None)
    micro = PREDICTORS["micro"].predict("SIM", None)
    vwap = PREDICTORS["vwap"].predict("SIM", None)
    candle = PREDICTORS["candle"].predict("SIM", None)

    code = build_condition_code(
        momentum=momentum,
        breakout=breakout,
        vwap=vwap,
        micro=micro,
        candle=candle
    )

    return code


# ============================================================
# Process market condition folder
# ============================================================

def process_market_conditions(folder):

    results = []

    files = sorted(os.listdir(folder))

    for file in files:

        if not file.endswith(".csv"):
            continue

        print("\nProcessing:", file)

        path = os.path.join(folder, file)

        df = pd.read_csv(path)

        # ensure timestamp format
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # keep latest candles
        df = df.tail(300)

        code = generate_condition_code(df)

        condition_name = file.replace(".csv", "")

        print("Condition Code →", code)

        results.append({
            "market_condition": condition_name,
            "condition_code": code
        })

    results_df = pd.DataFrame(results)

    results_df.to_csv("condition_code_map.csv", index=False)

    print("\nSaved → condition_code_map.csv")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    folder = "market_conditions"

    process_market_conditions(folder)