# ensemble_collector.py (MAIN.py)
# Collects outputs from all 7 predictors, adds OHLCV context, and asks Gemini for a final signal

import MetaTrader5 as mt5
import google.generativeai as genai
import json
import pandas as pd
import numpy as np
import warnings

from breakout import BreakoutPredictor
from candle import CandlePredictor
from macro import MacroPredictor
from microstructure import MicrostructurePredictor
from momentum import MomentumPredictor
from structure import StructurePredictor
from VWAP import VWAPPredictor

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)


class EnsembleCollector:
    def __init__(self):
        # Initialize all predictors
        self.breakout = BreakoutPredictor()
        self.candle = CandlePredictor()
        self.macro = MacroPredictor()
        self.micro = MicrostructurePredictor()
        self.momentum = MomentumPredictor()
        self.structure = StructurePredictor()
        self.vwap = VWAPPredictor()

    def collect(self, symbol="EURUSDm", timeframe=mt5.TIMEFRAME_M15):
        """
        Runs all predictors and returns a dictionary of results.
        """
        results = {
            "breakout": self.breakout.predict(symbol, timeframe),
            "candle": self.candle.predict(symbol, timeframe),
            "macro": self.macro.predict(symbol, timeframe),
            "micro": self.micro.predict(symbol, timeframe),
            "momentum": self.momentum.predict(symbol, timeframe),
            "structure": self.structure.predict(symbol, timeframe),
            "vwap": self.vwap.predict(symbol, timeframe),
        }
        return results


def get_recent_ohlcv(symbol="EURUSDm", timeframe=mt5.TIMEFRAME_M15, n=30):
    """Fetch last n candles from MT5."""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df[["time", "open", "high", "low", "close", "tick_volume"]]


def get_swing_summary(symbol="EURUSDm", timeframe=mt5.TIMEFRAME_M15, n=100):
    """Generate swing-level summary stats from last n candles."""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    closes = df["close"].values

    summary = {
        "trend_slope": float(np.polyfit(range(len(closes)), closes, 1)[0]),  # slope of close
        "atr14": float((df["high"] - df["low"]).rolling(14).mean().iloc[-1]),
        "swing_high": float(df["high"].max()),
        "swing_low": float(df["low"].min()),
        "last_close": float(closes[-1]),
        "vwap_bias": "above" if closes[-1] > (df["close"] * df["tick_volume"]).sum() / df["tick_volume"].sum() else "below",
    }

    return summary


def get_gemini_signal(predictor_outputs: dict, ohlcv: pd.DataFrame, swing_summary: dict) -> str:
    """Ask Gemini for final signal."""
    genai.configure(api_key="AIzaSyD28XS1hiMGszIKVT2Th0BKfuvaF_SvKc8")  # or set GOOGLE_API_KEY env var
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""
    You are a trading decision assistant.

    Given:
    1. Predictor outputs from 7 models
    2. Last 30 OHLCV candles (M15) for intraday context
    3. A swing summary (100 candles, ATR, trend, VWAP bias, HH/LL)

    Task:
    - Return exactly ONE word: buy, sell, or hold.
    - Do not explain, no punctuation, no extra words.

    Predictor outputs:
    {json.dumps(predictor_outputs, indent=2)}

    Last 30 OHLCV candles:
    {ohlcv.to_string(index=False)}

    Swing summary (100 candles):
    {json.dumps(swing_summary, indent=2)}
    """

    response = model.generate_content(prompt, generation_config={"temperature": 0.0})
    return response.text.strip().lower()


# Example usage
if __name__ == "__main__":
    if not mt5.initialize():
        print("❌ MT5 initialization failed")
    else:
        symbol = "EURUSDm"
        timeframe = mt5.TIMEFRAME_M15

        collector = EnsembleCollector()
        outputs = collector.collect(symbol, timeframe)

        ohlcv_recent = get_recent_ohlcv(symbol, timeframe, n=30)
        swing_summary = get_swing_summary(symbol, timeframe, n=100)

        final_signal = get_gemini_signal(outputs, ohlcv_recent, swing_summary)

        print("📊 Predictor outputs collected.")
        print("📈 Final Trading Signal (Gemini):", final_signal.upper())

        mt5.shutdown()
