# ensemble_collector.py
# Collects outputs from all 7 predictors in one place

import MetaTrader5 as mt5

from breakout import BreakoutPredictor
from candle import CandlePredictor
from macro import MacroPredictor
from microstructure import MicrostructurePredictor
from momentum import MomentumPredictor
from structure import StructurePredictor
from VWAP import VWAPPredictor


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


# Example usage
if __name__ == "__main__":
    # Initialize MT5 before running (make sure account is logged in)
    if not mt5.initialize():
        print("MT5 initialization failed")
    else:
        collector = EnsembleCollector()
        outputs = collector.collect("EURUSDm", mt5.TIMEFRAME_M15)

        # Print clean summary
        import json
        print(json.dumps(outputs, indent=2))
        
        mt5.shutdown()
