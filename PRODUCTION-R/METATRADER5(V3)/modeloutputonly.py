import MetaTrader5 as mt5
import logging

from momentumdeploy import MomentumPredictor
from breakoutdeploy import BreakoutPredictor
from microstructure import MicrostructurePredictor
from VWdeploy import VwapVolumePredictor
from candledeploy import CandlestickSequencePredictor
from code1 import evaluate_market_state, trading_conditions

SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_H1

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def ensure_symbol(symbol):
    info = mt5.symbol_info(symbol)
    if info is None:
        logging.error("Symbol %s not found.", symbol)
        return False
    if not info.visible:
        mt5.symbol_select(symbol, True)
    return True

def initialize_predictors():
    global PREDICTORS
    PREDICTORS = {
        "momentum": MomentumPredictor(),
        "breakout": BreakoutPredictor(),
        "micro":    MicrostructurePredictor(),
        "vwap":     VwapVolumePredictor(),
        "candle":   CandlestickSequencePredictor(),
    }


def get_raw_predictions(symbol, timeframe):
    raw = {}
 
    predictors_run = {
        "momentum": (PREDICTORS["momentum"], ["momentum_state", "momentum_slope_zone", "entry_bias"]),
        "breakout": (PREDICTORS["breakout"], ["squeeze_state", "breakout_bias", "entry_viability"]),
        "micro":    (PREDICTORS["micro"],    ["burst_type", "tail_pressure_direction", "micro_risk_zone"]),
        "vwap":     (PREDICTORS["vwap"],     ["vwap_zone", "volume_pressure", "liquidity_structure"]),
        "candle":   (PREDICTORS["candle"],   ["sequence_type", "candle_dominance", "pattern_strength"]),
    }
 
    print(f"\n{'─'*60}")
    print(f"  PREDICTIONS  |  {symbol}  [{TIMEFRAME}]")
    print(f"{'─'*60}")
    print(f"  {'Model':<12} {'Confidence':<12} {'Fields'}")
    print(f"{'─'*60}")
 
    for name, (model, keys) in predictors_run.items():
        r = model.predict(symbol, timeframe)
        conf = r.get("confidence", 0)
        fields = {k: r.get(k) for k in keys}
        raw[name] = r  # pass full dict to condition engine
        field_str = "  ".join(f"{k}={v}" for k, v in fields.items())
        print(f"  {name:<12} {conf:<12.4f} {field_str}")
 
    print(f"{'─'*60}\n")
    return raw
 

def main():
    if not mt5.initialize():
        logging.error("MT5 init failed.")
        return

    if not ensure_symbol(SYMBOL):
        return

    initialize_predictors()

    raw = get_raw_predictions(SYMBOL, TIMEFRAME)

    condition = evaluate_market_state(
        momentum=raw["momentum"],
        breakout=raw["breakout"],
        micro=raw["micro"],
        vwap=raw["vwap"],
        candle=raw["candle"],
        conditions=trading_conditions
    )

    logging.info(
        "CODE=%s | ID=%s | BIAS=%s | CONF=%.2f | CONSENSUS=%s | %s",
        condition["condition_code"],
        condition.get("condition_id"),
        condition["final_bias"],
        condition["confidence"],
        condition.get("consensus_used", False),
        condition["meaning"]
    )

    mt5.shutdown()

if __name__ == "__main__":
    main()