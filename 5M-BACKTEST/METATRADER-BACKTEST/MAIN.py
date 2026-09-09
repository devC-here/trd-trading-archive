import MetaTrader5 as mt5
import logging
from datetime import datetime

# ===============================
# Predictor imports
# ===============================
from momentumdeploy import MomentumPredictor
from breakoutdeploy import BreakoutPredictor
from microstructure import MicrostructurePredictor
from VWdeploy import VwapVolumePredictor
from candledeploy import CandlestickSequencePredictor

# ===============================
# Fusion / decision engine
# ===============================
from fusion import fuse_model_outputs as combine_model_outputs

# ===============================
# Config
# ===============================
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_H1

# How many candles ahead of each prediction to validate direction against.
# e.g. (1, 2, 3) checks the very next candle, the one after, and the one after that.
FORWARD_HORIZONS = (1, 2, 3)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

PREDICTORS = {}

# ------------------------------------------------------------
# Symbol check
# ------------------------------------------------------------
def ensure_symbol(symbol):
    info = mt5.symbol_info(symbol)
    if info is None:
        logging.error("Symbol %s not found in terminal.", symbol)
        return False
    if not info.visible:
        mt5.symbol_select(symbol, True)
    return True

# ------------------------------------------------------------
# Initialization
# ------------------------------------------------------------
def initialize_predictors():
    global PREDICTORS
    PREDICTORS = {
        "momentum": MomentumPredictor(),
        "breakout": BreakoutPredictor(),
        "micro":    MicrostructurePredictor(),
        "vwap":     VwapVolumePredictor(),
        "candle":   CandlestickSequencePredictor(),
    }
    logging.info("All predictors initialized.")

# ------------------------------------------------------------
# Prediction step (anchored at a given historical candle)
# ------------------------------------------------------------
def get_predictions(symbol, timeframe, start_candle):
    raw = {}
    for name, model in PREDICTORS.items():
        try:
            raw[name] = model.predict(symbol, timeframe, start_candle=start_candle)
            logging.info(
                "%s (candle %d) -> %s",
                name.upper(), start_candle, raw[name]
            )
        except Exception as e:
            logging.error("%s failed at candle %d: %s", name.upper(), start_candle, e)
            return None
    return raw

# ------------------------------------------------------------
# Decision engine (fusion-based)
# ------------------------------------------------------------
def get_decision(symbol, timeframe, start_candle):
    raw = get_predictions(symbol, timeframe, start_candle)
    if not raw:
        return "Hold", {}, {}

    fusion = combine_model_outputs(
        raw["momentum"], raw["breakout"], raw["micro"], raw["vwap"], raw["candle"]
    )

    direction = fusion.get("final_bias")
    viability = fusion.get("entry_viability")

    if direction == "long" and viability == "high":
        decision = "Buy"
    elif direction == "short" and viability == "high":
        decision = "Sell"
    else:
        decision = "Hold"

    return decision, fusion, raw

# ------------------------------------------------------------
# Forward validation: did price actually move as predicted?
# ------------------------------------------------------------
def get_candle_direction(symbol, timeframe, pos):
    """Return 'bull', 'bear', or 'flat' for the fully-closed candle at position `pos`."""
    if pos < 1:
        return None  # candle isn't closed yet, can't be validated
    rates = mt5.copy_rates_from_pos(symbol, timeframe, pos, 1)
    if rates is None or len(rates) == 0:
        return None
    bar = rates[0]
    if bar["close"] > bar["open"]:
        return "bull"
    elif bar["close"] < bar["open"]:
        return "bear"
    return "flat"

def evaluate_forward_candles(symbol, timeframe, start_candle, decision, horizons=FORWARD_HORIZONS):
    """
    Looks at the candle(s) that come *after* the prediction candle (i.e. closer
    to the present than start_candle) and marks whether each one closed in the
    direction implied by `decision`.

    Buy  -> True if that candle closed bullish (close > open)
    Sell -> True if that candle closed bearish (close < open)

    Returns {horizon: True/False/None}. None = flat candle or no data (skipped
    from scoring, not counted as a miss).
    """
    results = {}
    for h in horizons:
        pos = start_candle - h
        direction = get_candle_direction(symbol, timeframe, pos)
        if direction is None or direction == "flat":
            results[h] = None
            continue
        if decision == "Buy":
            results[h] = (direction == "bull")
        elif decision == "Sell":
            results[h] = (direction == "bear")
        else:
            results[h] = None
    return results

# ------------------------------------------------------------
# Single backtest cycle
# ------------------------------------------------------------
def run_single_cycle(start_candle):
    decision, fusion, raw = get_decision(SYMBOL, TIMEFRAME, start_candle)

    logging.info(
        "Candle %d -> Decision=%s | Bias=%s | Confidence=%s",
        start_candle, decision, fusion.get("final_bias"), fusion.get("confidence")
    )

    forward_results = {}
    if decision in ("Buy", "Sell"):
        forward_results = evaluate_forward_candles(SYMBOL, TIMEFRAME, start_candle, decision)
        for h, correct in forward_results.items():
            tag = "N/A" if correct is None else ("TRUE" if correct else "FALSE")
            logging.info("  +%d candle(s) ahead -> %s", h, tag)

    return {
        "start_candle": start_candle,
        "decision": decision,
        "confidence": fusion.get("confidence"),
        "final_bias": fusion.get("final_bias"),
        "entry_viability": fusion.get("entry_viability"),
        "supporting_factors": fusion.get("supporting_factors"),
        "raw_models": raw,
        "forward_results": forward_results,
    }

# ------------------------------------------------------------
# Distribution summary across all cycles
# ------------------------------------------------------------
def summarize_distribution(records, horizons=FORWARD_HORIZONS):
    summary = {}
    for h in horizons:
        true_count = false_count = na_count = 0
        for r in records:
            val = r["forward_results"].get(h)
            if val is True:
                true_count += 1
            elif val is False:
                false_count += 1
            else:
                na_count += 1

        total_scored = true_count + false_count
        win_rate = (true_count / total_scored * 100) if total_scored else 0.0

        summary[h] = {
            "true": true_count,
            "false": false_count,
            "skipped": na_count,
            "win_rate_pct": round(win_rate, 2),
        }
        logging.info(
            "Horizon +%d -> TRUE=%d  FALSE=%d  SKIPPED=%d  WIN RATE=%.2f%%",
            h, true_count, false_count, na_count, win_rate
        )
    return summary

# ------------------------------------------------------------
# Backtest runner
# ------------------------------------------------------------
def run_backtest(symbol=SYMBOL, timeframe=TIMEFRAME, start_candle=2000, end_candle=10, step=1):
    """
    Walks from `start_candle` (furthest back in history) down to `end_candle`
    (closest to the present, but must stay > max(FORWARD_HORIZONS) so the
    forward-validation candles are always fully closed).
    """
    if not mt5.initialize():
        logging.error("MT5 init failed.")
        return [], {}

    if not ensure_symbol(symbol):
        mt5.shutdown()
        return [], {}

    initialize_predictors()
    logging.info("Backtest started: candles %d -> %d (step %d)", start_candle, end_candle, step)

    records = []
    try:
        for idx in range(start_candle, end_candle - 1, -step):
            record = run_single_cycle(idx)
            records.append(record)
    except KeyboardInterrupt:
        logging.info("Backtest interrupted by user.")
    finally:
        mt5.shutdown()
        logging.info("MT5 shutdown.")

    summary = summarize_distribution(records)
    return records, summary

if __name__ == "__main__":
    # Placeholder default range — adjust start_candle/end_candle as needed.
    results, summary = run_backtest(SYMBOL, TIMEFRAME, start_candle=500, end_candle=10, step=1)
    logging.info("Backtest complete. %d cycles run.", len(results))