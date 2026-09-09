# run_all_models.py
# Single-shot runner for all 5 H1 models + ensemble signal.
# ─────────────────────────────────────────────────────────────────────
import logging
import MetaTrader5 as mt5

from deploy_momentum_h1       import MomentumPredictor
from deploy_vwap_h1           import VwapPredictor
from deploy_microstructure_h1 import MicrostructurePredictor
from deploy_candle_h1         import CandlePredictor
from deploy_breakout_h1       import BreakoutPredictor
from ensemble_h1              import from_predict, ensemble, log_ensemble

SYMBOL    = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_H1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def ensure_symbol(symbol: str) -> bool:
    info = mt5.symbol_info(symbol)
    if info is None:
        logging.error("Symbol not found: %s", symbol); return False
    if not info.visible:
        mt5.symbol_select(symbol, True)
    return True

def _fmt(r: dict) -> str:
    d = r["direction"]; ma = r["magnitude"]; rg = r["range_type"]
    cl = r["close_loc"]; bo = r["body_ratio"]; mo = r["momentum"]
    return (f"signal={d['label']:<8} conf={d['confidence']:.2%} | "
            f"mag={ma['label']:<8} range={rg['label']:<8} "
            f"loc={cl['label']:<6} body={bo['label']:<10} "
            f"mom={mo['label']:<10} trade_active={r['trade_active']}")

def main():
    if not mt5.initialize():
        logging.error("MT5 initialization failed."); return
    if not ensure_symbol(SYMBOL):
        mt5.shutdown(); return

    predictors = {
        "momentum":       MomentumPredictor(),
        "vwap":           VwapPredictor(),
        "microstructure": MicrostructurePredictor(),
        "candle":         CandlePredictor(),
        "breakout":       BreakoutPredictor(),
    }
    logging.info("All 5 predictors initialised.")

    try:
        logging.info("=" * 120)
        logging.info(f"SYMBOL: {SYMBOL}  |  TIMEFRAME: H1")
        logging.info("=" * 120)

        model_outputs = []
        for name, model in predictors.items():
            try:
                r = model.predict(SYMBOL, TIMEFRAME)
                if "error" in r:
                    logging.warning(f"  [{name.upper():<16}]  ERROR: {r['error']}")
                    continue
                logging.info(f"  [{name.upper():<16}]  {_fmt(r)}")
                model_outputs.append(from_predict(name, r))
            except Exception as e:
                logging.error(f"  [{name.upper():<16}]  FAILED: {e}")

        logging.info("-" * 120)

        if len(model_outputs) >= 3:
            result = ensemble(model_outputs)
            log_ensemble(result)
        else:
            logging.warning("Not enough model outputs for ensemble (need ≥ 3).")

    except Exception as e:
        logging.error(f"Runner error: {e}")
    finally:
        for m in predictors.values():
            try: m.shutdown()
            except: pass
        mt5.shutdown()
        logging.info("MT5 disconnected.")

if __name__ == "__main__":
    main()
