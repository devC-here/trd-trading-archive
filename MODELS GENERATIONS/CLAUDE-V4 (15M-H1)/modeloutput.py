# run_all_models.py
# Runs all 5 H1 models against a single symbol and prints results in one consolidated block.
# ─────────────────────────────────────────────────────────────────────
import logging
import MetaTrader5 as mt5

from deploy_momentum_h1       import MomentumPredictor
from deploy_vwap_h1           import VwapPredictor
from deploy_microstructure_h1 import MicrostructurePredictor
from deploy_candle_h1         import CandlePredictor
from deploy_breakout_h1       import BreakoutPredictor

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────
SYMBOL    = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_H1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def ensure_symbol(symbol: str) -> bool:
    info = mt5.symbol_info(symbol)
    if info is None:
        logging.error("Symbol not found: %s", symbol)
        return False
    if not info.visible:
        mt5.symbol_select(symbol, True)
    return True


def fmt(r: dict) -> str:
    """Format a single model result dict into a one-line string."""
    d   = r.get("direction",  {})
    mag = r.get("magnitude",  {})
    rng = r.get("range_type", {})
    loc = r.get("close_loc",  {})
    bdy = r.get("body_ratio", {})
    mom = r.get("momentum",   {})
    return (
        f"signal={d.get('label','?'):<8} | conf={d.get('confidence',0):.2%} | "
        f"mag={mag.get('label','?'):<7} | range={rng.get('label','?'):<7} | "
        f"loc={loc.get('label','?'):<6} | body={bdy.get('label','?'):<9} | "
        f"mom={mom.get('label','?'):<9} | trade_active={r.get('trade_active', False)}"
    )


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    if not mt5.initialize():
        logging.error("MT5 initialization failed.")
        return

    if not ensure_symbol(SYMBOL):
        mt5.shutdown()
        return

    predictors = {
        "momentum":       MomentumPredictor(),
        "vwap":           VwapPredictor(),
        "microstructure": MicrostructurePredictor(),
        "candle":         CandlePredictor(),
        "breakout":       BreakoutPredictor(),
    }
    logging.info("All 5 predictors initialized.")

    try:
        logging.info("=" * 120)
        logging.info(f"SYMBOL: {SYMBOL}  |  TIMEFRAME: H1")
        logging.info("=" * 120)

        for name, model in predictors.items():
            try:
                r = model.predict(SYMBOL, TIMEFRAME)
                if "error" in r:
                    logging.warning(f"  [{name.upper():<15}]  ERROR: {r['error']}")
                else:
                    logging.info(f"  [{name.upper():<15}]  {fmt(r)}")
            except Exception as e:
                logging.error(f"  [{name.upper():<15}]  prediction failed: {e}")

        logging.info("-" * 120)

    except Exception as e:
        logging.error(f"Run failed: {e}")

    finally:
        for model in predictors.values():
            try: model.shutdown()
            except: pass
        mt5.shutdown()
        logging.info("MT5 disconnected.")


if __name__ == "__main__":
    main()