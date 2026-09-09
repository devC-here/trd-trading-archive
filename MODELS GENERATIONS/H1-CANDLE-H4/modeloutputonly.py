import MetaTrader5 as mt5
import logging
from datetime import datetime
import numpy as np

# ===============================
# Predictor imports
# ===============================
from momentum_deploy import MomentumPredictor
from breakout_deploy import BreakoutPredictor
from microstructure_deploy import MicrostructurePredictor
from vwap_deploy import VwapPredictor
from candle_deploy import CandlePredictor

# ===============================
# Config
# ===============================
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
OUTPUT_STEPS = 4  # Captures t-1, t-2, t-3, t-4 panel steps

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
        logging.error("Symbol not found: %s", symbol)
        return False

    if not info.visible:
        mt5.symbol_select(symbol, True)

    return True


# ------------------------------------------------------------
# Predictor initialization
# ------------------------------------------------------------
def initialize_predictors():
    global PREDICTORS

    PREDICTORS = {
        "breakout": BreakoutPredictor(),
        "candle": CandlePredictor(),
        "microstructure": MicrostructurePredictor(),
        "momentum": MomentumPredictor(),
        "vwap": VwapPredictor(),
    }

    logging.info("All predictors initialized.")


# ------------------------------------------------------------
# Run predictors (Horizontal Log Formatting)
# ------------------------------------------------------------
def run_predictions(symbol, timeframe):
    # Iterate through models grouped by type
    for name, model in PREDICTORS.items():
        try:
            results = model.predict_multi(symbol, timeframe, n=OUTPUT_STEPS)
            
            # Print the header exactly matching your log style
            logging.info(f"MODEL → {name.upper()} (Total Panels: {len(results)})")
            
            # Sort steps chronologically from oldest (t-4) to newest (t-1)
            sorted_results = sorted(results, key=lambda x: int(x.get("bar", "t-1").split("-")[1]), reverse=True)
            
            horizontal_parts = []
            for r in sorted_results:
                bar_label = r.get("bar", "N/A")
                
                # Extract and format metadata based on each model's layout
                if name == "breakout":
                    meta = (f"signal={r['next_direction']['label']} | conf={r['next_direction']['confidence']:.2%} | "
                            f"mag={r['next_magnitude']['label']} | pat={r['next_pattern']['label']} | "
                            f"vol={r['next_volatility']['label']} | closeloc={r['next_close_loc']['label']} | "
                            f"follow={r['bk_follow_through']['label']} | trade_active={r['trade_active']}")
                            
                elif name == "candle":
                    meta = (f"signal={r['next_direction']['label']} | conf={r['next_direction']['confidence']:.2%} | "
                            f"size={r['next_body_size']['label']} | pat={r['next_pattern']['label']} | "
                            f"seq={r['seq_continuation']['label']} | closeloc={r['next_close_grav']['label']} | "
                            f"vol={r['next_volatility']['label']} | trade_active={r['trade_active']}")
                            
                elif name == "microstructure":
                    meta = (f"signal={r['next_flow_dir']['label']} | conf={r['next_flow_dir']['confidence']:.2%} | "
                            f"intent={r['next_bar_intent']['label']} | mag={r['next_magnitude']['label']} | "
                            f"rej={r['next_rejection']['label']} | follow={r['momentum_follow']['label']} | "
                            f"range={r['next_range_type']['label']} | trade_active={r['trade_active']}")
                            
                elif name == "momentum":
                    s = r.get("_summary", {})
                    meta = (f"signal={s.get('direction')} | conf={s.get('signal_conf', 0):.2%} | "
                            f"mag={s.get('magnitude')} | regime={s.get('regime')} | "
                            f"vol={s.get('volatility')} | bias={s.get('structure')} | "
                            f"quality={s.get('momentum')} | trade_active={s.get('trade_active')}")
                            
                elif name == "vwap":
                    meta = (f"signal={r['next_vwap_dir']['label']} | conf={r['next_vwap_dir']['confidence']:.2%} | "
                            f"volchar={r['next_vol_char']['label']} | interact={r['vwap_interaction']['label']} | "
                            f"instsig={r['next_inst_signal']['label']} | closeloc={r['next_close_location']['label']} | "
                            f"follow={r['vwap_follow_through']['label']} | trade_active={r['trade_active']}")

                # Wrap the step metrics string nicely
                horizontal_parts.append(f"[{bar_label}] -> {meta}")

            # Print out the aggregated time series block as a single consolidated row
            logging.info("  ||  ".join(horizontal_parts))
            logging.info("-" * 120)

        except Exception as e:
            logging.error(f"Prediction matrix aggregation failed for {name}: {str(e)}")


# ------------------------------------------------------------
# Main Script Entry Point (Single Shot Execution)
# ------------------------------------------------------------
def main():
    if not mt5.initialize():
        logging.error("MT5 initialization process failed.")
        return

    if not ensure_symbol(SYMBOL):
        mt5.shutdown()
        return

    try:
        initialize_predictors()
        run_predictions(SYMBOL, TIMEFRAME)
        
    except Exception as run_err:
        logging.error(f"Execution error encountered: {str(run_err)}")
        
    finally:
        mt5.shutdown()
        logging.info("MT5 interface disconnected cleanly.")


if __name__ == "__main__":
    main()