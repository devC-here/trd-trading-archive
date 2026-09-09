"""
MAIN_backtest.py
────────────────
Direction-accuracy backtest using the live deploy predictors unchanged.

HOW IT WORKS
  For each offset N (NUM_BARS → 1):
    • All 5 predictors called with  start=N   → prediction
    • Actual direction               start=N-1 → compare close vs close
      (N-1 is one bar newer, i.e. the "next" bar after the predicted bar)
  Results saved to CSV.

USAGE
  python MAIN_backtest.py
"""

import MetaTrader5 as mt5
import pandas as pd
import logging
import os

# ── live predictors, used as-is ──────────────────────────────
from momentumdeploy import MomentumPredictor
from breakoutdeploy import BreakoutPredictor
from microstructure import MicrostructurePredictor
from candledeploy   import CandlestickSequencePredictor
from VWdeploy       import VwapVolumePredictor

# ── condition engine (same as live) ──────────────────────────
from code1 import evaluate_market_state, trading_conditions

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════
SYMBOL    = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M15

NUM_BARS  = 2000          # how many bars back to test (start=NUM_BARS down to start=1)

OUTPUT_CSV = f"backtest_results_{SYMBOL}_{TIMEFRAME}.csv"
# ═══════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

TF_LABEL = {
    mt5.TIMEFRAME_M1:  "M1",  mt5.TIMEFRAME_M5:  "M5",
    mt5.TIMEFRAME_M15: "M15", mt5.TIMEFRAME_M30: "M30",
    mt5.TIMEFRAME_H1:  "H1",  mt5.TIMEFRAME_H4:  "H4",
    mt5.TIMEFRAME_D1:  "D1",  mt5.TIMEFRAME_W1:  "W1",
}.get(TIMEFRAME, str(TIMEFRAME))


def get_close(symbol, timeframe, start):
    """Fetch a single bar's close at the given start offset."""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, start, 1)
    if rates is None or len(rates) == 0:
        return None
    return float(rates[0]["close"])


def get_bar_time(symbol, timeframe, start):
    """Fetch a single bar's open time at the given start offset."""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, start, 1)
    if rates is None or len(rates) == 0:
        return None
    return pd.to_datetime(rates[0]["time"], unit="s")


def main():
    logging.info("═" * 60)
    logging.info("BACKTEST  %s [%s]  bars=%d", SYMBOL, TF_LABEL, NUM_BARS)
    logging.info("═" * 60)

    # Load all predictors once (they call mt5.initialize() internally)
    logging.info("Loading models…")
    momentum  = MomentumPredictor()
    breakout  = BreakoutPredictor()
    micro     = MicrostructurePredictor()
    candle    = CandlestickSequencePredictor()
    vwap      = VwapVolumePredictor()
    logging.info("All models loaded.")

    rows    = []
    skipped = 0

    # N = start offset for the prediction bar
    # N-1 = the bar that follows it (closer to present) → actual direction
    for n in range(NUM_BARS, 0, -1):

        # ── predictions at bar N ──────────────────────────────
        r_mom  = momentum.predict(SYMBOL, TIMEFRAME, start=n)
        r_brk  = breakout.predict(SYMBOL, TIMEFRAME, start=n)
        r_mic  = micro.predict(SYMBOL, TIMEFRAME, start=n)
        r_cnd  = candle.predict(SYMBOL, TIMEFRAME, start=n)
        r_vwp  = vwap.predict(SYMBOL, TIMEFRAME, start=n)

        # skip if any predictor failed
        if any("error" in r for r in (r_mom, r_brk, r_mic, r_cnd, r_vwp)):
            skipped += 1
            continue

        # ── condition engine ──────────────────────────────────
        condition = evaluate_market_state(
            momentum=r_mom,
            breakout=r_brk,
            micro=r_mic,
            vwap=r_vwp,
            candle=r_cnd,
            conditions=trading_conditions,
        )

        # ── actual direction: compare close[N] vs close[N-1] ──
        close_n   = get_close(SYMBOL, TIMEFRAME, n)
        close_nm1 = get_close(SYMBOL, TIMEFRAME, n - 1)

        if close_n is None or close_nm1 is None:
            skipped += 1
            continue

        actual    = "long" if close_nm1 > close_n else "short"
        predicted = condition["final_bias"]      # "long" | "short" | "hold" | "no_trade"

        if predicted in ("hold", "no_trade", None):
            direction_correct = "skip"
        else:
            direction_correct = "yes" if predicted == actual else "no"

        rows.append({
            "start_offset":     n,
            "datetime":         get_bar_time(SYMBOL, TIMEFRAME, n),
            "close_pred_bar":   close_n,
            "close_next_bar":   close_nm1,
            # individual confidences
            "momentum_conf":    r_mom.get("confidence", 0),
            "breakout_conf":    r_brk.get("confidence", 0),
            "micro_conf":       r_mic.get("confidence", 0),
            "vwap_conf":        r_vwp.get("confidence", 0),
            "candle_conf":      r_cnd.get("confidence", 0),
            # condition engine
            "condition_code":   condition["condition_code"],
            "condition_id":     condition.get("condition_id"),
            "final_bias":       predicted,
            "engine_conf":      condition["confidence"],
            "consensus_used":   condition.get("consensus_used", False),
            "meaning":          condition["meaning"],
            # result
            "actual_direction": actual,
            "direction_correct":direction_correct,
        })

        # progress log every 100 bars
        if len(rows) % 100 == 0:
            evaluated = [r for r in rows if r["direction_correct"] != "skip"]
            if evaluated:
                correct = sum(1 for r in evaluated if r["direction_correct"] == "yes")
                acc = correct / len(evaluated) * 100
                logging.info(
                    "N=%d | evaluated=%d  correct=%d  acc=%.1f%%",
                    n, len(evaluated), correct, acc,
                )

    # ── save CSV ──────────────────────────────────────────────
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_CSV, index=False)
    logging.info("Results saved → %s", os.path.abspath(OUTPUT_CSV))

    # ── summary ───────────────────────────────────────────────
    evaluated     = out_df[out_df["direction_correct"] != "skip"]
    skipped_preds = (out_df["direction_correct"] == "skip").sum()

    if len(evaluated) == 0:
        logging.warning("No actionable predictions (all skipped or errored).")
        mt5.shutdown()
        return

    correct  = (evaluated["direction_correct"] == "yes").sum()
    wrong    = (evaluated["direction_correct"] == "no").sum()
    accuracy = correct / len(evaluated) * 100

    logging.info("═" * 60)
    logging.info("BACKTEST SUMMARY  %s [%s]", SYMBOL, TF_LABEL)
    logging.info("  Total bars logged    : %d", len(out_df))
    logging.info("  Skipped (hold/error) : %d  (+%d model errors)", skipped_preds, skipped)
    logging.info("  Directional signals  : %d", len(evaluated))
    logging.info("  Correct              : %d", correct)
    logging.info("  Wrong                : %d", wrong)
    logging.info("  Direction Accuracy   : %.2f%%", accuracy)
    logging.info("═" * 60)

    if "condition_code" in out_df.columns:
        logging.info("Per-condition accuracy:")
        logging.info("  %-6s  %-30s  %s", "ID", "Code", "Correct/Total (Acc)")
        logging.info("  " + "─" * 56)
        for (cid, code), grp in evaluated.groupby(["condition_id", "condition_code"]):
            c = (grp["direction_correct"] == "yes").sum()
            n = len(grp)
            logging.info("  %-6s  %-30s  %d/%d  (%.1f%%)", cid, code, c, n, c / n * 100)

    mt5.shutdown()


if __name__ == "__main__":
    main()
