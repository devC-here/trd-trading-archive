"""
main.py — Raw Model Output Mapper
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For each H1 bar in history:
  1. Run all 5 models on M15 bars that closed BEFORE this H1 bar
  2. Record every raw model output value
  3. Label the NEXT H1 bar's candle type and direction

CANDLE LABELING
  label = 1   → definite bullish   (clear up-close, body dominates)
  label = 0   → definite bearish   (clear down-close, body dominates)
  label = -1  → excluded (directionless candle — see types below)

EXCLUDED CANDLE TYPES (label = -1):
  1. Classic doji        — body < 5% of total range
  2. Long-legged doji    — body < 10% of range AND both wicks > 30% each
  3. Spinning top        — body < 25% of range AND both wicks roughly equal
                           (upper_wick and lower_wick both > 25% of range)
  4. Gravestone doji     — tiny body at low, long upper wick (>70% of range)
  5. Dragonfly doji      — tiny body at high, long lower wick (>70% of range)
  6. High wave candle    — total wick > 3× body size (wicks overwhelm body)

These candles have no clear directional meaning and would add noise
to the label. Better to exclude them and work with clean signals.

OUTPUT: model_map_EURUSD.csv
"""

import warnings
warnings.filterwarnings("ignore")

import MetaTrader5 as mt5
import pandas as pd
import csv
import os
import logging
from datetime import datetime, timezone

from deploy_model1_momentum   import MomentumPredictor
from deploy_model2_structure  import StructurePredictor
from deploy_model3_volatility import VolatilityPredictor
from deploy_model4_volume     import VolumePredictor
from deploy_model5_geometry   import GeometryPredictor

# ══════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════
SYMBOL      = "EURUSD"
M15_TF      = mt5.TIMEFRAME_M15
H1_TF       = mt5.TIMEFRAME_H1
NUM_H1_BARS = 20000
OUTPUT_CSV  = f"model_map_{SYMBOL}.csv"
# ══════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

CSV_COLUMNS = [
    # identity
    "h1_offset", "datetime_utc", "hour_utc", "day_of_week",
    # H1 signal bar
    "h1_open", "h1_close", "h1_high", "h1_low",
    # M1
    "m1_p_bull", "m1_p_bear", "m1_confidence", "m1_direction",
    # M2
    "m2_Es", "m2_zone", "m2_fib382_dist", "m2_s1_dist", "m2_r1_dist",
    # M3
    "m3_p_breakout", "m3_p_chop", "m3_regime",
    "m3_bb_z", "m3_kc_squeeze", "m3_tr_z", "m3_fdi",
    # M4
    "m4_Ic", "m4_interpretation", "m4_mfa",
    "m4_vol_slope", "m4_vol_osc", "m4_rv_z",
    # M5
    "m5_Rp", "m5_interpretation",
    "m5_upper_wick", "m5_lower_wick", "m5_body_atr",
    # next H1 bar geometry (for your own inspection)
    "next_h1_open", "next_h1_close", "next_h1_high", "next_h1_low",
    "next_h1_body_pct",    # body as % of total range
    "next_h1_upper_wick_pct",
    "next_h1_lower_wick_pct",
    "next_h1_move_pips",
    # label
    "candle_type",  # "bullish" | "bearish" | "doji" | "spinning_top" |
                    # "long_legged_doji" | "gravestone" | "dragonfly" | "high_wave"
    "label",        # 1=bullish  0=bearish  -1=excluded
]


# ══════════════════════════════════════════════════════════════════
#  CANDLE CLASSIFIER
# ══════════════════════════════════════════════════════════════════

def classify_candle(o: float, c: float, h: float, l: float) -> tuple:
    """
    Returns (candle_type: str, label: int)
      label  1 = definite bullish
      label  0 = definite bearish
      label -1 = directionless / excluded

    Geometry definitions:
      total_range = high - low
      body        = abs(close - open)
      upper_wick  = high - max(open, close)
      lower_wick  = min(open, close) - low
      body_pct    = body / total_range
      upper_pct   = upper_wick / total_range
      lower_pct   = lower_wick / total_range
    """
    total = h - l
    if total < 1e-8:
        return ("doji", -1)

    body  = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l

    body_pct  = body  / total
    upper_pct = upper / total
    lower_pct = lower / total

    # ── Excluded types ─────────────────────────────────────────

    # 1. Classic doji — body almost nonexistent
    if body_pct < 0.05:
        return ("doji", -1)

    # 2. Gravestone doji — tiny body at the low, huge upper wick
    #    body in bottom 20% of range, upper wick > 70%
    if body_pct < 0.15 and upper_pct > 0.70:
        return ("gravestone", -1)

    # 3. Dragonfly doji — tiny body at the high, huge lower wick
    #    body in top 20% of range, lower wick > 70%
    if body_pct < 0.15 and lower_pct > 0.70:
        return ("dragonfly", -1)

    # 4. Long-legged doji — body small, both wicks substantial
    if body_pct < 0.10 and upper_pct > 0.30 and lower_pct > 0.30:
        return ("long_legged_doji", -1)

    # 5. Spinning top — body small, both wicks roughly balanced
    #    wicks together take up > 70% of range and are within 2× of each other
    if (body_pct < 0.25
            and upper_pct > 0.25 and lower_pct > 0.25
            and max(upper_pct, lower_pct) < 2.0 * min(upper_pct, lower_pct)):
        return ("spinning_top", -1)

    # 6. High wave — wicks are overwhelming regardless of body direction
    #    total wick > 3× body
    total_wick = upper + lower
    if body > 1e-8 and total_wick / body > 3.0:
        return ("high_wave", -1)

    # ── Directional ────────────────────────────────────────────

    if c > o:
        return ("bullish", 1)
    elif c < o:
        return ("bearish", 0)
    else:
        return ("doji", -1)


def get_rates(symbol, timeframe, start, count):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, start, count)
    if rates is None or len(rates) == 0:
        return None
    return rates


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    logging.info("═" * 62)
    logging.info("MODEL OUTPUT MAPPER  %s  H1 bars=%d", SYMBOL, NUM_H1_BARS)
    logging.info("Output → %s", os.path.abspath(OUTPUT_CSV))
    logging.info("Excluded candle types: doji, gravestone, dragonfly,")
    logging.info("  long-legged doji, spinning top, high wave")
    logging.info("═" * 62)

    if not mt5.initialize():
        logging.error("MT5 init failed: %s", mt5.last_error())
        return

    logging.info("Loading models…")
    m1 = MomentumPredictor  (model_path="model1_best.pth",      meta_path="model1_meta.pkl")
    m2 = StructurePredictor (model_path="model2_best.pth",      scaler_path="model2_scaler.pkl")
    m3 = VolatilityPredictor(model_path="model3_best.pth",      meta_path="model3_meta.pkl")
    m4 = VolumePredictor    (ensemble_path="model4_ensemble.pkl")
    m5 = GeometryPredictor  (ensemble_path="model5_ensemble.pkl")
    logging.info("All models loaded. Scanning…")

    rows          = []
    skip_err      = 0
    candle_counts = {}

    for h in range(NUM_H1_BARS, 1, -1):

        # ── H1 signal bar ─────────────────────────────────────
        h1_bars = get_rates(SYMBOL, H1_TF, h, 1)
        if h1_bars is None:
            skip_err += 1
            continue

        b     = h1_bars[0]
        h1_dt = datetime.fromtimestamp(int(b["time"]), tz=timezone.utc)

        # ── Models on M15 ending before this H1 bar ───────────
        predict_start = h * 4 + 4
        try:
            r1 = m1.predict(SYMBOL, M15_TF, start=predict_start)
            r2 = m2.predict(SYMBOL, M15_TF, start=predict_start)
            r3 = m3.predict(SYMBOL, M15_TF, start=predict_start)
            r4 = m4.predict(SYMBOL, M15_TF, start=predict_start)
            r5 = m5.predict(SYMBOL, M15_TF, start=predict_start)
        except Exception as e:
            logging.error("h=%d: %s", h, e)
            skip_err += 1
            continue

        if any("error" in r for r in (r1, r2, r3, r4, r5)):
            skip_err += 1
            continue

        # ── Next H1 bar — classify and label ──────────────────
        next_bars = get_rates(SYMBOL, H1_TF, h - 1, 1)
        if next_bars is None:
            skip_err += 1
            continue

        nb      = next_bars[0]
        nb_o    = float(nb["open"])
        nb_c    = float(nb["close"])
        nb_h    = float(nb["high"])
        nb_l    = float(nb["low"])
        nb_rng  = nb_h - nb_l if nb_h > nb_l else 1e-8
        nb_body = abs(nb_c - nb_o)
        nb_up   = nb_h - max(nb_o, nb_c)
        nb_dn   = min(nb_o, nb_c) - nb_l

        candle_type, label = classify_candle(nb_o, nb_c, nb_h, nb_l)
        candle_counts[candle_type] = candle_counts.get(candle_type, 0) + 1

        # ── Raw feature extraction ─────────────────────────────
        feats_m3 = r3.get("features", {})
        feats_m4 = r4.get("features", {})
        feats_m5 = r5.get("features", {})

        rows.append({
            # identity
            "h1_offset":             h,
            "datetime_utc":          h1_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "hour_utc":              h1_dt.hour,
            "day_of_week":           h1_dt.strftime("%A"),
            # H1 signal bar
            "h1_open":               round(float(b["open"]),  5),
            "h1_close":              round(float(b["close"]), 5),
            "h1_high":               round(float(b["high"]),  5),
            "h1_low":                round(float(b["low"]),   5),
            # M1
            "m1_p_bull":             round(r1["direction"]["probs"].get("bullish", 0), 5),
            "m1_p_bear":             round(r1["direction"]["probs"].get("bearish", 0), 5),
            "m1_confidence":         round(r1.get("confidence", 0), 5),
            "m1_direction":          r1["direction"].get("label", ""),
            # M2
            "m2_Es":                 round(r2["structural_position"]["E_s"], 5),
            "m2_zone":               r2["structural_position"].get("zone", ""),
            "m2_fib382_dist":        round(r2.get("features", {}).get("43_fib_382_dist", 0), 6),
            "m2_s1_dist":            round(r2.get("features", {}).get("49_s1_dist",      0), 6),
            "m2_r1_dist":            round(r2.get("features", {}).get("48_r1_dist",      0), 6),
            # M3
            "m3_p_breakout":         round(r3["regime"]["p_breakout"], 5),
            "m3_p_chop":             round(r3["regime"]["p_chop"],     5),
            "m3_regime":             r3["regime"].get("label", ""),
            "m3_bb_z":               round(feats_m3.get("62_bollinger_bandwidth_z_score", 0), 5),
            "m3_kc_squeeze":         round(feats_m3.get("63_keltner_channel_squeeze",     0), 5),
            "m3_tr_z":               round(feats_m3.get("72_true_range_z_score",          0), 5),
            "m3_fdi":                round(feats_m3.get("65_fractal_dimension_index",     0), 5),
            # M4
            "m4_Ic":                 round(r4["volume_intensity"]["I_c"], 5),
            "m4_interpretation":     r4["volume_intensity"].get("interpretation", ""),
            "m4_mfa":                round(feats_m4.get("90_money_flow_acceleration", 0), 3),
            "m4_vol_slope":          round(feats_m4.get("37_volume_trend_slope",      0), 5),
            "m4_vol_osc":            round(feats_m4.get("86_volume_oscillator",       0), 5),
            "m4_rv_z":               round(feats_m4.get("80_relative_volume_z_score", 0), 5),
            # M5
            "m5_Rp":                 round(r5["rejection_polarity"]["R_p"], 5),
            "m5_interpretation":     r5["rejection_polarity"].get("interpretation", ""),
            "m5_upper_wick":         round(feats_m5.get("97_final_upper_wick_ratio",      0), 5),
            "m5_lower_wick":         round(feats_m5.get("99_final_lower_wick_ratio",      0), 5),
            "m5_body_atr":           round(feats_m5.get("104_body_size_to_atr_ratio",     0), 5),
            # next H1 geometry
            "next_h1_open":          round(nb_o, 5),
            "next_h1_close":         round(nb_c, 5),
            "next_h1_high":          round(nb_h, 5),
            "next_h1_low":           round(nb_l, 5),
            "next_h1_body_pct":      round(nb_body / nb_rng * 100, 1),
            "next_h1_upper_wick_pct":round(nb_up   / nb_rng * 100, 1),
            "next_h1_lower_wick_pct":round(nb_dn   / nb_rng * 100, 1),
            "next_h1_move_pips":     round((nb_c - nb_o) / 0.0001, 1),
            # label
            "candle_type":           candle_type,
            "label":                 label,
        })

        if len(rows) % 500 == 0:
            pct  = (NUM_H1_BARS - h) / NUM_H1_BARS * 100
            bull = sum(1 for r in rows if r["label"] == 1)
            bear = sum(1 for r in rows if r["label"] == 0)
            excl = sum(1 for r in rows if r["label"] == -1)
            logging.info("h=%5d | rows=%5d  %.0f%%  "
                         "bull=%d  bear=%d  excluded=%d",
                         h, len(rows), pct, bull, bear, excl)

    # ── Write CSV ──────────────────────────────────────────────────
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    # ── Summary ────────────────────────────────────────────────────
    df   = pd.DataFrame(rows)
    bull = (df["label"] == 1).sum()
    bear = (df["label"] == 0).sum()
    excl = (df["label"] == -1).sum()

    logging.info("═" * 62)
    logging.info("DONE  %d rows → %s", len(rows), os.path.abspath(OUTPUT_CSV))
    logging.info("  Errors skipped : %d", skip_err)
    logging.info("  Label=1 bull   : %d  (%.1f%%)", bull, bull/len(df)*100)
    logging.info("  Label=0 bear   : %d  (%.1f%%)", bear, bear/len(df)*100)
    logging.info("  Label=-1 excl  : %d  (%.1f%%)", excl, excl/len(df)*100)
    logging.info("  Candle type breakdown:")
    for ctype, count in sorted(candle_counts.items(),
                                key=lambda x: -x[1]):
        logging.info("    %-20s : %d", ctype, count)
    logging.info("─" * 62)
    logging.info("Usable rows (label 0 or 1): %d", bull + bear)
    logging.info("═" * 62)

    mt5.shutdown()


if __name__ == "__main__":
    main()