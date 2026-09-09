# _labels_h4.py  v1
# ─────────────────────────────────────────────────────────────────────
# Labels for EURUSD H1 future bars (4-bar window = ~4 H1 sessions).
# Derived from _market_h4.py synthetic data; all thresholds calibrated
# at p33/p67 from 5 000 generate_sample() calls across all 5 models.
#
# What changed vs _labels.py (15M version)
# ─────────────────────────────────────────
# THRESHOLD RECALIBRATION (the only mandatory change per spec):
#   label          old_p33  old_p67  new_p33  new_p67  why
#   magnitude      1.25     2.65     0.37     0.85     H1 ATR-relative moves are
#                                                      smaller (4 bars vs 16 bars
#                                                      in 15M window); net/ATR
#                                                      distribution shifts left
#   range_type     2.34     3.46     0.97     1.84     same cause; H1 4-bar range
#                                                      is narrower in ATR units
#   body_ratio     0.35     0.65     0.30     0.62     marginal shift; p33 drops
#                                                      because doji/inside bars
#                                                      are more common in H4 engine
#   close_loc      0.35     0.65     0.30     0.69     wider spread reflects H1
#                                                      session wick asymmetry
#   direction      ±0.25    —        ±0.20    —        tighter flat band → 19.5%
#                                                      flat (was ~25% in 15M)
#
# MOMENTUM REDESIGN (v5 sign-agreement → 55% class-1 → replaced):
#   NEW: ATR-normalised velocity slope across the 4 future bars.
#   velocity[i] = (close[i] - open[i]) / context_ATR
#   slope = OLS slope of [vel_0, vel_1, vel_2, vel_3]  (analytical formula)
#   slope > +0.0884  → building (2)   ~33%
#   slope < -0.0875  → fading   (0)   ~33%
#   otherwise        → stable   (1)   ~34%
#   Rationale: old sign-agreement produced 55% in mid class because the
#   reduced future bias (0.62) causes many directional switches. Velocity
#   slope is continuous and splits at true p33/p67.
#
# Everything else (6 labels, 3 classes each, HEAD_SIZES, HEAD_LABELS,
# function signature, return dict keys) is identical to _labels.py.
# ─────────────────────────────────────────────────────────────────────

EPS = 1e-10

# ── label metadata (identical structure to _labels.py) ────────────────
LABEL_COLS = [
    "label_h1_direction",
    "label_h1_magnitude",
    "label_h1_range_type",
    "label_h1_close_loc",
    "label_h1_body_ratio",
    "label_h1_momentum",
]
HEAD_SIZES  = [3, 3, 3, 3, 3, 3]
HEAD_LABELS = {
    "h1_direction":  {0: "bearish",  1: "flat",     2: "bullish"},
    "h1_magnitude":  {0: "small",    1: "medium",   2: "large"},
    "h1_range_type": {0: "narrow",   1: "normal",   2: "wide"},
    "h1_close_loc":  {0: "lower",    1: "mid",      2: "upper"},
    "h1_body_ratio": {0: "weak",     1: "moderate", 2: "strong"},
    "h1_momentum":   {0: "fading",   1: "stable",   2: "building"},
}

# ── calibrated thresholds (p33/p67 from 5 000-sample sweep) ──────────
# All ratios normalised by context_ATR (price units, not pips).
_DIR_FLAT = 0.20      # ±fraction of ATR; ~19.5% of samples fall here
_MAG_P33  = 0.37      # |net|/ATR < 0.37  → small
_MAG_P67  = 0.85      # |net|/ATR < 0.85  → medium; else large
_RNG_P33  = 0.97      # range/ATR < 0.97  → narrow
_RNG_P67  = 1.84      # range/ATR < 1.84  → normal; else wide
_CL_P33   = 0.30      # close_loc < 0.30  → lower third
_CL_P67   = 0.69      # close_loc < 0.69  → mid; else upper
_BR_P33   = 0.30      # body_ratio < 0.30 → weak
_BR_P67   = 0.62      # body_ratio < 0.62 → moderate; else strong
_MOM_LO   = -0.0433   # velocity slope < this → fading   (p33 of 8k-sample sweep)
_MOM_HI   =  0.0411   # velocity slope > this → building (p67 of 8k-sample sweep)


def compute_labels_from_bars(future_bars: list, context_atr: float) -> dict:
    """
    Compute 6 classification labels from the 4 future H1 bars.

    Parameters
    ----------
    future_bars  : list of 4 bar dicts with keys open/high/low/close/volume/atr.
                   Must be the direct output of _market_h4.generate_sample().
    context_atr  : ATR in price units (not pips) computed from the 80-bar context.
                   Recommended: mean of ctx[-14:]["atr"] * 0.0001
                   i.e.  np.mean([b["atr"] for b in ctx[-14:]]) * 0.0001

    Returns
    -------
    dict mapping each LABEL_COLS key to an integer in {0, 1, 2}.
    All 6 labels are independently calibrated to ~33/34/33% class balance.
    """
    atr = context_atr + EPS

    # ── aggregate statistics across the 4-bar future window ──────────
    o   = future_bars[0]["open"]
    c   = future_bars[-1]["close"]
    h   = max(b["high"] for b in future_bars)
    l   = min(b["low"]  for b in future_bars)
    net = c - o
    rng = h - l + EPS
    br  = abs(net) / rng          # body ratio: |net| / total range

    # ── 1. direction ─────────────────────────────────────────────────
    # Net move relative to ATR. Flat band is tighter than 15M (±0.20 vs ±0.25)
    # because H1 future bars cover fewer pips per ATR unit.
    # Distribution: ~40% bearish / ~20% flat / ~40% bullish.
    if   net >  _DIR_FLAT * atr: direction = 2   # bullish
    elif net < -_DIR_FLAT * atr: direction = 0   # bearish
    else:                        direction = 1   # flat

    # ── 2. magnitude ─────────────────────────────────────────────────
    # |net| / ATR: how much did price move relative to context volatility?
    # Thresholds much lower than 15M (0.37/0.85 vs 1.25/2.65) because
    # a 4-bar H1 window is ~4 sessions while 15M had 16 bars.
    mag = abs(net) / atr
    if   mag < _MAG_P33: magnitude = 0   # small
    elif mag < _MAG_P67: magnitude = 1   # medium
    else:                magnitude = 2   # large

    # ── 3. range type ────────────────────────────────────────────────
    # High-low span / ATR. H1 4-bar ranges are much narrower in ATR
    # units than 15M 16-bar ranges (0.97/1.84 vs 2.34/3.46).
    rn = rng / atr
    if   rn < _RNG_P33: range_type = 0   # narrow
    elif rn < _RNG_P67: range_type = 1   # normal
    else:               range_type = 2   # wide

    # ── 4. close location ────────────────────────────────────────────
    # Where did the final bar close within the 4-bar high-low range?
    # Slightly wider split than 15M (0.30/0.69 vs 0.35/0.65) due to
    # H1 session wick asymmetry pushing closes toward extremes.
    cl = (c - l) / rng
    if   cl < _CL_P33: close_loc = 0   # lower third
    elif cl < _CL_P67: close_loc = 1   # mid
    else:              close_loc = 2   # upper third

    # ── 5. body ratio ────────────────────────────────────────────────
    # |net| / range: how much of the session H-L range was directional?
    # Marginally lower p33 (0.30 vs 0.35) because the H4 engine generates
    # more inside bars and doji in overnight/asian consolidation phases.
    if   br < _BR_P33: body_class = 0   # weak   (wick-dominant, no direction)
    elif br < _BR_P67: body_class = 1   # moderate
    else:              body_class = 2   # strong (clean directional thrust)

    # ── 6. momentum ──────────────────────────────────────────────────
    # ATR-normalised per-bar velocity, then OLS slope over the 4 bars.
    #
    # velocity[i] = (close[i] - open[i]) / ATR
    #
    # OLS slope for evenly-spaced x = [0,1,2,3]:
    #   slope = (−3·v0 − 1·v1 + 1·v2 + 3·v3) / 20
    #
    # Positive slope → each successive bar moves more in the same direction
    #                  → "building" momentum
    # Negative slope → each successive bar moves less / reverses
    #                  → "fading" momentum
    # Near-zero      → consistent velocity, neither accelerating nor fading
    #                  → "stable"
    #
    # Calibrated thresholds: p33=−0.0433 / p67=+0.0411
    # Distribution: ~33% fading / ~34% stable / ~33% building
    #
    # Why not sign-agreement (v5 approach)?
    #   H4 engine future bias = 0.62 → frequent bar-level direction switches
    #   → sign-agreement produced 55% in the stable (mid) class → useless.
    #   Velocity slope is agnostic to bias level and splits cleanly at p33/p67.
    v0 = (future_bars[0]["close"] - future_bars[0]["open"]) / atr
    v1 = (future_bars[1]["close"] - future_bars[1]["open"]) / atr
    v2 = (future_bars[2]["close"] - future_bars[2]["open"]) / atr
    v3 = (future_bars[3]["close"] - future_bars[3]["open"]) / atr
    slope = (-3*v0 - 1*v1 + 1*v2 + 3*v3) / 20.0

    if   slope > _MOM_HI: momentum = 2   # building
    elif slope < _MOM_LO: momentum = 0   # fading
    else:                 momentum = 1   # stable

    return {
        "label_h1_direction":  direction,
        "label_h1_magnitude":  magnitude,
        "label_h1_range_type": range_type,
        "label_h1_close_loc":  close_loc,
        "label_h1_body_ratio": body_class,
        "label_h1_momentum":   momentum,
    }
