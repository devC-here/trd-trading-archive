# _labels.py  v5 — labels computed from actual future bar prices
# Thresholds calibrated to synthetic EURUSD 15M move library (p33/p67).
#
# v5 change — momentum label redefined:
#   OLD: last2_net - first2_net > ±0.25*ATR  (noisy, measures acceleration quantum)
#   NEW: sign(first_half) == sign(second_half)  (sustained vs mixed, ~33/31/35 balance)
#   Motivation: old definition had ~23% stable class and 31% agreement with new.
#   New definition is a cleaner categorical: did the move sustain through the full H1?

EPS = 1e-10

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
    "h1_direction":  {0:"bearish",  1:"flat",     2:"bullish"},
    "h1_magnitude":  {0:"small",    1:"medium",   2:"large"},
    "h1_range_type": {0:"narrow",   1:"normal",   2:"wide"},
    "h1_close_loc":  {0:"lower",    1:"mid",      2:"upper"},
    "h1_body_ratio": {0:"weak",     1:"moderate", 2:"strong"},
    "h1_momentum":   {0:"fading",   1:"stable",   2:"building"},
}


def compute_labels_from_bars(future_bars: list, context_atr: float) -> dict:
    """
    Labels derived entirely from future bar prices — no hardcoding.
    Thresholds set at p33/p67 of the move library distribution.
    """
    atr = context_atr + EPS

    o   = future_bars[0]["open"]
    c   = future_bars[-1]["close"]
    h   = max(b["high"] for b in future_bars)
    l   = min(b["low"]  for b in future_bars)
    net = c - o
    rng = h - l + EPS
    br  = abs(net) / rng

    # direction — flat band ±0.25 ATR (~2.5 pips)
    if   net >  0.25 * atr: direction = 2
    elif net < -0.25 * atr: direction = 0
    else:                   direction = 1

    # magnitude — calibrated p33/p67 of |net|/ATR
    mag = abs(net) / atr
    if   mag < 1.25: magnitude = 0   # small
    elif mag < 2.65: magnitude = 1   # medium
    else:            magnitude = 2   # large

    # range type — calibrated p33/p67 of range/ATR
    rn = rng / atr
    if   rn < 2.34: range_type = 0   # narrow
    elif rn < 3.46: range_type = 1   # normal
    else:           range_type = 2   # wide

    # close location in H1 range
    cl = (c - l) / rng
    if   cl < 0.35: close_loc = 0
    elif cl < 0.65: close_loc = 1
    else:           close_loc = 2

    # body ratio
    if   br < 0.35: body_class = 0
    elif br < 0.65: body_class = 1
    else:           body_class = 2

    # momentum v5 — sign agreement between first and second half of H1
    # first_half: bars 0-1, second_half: bars 2-3
    # Both positive  → building (2): move sustained bullishly through full H1
    # Both negative  → fading   (0): move sustained bearishly through full H1
    # Mixed signs    → stable   (1): move changed character mid-H1
    # Distribution: ~33% / 31% / 35% — near-balanced, vs old 37%/23%/40%
    first_half = future_bars[1]["close"] - future_bars[0]["open"]
    second_half = future_bars[3]["close"] - future_bars[2]["open"]
    if   first_half > 0 and second_half > 0: momentum = 2   # building
    elif first_half < 0 and second_half < 0: momentum = 0   # fading
    else:                                    momentum = 1   # stable / mixed

    return {
        "label_h1_direction":  direction,
        "label_h1_magnitude":  magnitude,
        "label_h1_range_type": range_type,
        "label_h1_close_loc":  close_loc,
        "label_h1_body_ratio": body_class,
        "label_h1_momentum":   momentum,
    }
