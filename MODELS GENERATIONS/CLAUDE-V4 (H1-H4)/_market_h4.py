# _market_h4.py  v1
# ─────────────────────────────────────────────────────────────────────
# EURUSD H1 market simulation engine (H4-context / H1-bar resolution)
# 35 named moves. Labels computed from ACTUAL future bar prices (not hardcoded).
# Per-model move weights ensure each model learns different aspects.
# ─────────────────────────────────────────────────────────────────────
import numpy as np
import random

PIP = 0.0001
EPS = 1e-10

# ── core bar builder ──────────────────────────────────────────────────
def _bar(o, direction, body_pips, wick_pips, vol):
    b = abs(body_pips) * PIP
    w = abs(wick_pips) * PIP
    d = 1 if direction >= 0 else -1
    c = o + d * b
    h = max(o, c) + w + abs(np.random.normal(0, b * 0.06))
    l = min(o, c) - w - abs(np.random.normal(0, b * 0.06))
    return {"open": float(o), "high": float(h), "low": float(l), "close": float(c), "volume": int(max(1, vol))}, float(c)

def _n(val, pct=0.30):
    return val * (1.0 + np.random.uniform(-pct, pct))

# ── H1 session profiles ───────────────────────────────────────────────
SESS = {
    "asian":            dict(body=(5,  15), wick=(3,  8),  vol=(180,  600)),
    "london_open":      dict(body=(25, 55), wick=(8,  18), vol=(1800, 5000)),
    "london_main":      dict(body=(15, 40), wick=(6,  14), vol=(1200, 3500)),
    "ny_open":          dict(body=(20, 45), wick=(7,  16), vol=(1500, 4500)),
    "ny_main":          dict(body=(12, 32), wick=(5,  12), vol=(900,  3000)),
    "london_close":     dict(body=(8,  20), wick=(4,  10), vol=(500,  1800)),
    "overnight":        dict(body=(3,  10), wick=(2,   7), vol=(80,   400)),
    "news":             dict(body=(60,120), wick=(15, 40), vol=(5000,15000)),
    "news_medium":      dict(body=(30, 60), wick=(8,  20), vol=(2500, 7000)),
    "friday_close":     dict(body=(4,  12), wick=(3,  10), vol=(150,  600)),
    "monday_open":      dict(body=(5,  18), wick=(3,  10), vol=(200,  700)),
    "london_ny_overlap":dict(body=(22, 50), wick=(8,  18), vol=(2000, 6000)),
}

def _sess_bar(o, sess_name, direction, bias=0.68, gap_pips=0.0):
    """Build one bar using named session parameters."""
    s = SESS[sess_name]
    body  = np.random.uniform(*s["body"])
    wick  = np.random.uniform(*s["wick"])
    vol   = np.random.uniform(*s["vol"])
    d     = direction if random.random() < bias else -direction
    o_eff = o + gap_pips * PIP
    return _bar(o_eff, d, _n(body), _n(wick), _n(vol))

# ── day-cycle arrays (24 H1 bars per day) ────────────────────────────
DAY_NORMAL = [
    "overnight", "overnight",                              # 00-02
    "asian", "asian", "asian",                             # 02-05
    "london_open", "london_open",                          # 05-07
    "london_main", "london_main", "london_main",           # 07-10
    "ny_open", "ny_open",                                  # 10-12
    "ny_main", "ny_main", "ny_main",                       # 12-15
    "london_close", "london_close",                        # 15-17
    "ny_main", "ny_main",                                  # 17-19
    "overnight", "overnight", "overnight", "overnight", "overnight",  # 19-24
]

DAY_MONDAY = [
    "overnight", "overnight", "overnight",                 # 00-03 Sunday gap zone
    "monday_open", "monday_open", "monday_open",           # 03-06 cautious Monday open
    "london_open", "london_open",                          # 06-08 London sets weekly direction
    "london_main", "london_main", "london_main",           # 08-11
    "ny_open", "ny_open",                                  # 11-13
    "ny_main", "ny_main", "ny_main",                       # 13-16
    "london_close", "london_close",                        # 16-18
    "ny_main", "ny_main",                                  # 18-20
    "overnight", "overnight", "overnight", "overnight",    # 20-24
]

DAY_FRIDAY = [
    "overnight", "overnight",                              # 00-02
    "asian", "asian", "asian",                             # 02-05
    "london_open", "london_open",                          # 05-07
    "london_main", "london_main", "london_main",           # 07-10
    "ny_open", "ny_open",                                  # 10-12
    "ny_main", "ny_main", "ny_main",                       # 12-15
    "friday_close", "friday_close", "friday_close", "friday_close",  # 15-19
    "overnight", "overnight", "overnight", "overnight", "overnight",  # 19-24
]

def _day_bar(price, bar_idx, direction, bias=0.68, day_type="normal", gap_pips=0.0):
    cycle = DAY_MONDAY if day_type == "monday" else (DAY_FRIDAY if day_type == "friday" else DAY_NORMAL)
    sess  = cycle[bar_idx % len(cycle)]
    return _sess_bar(price, sess, direction, bias, gap_pips)

# ── background generators ─────────────────────────────────────────────
def _bg_trend(price, n, direction, vol=None):
    bars = []
    for i in range(n):
        day_type = "monday" if i < 8 else ("friday" if i > 58 else "normal")
        bar, price = _day_bar(price, i, direction, bias=0.68, day_type=day_type)
        bars.append(bar)
    return bars, price

def _bg_range(price, n, center=None, span=60, vol=None):
    bars = []; c0 = center or price
    for i in range(n):
        day_type = "monday" if i < 8 else ("friday" if i > 58 else "normal")
        d = 1 if price < c0 else -1
        if random.random() < 0.45: d = -d
        bar, price = _day_bar(price, i, d, bias=0.55, day_type=day_type)
        price = float(np.clip(price, c0 - span * PIP, c0 + span * PIP))
        bars.append(bar)
    return bars, price

def _bg_squeeze(price, n):
    """Asian-only bars — very tight, pre-breakout coiling."""
    bars = []
    for _ in range(n):
        bar, price = _sess_bar(price, "asian", random.choice([-1, 1]), bias=0.50)
        bars.append(bar)
    return bars, price

def _bg_volatile(price, n):
    """Mixed London/NY bars — large ranges, post-news environment."""
    sess_pool = ["london_open", "london_main", "ny_open", "ny_main"]
    bars = []
    for _ in range(n):
        sess = random.choice(sess_pool)
        bar, price = _sess_bar(price, sess, random.choice([-1, 1]), bias=0.50)
        bars.append(bar)
    return bars, price

def _bg_weekly(price, n):
    """Weekly range: Monday drift + mid-week containment."""
    bars = []
    center = price
    for i in range(n):
        if i < 8:
            d = 1 if random.random() < 0.55 else -1
            bar, price = _sess_bar(price, "monday_open", d, bias=0.55)
        else:
            d = 1 if price < center else -1
            if random.random() < 0.40: d = -d
            bar, price = _day_bar(price, i, d, bias=0.55, day_type="normal")
            price = float(np.clip(price, center - 80 * PIP, center + 80 * PIP))
        bars.append(bar)
    return bars, price

# ── future bar generator ──────────────────────────────────────────────
def _future(price, direction, sessions, bias=0.80, accel=False, fade=False):
    """
    sessions = list of 4 session names for each future H1 bar
    e.g. ["ny_main","london_close","overnight","asian"]
    """
    bars = []
    for i, sess in enumerate(sessions):
        s    = SESS[sess]
        body = np.random.uniform(*s["body"])
        wick = np.random.uniform(*s["wick"])
        vol  = np.random.uniform(*s["vol"])
        if accel:
            body *= (1 + i * 0.18); wick *= max(0.4, 1 - i * 0.12); vol *= (1 + i * 0.12)
        elif fade:
            body *= max(0.25, 1 - i * 0.22); wick *= (1 + i * 0.18); vol *= max(0.35, 1 - i * 0.18)
        else:
            body = _n(body, 0.20); wick = _n(wick, 0.20); vol = _n(vol, 0.20)
        d = direction if random.random() < bias else -direction
        bar, price = _bar(price, d, body, wick, vol)
        bars.append(bar)
    return bars

# Convenience: continuation future session sequence
_FUT_CONT  = ["ny_main", "london_close", "overnight", "asian"]
# Reversal future session sequence (into next London)
_FUT_REV   = ["london_open", "london_main", "ny_open", "ny_main"]
# Fade sequence
_FUT_FADE  = ["london_open", "london_main", "overnight", "asian"]

# ── 35 moves ─────────────────────────────────────────────────────────

# ── Trending moves ────────────────────────────────────────────────────
def move_bull_impulse(p):
    bg, p = _bg_trend(p, 68, 1)
    s = []
    # asian x4 → london_open x3 → london_main x2 → ny_open x2 → ny_main x1
    seq = ["asian"]*4 + ["london_open"]*3 + ["london_main"]*2 + ["ny_open"]*2 + ["ny_main"]*1
    for i, sess in enumerate(seq):
        bar, p = _sess_bar(p, sess, 1, bias=0.80)
        s.append(bar)
    return bg + s, _future(p, 1, _FUT_CONT, accel=True)

def move_bear_impulse(p):
    bg, p = _bg_trend(p, 68, -1)
    s = []
    seq = ["asian"]*4 + ["london_open"]*3 + ["london_main"]*2 + ["ny_open"]*2 + ["ny_main"]*1
    for sess in seq:
        bar, p = _sess_bar(p, sess, -1, bias=0.80)
        s.append(bar)
    return bg + s, _future(p, -1, _FUT_CONT, accel=True)

def move_bull_grind(p):
    bg, p = _bg_trend(p, 68, 1)
    s = []
    seq = ["asian"]*4 + ["london_open"]*2 + ["london_main"]*3 + ["ny_open"]*2 + ["ny_main"]*1
    for sess in seq:
        d = 1 if random.random() < 0.72 else -1
        bar, p = _sess_bar(p, sess, d, bias=0.72)
        s.append(bar)
    return bg + s, _future(p, 1, _FUT_CONT)

def move_bear_grind(p):
    bg, p = _bg_trend(p, 68, -1)
    s = []
    seq = ["asian"]*4 + ["london_open"]*2 + ["london_main"]*3 + ["ny_open"]*2 + ["ny_main"]*1
    for sess in seq:
        d = -1 if random.random() < 0.72 else 1
        bar, p = _sess_bar(p, sess, d, bias=0.72)
        s.append(bar)
    return bg + s, _future(p, -1, _FUT_CONT)

def move_bull_acceleration(p):
    bg, p = _bg_range(p, 68)
    s = []
    seq = ["asian"]*4 + ["london_open"]*2 + ["london_main"]*2 + ["ny_open"]*2 + ["ny_main"]*2
    for i, sess in enumerate(seq):
        if i < 6:
            d = 1 if random.random() < 0.55 else -1
            bar, p = _sess_bar(p, sess, d, bias=0.55)
        else:
            bar, p = _sess_bar(p, sess, 1, bias=0.88)
        s.append(bar)
    return bg + s, _future(p, 1, _FUT_CONT, accel=True)

def move_bear_acceleration(p):
    bg, p = _bg_range(p, 68)
    s = []
    seq = ["asian"]*4 + ["london_open"]*2 + ["london_main"]*2 + ["ny_open"]*2 + ["ny_main"]*2
    for i, sess in enumerate(seq):
        if i < 6:
            d = -1 if random.random() < 0.55 else 1
            bar, p = _sess_bar(p, sess, d, bias=0.55)
        else:
            bar, p = _sess_bar(p, sess, -1, bias=0.88)
        s.append(bar)
    return bg + s, _future(p, -1, _FUT_CONT, accel=True)

def move_bull_pullback_resume(p):
    bg, p = _bg_trend(p, 68, 1)
    s = []
    seq = ["asian"]*3 + ["london_open"]*2 + ["london_main"]*2 + ["ny_open"]*2 + ["ny_main"]*3
    for i, sess in enumerate(seq):
        if i < 5:   bar, p = _sess_bar(p, sess, 1,  bias=0.80)
        elif i < 8: bar, p = _sess_bar(p, sess, -1, bias=0.70)
        else:       bar, p = _sess_bar(p, sess, 1,  bias=0.78)
        s.append(bar)
    return bg + s, _future(p, 1, _FUT_CONT)

def move_bear_pullback_resume(p):
    bg, p = _bg_trend(p, 68, -1)
    s = []
    seq = ["asian"]*3 + ["london_open"]*2 + ["london_main"]*2 + ["ny_open"]*2 + ["ny_main"]*3
    for i, sess in enumerate(seq):
        if i < 5:   bar, p = _sess_bar(p, sess, -1, bias=0.80)
        elif i < 8: bar, p = _sess_bar(p, sess, 1,  bias=0.70)
        else:       bar, p = _sess_bar(p, sess, -1, bias=0.78)
        s.append(bar)
    return bg + s, _future(p, -1, _FUT_CONT)

# ── Exhaustion moves ──────────────────────────────────────────────────
def move_bull_exhaustion(p):
    bg, p = _bg_trend(p, 68, 1)
    s = []
    seq = ["asian"]*3 + ["london_open"]*2 + ["london_main"]*3 + ["ny_open"]*2 + ["ny_main"]*2
    for i, sess in enumerate(seq):
        sp = SESS[sess]
        if i < 8:
            bar, p = _sess_bar(p, sess, 1, bias=0.80)
        else:
            # shrink body 20-25%, inflate wick 130-220%
            body = np.random.uniform(*sp["body"]) * np.random.uniform(0.20, 0.25)
            wick = np.random.uniform(*sp["wick"]) * np.random.uniform(1.30, 2.20)
            vol  = np.random.uniform(*sp["vol"])  * np.random.uniform(0.35, 0.55)
            d    = 1 if random.random() < 0.40 else -1
            bar, p = _bar(p, d, body, wick, vol)
        s.append(bar)
    return bg + s, _future(p, -1, _FUT_REV, fade=True)

def move_bear_exhaustion(p):
    bg, p = _bg_trend(p, 68, -1)
    s = []
    seq = ["asian"]*3 + ["london_open"]*2 + ["london_main"]*3 + ["ny_open"]*2 + ["ny_main"]*2
    for i, sess in enumerate(seq):
        sp = SESS[sess]
        if i < 8:
            bar, p = _sess_bar(p, sess, -1, bias=0.80)
        else:
            body = np.random.uniform(*sp["body"]) * np.random.uniform(0.20, 0.25)
            wick = np.random.uniform(*sp["wick"]) * np.random.uniform(1.30, 2.20)
            vol  = np.random.uniform(*sp["vol"])  * np.random.uniform(0.35, 0.55)
            d    = -1 if random.random() < 0.40 else 1
            bar, p = _bar(p, d, body, wick, vol)
        s.append(bar)
    return bg + s, _future(p, 1, _FUT_REV, fade=True)

# ── Pin bar moves ─────────────────────────────────────────────────────
def move_bull_pin_top(p):
    """Pin bar forms at london_open — H4 key-level rejection."""
    bg, p = _bg_trend(p, 68, 1)
    s = []
    seq = ["asian"]*4 + ["london_open"]*2 + ["london_main"]*2 + ["ny_open"]*2 + ["ny_main"]*2
    for i, sess in enumerate(seq):
        if i < 9:
            bar, p = _sess_bar(p, sess, 1, bias=0.78)
            s.append(bar)
        elif i == 9:
            # london_open pin bar: spike 55-70 pips, body 8-12 pips
            o = p
            spike = np.random.uniform(55, 70) * PIP
            body  = np.random.uniform(8, 12)  * PIP
            h = o + spike
            l = o - np.random.uniform(2, 4) * PIP
            c = o - body
            bar = {"open": float(o), "high": float(h), "low": float(l), "close": float(c), "volume": int(_n(4500))}
            p = float(c); s.append(bar)
        else:
            d = -1 if random.random() < 0.65 else 1
            bar, p = _sess_bar(p, sess, d, bias=0.65)
            s.append(bar)
    return bg + s, _future(p, -1, _FUT_REV, fade=True)

def move_bear_pin_bottom(p):
    bg, p = _bg_trend(p, 68, -1)
    s = []
    seq = ["asian"]*4 + ["london_open"]*2 + ["london_main"]*2 + ["ny_open"]*2 + ["ny_main"]*2
    for i, sess in enumerate(seq):
        if i < 9:
            bar, p = _sess_bar(p, sess, -1, bias=0.78)
            s.append(bar)
        elif i == 9:
            o = p
            spike = np.random.uniform(55, 70) * PIP
            body  = np.random.uniform(8, 12)  * PIP
            l = o - spike
            h = o + np.random.uniform(2, 4) * PIP
            c = o + body
            bar = {"open": float(o), "high": float(h), "low": float(l), "close": float(c), "volume": int(_n(4500))}
            p = float(c); s.append(bar)
        else:
            d = 1 if random.random() < 0.65 else -1
            bar, p = _sess_bar(p, sess, d, bias=0.65)
            s.append(bar)
    return bg + s, _future(p, 1, _FUT_REV, fade=True)

# ── Engulf moves ──────────────────────────────────────────────────────
def move_bearish_engulf(p):
    bg, p = _bg_trend(p, 68, 1)
    s = []
    seq = ["asian"]*3 + ["london_main"]*5 + ["london_main"]*1 + ["london_open"]*1 + ["london_main"]*2
    for i, sess in enumerate(seq):
        if i < 10:
            d = 1 if random.random() < 0.65 else -1
            bar, p = _sess_bar(p, sess, d, bias=0.65)
        elif i == 10:
            # small london_main bar
            bar, p = _sess_bar(p, "london_main", 1, bias=0.85)
        else:
            # engulf: full london_open force
            bar, p = _sess_bar(p, "london_open", -1, bias=0.92)
        s.append(bar)
    return bg + s, _future(p, -1, _FUT_CONT, accel=True)

def move_bullish_engulf(p):
    bg, p = _bg_trend(p, 68, -1)
    s = []
    seq = ["asian"]*3 + ["london_main"]*5 + ["london_main"]*1 + ["london_open"]*1 + ["london_main"]*2
    for i, sess in enumerate(seq):
        if i < 10:
            d = -1 if random.random() < 0.65 else 1
            bar, p = _sess_bar(p, sess, d, bias=0.65)
        elif i == 10:
            bar, p = _sess_bar(p, "london_main", -1, bias=0.85)
        else:
            bar, p = _sess_bar(p, "london_open", 1, bias=0.92)
        s.append(bar)
    return bg + s, _future(p, 1, _FUT_CONT, accel=True)

# ── Fakeout / trap moves ──────────────────────────────────────────────
def move_fakeout_bull_trap(p):
    bg, p = _bg_trend(p, 68, 1)
    s = []
    seq = ["asian"]*3 + ["london_open"]*2 + ["london_main"]*3 + ["london_open"]*2 + ["london_main"]*2
    for i, sess in enumerate(seq):
        if i < 8:    bar, p = _sess_bar(p, sess, 1,  bias=0.72)
        elif i < 10: bar, p = _sess_bar(p, sess, 1,  bias=0.88)   # push
        else:        bar, p = _sess_bar(p, sess, -1, bias=0.90)   # hard reversal
        s.append(bar)
    return bg + s, _future(p, -1, _FUT_CONT, accel=True)

def move_fakeout_bear_trap(p):
    bg, p = _bg_trend(p, 68, -1)
    s = []
    seq = ["asian"]*3 + ["london_open"]*2 + ["london_main"]*3 + ["london_open"]*2 + ["london_main"]*2
    for i, sess in enumerate(seq):
        if i < 8:    bar, p = _sess_bar(p, sess, -1, bias=0.72)
        elif i < 10: bar, p = _sess_bar(p, sess, -1, bias=0.88)
        else:        bar, p = _sess_bar(p, sess, 1,  bias=0.90)
        s.append(bar)
    return bg + s, _future(p, 1, _FUT_CONT, accel=True)

# ── Squeeze break moves ───────────────────────────────────────────────
def move_bull_squeeze_break(p):
    bg, p = _bg_squeeze(p, 68)
    s = []; center = p
    for i in range(12):
        if i < 9:
            bar, p = _sess_bar(p, "asian", random.choice([-1, 1]), bias=0.50)
            p = float(np.clip(p, center - 22 * PIP, center + 22 * PIP))
        elif i < 11:
            bar, p = _sess_bar(p, "london_open", 1, bias=0.90)
        else:
            bar, p = _sess_bar(p, "london_main", 1, bias=0.88)
        s.append(bar)
    return bg + s, _future(p, 1, _FUT_CONT, accel=True)

def move_bear_squeeze_break(p):
    bg, p = _bg_squeeze(p, 68)
    s = []; center = p
    for i in range(12):
        if i < 9:
            bar, p = _sess_bar(p, "asian", random.choice([-1, 1]), bias=0.50)
            p = float(np.clip(p, center - 22 * PIP, center + 22 * PIP))
        elif i < 11:
            bar, p = _sess_bar(p, "london_open", -1, bias=0.90)
        else:
            bar, p = _sess_bar(p, "london_main", -1, bias=0.88)
        s.append(bar)
    return bg + s, _future(p, -1, _FUT_CONT, accel=True)

# ── Flag moves ────────────────────────────────────────────────────────
def move_bull_flag_break(p):
    bg, p = _bg_trend(p, 68, 1)
    s = []; fh = p
    # Pole: london_open + london_main + ny_open + ny_main
    # Flag consolidation: london_close + overnight x2 + asian x2
    # Break: london_open x2
    seq = (["london_open"]*2 + ["london_main"]*1 + ["ny_open"]*1 + ["ny_main"]*1 +
           ["london_close"]*1 + ["overnight"]*2 + ["asian"]*2 +
           ["london_open"]*2)
    for i, sess in enumerate(seq):
        if i < 5:
            bar, p = _sess_bar(p, sess, 1, bias=0.82)
            fh = p
        elif i < 9:
            d = random.choice([-1, 1])
            bar, p = _sess_bar(p, sess, d, bias=0.50)
            p = float(np.clip(p, fh - 45 * PIP, fh + 8 * PIP))
        else:
            bar, p = _sess_bar(p, sess, 1, bias=0.88)
        s.append(bar)
    return bg + s, _future(p, 1, _FUT_CONT)

def move_bear_flag_break(p):
    bg, p = _bg_trend(p, 68, -1)
    s = []; fl = p
    seq = (["london_open"]*2 + ["london_main"]*1 + ["ny_open"]*1 + ["ny_main"]*1 +
           ["london_close"]*1 + ["overnight"]*2 + ["asian"]*2 +
           ["london_open"]*2)
    for i, sess in enumerate(seq):
        if i < 5:
            bar, p = _sess_bar(p, sess, -1, bias=0.82)
            fl = p
        elif i < 9:
            d = random.choice([-1, 1])
            bar, p = _sess_bar(p, sess, d, bias=0.50)
            p = float(np.clip(p, fl - 8 * PIP, fl + 45 * PIP))
        else:
            bar, p = _sess_bar(p, sess, -1, bias=0.88)
        s.append(bar)
    return bg + s, _future(p, -1, _FUT_CONT)

# ── Failed break moves ────────────────────────────────────────────────
def move_failed_bull_break(p):
    bg, p = _bg_range(p, 68)
    s = []
    seq = (["london_main"]*5 + ["london_open"]*2 +
           ["london_open"]*1 + ["london_main"]*1 +
           ["london_main"]*3)
    for i, sess in enumerate(seq):
        if i < 7:
            d = 1 if random.random() < 0.55 else -1
            bar, p = _sess_bar(p, sess, d, bias=0.55)
        elif i < 9:
            bar, p = _sess_bar(p, "london_open", 1, bias=0.85)   # push
        elif i == 9:
            # rejection bar: 60-65 pip spike, 12-14 pip body, closes against direction
            o = p
            spike = np.random.uniform(60, 65) * PIP
            body  = np.random.uniform(12, 14) * PIP
            h = o + spike; l = o - np.random.uniform(3, 5) * PIP; c = o - body
            bar = {"open": float(o), "high": float(h), "low": float(l), "close": float(c), "volume": int(_n(5500))}
            p = float(c)
        else:
            bar, p = _sess_bar(p, "london_main", -1, bias=0.78)  # confirming
        s.append(bar)
    return bg + s, _future(p, -1, _FUT_FADE, fade=True)

def move_failed_bear_break(p):
    bg, p = _bg_range(p, 68)
    s = []
    seq = (["london_main"]*5 + ["london_open"]*2 +
           ["london_open"]*1 + ["london_main"]*1 +
           ["london_main"]*3)
    for i, sess in enumerate(seq):
        if i < 7:
            d = -1 if random.random() < 0.55 else 1
            bar, p = _sess_bar(p, sess, d, bias=0.55)
        elif i < 9:
            bar, p = _sess_bar(p, "london_open", -1, bias=0.85)
        elif i == 9:
            o = p
            spike = np.random.uniform(60, 65) * PIP
            body  = np.random.uniform(12, 14) * PIP
            l = o - spike; h = o + np.random.uniform(3, 5) * PIP; c = o + body
            bar = {"open": float(o), "high": float(h), "low": float(l), "close": float(c), "volume": int(_n(5500))}
            p = float(c)
        else:
            bar, p = _sess_bar(p, "london_main", 1, bias=0.78)
        s.append(bar)
    return bg + s, _future(p, 1, _FUT_FADE, fade=True)

# ── Range / chop moves ────────────────────────────────────────────────
def move_asian_range(p):
    bg, p = _bg_range(p, 68, span=40)
    s = []; center = p
    for _ in range(12):
        d = 1 if p < center else -1
        if random.random() < 0.45: d = -d
        bar, p = _sess_bar(p, "asian", d, bias=0.52)
        p = float(np.clip(p, center - 18 * PIP, center + 18 * PIP))
        s.append(bar)
    return bg + s, _future(p, random.choice([-1, 1]), ["asian", "london_open", "london_open", "london_main"])

def move_monday_weekly_open(p):
    """Monday gap + drift + London sets weekly direction."""
    bg, p = _bg_weekly(p, 68)
    s = []; direction = 1 if random.random() < 0.50 else -1
    gap_pips = np.random.uniform(-18, 18)
    seq = ["monday_open"]*3 + ["monday_open"]*3 + ["london_open"]*2 + ["london_main"]*2 + ["ny_open"]*2
    for i, sess in enumerate(seq):
        gp = gap_pips if i == 0 else 0.0
        if i < 6:
            d = direction if random.random() < 0.62 else -direction
            bar, p = _sess_bar(p, sess, d, bias=0.62, gap_pips=gp)
        else:
            bar, p = _sess_bar(p, sess, direction, bias=0.80)
        s.append(bar)
    return bg + s, _future(p, direction, _FUT_CONT)

def move_post_news_chop(p):
    bg, p = _bg_volatile(p, 68)
    s = []; direction = random.choice([-1, 1])
    alt_sess = ["london_main", "ny_open", "ny_main", "london_close",
                "ny_main", "london_main", "ny_open", "london_close",
                "overnight", "asian", "london_open", "london_main"]
    for i, sess in enumerate(alt_sess):
        if i == 0:
            bar, p = _sess_bar(p, "news_medium", direction, bias=0.90)
        else:
            d = direction if i % 2 == 0 else -direction
            bar, p = _sess_bar(p, sess, d, bias=0.55)
        s.append(bar)
    return bg + s, _future(p, random.choice([-1, 1]), ["london_open", "london_main", "ny_open", "ny_main"])

def move_midweek_consolidation(p):
    """Wednesday/Thursday inside range — full day cycle contained ±55 pips."""
    bg, p = _bg_range(p, 68, span=60)
    s = []; center = p
    seq = ["overnight"]*2 + ["asian"]*3 + ["london_open"]*2 + ["london_main"]*2 + ["ny_open"]*1 + ["ny_main"]*2
    for sess in seq:
        d = 1 if p < center else -1
        if random.random() < 0.42: d = -d
        bar, p = _sess_bar(p, sess, d, bias=0.52)
        p = float(np.clip(p, center - 55 * PIP, center + 55 * PIP))
        s.append(bar)
    return bg + s, _future(p, random.choice([-1, 1]), ["london_open", "london_main", "ny_open", "ny_main"])

def move_bull_flag_tight(p):
    bg, p = _bg_trend(p, 68, 1)
    s = []; fh = p
    seq = (["london_open"]*2 + ["london_main"]*1 + ["ny_open"]*1 + ["ny_main"]*1 +
           ["london_close"]*2 + ["overnight"]*1 + ["asian"]*2 +
           ["london_open"]*2)
    for i, sess in enumerate(seq):
        if i < 5:
            bar, p = _sess_bar(p, sess, 1, bias=0.84)
            fh = p
        elif i < 9:
            d = random.choice([-1, 1])
            bar, p = _sess_bar(p, sess, d, bias=0.50)
            p = float(np.clip(p, fh - 30 * PIP, fh + 5 * PIP))
        else:
            bar, p = _sess_bar(p, sess, 1, bias=0.88)
        s.append(bar)
    return bg + s, _future(p, 1, _FUT_CONT)

def move_bear_flag_tight(p):
    bg, p = _bg_trend(p, 68, -1)
    s = []; fl = p
    seq = (["london_open"]*2 + ["london_main"]*1 + ["ny_open"]*1 + ["ny_main"]*1 +
           ["london_close"]*2 + ["overnight"]*1 + ["asian"]*2 +
           ["london_open"]*2)
    for i, sess in enumerate(seq):
        if i < 5:
            bar, p = _sess_bar(p, sess, -1, bias=0.84)
            fl = p
        elif i < 9:
            d = random.choice([-1, 1])
            bar, p = _sess_bar(p, sess, d, bias=0.50)
            p = float(np.clip(p, fl - 5 * PIP, fl + 30 * PIP))
        else:
            bar, p = _sess_bar(p, sess, -1, bias=0.88)
        s.append(bar)
    return bg + s, _future(p, -1, _FUT_CONT)

def move_friday_squaring(p):
    """First 6 bars week direction (Thursday), last 6 bars reversal + fade (friday_close)."""
    bg, p = _bg_trend(p, 68, 1)
    s = []; direction = 1 if random.random() < 0.50 else -1
    seq = (["ny_main"]*2 + ["london_close"]*2 + ["overnight"]*2 +   # Thursday tail
           ["friday_close"]*2 + ["friday_close"]*2 + ["friday_close"]*2)  # Friday squaring
    for i, sess in enumerate(seq):
        if i < 6:
            bar, p = _sess_bar(p, sess, direction, bias=0.68)
        else:
            bar, p = _sess_bar(p, sess, -direction, bias=0.72)   # reversal + fade
        s.append(bar)
    return bg + s, _future(p, -direction, _FUT_FADE, fade=True)

# ── Microstructure moves ──────────────────────────────────────────────
def move_bull_absorption(p):
    bg, p = _bg_trend(p, 68, -1)
    s = []
    seq = ["asian"]*2 + ["london_main"]*2 + ["ny_open"]*2 + ["london_open"]*3 + ["london_main"]*3
    for i, sess in enumerate(seq):
        if i < 6:
            bar, p = _sess_bar(p, sess, -1, bias=0.78)
        else:
            # High-vol absorption: london_open vol regardless of session label, tiny body
            sp = SESS["london_open"]
            vol  = np.random.uniform(*sp["vol"]) * 1.6
            body = np.random.uniform(*SESS[sess]["body"]) * np.random.uniform(0.18, 0.45)
            wick = np.random.uniform(*SESS[sess]["wick"])
            d    = -1 if random.random() < 0.42 else 1
            bar, p = _bar(p, d, body, wick, vol)
        s.append(bar)
    return bg + s, _future(p, 1, _FUT_CONT)

def move_bear_absorption(p):
    bg, p = _bg_trend(p, 68, 1)
    s = []
    seq = ["asian"]*2 + ["london_main"]*2 + ["ny_open"]*2 + ["london_open"]*3 + ["london_main"]*3
    for i, sess in enumerate(seq):
        if i < 6:
            bar, p = _sess_bar(p, sess, 1, bias=0.78)
        else:
            sp = SESS["london_open"]
            vol  = np.random.uniform(*sp["vol"]) * 1.6
            body = np.random.uniform(*SESS[sess]["body"]) * np.random.uniform(0.18, 0.45)
            wick = np.random.uniform(*SESS[sess]["wick"])
            d    = 1 if random.random() < 0.42 else -1
            bar, p = _bar(p, d, body, wick, vol)
        s.append(bar)
    return bg + s, _future(p, -1, _FUT_CONT)

def move_news_bull_catalyst(p):
    bg, p = _bg_range(p, 68)
    s = []
    seq = (["asian"]*3 + ["london_main"]*3 +
           ["news"]*1 +                               # news bar: 60-120 pip body
           ["ny_open"]*2 + ["ny_main"]*2 + ["london_close"]*1)  # sustained continuation
    for i, sess in enumerate(seq):
        if i < 6:
            bar, p = _sess_bar(p, sess, random.choice([-1, 1]), bias=0.50)
        elif i == 6:
            bar, p = _sess_bar(p, "news", 1, bias=0.92)
        else:
            bar, p = _sess_bar(p, sess, 1, bias=0.82)   # sustained, not retrace
        s.append(bar)
    return bg + s, _future(p, 1, ["ny_main", "london_close", "overnight", "asian"])

def move_news_bear_catalyst(p):
    bg, p = _bg_range(p, 68)
    s = []
    seq = (["asian"]*3 + ["london_main"]*3 +
           ["news"]*1 +
           ["ny_open"]*2 + ["ny_main"]*2 + ["london_close"]*1)
    for i, sess in enumerate(seq):
        if i < 6:
            bar, p = _sess_bar(p, sess, random.choice([-1, 1]), bias=0.50)
        elif i == 6:
            bar, p = _sess_bar(p, "news", -1, bias=0.92)
        else:
            bar, p = _sess_bar(p, sess, -1, bias=0.82)
        s.append(bar)
    return bg + s, _future(p, -1, ["ny_main", "london_close", "overnight", "asian"])

# ── New H4-specific reversal moves ───────────────────────────────────
def move_london_reversal_of_asia(p):
    """Most common H4 EURUSD pattern: Asian drift → London hard reversal."""
    bg, p = _bg_trend(p, 68, 1)
    s = []
    # Asian bars × 6 drifting one way, then london_open × 3 hard opposite
    seq = ["asian"]*6 + ["london_open"]*3 + ["london_main"]*2 + ["ny_open"]*1
    direction = 1 if random.random() < 0.50 else -1
    for i, sess in enumerate(seq):
        if i < 6:
            bar, p = _sess_bar(p, sess, direction, bias=0.68)
        else:
            bar, p = _sess_bar(p, sess, -direction, bias=0.88)
        s.append(bar)
    return bg + s, _future(p, -direction, _FUT_CONT)

def move_ny_reversal_of_london(p):
    """Second most common H4 reversal: London move → NY hard opposite."""
    bg, p = _bg_trend(p, 68, -1)
    s = []
    seq = (["london_open"]*2 + ["london_main"]*3 + ["london_main"]*3 +
           ["ny_open"]*2 + ["ny_main"]*2)
    direction = 1 if random.random() < 0.50 else -1
    for i, sess in enumerate(seq):
        if i < 8:
            bar, p = _sess_bar(p, sess, direction, bias=0.72)
        else:
            bar, p = _sess_bar(p, sess, -direction, bias=0.90)
        s.append(bar)
    return bg + s, _future(p, -direction, _FUT_CONT)

def move_news_reversal(p):
    """Prior trend → news bar opposite → sustained reversal continuation."""
    prior_dir = 1 if random.random() < 0.50 else -1
    bg, p = _bg_trend(p, 68, prior_dir)
    s = []
    seq = (["asian"]*2 + ["london_main"]*2 + ["ny_main"]*2 +
           ["news"]*1 +                              # news bar contradicts prior trend
           ["ny_open"]*2 + ["ny_main"]*2 + ["london_close"]*1)
    for i, sess in enumerate(seq):
        if i < 6:
            bar, p = _sess_bar(p, sess, prior_dir, bias=0.72)
        elif i == 6:
            bar, p = _sess_bar(p, "news", -prior_dir, bias=0.92)
        else:
            bar, p = _sess_bar(p, sess, -prior_dir, bias=0.82)
        s.append(bar)
    return bg + s, _future(p, -prior_dir, _FUT_CONT)

def move_overnight_gap_fill(p):
    """Session gap (8-20 pips) → London fills it."""
    bg, p = _bg_range(p, 68, span=50)
    s = []; gap_dir = 1 if random.random() < 0.50 else -1
    gap_pips = np.random.uniform(8, 20) * gap_dir
    seq = (["overnight"]*2 + ["asian"]*2 +
           ["london_open"]*3 + ["london_main"]*3 + ["ny_open"]*2)
    for i, sess in enumerate(seq):
        gp = gap_pips if i == 0 else 0.0
        if i < 4:
            bar, p = _sess_bar(p, sess, gap_dir, bias=0.65, gap_pips=gp)
        else:
            # fill: london goes against gap direction
            bar, p = _sess_bar(p, sess, -gap_dir, bias=0.82)
        s.append(bar)
    return bg + s, _future(p, -gap_dir, _FUT_CONT)

# ── registry ──────────────────────────────────────────────────────────
ALL_MOVES = [
    move_bull_impulse,          move_bear_impulse,           #  0, 1
    move_bull_grind,            move_bear_grind,             #  2, 3
    move_bull_acceleration,     move_bear_acceleration,      #  4, 5
    move_bull_pullback_resume,  move_bear_pullback_resume,   #  6, 7
    move_bull_exhaustion,       move_bear_exhaustion,        #  8, 9
    move_bull_pin_top,          move_bear_pin_bottom,        # 10,11
    move_bearish_engulf,        move_bullish_engulf,         # 12,13
    move_fakeout_bull_trap,     move_fakeout_bear_trap,      # 14,15
    move_bull_squeeze_break,    move_bear_squeeze_break,     # 16,17
    move_bull_flag_break,       move_bear_flag_break,        # 18,19
    move_failed_bull_break,     move_failed_bear_break,      # 20,21
    move_asian_range,           move_midweek_consolidation,  # 22,23  (replaces asia_tight/midrange)
    move_bull_flag_tight,       move_bear_flag_tight,        # 24,25
    move_bull_absorption,       move_bear_absorption,        # 26,27
    move_news_bull_catalyst,    move_news_bear_catalyst,     # 28,29
    move_london_reversal_of_asia, move_ny_reversal_of_london,# 30,31  NEW
    move_monday_weekly_open,    move_friday_squaring,        # 32,33  NEW
    move_news_reversal,                                      # 34     NEW
    move_overnight_gap_fill,                                 # 35     NEW (total 36 fns = 35 named + 0-indexed 35)
    move_post_news_chop,                                     # 36     extra (was in spec as chop)
]

# ── Per-model weights (one weight per move, same order as ALL_MOVES) ──
MODEL_WEIGHTS = {
    #                  0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18   19   20   21   22   23   24   25   26   27   28   29   30   31   32   33   34   35   36
    "momentum":     [3.5, 3.5, 3.0, 3.0, 3.5, 3.5, 3.5, 3.5, 2.5, 2.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.0, 1.0, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5, 1.0, 1.0, 2.5, 2.5, 2.0, 1.5, 1.5, 2.0, 1.0],
    "vwap":         [2.0, 2.0, 1.5, 1.5, 2.0, 2.0, 2.5, 2.5, 2.0, 2.0, 1.5, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.5, 1.5, 2.0, 2.0, 2.0, 2.0, 3.5, 3.5, 3.0, 3.0, 2.0, 2.0, 1.5, 1.5, 3.0, 3.5, 2.0],
    "candle":       [3.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 3.0, 3.0, 3.5, 3.5, 3.5, 3.5, 2.5, 2.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5, 2.5, 2.0, 2.0, 1.5, 1.5, 3.5, 3.0, 1.5, 2.0, 2.0, 1.5, 1.5],
    "breakout":     [2.0, 2.0, 1.5, 1.5, 2.0, 2.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 2.5, 2.5, 3.5, 3.5, 3.0, 3.0, 3.0, 3.0, 3.0, 2.5, 3.5, 3.5, 1.5, 1.5, 1.5, 1.5, 2.0, 2.0, 2.5, 2.0, 1.5, 2.0, 1.5],
    "microstructure":[3.5,3.5, 1.5, 1.5, 2.5, 2.5, 2.0, 2.0, 3.0, 3.0, 2.0, 2.0, 3.0, 3.0, 2.5, 2.5, 2.0, 2.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 3.5, 3.5, 3.5, 3.5, 2.0, 2.0, 2.0, 1.5, 3.5, 3.5, 2.5],
}

def generate_sample(model_name="momentum", base_price=None):
    if base_price is None:
        base_price = 1.0800 + np.random.uniform(-0.0200, 0.0200)
    weights = MODEL_WEIGHTS[model_name]
    fn = random.choices(ALL_MOVES, weights=weights, k=1)[0]
    ctx, fut = fn(base_price)
    assert len(ctx) == 80, f"{fn.__name__} returned {len(ctx)} ctx bars"
    assert len(fut) == 4,  f"{fn.__name__} returned {len(fut)} fut bars"
    return ctx, fut, fn.__name__
