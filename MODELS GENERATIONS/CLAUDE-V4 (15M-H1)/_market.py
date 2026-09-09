# _market.py  v4
# ─────────────────────────────────────────────────────────────────────
# EURUSD 15M market simulation engine
# 30 named moves. Labels computed from ACTUAL future bar prices (not hardcoded).
# Per-model move weights ensure each model learns different aspects.
# ─────────────────────────────────────────────────────────────────────
import numpy as np
import random

PIP = 0.0001
EPS = 1e-10

def _bar(o, direction, body_pips, wick_pips, vol):
    b = abs(body_pips) * PIP
    w = abs(wick_pips) * PIP
    d = 1 if direction >= 0 else -1
    c = o + d * b
    h = max(o,c) + w + abs(np.random.normal(0, b*0.06))
    l = min(o,c) - w - abs(np.random.normal(0, b*0.06))
    return {"open":float(o),"high":float(h),"low":float(l),"close":float(c),"volume":int(max(1,vol))}, float(c)

def _n(val, pct=0.22):
    return val * (1.0 + np.random.uniform(-pct, pct))

# ── backgrounds ──────────────────────────────────────────────────────
def _bg_trend(price, n, direction, body=8, vol=1000):
    bars = []
    for _ in range(n):
        d = direction if random.random()<0.68 else -direction
        bar, price = _bar(price, d, _n(body), _n(body*0.30), _n(vol))
        bars.append(bar)
    return bars, price

def _bg_range(price, n, center=None, span=20, vol=650):
    bars = []; c0 = center or price
    for _ in range(n):
        d = 1 if price<c0 else -1
        if random.random()<0.45: d=-d
        bar, price = _bar(price, d, _n(4), _n(3), _n(vol))
        price = float(np.clip(price, c0-span*PIP, c0+span*PIP))
        bars.append(bar)
    return bars, price

def _bg_squeeze(price, n, vol=380):
    bars = []
    for _ in range(n):
        bar, price = _bar(price, random.choice([-1,1]), _n(2.5), _n(1.5), _n(vol))
        bars.append(bar)
    return bars, price

def _bg_volatile(price, n, vol=2200):
    bars = []
    for _ in range(n):
        bar, price = _bar(price, random.choice([-1,1]), _n(13), _n(9), _n(vol))
        bars.append(bar)
    return bars, price

# ── future generator ─────────────────────────────────────────────────
def _future(price, direction, body, wick, vol, accel=False, fade=False):
    bars = []
    for i in range(4):
        if accel:   b_,w_,v_ = body*(1+i*0.18), wick*max(0.4,1-i*0.12), vol*(1+i*0.12)
        elif fade:  b_,w_,v_ = body*max(0.25,1-i*0.22), wick*(1+i*0.18), vol*max(0.35,1-i*0.18)
        else:       b_,w_,v_ = _n(body,0.20), _n(wick,0.20), _n(vol,0.20)
        d = direction if random.random()<0.80 else -direction
        bar, price = _bar(price, d, b_, w_, v_)
        bars.append(bar)
    return bars

# ── 30 moves ─────────────────────────────────────────────────────────
def move_bull_impulse(p):
    bg,p=_bg_trend(p,68,1,body=7,vol=900)
    s=[]
    for i in range(12):
        bar,p=_bar(p,1,_n(11+i*0.3),_n(2.5),_n(1800+i*60)); s.append(bar)
    return bg+s, _future(p,1,12,2.5,2100,accel=True)

def move_bear_impulse(p):
    bg,p=_bg_trend(p,68,-1,body=7,vol=900)
    s=[]
    for i in range(12):
        bar,p=_bar(p,-1,_n(11+i*0.3),_n(2.5),_n(1800+i*60)); s.append(bar)
    return bg+s, _future(p,-1,12,2.5,2100,accel=True)

def move_bull_grind(p):
    bg,p=_bg_trend(p,68,1,body=4,vol=600)
    s=[]
    for _ in range(12):
        d=1 if random.random()<0.72 else -1
        bar,p=_bar(p,d,_n(4),_n(2.5),_n(650)); s.append(bar)
    return bg+s, _future(p,1,5,3,700)

def move_bear_grind(p):
    bg,p=_bg_trend(p,68,-1,body=4,vol=600)
    s=[]
    for _ in range(12):
        d=-1 if random.random()<0.72 else 1
        bar,p=_bar(p,d,_n(4),_n(2.5),_n(650)); s.append(bar)
    return bg+s, _future(p,-1,5,3,700)

def move_bull_acceleration(p):
    bg,p=_bg_range(p,68,vol=700); s=[]
    for i in range(12):
        if i<6: d=1 if random.random()<0.55 else -1; bar,p=_bar(p,d,_n(4),_n(3),_n(700))
        else: bar,p=_bar(p,1,_n(8+(i-6)*1.5),_n(2),_n(1400+(i-6)*150))
        s.append(bar)
    return bg+s, _future(p,1,13,2,2300,accel=True)

def move_bear_acceleration(p):
    bg,p=_bg_range(p,68,vol=700); s=[]
    for i in range(12):
        if i<6: d=-1 if random.random()<0.55 else 1; bar,p=_bar(p,d,_n(4),_n(3),_n(700))
        else: bar,p=_bar(p,-1,_n(8+(i-6)*1.5),_n(2),_n(1400+(i-6)*150))
        s.append(bar)
    return bg+s, _future(p,-1,13,2,2300,accel=True)

def move_bull_pullback_resume(p):
    bg,p=_bg_trend(p,68,1,body=8); s=[]
    for i in range(12):
        if i<5: bar,p=_bar(p,1,_n(9),_n(2.5),_n(1200))
        elif i<8: bar,p=_bar(p,-1,_n(5),_n(3),_n(700))
        else: bar,p=_bar(p,1,_n(8),_n(2),_n(1100))
        s.append(bar)
    return bg+s, _future(p,1,9,2.5,1300)

def move_bear_pullback_resume(p):
    bg,p=_bg_trend(p,68,-1,body=8); s=[]
    for i in range(12):
        if i<5: bar,p=_bar(p,-1,_n(9),_n(2.5),_n(1200))
        elif i<8: bar,p=_bar(p,1,_n(5),_n(3),_n(700))
        else: bar,p=_bar(p,-1,_n(8),_n(2),_n(1100))
        s.append(bar)
    return bg+s, _future(p,-1,9,2.5,1300)

def move_bull_exhaustion(p):
    bg,p=_bg_trend(p,68,1,body=9); s=[]
    for i in range(12):
        if i<8: bar,p=_bar(p,1,_n(9),_n(2),_n(1500))
        else:
            d=1 if random.random()<0.40 else -1
            bar,p=_bar(p,d,_n(3),_n(5.5),_n(580))
        s.append(bar)
    return bg+s, _future(p,-1,7,5,900,fade=True)

def move_bear_exhaustion(p):
    bg,p=_bg_trend(p,68,-1,body=9); s=[]
    for i in range(12):
        if i<8: bar,p=_bar(p,-1,_n(9),_n(2),_n(1500))
        else:
            d=-1 if random.random()<0.40 else 1
            bar,p=_bar(p,d,_n(3),_n(5.5),_n(580))
        s.append(bar)
    return bg+s, _future(p,1,7,5,900,fade=True)

def move_bull_pin_top(p):
    bg,p=_bg_trend(p,68,1,body=8); s=[]
    for i in range(12):
        if i<8: bar,p=_bar(p,1,_n(8),_n(2.5),_n(1200)); s.append(bar)
        elif i==9:
            o=p; h=o+_n(16)*PIP; l=o-_n(2)*PIP; c=o-_n(2)*PIP
            bar={"open":float(o),"high":float(h),"low":float(l),"close":float(c),"volume":int(_n(1900))}
            p=float(c); s.append(bar)
        else:
            d=-1 if random.random()<0.65 else 1
            bar,p=_bar(p,d,_n(4),_n(3),_n(850)); s.append(bar)
    return bg+s, _future(p,-1,8,5,1100,fade=True)

def move_bear_pin_bottom(p):
    bg,p=_bg_trend(p,68,-1,body=8); s=[]
    for i in range(12):
        if i<8: bar,p=_bar(p,-1,_n(8),_n(2.5),_n(1200)); s.append(bar)
        elif i==9:
            o=p; l=o-_n(16)*PIP; h=o+_n(2)*PIP; c=o+_n(2)*PIP
            bar={"open":float(o),"high":float(h),"low":float(l),"close":float(c),"volume":int(_n(1900))}
            p=float(c); s.append(bar)
        else:
            d=1 if random.random()<0.65 else -1
            bar,p=_bar(p,d,_n(4),_n(3),_n(850)); s.append(bar)
    return bg+s, _future(p,1,8,5,1100,fade=True)

def move_bearish_engulf(p):
    bg,p=_bg_trend(p,68,1,body=7); s=[]
    for i in range(12):
        if i<10: d=1 if random.random()<0.65 else -1; bar,p=_bar(p,d,_n(7),_n(3),_n(1000))
        elif i==10: bar,p=_bar(p,1,_n(5),_n(2),_n(800))
        else: bar,p=_bar(p,-1,_n(14),_n(2),_n(2300))
        s.append(bar)
    return bg+s, _future(p,-1,10,2.5,1900,accel=True)

def move_bullish_engulf(p):
    bg,p=_bg_trend(p,68,-1,body=7); s=[]
    for i in range(12):
        if i<10: d=-1 if random.random()<0.65 else 1; bar,p=_bar(p,d,_n(7),_n(3),_n(1000))
        elif i==10: bar,p=_bar(p,-1,_n(5),_n(2),_n(800))
        else: bar,p=_bar(p,1,_n(14),_n(2),_n(2300))
        s.append(bar)
    return bg+s, _future(p,1,10,2.5,1900,accel=True)

def move_fakeout_bull_trap(p):
    bg,p=_bg_trend(p,68,1,body=6); s=[]
    for i in range(12):
        if i<8: bar,p=_bar(p,1,_n(6),_n(3),_n(900))
        elif i<10: bar,p=_bar(p,1,_n(10),_n(3),_n(1700))
        else: bar,p=_bar(p,-1,_n(13),_n(2),_n(2100))
        s.append(bar)
    return bg+s, _future(p,-1,11,3,1900,accel=True)

def move_fakeout_bear_trap(p):
    bg,p=_bg_trend(p,68,-1,body=6); s=[]
    for i in range(12):
        if i<8: bar,p=_bar(p,-1,_n(6),_n(3),_n(900))
        elif i<10: bar,p=_bar(p,-1,_n(10),_n(3),_n(1700))
        else: bar,p=_bar(p,1,_n(13),_n(2),_n(2100))
        s.append(bar)
    return bg+s, _future(p,1,11,3,1900,accel=True)

def move_bull_squeeze_break(p):
    bg,p=_bg_range(p,68,vol=700); s=[]
    for i in range(12):
        if i<9: bar,p=_bar(p,random.choice([-1,1]),_n(2.5),_n(1.5),_n(370))
        else: bar,p=_bar(p,1,_n(15+(i-9)*3),_n(2),_n(2900+(i-9)*400))
        s.append(bar)
    return bg+s, _future(p,1,14,2,2600,accel=True)

def move_bear_squeeze_break(p):
    bg,p=_bg_range(p,68,vol=700); s=[]
    for i in range(12):
        if i<9: bar,p=_bar(p,random.choice([-1,1]),_n(2.5),_n(1.5),_n(370))
        else: bar,p=_bar(p,-1,_n(15+(i-9)*3),_n(2),_n(2900+(i-9)*400))
        s.append(bar)
    return bg+s, _future(p,-1,14,2,2600,accel=True)

def move_bull_flag_break(p):
    bg,p=_bg_trend(p,68,1,body=8); s=[]
    for i in range(12):
        if i<5: bar,p=_bar(p,1,_n(10),_n(2),_n(1700))
        elif i<10: bar,p=_bar(p,random.choice([-1,1]),_n(3),_n(2),_n(500))
        else: bar,p=_bar(p,1,_n(9),_n(2),_n(1500))
        s.append(bar)
    return bg+s, _future(p,1,10,2.5,1600)

def move_bear_flag_break(p):
    bg,p=_bg_trend(p,68,-1,body=8); s=[]
    for i in range(12):
        if i<5: bar,p=_bar(p,-1,_n(10),_n(2),_n(1700))
        elif i<10: bar,p=_bar(p,random.choice([-1,1]),_n(3),_n(2),_n(500))
        else: bar,p=_bar(p,-1,_n(9),_n(2),_n(1500))
        s.append(bar)
    return bg+s, _future(p,-1,10,2.5,1600)

def move_failed_bull_break(p):
    bg,p=_bg_range(p,68); s=[]
    for i in range(12):
        if i<7:
            d=1 if random.random()<0.55 else -1; bar,p=_bar(p,d,_n(6),_n(3),_n(800)); s.append(bar)
        elif i<10:
            bar,p=_bar(p,1,_n(9),_n(4),_n(1500)); s.append(bar)
        else:
            o=p; h=o+_n(14)*PIP; l=o-_n(2)*PIP; c=o-_n(4)*PIP
            bar={"open":float(o),"high":float(h),"low":float(l),"close":float(c),"volume":int(_n(1900))}
            s.append(bar); p=float(c)
    return bg+s, _future(p,-1,8,4.5,1300,fade=True)

def move_failed_bear_break(p):
    bg,p=_bg_range(p,68); s=[]
    for i in range(12):
        if i<7:
            d=-1 if random.random()<0.55 else 1; bar,p=_bar(p,d,_n(6),_n(3),_n(800)); s.append(bar)
        elif i<10:
            bar,p=_bar(p,-1,_n(9),_n(4),_n(1500)); s.append(bar)
        else:
            o=p; l=o-_n(14)*PIP; h=o+_n(2)*PIP; c=o+_n(4)*PIP
            bar={"open":float(o),"high":float(h),"low":float(l),"close":float(c),"volume":int(_n(1900))}
            s.append(bar); p=float(c)
    return bg+s, _future(p,1,8,4.5,1300,fade=True)

def move_asia_tight_range(p):
    bg,p=_bg_range(p,68,vol=500); center=p; s=[]
    for _ in range(12):
        d=1 if p<center else -1
        if random.random()<0.45: d=-d
        bar,p=_bar(p,d,_n(2.5),_n(2),_n(310))
        p=float(np.clip(p,center-8*PIP,center+8*PIP)); s.append(bar)
    return bg+s, _future(p,random.choice([-1,1]),3.5,2.5,380)

def move_midrange_oscillation(p):
    bg,p=_bg_range(p,68,vol=700); center=p; swing=1; s=[]
    for i in range(12):
        if i%4==0: swing=-swing
        bar,p=_bar(p,swing,_n(5),_n(3.5),_n(650))
        p=float(np.clip(p,center-15*PIP,center+15*PIP)); s.append(bar)
    return bg+s, _future(p,random.choice([-1,1]),4.5,4,700)

def move_bull_flag_tight(p):
    bg,p=_bg_trend(p,68,1,body=9); fh=p; s=[]
    for i in range(12):
        if i<5: bar,p=_bar(p,1,_n(11),_n(2.5),_n(1800)); fh=p
        else:
            bar,p=_bar(p,random.choice([-1,1]),_n(2.5),_n(2),_n(420))
            p=float(np.clip(p,fh-12*PIP,fh+3*PIP))
        s.append(bar)
    return bg+s, _future(p,1,9,2,1500)

def move_bear_flag_tight(p):
    bg,p=_bg_trend(p,68,-1,body=9); fl=p; s=[]
    for i in range(12):
        if i<5: bar,p=_bar(p,-1,_n(11),_n(2.5),_n(1800)); fl=p
        else:
            bar,p=_bar(p,random.choice([-1,1]),_n(2.5),_n(2),_n(420))
            p=float(np.clip(p,fl-3*PIP,fl+12*PIP))
        s.append(bar)
    return bg+s, _future(p,-1,9,2,1500)

def move_bull_absorption(p):
    bg,p=_bg_trend(p,68,-1,body=7); s=[]
    for i in range(12):
        if i<6: bar,p=_bar(p,-1,_n(7),_n(2.5),_n(1100))
        else:
            d=-1 if random.random()<0.42 else 1
            bar,p=_bar(p,d,_n(2.5),_n(3.5),_n(2900))
        s.append(bar)
    return bg+s, _future(p,1,9,3,1600)

def move_bear_absorption(p):
    bg,p=_bg_trend(p,68,1,body=7); s=[]
    for i in range(12):
        if i<6: bar,p=_bar(p,1,_n(7),_n(2.5),_n(1100))
        else:
            d=1 if random.random()<0.42 else -1
            bar,p=_bar(p,d,_n(2.5),_n(3.5),_n(2900))
        s.append(bar)
    return bg+s, _future(p,-1,9,3,1600)

def move_news_bull_spike(p):
    bg,p=_bg_range(p,68,vol=700); s=[]
    for i in range(12):
        if i<6: bar,p=_bar(p,random.choice([-1,1]),_n(5),_n(3),_n(700))
        elif i<8: bar,p=_bar(p,1,_n(28),_n(4),_n(5800))
        else:
            d=-1 if random.random()<0.55 else 1
            bar,p=_bar(p,d,_n(6),_n(5),_n(1300))
        s.append(bar)
    return bg+s, _future(p,1,8,6,1200)

def move_news_bear_spike(p):
    bg,p=_bg_range(p,68,vol=700); s=[]
    for i in range(12):
        if i<6: bar,p=_bar(p,random.choice([-1,1]),_n(5),_n(3),_n(700))
        elif i<8: bar,p=_bar(p,-1,_n(28),_n(4),_n(5800))
        else:
            d=1 if random.random()<0.55 else -1
            bar,p=_bar(p,d,_n(6),_n(5),_n(1300))
        s.append(bar)
    return bg+s, _future(p,-1,8,6,1200)

# ── registry ─────────────────────────────────────────────────────────
ALL_MOVES = [
    move_bull_impulse, move_bear_impulse,         # 0,1
    move_bull_grind, move_bear_grind,             # 2,3
    move_bull_acceleration, move_bear_acceleration, # 4,5
    move_bull_pullback_resume, move_bear_pullback_resume, # 6,7
    move_bull_exhaustion, move_bear_exhaustion,   # 8,9
    move_bull_pin_top, move_bear_pin_bottom,      # 10,11
    move_bearish_engulf, move_bullish_engulf,     # 12,13
    move_fakeout_bull_trap, move_fakeout_bear_trap, # 14,15
    move_bull_squeeze_break, move_bear_squeeze_break, # 16,17
    move_bull_flag_break, move_bear_flag_break,   # 18,19
    move_failed_bull_break, move_failed_bear_break, # 20,21
    move_asia_tight_range, move_midrange_oscillation, # 22,23
    move_bull_flag_tight, move_bear_flag_tight,   # 24,25
    move_bull_absorption, move_bear_absorption,   # 26,27
    move_news_bull_spike, move_news_bear_spike,   # 28,29
]

MODEL_WEIGHTS = {
    "momentum":      [3.5,3.5,3.0,3.0,3.5,3.5,3.5,3.5,2.5,2.5,1.5,1.5,1.5,1.5,1.5,1.5,1.0,1.0,1.5,1.5,1.0,1.0,1.0,1.0,1.5,1.5,1.5,1.5,1.0,1.0],
    "vwap":          [2.0,2.0,1.5,1.5,2.0,2.0,2.5,2.5,2.0,2.0,1.5,1.5,1.5,1.5,2.0,2.0,2.0,2.0,2.0,2.0,1.5,1.5,2.0,2.0,2.0,2.0,3.5,3.5,3.0,3.0],
    "candle":        [3.0,3.0,2.0,2.0,2.0,2.0,2.5,2.5,3.0,3.0,3.5,3.5,3.5,3.5,2.5,2.5,1.5,1.5,2.0,2.0,2.0,2.0,2.5,2.5,2.5,2.5,2.0,2.0,1.5,1.5],
    "breakout":      [2.0,2.0,1.5,1.5,2.0,2.0,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,2.5,2.5,3.5,3.5,3.0,3.0,3.0,3.0,3.0,3.0,3.5,3.5,1.5,1.5,1.5,1.5],
    "microstructure":[3.5,3.5,1.5,1.5,2.5,2.5,2.0,2.0,3.0,3.0,2.0,2.0,3.0,3.0,2.5,2.5,2.0,2.0,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,3.5,3.5,3.5,3.5],
}

def generate_sample(model_name="momentum", base_price=None):
    if base_price is None:
        base_price = 1.0800 + np.random.uniform(-0.0100, 0.0100)
    weights = MODEL_WEIGHTS[model_name]
    fn = random.choices(ALL_MOVES, weights=weights, k=1)[0]
    ctx, fut = fn(base_price)
    assert len(ctx)==80, f"{fn.__name__} returned {len(ctx)} ctx bars"
    assert len(fut)==4,  f"{fn.__name__} returned {len(fut)} fut bars"
    return ctx, fut, fn.__name__
