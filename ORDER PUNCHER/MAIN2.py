# =============================================================
# MT5 M30 CHAIN FLIP PUNCHER (EXTENSION OF MAIN ENGINE)
# =============================================================
# BUILDS ON PREVIOUS MAIN FILE FEATURES:
# - BUY / SELL separate configs
# - CHAIN MODE when spacing_pts = 0
# - Bid / Ask correct pricing
# - M30 calibrated timing
# - SAME inputs & structure
#
# NEW BEHAVIOR:
# - Orders are NOT placed once and exit
# - Engine runs continuously
# - When ALL orders + positions of one side are cleared
#   -> immediately places the OPPOSITE chain
# - Specification (start, TP, SL, spacing) REMAINS SAME
# =============================================================

import MetaTrader5 as mt5
import time
from datetime import datetime

# ================= CONFIG (SAME AS MAIN) =================
SYMBOL = "EURUSD.e"

# --- BUY SIDE ---
BUY_ORDERS      = 4
BUY_START_PTS   = 40
BUY_SPACING_PTS = 0        # 0 = CHAIN MODE
BUY_TP_PTS      = 100
BUY_SL_PTS      = 6

# --- SELL SIDE ---
SELL_ORDERS      = 4
SELL_START_PTS   = 40
SELL_SPACING_PTS = 0       # 0 = CHAIN MODE
SELL_TP_PTS      = 100
SELL_SL_PTS      = 6

VOLUME          = 0.01
MAGIC           = 260210
DEVIATION       = 10
OFFSET_SECONDS  = -2       # M30 calibration
REMOVE_EXISTING = True

POLL_INTERVAL   = 0.5      # state check loop
# ==========================================================


def get_m30_calibrated(symbol, offset):
    mt5.symbol_select(symbol, True)
    info = mt5.symbol_info(symbol)
    if info is None:
        return None
    remaining = 1800 - (info.time % 1800)
    return remaining - (time.time() % 1) + offset


def pts(symbol):
    return mt5.symbol_info(symbol).point


def norm(price, symbol):
    return round(price, mt5.symbol_info(symbol).digits)


def active_orders():
    o = mt5.orders_get(symbol=SYMBOL)
    return o if o else []


def active_positions():
    p = mt5.positions_get(symbol=SYMBOL)
    return p if p else []


def remove_existing(symbol, magic):
    orders = mt5.orders_get(symbol=symbol)
    if not orders:
        return
    for o in orders:
        if o.magic == magic:
            mt5.order_send({
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": o.ticket,
                "symbol": symbol,
            })


def place_pending(order_type, entry, sl, tp):
    req = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": SYMBOL,
        "volume": VOLUME,
        "type": order_type,
        "price": norm(entry, SYMBOL),
        "sl": norm(sl, SYMBOL),
        "tp": norm(tp, SYMBOL),
        "deviation": DEVIATION,
        "magic": MAGIC,
        "type_time": mt5.ORDER_TIME_GTC,
        "comment": "m30_chain_flip",
    }
    mt5.order_send(req)


# -------- CHAIN PLACERS (REUSED LOGIC) --------

def place_buy_chain(base_price):
    p = pts(SYMBOL)
    entry = base_price + BUY_START_PTS * p
    for i in range(BUY_ORDERS):
        tp = entry + BUY_TP_PTS * p
        sl = entry - BUY_SL_PTS * p
        place_pending(mt5.ORDER_TYPE_BUY_STOP, entry, sl, tp)
        if BUY_SPACING_PTS > 0:
            entry = base_price + (BUY_START_PTS + BUY_SPACING_PTS * (i + 1)) * p
        else:
            entry = tp


def place_sell_chain(base_price):
    p = pts(SYMBOL)
    entry = base_price - SELL_START_PTS * p
    for i in range(SELL_ORDERS):
        tp = entry - SELL_TP_PTS * p
        sl = entry + SELL_SL_PTS * p
        place_pending(mt5.ORDER_TYPE_SELL_STOP, entry, sl, tp)
        if SELL_SPACING_PTS > 0:
            entry = base_price - (SELL_START_PTS + SELL_SPACING_PTS * (i + 1)) * p
        else:
            entry = tp


# ================= MAIN ENGINE =================
if not mt5.initialize():
    raise RuntimeError("MT5 init failed")

mt5.symbol_select(SYMBOL, True)

# Initial M30 sync (ONLY ON START)
wait = get_m30_calibrated(SYMBOL, OFFSET_SECONDS)
if wait is None or wait <= 0:
    raise RuntimeError("Invalid M30 timing")

print(f"Initial sync: waiting {wait:.2f}s")
time.sleep(wait)

if REMOVE_EXISTING:
    remove_existing(SYMBOL, MAGIC)

# Bootstrap with BOTH sides
# (same behavior as main puncher)
tick = mt5.symbol_info_tick(SYMBOL)
place_buy_chain(tick.ask)
place_sell_chain(tick.bid)

last_flip = None
print("M30 Chain Flip Puncher running")

while True:
    time.sleep(POLL_INTERVAL)

    # If nothing active → determine flip side
    if not active_orders() and not active_positions():
        tick = mt5.symbol_info_tick(SYMBOL)

        # Flip based on last direction
        if last_flip == "buy":
            place_sell_chain(tick.bid)
            last_flip = "sell"
            print("FLIP → SELL chain")
        else:
            place_buy_chain(tick.ask)
            last_flip = "buy"
            print("FLIP → BUY chain")
