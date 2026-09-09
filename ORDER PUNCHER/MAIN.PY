# =============================================================
# MT5 MULTI STOP PUNCHER (CHAIN MODE + M30 CALIBRATED)
# =============================================================
# spacing_pts > 0 : classic fixed-distance ladder
# spacing_pts = 0 : CHAIN MODE (next order starts where previous TP ends)
# Orders are punched just before new M30 candle
# =============================================================

import MetaTrader5 as mt5
import time
from datetime import datetime

# ================= CONFIG =================
SYMBOL = "XAUUSD"

# --- BUY SIDE ---
BUY_ORDERS      = 5
BUY_START_PTS   = 150       # first order offset from price
BUY_SPACING_PTS = 0        # 0 = CHAIN MODE
BUY_TP_PTS      = 200
BUY_SL_PTS      = None

# --- SELL SIDE ---
SELL_ORDERS      = 5
SELL_START_PTS   = 40
SELL_SPACING_PTS = 0       # 0 = CHAIN MODE
SELL_TP_PTS      = 200
SELL_SL_PTS      = None

VOLUME          = 0.01
MAGIC           = 260210
DEVIATION       = 10
OFFSET_SECONDS  = -1       # punch before M30 open
REMOVE_EXISTING = True
# =========================================


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


def place_pending(order_type, price, sl, tp):
    req = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": SYMBOL,
        "volume": VOLUME,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": DEVIATION,
        "magic": MAGIC,
        "comment": "m30_chain_puncher",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    res = mt5.order_send(req)
    if res.retcode != mt5.TRADE_RETCODE_DONE:
        print("FAILED", order_type, res.retcode, res.comment)
    else:
        print(f"PLACED {order_type} @ {price}")


# ================= MAIN =================
if not mt5.initialize():
    raise RuntimeError("MT5 init failed")

if not mt5.symbol_select(SYMBOL, True):
    raise RuntimeError("Symbol not found")

wait = get_m30_calibrated(SYMBOL, OFFSET_SECONDS)
if wait is None or wait <= 0:
    raise RuntimeError("Invalid M30 timing")

print(f"Waiting {wait:.2f}s for M30 sync")
time.sleep(wait)

bid = mt5.symbol_info_tick(SYMBOL).bid
p = pts(SYMBOL)

if REMOVE_EXISTING:
    remove_existing(SYMBOL, MAGIC)

# -------- BUY SIDE --------
entry = bid + BUY_START_PTS * p
for i in range(BUY_ORDERS):
    tp = entry + BUY_TP_PTS * p
    sl = None

    place_pending(
        mt5.ORDER_TYPE_BUY_STOP,
        norm(entry, SYMBOL),
        norm(sl, SYMBOL),
        norm(tp, SYMBOL),
    )

    # spacing logic
    if BUY_SPACING_PTS > 0:
        entry = bid + (BUY_START_PTS + BUY_SPACING_PTS * (i + 1)) * p
    else:
        entry = tp  # CHAIN MODE

# -------- SELL SIDE --------
entry = bid - SELL_START_PTS * p
for i in range(SELL_ORDERS):
    tp = entry - SELL_TP_PTS * p
    sl = entry + SELL_SL_PTS * p

    place_pending(
        mt5.ORDER_TYPE_SELL_STOP,
        norm(entry, SYMBOL),
        norm(sl, SYMBOL),
        norm(tp, SYMBOL),
    )

    if SELL_SPACING_PTS > 0:
        entry = bid - (SELL_START_PTS + SELL_SPACING_PTS * (i + 1)) * p
    else:
        entry = tp  # CHAIN MODE

print("\nOrders punched at", datetime.now())
mt5.shutdown()
