import MetaTrader5 as mt5
from datetime import datetime
import math
import time
import logging

def initialize_mt5():
    """Initialize MT5 connection if not already connected."""
    if not mt5.initialize():
        print("MT5 initialization failed!", mt5.last_error())
        return False
    return True


def get_account_balance():
    """Retrieve account balance."""
    account_info = mt5.account_info()
    if account_info is None:
        print("Failed to retrieve account info.", mt5.last_error())
        return None
    return account_info.balance


def calculate_lot_size(balance):
    """
    Calculate lot size:
    - 0.01 lots per $40 balance
    - Minimum lot size = 0.01
    """
    if balance is None:
        return 0.01  # fallback minimum lot

    lot_units = math.floor(balance /20)
    lot_size = lot_units * 0.01
    lot_size = max(lot_size, 0.01)  # Ensure at least 0.01 lots

    print(f"💰 Account Balance: {balance}, Calculated Lot Size: {lot_size}")
    return lot_size


def has_open_position(symbol, trade_type):
    """Check if there is an existing opposite position."""
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return False
    for position in positions:
        if (trade_type == "buy" and position.type == mt5.ORDER_TYPE_SELL) or \
           (trade_type == "sell" and position.type == mt5.ORDER_TYPE_BUY):
            print(f"⚠️ Open opposite trade found for {symbol}. Blocking trade.")
            return True
    return False
#------------------------------------------------------------------------------------------------------------

def place_trade(symbol, trade_type, estimate):
    if not initialize_mt5():
        return False

    # =============================
    # SYMBOL SETUP
    # =============================
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"❌ Symbol not found: {symbol}")
        return False

    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            print(f"❌ Failed to select {symbol}")
            return False

    # =============================
    # PRE-CHECKS
    # =============================
    opposite = "sell" if trade_type.lower() == "buy" else "buy"
    if has_open_position(symbol, opposite):
        print("❌ Opposite position exists")
        return False

    # =============================
    # LOT SIZE CALCULATION
    # =============================
    balance = get_account_balance()
    lot_size = calculate_lot_size(balance)

    step = symbol_info.volume_step
    market_lot = round((lot_size * 0.33) / step) * step
    limit_lot  = round((lot_size * 0.66) / step) * step

    # =============================
    # PRICE DATA
    # =============================
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        print("❌ No tick data")
        return False

    digits = symbol_info.digits

    e = estimate  # price distance (already in price units)

    current_price = tick.ask if trade_type.lower() == "buy" else tick.bid

    # =============================
    # DIRECTION LOGIC
    # =============================
    if trade_type.lower() == "buy":
        market_type = mt5.ORDER_TYPE_BUY
        limit_type  = mt5.ORDER_TYPE_BUY_LIMIT

        tp = current_price + e
        sl = current_price - e
        limit_entry = current_price - (0.5 * e)

    else:
        market_type = mt5.ORDER_TYPE_SELL
        limit_type  = mt5.ORDER_TYPE_SELL_LIMIT

        tp = current_price - e
        sl = current_price + e
        limit_entry = current_price + (0.5 * e)

    # =============================
    # NORMALIZATION
    # =============================
    current_price = round(current_price, digits)
    tp = round(tp, digits)
    sl = round(sl, digits)
    limit_entry = round(limit_entry, digits)

    success = False

    # =============================
    # 1️⃣ MARKET ORDER (FOK)
    # =============================
    market_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": market_lot,
        "type": market_type,
        "price": current_price,
        "sl": sl,
        "tp": tp,
        "deviation": 10,
        "magic": 10000,
        "comment": "MKT",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    result = mt5.order_send(market_request)

    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"✅ Market order placed: {result.order}")
        success = True
    else:
        print(f"❌ Market failed: {result.retcode if result else mt5.last_error()}")

    # =============================
    # 2️⃣ LIMIT ORDER (RETURN)
    # =============================
    limit_request = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "volume": limit_lot,
        "type": limit_type,
        "price": limit_entry,
        "sl": sl,
        "tp": tp,
        "magic": 10000,
        "comment": "LMT",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }

    result = mt5.order_send(limit_request)

    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"✅ Limit order placed: {result.order}")
        success = True
    else:
        print(f"❌ Limit failed: {result.retcode if result else mt5.last_error()}")

    # =============================
    # FINAL STATUS
    # =============================
    if success:
        print("🚀 Orders executed successfully")
    else:
        print("💥 All orders failed")

    return success

#-------------------------------------------------------------------------------------------------------
def execute_automated_trade(symbol, prediction, estimate):
    """Execute trade based on prediction using estimate (e)."""

    if not prediction:
        print("❌ No valid prediction received, skipping trade.")
        return None

    prediction = prediction.lower()

    # Ensure MT5 is ready
    if not mt5.initialize():
        print("❌ MT5 not initialized")
        return None

    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        print("❌ No tick data")
        return None

    # 🔥 Convert estimate if needed (IMPORTANT)
    symbol_info = mt5.symbol_info(symbol)
    e = estimate

    # 👉 If estimate is in PIPS, uncomment this:
    # e = estimate * symbol_info.point

    if prediction == "buy":
        print(f"{datetime.now()} - 📈 Prediction: BUY, placing trade on {symbol}...")
        return place_trade(symbol, "buy", e)

    elif prediction == "sell":
        print(f"{datetime.now()} - 📉 Prediction: SELL, placing trade on {symbol}...")
        return place_trade(symbol, "sell", e)

    elif prediction == "no trade":
        print(f"{datetime.now()} - 🛑 Prediction: NO TRADE, skipping execution.")
        return None

    print("❌ Invalid prediction received, skipping trade.")
    return None


def close_position(pos):
    """
    Closes either:
    - Open position
    - Pending LIMIT order
    """

    # --------------------------------------------------------
    # ✅ Case 1: Open Position
    # --------------------------------------------------------
    if hasattr(pos, "type") and pos.type in (mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_SELL):

        close_type = (
            mt5.ORDER_TYPE_SELL
            if pos.type == mt5.ORDER_TYPE_BUY
            else mt5.ORDER_TYPE_BUY
        )

        tick = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            logging.warning("No tick data for %s", pos.symbol)
            return False

        price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": pos.ticket,
            "price": price,
            "deviation": 50,
            "magic": 10000,
            "comment": "Auto-close position",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

    # --------------------------------------------------------
    # ✅ Case 2: Pending Order (LIMIT / others)
    # --------------------------------------------------------
    elif hasattr(pos, "type") and pos.type in (
        mt5.ORDER_TYPE_BUY_LIMIT,
        mt5.ORDER_TYPE_SELL_LIMIT,
        mt5.ORDER_TYPE_BUY_STOP,
        mt5.ORDER_TYPE_SELL_STOP,
    ):

        req = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": pos.ticket,
            "symbol": pos.symbol,
            "magic": 10000,
            "comment": "Auto-remove pending",
        }

    else:
        logging.warning("Unknown object type for ticket=%s", getattr(pos, "ticket", "N/A"))
        return False

    # --------------------------------------------------------
    # Execute
    # --------------------------------------------------------
    result = mt5.order_send(req)

    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        logging.info("Closed/Removed ticket=%d", pos.ticket)
        return True

    logging.warning("Close/Remove failed ticket=%d, retcode=%s", pos.ticket, getattr(result, "retcode", None))
    return False
# ------------------------------------------------------------
# Close profitable trades
# ------------------------------------------------------------
def close_profitable_trades(symbol):

    # ----------------------------------------
    # 1. Close profitable OPEN positions
    # ----------------------------------------
    positions = mt5.positions_get(symbol=symbol) or []

    for pos in positions:
        if pos.profit > 0:
            logging.info(
                "Closing profitable trade ticket=%d profit=%.2f",
                pos.ticket,
                pos.profit
            )
            close_position(pos)

    # ----------------------------------------
    # 2. Remove ALL pending (unfilled) orders
    # ----------------------------------------
    orders = mt5.orders_get(symbol=symbol) or []

    for order in orders:
        logging.info(
            "Removing pending order ticket=%d type=%d price=%.5f",
            order.ticket,
            order.type,
            order.price_open
        )

        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": order.ticket,
        }

        result = mt5.order_send(request)

        if result is None:
            logging.error(f"❌ Failed to remove order {order.ticket}: {mt5.last_error()}")
        elif result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"❌ Order removal failed {order.ticket}, retcode={result.retcode}")
        else:
            logging.info(f"✅ Removed pending order {order.ticket}")

# ------------------------------------------------------------
# Update SLTP
# ------------------------------------------------------------import MetaTrader5 as mt5
import MetaTrader5 as mt5
import logging
import time
from sltp import compute_sltp
def update_existing_trade_sltp(pos, symbol, timeframe, retries=3):
    """
    Safely updates SL/TP for an existing position.
    """

    try:
        # 🔥 Always ensure correct symbol from position
        symbol = pos.symbol

        # 🔥 Ensure symbol is selected (import-safe)
        if not mt5.symbol_select(symbol, True):
            logging.warning("Failed to select symbol %s", symbol)
            return False

        # 🔥 Retry tick fetch (fixes import timing issue)
        tick = None
        info = None
        for _ in range(5):
            tick = mt5.symbol_info_tick(symbol)
            info = mt5.symbol_info(symbol)
            if tick and info:
                break
            time.sleep(0.2)

        if not tick or not info:
            logging.warning("No tick/info for %s", symbol)
            return False

        # =========================
        # SLTP ESTIMATE
        # =========================
        estimate = compute_sltp(symbol, timeframe)
        if estimate is None:
            logging.warning("No SLTP estimate")
            return False

        # 🔥 CRITICAL FIX → remove numpy type
        estimate = float(estimate)

        # =========================
        # CURRENT PRICE
        # =========================
        price = tick.ask if pos.type == mt5.ORDER_TYPE_BUY else tick.bid

        # =========================
        # BROKER CONSTRAINTS
        # =========================
        min_stop = info.trade_stops_level * info.point
        buffer = 2 * info.point
        min_dist = float(min_stop + buffer)

        # ensure estimate respects broker rules
        estimate = max(estimate, min_dist)

        # =========================
        # RAW SL / TP
        # =========================
        if pos.type == mt5.ORDER_TYPE_BUY:
            sl = price - estimate
            tp = price + estimate
        else:
            sl = price + estimate
            tp = price - estimate

        # =========================
        # ENFORCE MIN DISTANCE
        # =========================
        if pos.type == mt5.ORDER_TYPE_BUY:
            sl = min(sl, price - min_dist)
            tp = max(tp, price + min_dist)
        else:
            sl = max(sl, price + min_dist)
            tp = min(tp, price - min_dist)

        # =========================
        # FINAL CLEAN TYPES (🔥 IMPORTANT)
        # =========================
        sl = float(round(sl, info.digits))
        tp = float(round(tp, info.digits))

        # =========================
        # REQUEST
        # =========================
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": pos.ticket,
            "sl": sl,
            "tp": tp,
        }

        logging.info("SLTP REQUEST → %s", request)

        # =========================
        # RETRY LOGIC
        # =========================
        result = None

        for attempt in range(retries):
            result = mt5.order_send(request)

            if result is not None:
                break

            logging.warning("order_send None → retry %d", attempt + 1)
            time.sleep(0.3)

        # =========================
        # RESULT HANDLING
        # =========================
        if result is None:
            logging.error("SLTP FAILED → MT5 busy (None result)")
            return False

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(
                "SLTP FAILED → retcode=%s comment=%s",
                result.retcode,
                result.comment
            )
            return False

        logging.info(
            "SLTP SUCCESS → ticket=%s SL=%.5f TP=%.5f",
            pos.ticket, sl, tp
        )

        time.sleep(0.2)
        return True

    except Exception as e:
        logging.error("SLTP EXCEPTION → %s", str(e))
        return False
#calibration ---------------------------------------------------------------------------
import MetaTrader5 as mt5
import time

def get_m30_calibrated(symbol="EURUSD", offset= -2):
    """
    Calculates seconds until the next 30m mark + an additional offset.
    Returns float (seconds.milliseconds).
    """
    # 1. Ensure symbol is selected so info isn't None
    mt5.symbol_select(symbol, True)
    
    info = mt5.symbol_info(symbol)
    if info is None:
        return None
    
    # 2. Calculate raw seconds remaining in the 1800s (30m) block
    # info.time % 1800 = seconds passed since last :00 or :30
    remaining = 3600 - (info.time % 3600)
    
    # 3. Add millisecond precision from local clock and the 2s addition
    # Subtracting (time.time() % 1) aligns us to the 000m s start of the second
    total_wait = remaining - (time.time() % 1) + offset
    
    return total_wait