import MetaTrader5 as mt5
import logging
import time
from datetime import datetime
import numpy as np
import pandas as pd

from classifier import CandlePredictor
from placer import execute_automated_trade


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SYMBOLS = ["EURUSD.e"]
TIMEFRAME = mt5.TIMEFRAME_M30
MAGIC = 2   # Only for trades opened by THIS bot


# ==========================================================
# ------------------- MT5-STYLE EMA ------------------------
# ==========================================================

def mt5_style_ema(closes, period=10):
    alpha = 2 / (period + 1)
    ema = np.zeros(len(closes))
    ema[0] = closes[0]

    for i in range(1, len(closes)):
        ema[i] = ema[i - 1] + alpha * (closes[i] - ema[i - 1])

    return ema


# ==========================================================
# -------- EMA BODY INTERACTION (LAST 2 ONLY) --------------
# ==========================================================

def check_recent_crossover(rates):
    """
    EMA must pass through the BODY of
    one of the LAST TWO closed candles (-2, -1)
    """
    closes = np.array([r["close"] for r in rates], float)

    if len(closes) < 15:
        return False

    ema = mt5_style_ema(closes, period=10)

    for i in [-2, -1]:
        o = rates[i]["open"]
        c = rates[i]["close"]
        e = ema[i]

        if min(o, c) <= e <= max(o, c):
            logging.info(f"✅ EMA body interaction confirmed on candle {i}")
            return True

    return False


# ==========================================================
# ---------------- BOT POSITION CHECK ----------------------
# ==========================================================

def get_bot_position(symbol):
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return None

    for pos in positions:
        if pos.magic == MAGIC:
            return pos

    return None


# ==========================================================
# ---------------- CLOSE OPPOSITE TRADE --------------------
# ==========================================================

def close_opposite_trade(symbol, position):
    close_type = (
        mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY
        else mt5.ORDER_TYPE_BUY
    )

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": position.volume,
        "type": close_type,
        "position": position.ticket,
        "deviation": 20,
        "magic": MAGIC,
        "comment": "Auto reverse",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    result = mt5.order_send(request)
    logging.info(f"🔄 Closed position {position.ticket} → {result}")


# ==========================================================
# ------------------- STRATEGY LOGIC -----------------------
# ==========================================================

def check_strategy(predictor, symbol):
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 1, 60)
    if rates is None or len(rates) < 50:
        return "hold"

    # ---- EMA BODY INTERACTION GATE ----
    if not check_recent_crossover(rates):
        return "hold"

    prev = rates[-2]
    last = rates[-1]

    o, c = last["open"], last["close"]
    last_body = abs(c - o)

    bodies = [abs(r["close"] - r["open"]) for r in rates[-9:-1]]
    avg_body8 = np.mean(bodies)
    bigger_count = sum(last_body > b for b in bodies)

    # ---- Classifier ----
    result = predictor.predict(symbol, TIMEFRAME, 50)
    pred = result["prediction"]

    # ---- EMA ----
    closes = np.array([r["close"] for r in rates], float)
    ema = mt5_style_ema(closes, 10)

    last_ema = ema[-1]
    prev_ema = ema[-2]

    crossed_up = prev["close"] < prev_ema and c > last_ema
    crossed_down = prev["close"] > prev_ema and c < last_ema

    # ---- Ratio logic ----
    high_body = max(o, c)
    low_body = min(o, c)
    ema_clamped = np.clip(last_ema, low_body, high_body)
    body_size = last_body if last_body > 0 else 1e-6

    upper_ratio = (high_body - ema_clamped) / body_size
    lower_ratio = (ema_clamped - low_body) / body_size

    # ---------------- DECISION RULES ---------------- #

    # 🥇 Rule 1: EMA crossover + strength
    if crossed_up and last_body >= avg_body8 * 0.5:
        return "buy"
    if crossed_down and last_body >= avg_body8 * 0.5:
        return "sell"

    # 🥈 Rule 2: Classifier + dominance
    if pred == 2 and bigger_count >= 4:
        return "buy"
    if pred == 0 and bigger_count >= 4:
        return "sell"

    # 🥉 Rule 3: Ratio conviction
    if upper_ratio > 0.75:
        return "buy"
    if lower_ratio > 0.75:
        return "sell"

    return "hold"


# ==========================================================
# ---------------------- MAIN LOOP -------------------------
# ==========================================================

def main_loop():
    print("🚀 Starting trading bot...")

    if not mt5.initialize():
        logging.error("❌ MT5 initialization failed.")
        return

    predictor = CandlePredictor()
    last_timestamp = None

    try:
        while True:
            now = datetime.now()
            candle_minute = (now.minute // 15) * 15
            candle_time = now.replace(minute=candle_minute, second=0, microsecond=0)

            if now.minute % 15 == 0 and now.second < 2 and candle_time != last_timestamp:
                last_timestamp = candle_time

                for symbol in SYMBOLS:
                    decision = check_strategy(predictor, symbol)
                    pos = get_bot_position(symbol)

                    if decision in ["buy", "sell"]:
                        if pos:
                            pos_dir = "buy" if pos.type == mt5.ORDER_TYPE_BUY else "sell"

                            # Same direction → do nothing
                            if pos_dir == decision:
                                logging.info(f"⏸ {symbol}: Same direction trade exists — skipping.")
                                continue

                            # Opposite → close & reverse
                            logging.info(f"🔄 {symbol}: Reversing position.")
                            close_opposite_trade(symbol, pos)
                            execute_automated_trade(symbol, decision)

                        else:
                            execute_automated_trade(symbol, decision)

                    else:
                        logging.info(f"{symbol}: HOLD")

                time.sleep(13 * 60)
            else:
                time.sleep(1)

    except KeyboardInterrupt:
        logging.info("🛑 Bot stopped manually.")
    finally:
        predictor.shutdown()
        mt5.shutdown()


# ==========================================================
# ------------------------- RUN -----------------------------
# ==========================================================

if __name__ == "__main__":
    main_loop()
