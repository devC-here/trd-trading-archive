import MetaTrader5 as mt5
import logging
import time
from datetime import datetime
import pandas as pd
import numpy as np

from placer import execute_automated_trade


# ==========================
# CONFIG
# ==========================
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M5
BARS = 200
MAX_SPREAD = 0.0003

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


# ------------------------------------------------
# Ensure symbol
# ------------------------------------------------
def ensure_symbol(symbol):

    info = mt5.symbol_info(symbol)

    if info is None:
        logging.error("Symbol not found")
        return False

    if not info.visible:
        mt5.symbol_select(symbol, True)

    return True


# ------------------------------------------------
# Fetch market data
# ------------------------------------------------
def get_data():

    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, BARS)

    if rates is None:
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    return df

def mt5_style_ema(closes, period=10):
    alpha = 2 / (period + 1)

    ema = np.zeros(len(closes))
    ema[0] = closes[0]

    for i in range(1, len(closes)):
        ema[i] = ema[i - 1] + alpha * (closes[i] - ema[i - 1])

    return ema

# ------------------------------------------------
# Strategy (EMA + Candle)
# ------------------------------------------------
def strategy(df, position=None):

    closes = df["close"].values

    ema9 = mt5_style_ema(closes, 9)[-1]
    ema15 = mt5_style_ema(closes, 15)[-1]

    row = df.iloc[-1]

    o = row.open
    c = row.close
    h = row.high
    l = row.low

    body = abs(c - o)
    rng = h - l

    if rng == 0:
        return {"signal": "NONE"}

    upper = h - max(o, c)
    lower = min(o, c) - l

    hammer = lower > body * 2 and upper < body
    bigbull = c > o and body > rng * 0.6
    bigbear = c < o and body > rng * 0.6

    bull = hammer or bigbull
    bear = bigbear

    buy_signal = ema9 > ema15 and bull
    sell_signal = ema9 < ema15 and bear

    if position == "BUY" and sell_signal:
        return {"signal": "CLOSE"}

    if position == "SELL" and buy_signal:
        return {"signal": "CLOSE"}

    if position is None:

        if buy_signal:
            return {"signal": "BUY", "entry": h, "sl": l}

        if sell_signal:
            return {"signal": "SELL", "entry": l, "sl": h}

    return {"signal": "NONE"}
# ------------------------------------------------
# Close position
# ------------------------------------------------
def close_position(pos):

    close_type = (
        mt5.ORDER_TYPE_SELL
        if pos.type == mt5.ORDER_TYPE_BUY
        else mt5.ORDER_TYPE_BUY
    )

    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": pos.symbol,
        "volume": pos.volume,
        "type": close_type,
        "position": pos.ticket,
        "deviation": 50,
        "magic": 10000,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    mt5.order_send(req)


# ------------------------------------------------
# Run strategy
# ------------------------------------------------
def run_strategy():

    df = get_data()

    if df is None or len(df) < 20:
        return

    tick = mt5.symbol_info_tick(SYMBOL)

    spread = tick.ask - tick.bid

    if spread > MAX_SPREAD:
        logging.info("Spread too high")
        return

    positions = mt5.positions_get(symbol=SYMBOL) or []

    position_type = None

    if positions:
        p = positions[0]
        position_type = "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL"

    signal = strategy(df, position_type)

    logging.info("Signal → %s", signal)

    if signal["signal"] == "NONE":
        return

    if signal["signal"] == "CLOSE":

        for pos in positions:
            close_position(pos)

        return

    if positions:
        return

    if signal["signal"] in ["BUY", "SELL"]:

        direction = "Buy" if signal["signal"] == "BUY" else "Sell"

        execute_automated_trade(
            SYMBOL,
            direction,
            sl=round(signal["sl"], 5),
            tp=None
        )


# ------------------------------------------------
# Wait for new candle
# ------------------------------------------------
def wait_for_new_candle():

    now = datetime.now()

    seconds_passed = (now.minute % 5) * 60 + now.second
    sleep_time = 300 - seconds_passed

    logging.info("Sleeping %.2f sec", sleep_time)

    time.sleep(sleep_time)


# ------------------------------------------------
# Main engine
# ------------------------------------------------
def main():

    if not mt5.initialize():
        logging.error("MT5 init failed")
        return

    if not ensure_symbol(SYMBOL):
        return

    logging.info("Engine started")

    try:

        while True:

            wait_for_new_candle()

            run_strategy()

    except KeyboardInterrupt:

        logging.info("Stopped")

    finally:

        mt5.shutdown()


if __name__ == "__main__":
    main()