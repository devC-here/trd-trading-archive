import MetaTrader5 as mt5
import logging
import time
from datetime import datetime
import pandas as pd

# ==================================
# STRATEGY IMPORT (PLUG HERE)
# ==================================
from strategy import ema_candle_strategy

# ==================================
# UTILITIES
# ==================================
from placer import execute_automated_trade


# ==================================
# CONFIG
# ==================================
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M5
BARS = 200

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


# ------------------------------------------------------------
# Ensure symbol available
# ------------------------------------------------------------
def ensure_symbol(symbol):

    info = mt5.symbol_info(symbol)

    if info is None:
        logging.error("Symbol not found: %s", symbol)
        return False

    if not info.visible:
        mt5.symbol_select(symbol, True)

    return True


# ------------------------------------------------------------
# Fetch market data
# ------------------------------------------------------------
def get_data(symbol, timeframe, bars=BARS):

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)

    if rates is None:
        logging.error("Failed to fetch data")
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    return df


# ------------------------------------------------------------
# Close position
# ------------------------------------------------------------
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
        "comment": "Strategy close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    result = mt5.order_send(req)

    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        logging.info("Closed position ticket=%d", pos.ticket)
        return True

    logging.warning("Close failed ticket=%d", pos.ticket)
    return False


# ------------------------------------------------------------
# Run strategy
# ------------------------------------------------------------
def run_strategy():

    df = get_data(SYMBOL, TIMEFRAME)

    if df is None or len(df) < 20:
        return

    positions = mt5.positions_get(symbol=SYMBOL) or []

    position_type = None

    if positions:
        p = positions[0]
        position_type = "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL"

    signal = ema_candle_strategy(df, position_type)

    logging.info("Strategy output: %s", signal)

    if signal["signal"] == "NONE":
        return

    # CLOSE POSITION
    if signal["signal"] == "CLOSE":

        for pos in positions:
            close_position(pos)

        return

    # OPEN TRADE
    if signal["signal"] in ["BUY", "SELL"]:

        direction = "Buy" if signal["signal"] == "BUY" else "Sell"

        execute_automated_trade(
            SYMBOL,
            direction,
            sl=round(signal["sl"], 5),
            tp=None
        )


# ------------------------------------------------------------
# Scheduler (sync with new M5 candle)
# ------------------------------------------------------------
def wait_for_new_candle():

    now = datetime.now()

    seconds_passed = (now.minute % 5) * 60 + now.second
    sleep_time = 300 - seconds_passed

    logging.info("Sleeping %.2f seconds until next candle", sleep_time)

    time.sleep(sleep_time)


# ------------------------------------------------------------
# Main engine
# ------------------------------------------------------------
def main():

    if not mt5.initialize():
        logging.error("MT5 initialization failed")
        return

    if not ensure_symbol(SYMBOL):
        return

    logging.info("Strategy engine started.")

    try:

        while True:

            wait_for_new_candle()

            run_strategy()

    except KeyboardInterrupt:

        logging.info("Stopped by user.")

    finally:

        mt5.shutdown()
        logging.info("MT5 shutdown.")


if __name__ == "__main__":
    main()