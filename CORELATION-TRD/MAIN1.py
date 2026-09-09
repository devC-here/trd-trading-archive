import MetaTrader5 as mt5
import time
import json
from datetime import datetime

SYMBOLS = ["EURUSD", "USDJPY", "XAUUSD"]

# state storage
candle_open = {s: None for s in SYMBOLS}
current_minute = {s: None for s in SYMBOLS}
last_tick_time = {s: 0 for s in SYMBOLS}
current_price = {s: 0.0 for s in SYMBOLS}
pressure = {s: 0.0 for s in SYMBOLS}
direction = {s: "FLAT" for s in SYMBOLS}

LOG_FILE = "tick_pressure_log.jsonl"


# connect MT5
if not mt5.initialize():
    print("MT5 initialization failed")
    quit()

for s in SYMBOLS:
    mt5.symbol_select(s, True)

print("Continuous tick logging with BULL/BEAR started...")


def get_direction(diff):

    if diff > 0:
        return "BULL"
    elif diff < 0:
        return "BEAR"
    else:
        return "FLAT"


def update_symbol(symbol):

    tick = mt5.symbol_info_tick(symbol)

    if tick is None:
        return False

    if tick.time_msc == last_tick_time[symbol]:
        return False

    last_tick_time[symbol] = tick.time_msc

    price = tick.bid
    tick_time = datetime.fromtimestamp(tick.time)
    minute = tick_time.minute

    # detect new candle
    if current_minute[symbol] != minute:
        current_minute[symbol] = minute
        candle_open[symbol] = price

    current_price[symbol] = price
    pressure[symbol] = price - candle_open[symbol]
    direction[symbol] = get_direction(pressure[symbol])

    return True


def save_tick():

    record = {
        "timestamp": datetime.utcnow().isoformat(),

        "EURUSD": {
            "price": current_price["EURUSD"],
            "pressure": pressure["EURUSD"],
            "direction": direction["EURUSD"]
        },

        "USDJPY": {
            "price": current_price["USDJPY"],
            "pressure": pressure["USDJPY"],
            "direction": direction["USDJPY"]
        },

        "XAUUSD": {
            "price": current_price["XAUUSD"],
            "pressure": pressure["XAUUSD"],
            "direction": direction["XAUUSD"]
        }
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


while True:

    updated = False

    for s in SYMBOLS:
        if update_symbol(s):
            updated = True

    if updated:

        # print with BULL/BEAR
        print(
            f"EURUSD {current_price['EURUSD']:.5f} ({pressure['EURUSD']:+.5f}) {direction['EURUSD']} | "
            f"USDJPY {current_price['USDJPY']:.3f} ({pressure['USDJPY']:+.3f}) {direction['USDJPY']} | "
            f"XAUUSD {current_price['XAUUSD']:.2f} ({pressure['XAUUSD']:+.2f}) {direction['XAUUSD']}"
        )

        save_tick()

    time.sleep(0.001)