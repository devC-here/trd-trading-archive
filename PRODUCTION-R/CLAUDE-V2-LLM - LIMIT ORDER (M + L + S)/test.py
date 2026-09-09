import MetaTrader5 as mt5
import time
from sltp import compute_sltp
SYMBOL = "BTCUSD"

print("🔌 Initializing MT5...")
mt5.initialize()
positions = mt5.positions_get(symbol=SYMBOL)
print ( compute_sltp(SYMBOL , timeframe= mt5.TIMEFRAME_H1 ) )
mt5.initialize()

if positions:
    pos = positions[0]

    tick = mt5.symbol_info_tick(pos.symbol)
    info = mt5.symbol_info(pos.symbol)

    price = tick.ask if pos.type == mt5.ORDER_TYPE_BUY else tick.bid

    shift = 100  # 🔥 +/- 10

    if pos.type == mt5.ORDER_TYPE_BUY:
        sl = price - shift
        tp = price + shift
    else:
        sl = price + shift
        tp = price - shift

    sl = float(round(sl, info.digits))
    tp = float(round(tp, info.digits))

    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": pos.symbol,
        "position": pos.ticket,
        "sl": sl,
        "tp": tp,
    }

    print("Sending:", request)

    time.sleep(0.5)  # 🔥 important for MT5

    result = mt5.order_send(request)

    print("Result:", result)

else:
    print("No positions found")


mt5.shutdown()