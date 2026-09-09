import MetaTrader5 as mt5
import time
import logging
from sltp import compute_sltp
from placer import update_existing_trade_sltp
# =========================
# CONFIG
# =========================
SYMBOL = "BTCUSD"
TIMEFRAME = mt5.TIMEFRAME_M30


# =========================
# CORE FUNCTION


# MOCK SLTP CALC (REPLACE YOUR OWN)
# =========================


# =========================
# USE CASE / TEST SCRIPT
# =========================
def test_sltp_update():

    logging.basicConfig(level=logging.INFO)

    print("🔌 Initializing MT5...")
    if not mt5.initialize():
        print("❌ MT5 init failed:", mt5.last_error())
        return

    print("✅ Connected to MT5")

    if not mt5.symbol_select(SYMBOL, True):
        print("❌ Failed to select symbol")
        return

    positions = mt5.positions_get(symbol=SYMBOL)

    if not positions:
        print("❌ No open positions found")
        mt5.shutdown()
        return

    print(f"📊 Found {len(positions)} position(s)")

    for pos in positions:
        print("\n--- Testing Position ---")
        print("Ticket:", pos.ticket)
        print("Type:", "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL")

        success = update_existing_trade_sltp(
            pos,
            SYMBOL,
            TIMEFRAME
        )

        print("Result:", "✅ SUCCESS" if success else "❌ FAILED")

    mt5.shutdown()
    print("\n🔌 MT5 shutdown")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    test_sltp_update()