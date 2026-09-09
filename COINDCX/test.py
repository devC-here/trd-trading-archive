import MetaTrader5 as mt5

symbol = "EURUSD"   # change to your symbol

def test_filling_type(filling):
    print(f"\n=== Testing Filling Type: {filling} ===")
    
    if not mt5.initialize():
        print("Initialization failed:", mt5.last_error())
        return
    
    # Ensure symbol is visible
    if not mt5.symbol_select(symbol, True):
        print("Symbol select failed:", mt5.last_error())
        return
    
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print("Cannot get tick:", mt5.last_error())
        return
    
    price = tick.ask

    # Send a *validation only* order (volume = 0 prevents execution)
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": 0.0,        # <- ZERO volume: safe test
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "deviation": 200,
        "type_filling": filling,
    }

    result = mt5.order_send(request)

    if result is None:
        print("order_send failed:", mt5.last_error())
    else:
        print("retcode:", result.retcode)
        print("comment:", result.comment)
        print("retcode_external:", result.retcode_external)

    mt5.shutdown()


# TEST ALL FILLINGS
test_filling_type(mt5.ORDER_FILLING_FOK)
test_filling_type(mt5.ORDER_FILLING_IOC)
test_filling_type(mt5.ORDER_FILLING_RETURN)
