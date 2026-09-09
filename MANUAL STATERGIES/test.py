import MetaTrader5 as mt5 
mt5.initialize()
symbol ='EURUSDm'
timeframe = mt5.TIMEFRAME_M15




def get_last_closed_candle(symbol, timeframe):
    """Fetch the last fully closed candle."""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, 4)
    if rates is None or len(rates) < 2:
        return None
    return rates[-1]   # ✅ last closed candle only

print(get_last_closed_candle(symbol , timeframe))