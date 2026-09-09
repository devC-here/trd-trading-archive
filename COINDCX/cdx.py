from coindcx_plugin import fetch_candles as _fetch_candles

def fetch_candles(symbol, timeframe, start_index, num_candles):
    """
    Self-contained CoinDCX candle fetcher compatible with existing MT5-based prediction code.

    Args:
        symbol (str): CoinDCX symbol, e.g., "B-BTC_USDT" or "B-EUR_USDT"
        timeframe (str): Timeframe in minutes or "1D", e.g., "1", "5", "15", "60", "1D"
        start_index (int): Starting candle index (1 = latest)
        num_candles (int): Number of previous candles to fetch

    Returns:
        pd.DataFrame: DataFrame with columns: 
            ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
    """
    return _fetch_candles(symbol=symbol, timeframe=timeframe, start_index=start_index, num_candles=num_candles)
