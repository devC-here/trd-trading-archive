import MetaTrader5 as mt5
import pandas as pd

mt5.initialize()
symbol = 'XAUUSD'

digits = mt5.symbol_info(symbol).digits
print(digits)