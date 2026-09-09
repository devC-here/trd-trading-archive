import MetaTrader5 as mt5
import pandas as pd
import numpy as np

# ==============================
# CONFIG
# ==============================

LOOKBACK = 1200
REF_CANDLES = 6
EPS = 1e-8

symbols = ["EURUSD","USDJPY","XAUUSD","DXYUSD", "BA.N", "US SP 500" , "XAGUSD" , "BTCUSD" ,"GBPUSD", "US Oil" , "ETHUSD" , "NGAS"]

timeframes = {
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1
}

# ==============================
# SESSION RESTRICTION
# ==============================

def is_restricted_time(ts):

    minutes = ts.hour*60 + ts.minute

    asian_open = 0*60
    london_open = 8*60
    ny_open = 13*60

    if asian_open <= minutes < asian_open+60:
        return True

    if london_open <= minutes < london_open+60:
        return True

    if ny_open <= minutes < ny_open+60:
        return True

    return False


# ==============================
# UTILS
# ==============================

def candle_size(o,c):
    return abs(o-c)


def variation_weighted_avg(sizes):

    mean = np.mean(sizes)

    weights = [1/(abs(s-mean)+EPS) for s in sizes]

    num = np.sum(np.array(sizes)*np.array(weights))
    den = np.sum(weights)

    return num/den


# ==============================
# MT5 INIT
# ==============================

if not mt5.initialize():
    raise RuntimeError("MT5 init failed")


results = []

# ==============================
# MAIN TEST LOOP
# ==============================

for symbol in symbols:

    if not mt5.symbol_select(symbol,True):
        print("Symbol failed:",symbol)
        continue

    for tf_name, tf in timeframes.items():

        rates = mt5.copy_rates_from_pos(symbol,tf,0,LOOKBACK)

        if rates is None:
            print("No data:",symbol,tf_name)
            continue

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'],unit='s')

        hit1 = 0
        hit2 = 0
        hit3 = 0
        total = 0

        for i in range(10,len(df)-4):

            ts = df.iloc[i]['time']

            if is_restricted_time(ts):
                continue

            sizes = []
            j = i+1

            while j < len(df) and len(sizes) < REF_CANDLES:

                row = df.iloc[j]

                if not is_restricted_time(row.time):
                    sizes.append(candle_size(row.open,row.close))

                j+=1

            if len(sizes) < REF_CANDLES:
                continue

            war = variation_weighted_avg(sizes)

            open_price = df.iloc[i]['open']

            upper = open_price + war
            lower = open_price - war

            c1 = df.iloc[i-1]
            c2 = df.iloc[i-2]
            c3 = df.iloc[i-3]

            total += 1

            if c1.high >= upper or c1.low <= lower:
                hit1 += 1

            if (
                c1.high >= upper or c1.low <= lower or
                c2.high >= upper or c2.low <= lower
            ):
                hit2 += 1

            if (
                c1.high >= upper or c1.low <= lower or
                c2.high >= upper or c2.low <= lower or
                c3.high >= upper or c3.low <= lower
            ):
                hit3 += 1

        results.append({
            "symbol":symbol,
            "timeframe":tf_name,
            "tests":total,
            "hit1":hit1/total,
            "hit2":hit2/total,
            "hit3":hit3/total
        })


mt5.shutdown()


# ==============================
# RESULTS TABLE
# ==============================

df_results = pd.DataFrame(results)

print("\n===== PROJECTION MATRIX =====\n")
print(df_results)


# ==============================
# PIVOT MATRIX (OPTIONAL)
# ==============================

pivot = df_results.pivot(index="symbol",columns="timeframe",values="hit3")

print("\n===== +3 CANDLE HIT MATRIX =====\n")
print(pivot)