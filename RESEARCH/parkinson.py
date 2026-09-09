import MetaTrader5 as mt5
import pandas as pd
import numpy as np

# ==============================
# CONFIG
# ==============================

LOOKBACK = 1200
REF_CANDLES = 6
ATR_PERIOD = 14
PARK_PERIOD = 14
EPS = 1e-8

symbols = [
    "EURUSD","USDJPY","XAUUSD","DXYUSD",
    "BA.N","US SP 500","XAGUSD","BTCUSD",
    "GBPUSD","US Oil","ETHUSD","NGAS"
]

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
# ATR
# ==============================

def compute_atr(df,period):

    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())

    tr = pd.concat([tr1,tr2,tr3],axis=1).max(axis=1)

    atr = tr.rolling(period).mean()

    return atr


# ==============================
# PARKINSON VOLATILITY
# ==============================

def compute_parkinson(df,period):

    hl = np.log(df["high"]/df["low"])**2

    park = np.sqrt(
        (1/(4*np.log(2))) * hl.rolling(period).mean()
    )

    return park


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
        df["time"] = pd.to_datetime(df["time"],unit="s")

        df["atr"] = compute_atr(df,ATR_PERIOD)
        df["parkinson"] = compute_parkinson(df,PARK_PERIOD)

        hit1 = 0
        hit2 = 0
        hit3 = 0
        total = 0

        war_values = []
        atr_values = []
        park_values = []

        for i in range(10,len(df)-4):

            ts = df.iloc[i]["time"]

            if is_restricted_time(ts):
                continue

            sizes = []
            j = i+1

            while j < len(df) and len(sizes) < REF_CANDLES:

                row = df.iloc[j]

                if not is_restricted_time(row.time):
                    sizes.append(candle_size(row.open,row.close))

                j += 1

            if len(sizes) < REF_CANDLES:
                continue

            war = variation_weighted_avg(sizes)

            atr = df.iloc[i]["atr"]
            park = df.iloc[i]["parkinson"]

            if pd.isna(atr) or pd.isna(park):
                continue

            war_values.append(war)
            atr_values.append(atr)
            park_values.append(park)

            open_price = df.iloc[i]["open"]

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


        if total == 0:
            continue

        avg_war = np.mean(war_values)
        avg_atr = np.mean(atr_values)
        avg_park = np.mean(park_values)

        size_vs_atr = avg_war / (avg_atr + EPS)
        war_vs_park = avg_war / (avg_park + EPS)

        hit1_rate = hit1/total
        hit2_rate = hit2/total
        hit3_rate = hit3/total

        efficiency = hit3_rate / (size_vs_atr + EPS)

        results.append({
            "symbol":symbol,
            "timeframe":tf_name,
            "tests":total,
            "hit1":hit1_rate,
            "hit2":hit2_rate,
            "hit3":hit3_rate,
            "size_vs_atr":size_vs_atr,
            "efficiency":efficiency,
            "parkinson":avg_park,
            "war_vs_park":war_vs_park
        })


mt5.shutdown()


# ==============================
# RESULTS TABLE
# ==============================

df_results = pd.DataFrame(results)

print("\n===== PROJECTION MATRIX =====\n")
print(df_results)


# ==============================
# +3 HIT MATRIX
# ==============================

pivot_hit = df_results.pivot(index="symbol",columns="timeframe",values="hit3")

print("\n===== +3 CANDLE HIT MATRIX =====\n")
print(pivot_hit)


# ==============================
# EFFICIENCY MATRIX
# ==============================

pivot_eff = df_results.pivot(index="symbol",columns="timeframe",values="efficiency")

print("\n===== EFFICIENCY MATRIX =====\n")
print(pivot_eff)