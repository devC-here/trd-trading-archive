"""
Strategy 54: Renko Brick Trend + 10 SMA + OBV Scanner
Creator Video ID: IkCC_gC-wXQ (+65.4% Annual Return, 62.2% Win Rate, 3.0% Max DD)
Markets: Forex Majors/Crosses & Gold (XAUUSD)
Execution: SIGNAL ONLY (No order placement)
"""

import sys
import os
import argparse
import datetime
import pandas as pd
import numpy as np

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def resolve_symbol(symbol_name):
    if not mt5:
        return symbol_name
    symbols = mt5.symbols_get()
    if not symbols:
        return symbol_name
    names = [s.name for s in symbols]
    if symbol_name in names:
        return symbol_name
    for n in names:
        if symbol_name.upper() in n.upper():
            return n
    return symbol_name


def get_rates(symbol, timeframe, n_bars=300):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


def build_renko(df, brick_size):
    # Construct Renko bricks from standard OHLC prices
    bricks = []
    if len(df) == 0 or brick_size <= 0:
        return pd.DataFrame()

    first_close = df.iloc[0]["close"]
    prev_close = first_close
    trend = 0  # 1 = up, -1 = down
    obv = 0

    for idx, row in df.iterrows():
        c = row["close"]
        v = row.get("tick_volume", 100)

        if trend == 0:
            if c >= prev_close + brick_size:
                trend = 1
                prev_close = prev_close + brick_size
                obv += v
                bricks.append({"time": row["time"], "open": prev_close - brick_size, "close": prev_close, "type": 1, "obv": obv})
            elif c <= prev_close - brick_size:
                trend = -1
                prev_close = prev_close - brick_size
                obv -= v
                bricks.append({"time": row["time"], "open": prev_close + brick_size, "close": prev_close, "type": -1, "obv": obv})
        elif trend == 1:
            while c >= prev_close + brick_size:
                prev_close += brick_size
                obv += v
                bricks.append({"time": row["time"], "open": prev_close - brick_size, "close": prev_close, "type": 1, "obv": obv})
            while c <= prev_close - (2 * brick_size):
                trend = -1
                prev_close -= (2 * brick_size)
                obv -= v
                bricks.append({"time": row["time"], "open": prev_close + brick_size, "close": prev_close, "type": -1, "obv": obv})
        elif trend == -1:
            while c <= prev_close - brick_size:
                prev_close -= brick_size
                obv -= v
                bricks.append({"time": row["time"], "open": prev_close + brick_size, "close": prev_close, "type": -1, "obv": obv})
            while c >= prev_close + (2 * brick_size):
                trend = 1
                prev_close += (2 * brick_size)
                obv += v
                bricks.append({"time": row["time"], "open": prev_close - brick_size, "close": prev_close, "type": 1, "obv": obv})

    return pd.DataFrame(bricks)


def scan_symbol(symbol="EURUSD"):
    res = {
        "strategy": "Renko Brick 10 SMA + OBV",
        "symbol": symbol,
        "signal": "NEUTRAL",
        "timeframe": "Renko (M15 Derived)",
        "entry": None,
        "sl": None,
        "tp": None,
        "rr": 1.5,
        "details": ""
    }

    if not mt5:
        res["details"] = "MetaTrader5 library not installed."
        return res

    sym = resolve_symbol(symbol)
    if not mt5.symbol_select(sym, True):
        res["details"] = f"Failed to select symbol {sym}"
        return res

    df = get_rates(sym, mt5.TIMEFRAME_M15, 300)
    if df is None or len(df) < 100:
        res["details"] = "Insufficient M15 rates for Renko construction"
        return res

    # Determine dynamic brick size using 14-period ATR
    tr1 = df["high"] - df["low"]
    tr2 = abs(df["high"] - df["close"].shift(1))
    tr3 = abs(df["low"] - df["close"].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    brick_size = max(round(atr * 0.75, 5), 0.0005)

    renko_df = build_renko(df, brick_size)
    if len(renko_df) < 25:
        res["details"] = f"Renko bricks building ({len(renko_df)}/25 bricks). Brick Size: {brick_size:.5f}"
        return res

    renko_df["sma10"] = renko_df["close"].rolling(10).mean()

    curr_b = renko_df.iloc[-1]
    prev_b = renko_df.iloc[-2]
    prev2_b = renko_df.iloc[-3]

    curr_price = float(df.iloc[-1]["close"])
    spread = mt5.symbol_info(sym).spread * mt5.symbol_info(sym).point

    # Bullish Alignment:
    # 1. New Green Brick (type == 1)
    # 2. Brick close > 10 SMA
    # 3. 10 SMA sloping upwards (curr_b["sma10"] > prev_b["sma10"])
    # 4. OBV rising (curr_b["obv"] > prev_b["obv"])
    bull_brick = curr_b["type"] == 1
    bull_sma_pos = curr_b["close"] > curr_b["sma10"]
    bull_sma_slope = curr_b["sma10"] > prev_b["sma10"]
    bull_obv = curr_b["obv"] > prev_b["obv"]

    new_bull_flip = (prev_b["type"] == -1 or not (prev_b["close"] > prev_b["sma10"])) and bull_brick

    if bull_brick and bull_sma_pos and bull_sma_slope and bull_obv:
        entry = curr_price
        sl = float(curr_b["close"] - (2 * brick_size) - (spread * 2))
        tp = float(curr_b["close"] + (3 * brick_size))
        risk = entry - sl
        rr = round((tp - entry) / max(risk, 1e-5), 2)
        res.update({
            "signal": "BUY",
            "entry": round(entry, 5),
            "sl": round(sl, 5),
            "tp": round(tp, 5),
            "rr": rr,
            "details": f"Bullish Renko: Green brick above rising 10 SMA ({curr_b['sma10']:.5f}) with rising OBV ({curr_b['obv']}). Brick Size={brick_size:.5f}, SL 2 bricks below, TP 3 bricks above."
        })
        return res

    # Bearish Alignment:
    bear_brick = curr_b["type"] == -1
    bear_sma_pos = curr_b["close"] < curr_b["sma10"]
    bear_sma_slope = curr_b["sma10"] < prev_b["sma10"]
    bear_obv = curr_b["obv"] < prev_b["obv"]

    if bear_brick and bear_sma_pos and bear_sma_slope and bear_obv:
        entry = curr_price
        sl = float(curr_b["close"] + (2 * brick_size) + (spread * 2))
        tp = float(curr_b["close"] - (3 * brick_size))
        risk = sl - entry
        rr = round((entry - tp) / max(risk, 1e-5), 2)
        res.update({
            "signal": "SELL",
            "entry": round(entry, 5),
            "sl": round(sl, 5),
            "tp": round(tp, 5),
            "rr": rr,
            "details": f"Bearish Renko: Red brick below falling 10 SMA ({curr_b['sma10']:.5f}) with falling OBV ({curr_b['obv']}). Brick Size={brick_size:.5f}, SL 2 bricks above, TP 3 bricks below."
        })
        return res

    b_type_str = "Green" if bull_brick else "Red"
    res["details"] = f"Renko Brick={b_type_str} (Close={curr_b['close']:.5f}), SMA10={curr_b['sma10']:.5f} (Sloping {'Up' if bull_sma_slope else 'Down'}), OBV={curr_b['obv']} (Change={curr_b['obv'] - prev_b['obv']}). Alignment pending."
    return res


def main():
    parser = argparse.ArgumentParser(description="Renko Brick 10 SMA + OBV Scanner")
    parser.add_argument("symbol", nargs="?", default="EURUSD", help="Symbol to scan")
    parser.add_argument("--loop", action="store_true", help="Run continuously every 60s")
    args = parser.parse_args()

    if mt5 and not mt5.initialize():
        print(f"[ERROR] MT5 initialization failed: {mt5.last_error()}")
        return

    import time
    while True:
        res = scan_symbol(args.symbol)
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("=" * 60)
        print(f"[{now_str}] Renko Brick Trend Scanner -> {res['symbol']}")
        print(f"Signal: {res['signal']} | Timeframe: {res['timeframe']}")
        if res["signal"] != "NEUTRAL":
            print(f"Entry: {res['entry']} | SL: {res['sl']} | TP: {res['tp']} | RR: {res['rr']}")
        print(f"Details: {res['details']}")
        print("=" * 60)

        if not args.loop:
            break
        time.sleep(60)

    if mt5:
        mt5.shutdown()


if __name__ == "__main__":
    main()
