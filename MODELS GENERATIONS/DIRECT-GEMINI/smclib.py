"""
smclib.py — Smart Money Concepts / ICT Library for MetaTrader 5
===============================================================
Single-file library. Drop it in your project folder and import.

Requirements
------------
    pip install MetaTrader5 pandas numpy

MetaTrader 5 terminal must be installed, running, and logged in
before calling any get_* / fetch_* function.

Quick start
-----------
    from smclib import connect, get_eurusd_1h, get_dxy_1h, build_features

    connect()                        # connects to running MT5 terminal

    eu = get_eurusd_1h(300)          # 300 x 1h EURUSD candles
    dx = get_dxy_1h(300)             # 300 x 1h DXY candles

    result = build_features(eu)      # full SMC pipeline

    disconnect()                     # clean shutdown

DXY in MT5
----------
The DXY (US Dollar Index) symbol name depends on your broker.
Common names: "USDX", "DXY", "DOLLAR", "DX", "USDIndex".
The library tries them all automatically, or you can pass yours:
    get_dxy_1h(symbol="USDX")
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Optional


# ============================================================================
#  MT5 CONNECTION MANAGEMENT
# ============================================================================

def connect(
    login:    int | None = None,
    password: str | None = None,
    server:   str | None = None,
    path:     str | None = None,
) -> bool:
    """
    Initialize connection to the MetaTrader 5 terminal.

    If login / password / server are omitted, MT5 connects to whichever
    account is already open in the running terminal — the most common
    workflow for manual + Python hybrid setups.

    Parameters
    ----------
    login    : account number (int). Optional if terminal is already logged in.
    password : account password. Optional.
    server   : broker server name e.g. "ICMarkets-Demo". Optional.
    path     : path to terminal64.exe if MT5 is not in the default location.
               e.g. r"C:\\Program Files\\MetaTrader 5\\terminal64.exe"

    Returns
    -------
    True if connected successfully.

    Raises
    ------
    RuntimeError if MT5 fails to initialize.
    ImportError  if MetaTrader5 package is not installed.

    Example
    -------
        # already logged in terminal — no credentials needed
        connect()

        # explicit credentials
        connect(login=123456, password="mypass", server="ICMarkets-Demo")
    """
    mt5 = _import_mt5()

    kwargs: dict = {}
    if path:     kwargs["path"]     = path
    if login:    kwargs["login"]    = login
    if password: kwargs["password"] = password
    if server:   kwargs["server"]   = server

    if not mt5.initialize(**kwargs):
        err = mt5.last_error()
        raise RuntimeError(
            f"MT5 initialize() failed — error {err}\n"
            "Make sure MetaTrader 5 is installed, running, and logged in."
        )
    return True


def disconnect() -> None:
    """
    Cleanly shut down the MT5 connection.
    Always call this when your script finishes.
    """
    mt5 = _import_mt5()
    mt5.shutdown()


def _import_mt5():
    """Import MetaTrader5 or raise a clear error."""
    try:
        import MetaTrader5 as mt5
        return mt5
    except ImportError:
        raise ImportError(
            "MetaTrader5 package not found.\n"
            "Install it with:  pip install MetaTrader5\n"
            "Note: MetaTrader5 Python package only works on Windows."
        )


# ============================================================================
#  TIMEFRAME MAP
# ============================================================================

# Built lazily on first use to avoid importing MT5 at module load time.
_TF_MAP: dict[str, int] | None = None

def _get_tf_map() -> dict[str, int]:
    global _TF_MAP
    if _TF_MAP is None:
        mt5 = _import_mt5()
        _TF_MAP = {
            "M1":  mt5.TIMEFRAME_M1,
            "M5":  mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1":  mt5.TIMEFRAME_H1,
            "H4":  mt5.TIMEFRAME_H4,
            "D1":  mt5.TIMEFRAME_D1,
            "W1":  mt5.TIMEFRAME_W1,
            "MN":  mt5.TIMEFRAME_MN1,
        }
    return _TF_MAP


# ============================================================================
#  CORE MT5 DATA FETCHER
# ============================================================================

# Common DXY symbol names across brokers — tried in order
_DXY_ALIASES = ["USDX", "DXY", "DOLLAR", "DX", "USDIndex", "DX-Y.NYB", "USDIDX"]


def fetch(symbol: str, timeframe: str, candles: int) -> pd.DataFrame:
    """
    Fetch OHLCV candles from the MT5 terminal.

    Parameters
    ----------
    symbol    : MT5 symbol name e.g. "EURUSD", "USDX", "XAUUSD"
    timeframe : "M1" | "M5" | "M15" | "M30" | "H1" | "H4" | "D1" | "W1" | "MN"
    candles   : number of bars to retrieve (most recent, position 0)

    Returns
    -------
    pd.DataFrame with UTC DatetimeIndex and columns:
        open, high, low, close, volume
    (volume = tick_volume from MT5 — number of ticks per bar)

    Raises
    ------
    RuntimeError if MT5 returns no data.

    Example
    -------
        df = fetch("EURUSD", "H1", 300)
        df = fetch("XAUUSD", "H4", 150)
    """
    mt5    = _import_mt5()
    tf_map = _get_tf_map()

    tf_upper = timeframe.upper()
    if tf_upper not in tf_map:
        raise ValueError(
            f"Unknown timeframe '{timeframe}'. "
            f"Choose from: {list(tf_map.keys())}"
        )

    # ensure symbol is visible in MarketWatch before fetching
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(
            f"Symbol '{symbol}' not found or could not be selected in MT5.\n"
            "Check the symbol name in your broker's Market Watch."
        )

    rates = mt5.copy_rates_from_pos(symbol, tf_map[tf_upper], 0, candles)

    if rates is None or len(rates) == 0:
        err = mt5.last_error()
        raise RuntimeError(
            f"MT5 returned no data for '{symbol}' {timeframe}.\n"
            f"MT5 error code: {err}\n"
            "Possible causes:\n"
            "  • Symbol not subscribed in Market Watch\n"
            "  • Not enough history — scroll chart back in MT5 first\n"
            "  • Terminal not connected to broker"
        )

    df = pd.DataFrame(rates)

    # MT5 returns bar open time as Unix seconds in UTC
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time")
    df.index.name = "time"

    # tick_volume = number of ticks per bar (proxy for volume in forex)
    df = df.rename(columns={"tick_volume": "volume"})
    df = df[["open", "high", "low", "close", "volume"]].copy()

    return df


def fetch_range(
    symbol:    str,
    timeframe: str,
    date_from: datetime,
    date_to:   datetime,
) -> pd.DataFrame:
    """
    Fetch OHLCV candles for a specific date range from MT5.

    Parameters
    ----------
    symbol    : MT5 symbol name
    timeframe : "H1" | "H4" | "D1" etc.
    date_from : start datetime (tz-aware recommended; naive treated as UTC)
    date_to   : end datetime

    Returns
    -------
    pd.DataFrame with UTC DatetimeIndex and columns: open, high, low, close, volume

    Example
    -------
        from datetime import datetime, timezone
        df = fetch_range(
            "EURUSD", "H1",
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 3, 1, tzinfo=timezone.utc),
        )
    """
    mt5    = _import_mt5()
    tf_map = _get_tf_map()

    tf_upper = timeframe.upper()
    if tf_upper not in tf_map:
        raise ValueError(f"Unknown timeframe '{timeframe}'. Choose from: {list(tf_map.keys())}")

    if date_from.tzinfo is None:
        date_from = date_from.replace(tzinfo=timezone.utc)
    if date_to.tzinfo is None:
        date_to = date_to.replace(tzinfo=timezone.utc)

    mt5.symbol_select(symbol, True)
    rates = mt5.copy_rates_range(symbol, tf_map[tf_upper], date_from, date_to)

    if rates is None or len(rates) == 0:
        raise RuntimeError(
            f"MT5 returned no data for '{symbol}' {timeframe} in range "
            f"{date_from.date()} → {date_to.date()}."
        )

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time")
    df = df.rename(columns={"tick_volume": "volume"})
    df = df[["open", "high", "low", "close", "volume"]].copy()
    return df


# ============================================================================
#  NAMED PAIR FETCHERS
# ============================================================================

def get_eurusd_1h(candles: int = 300) -> pd.DataFrame:
    """
    Fetch the last *candles* 1-hour bars for EURUSD from MT5.

    Parameters
    ----------
    candles : number of bars (default 300, ~12 trading days)

    Returns
    -------
    pd.DataFrame — UTC DatetimeIndex, columns: open, high, low, close, volume

    Example
    -------
        connect()
        df = get_eurusd_1h()       # 300 bars
        df = get_eurusd_1h(500)    # 500 bars
    """
    return fetch("EURUSD", "H1", candles)


def get_eurusd_4h(candles: int = 150) -> pd.DataFrame:
    """
    Fetch the last *candles* 4-hour bars for EURUSD from MT5.

    MT5 has native H4 — no resampling.

    Parameters
    ----------
    candles : number of bars (default 150, ~25 trading days)

    Returns
    -------
    pd.DataFrame — UTC DatetimeIndex, columns: open, high, low, close, volume
    """
    return fetch("EURUSD", "H4", candles)


def get_dxy_1h(candles: int = 300, symbol: str | None = None) -> pd.DataFrame:
    """
    Fetch the last *candles* 1-hour bars for the DXY (US Dollar Index) from MT5.

    DXY symbol name varies by broker. If *symbol* is not provided, the
    library auto-detects by trying these names in order:
        USDX, DXY, DOLLAR, DX, USDIndex, DX-Y.NYB, USDIDX

    Parameters
    ----------
    candles : number of bars (default 300)
    symbol  : your broker's DXY symbol name. Pass this to skip auto-detection.

    Returns
    -------
    pd.DataFrame — UTC DatetimeIndex, columns: open, high, low, close, volume

    Example
    -------
        df = get_dxy_1h()                  # auto-detect DXY symbol
        df = get_dxy_1h(symbol="USDX")    # skip detection, use directly
    """
    return _fetch_dxy("H1", candles, symbol)


def get_dxy_4h(candles: int = 150, symbol: str | None = None) -> pd.DataFrame:
    """
    Fetch the last *candles* 4-hour bars for the DXY from MT5.

    Parameters
    ----------
    candles : number of bars (default 150)
    symbol  : your broker's DXY symbol name (auto-detected if None)

    Returns
    -------
    pd.DataFrame — UTC DatetimeIndex, columns: open, high, low, close, volume
    """
    return _fetch_dxy("H4", candles, symbol)


def _fetch_dxy(timeframe: str, candles: int, symbol: str | None) -> pd.DataFrame:
    """Try each known DXY alias until one works in this terminal."""
    mt5 = _import_mt5()
    candidates = [symbol] if symbol else _DXY_ALIASES
    last_err   = ""

    for sym in candidates:
        if mt5.symbol_select(sym, True):
            try:
                return fetch(sym, timeframe, candles)
            except RuntimeError as e:
                last_err = str(e)
                continue

    raise RuntimeError(
        f"Could not find a DXY symbol in your MT5 terminal.\n"
        f"Tried: {candidates}\n"
        f"Last error: {last_err}\n\n"
        "Fix: pass your broker's DXY symbol name explicitly:\n"
        "    get_dxy_1h(symbol='USDX')\n"
        "Find it in MT5 → View → Symbols → search 'dollar' or 'dxy'."
    )


def get_all_data(
    eurusd_1h_candles: int = 300,
    eurusd_4h_candles: int = 150,
    dxy_1h_candles:    int = 300,
    dxy_4h_candles:    int = 150,
    dxy_symbol:        str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Fetch EURUSD and DXY at both 1h and 4h in one call.

    Parameters
    ----------
    eurusd_1h_candles : bars for EURUSD 1h (default 300)
    eurusd_4h_candles : bars for EURUSD 4h (default 150)
    dxy_1h_candles    : bars for DXY 1h    (default 300)
    dxy_4h_candles    : bars for DXY 4h    (default 150)
    dxy_symbol        : your broker's DXY symbol (auto-detected if None)

    Returns
    -------
    dict with keys: "eurusd_1h", "eurusd_4h", "dxy_1h", "dxy_4h"

    Example
    -------
        connect()
        data = get_all_data()

        result = build_features(data["eurusd_1h"])
        choch  = get_choch(result, direction="bear")
        disconnect()
    """
    return {
        "eurusd_1h": get_eurusd_1h(eurusd_1h_candles),
        "eurusd_4h": get_eurusd_4h(eurusd_4h_candles),
        "dxy_1h":    get_dxy_1h(dxy_1h_candles, dxy_symbol),
        "dxy_4h":    get_dxy_4h(dxy_4h_candles, dxy_symbol),
    }


def get_candles(symbol: str, timeframe: str, candles: int) -> pd.DataFrame:
    """
    Generic fetcher for any symbol / timeframe in your MT5 terminal.

    Parameters
    ----------
    symbol    : any MT5 symbol e.g. "GBPUSD", "XAUUSD", "US30", "BTCUSD"
    timeframe : "M1" | "M5" | "M15" | "M30" | "H1" | "H4" | "D1" | "W1" | "MN"
    candles   : number of bars

    Returns
    -------
    pd.DataFrame — UTC DatetimeIndex, columns: open, high, low, close, volume

    Example
    -------
        gbp  = get_candles("GBPUSD", "H1",  300)
        gold = get_candles("XAUUSD", "H4",  150)
        nas  = get_candles("NAS100", "D1",  252)
    """
    return fetch(symbol, timeframe, candles)


# ============================================================================
#  VALIDATION HELPERS
# ============================================================================

def _require_tz(df: pd.DataFrame) -> None:
    if df.index.tz is None:
        raise ValueError(
            "DataFrame index must be timezone-aware. "
            "Data from fetch() is always UTC. "
            "If you loaded data manually: df.index = df.index.tz_localize('UTC')"
        )


def _require_cols(df: pd.DataFrame, *cols: str) -> None:
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


# ============================================================================
#  ATR
# ============================================================================

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Wilder's Average True Range (EWM, adjust=False — matches MT5 / TradingView).

    TR = max( H−L,  |H − Cprev|,  |L − Cprev| )
    ATR = EMA(TR, span=period)

    Parameters
    ----------
    df     : OHLCV DataFrame
    period : smoothing period (default 14)

    Returns
    -------
    pd.Series of ATR values aligned to df.index
    """
    _require_cols(df, "high", "low", "close")
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"]  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ============================================================================
#  PREVIOUS DAY / WEEK / MONTH HIGH & LOW
# ============================================================================

def _resample_hl_shift(df: pd.DataFrame, freq: str, prefix: str) -> pd.DataFrame:
    hi = df["high"].resample(freq).max().shift(1)
    lo = df["low"].resample(freq).min().shift(1)
    df[f"{prefix}H"] = hi.reindex(df.index, method="ffill")
    df[f"{prefix}L"] = lo.reindex(df.index, method="ffill")
    return df


def previous_day_hl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add PDH (Previous Day High) and PDL (Previous Day Low).

    Uses resample("D") — timezone-aware and DST-safe.
    Each bar sees yesterday's H/L, forward-filled across the whole session.

    Parameters
    ----------
    df : OHLCV with tz-aware DatetimeIndex (UTC from MT5)

    Returns
    -------
    df copy with added columns: PDH, PDL
    """
    _require_tz(df)
    return _resample_hl_shift(df.copy(), "D", "PD")


def previous_week_hl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add PWH (Previous Week High) and PWL (Previous Week Low).
    Week boundary = Monday 00:00 UTC.

    Returns
    -------
    df copy with added columns: PWH, PWL
    """
    _require_tz(df)
    return _resample_hl_shift(df.copy(), "W-MON", "PW")


def previous_month_hl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add PMH (Previous Month High) and PML (Previous Month Low).

    Returns
    -------
    df copy with added columns: PMH, PML
    """
    _require_tz(df)
    return _resample_hl_shift(df.copy(), "MS", "PM")


def all_reference_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute PDH/PDL, PWH/PWL, and PMH/PML in one call.

    Returns
    -------
    df copy with added columns: PDH, PDL, PWH, PWL, PMH, PML
    """
    df = previous_day_hl(df)
    df = previous_week_hl(df)
    df = previous_month_hl(df)
    return df


# ============================================================================
#  SWING HIGH / LOW DETECTION
# ============================================================================

def detect_swings(
    df: pd.DataFrame,
    lookback:       int   = 3,
    atr_period:     int   = 14,
    atr_multiplier: float = 0.5,
) -> pd.DataFrame:
    """
    ATR-adaptive fractal swing high / low detection.

    A bar at index i is a swing HIGH when:
      • df['high'][i] is the maximum in window [i-lookback, i+lookback]
      • Protrusion above window minimum > atr_multiplier × ATR
        (this rejects noise-level wicks in compressed ranges)

    Symmetric logic for swing LOW.

    Zero look-ahead: a pivot is only confirmed once the full right-side
    window of *lookback* bars has closed.

    Uses numpy .values arrays — ~5x faster than pandas .iloc in the loop.

    Parameters
    ----------
    df             : OHLCV with tz-aware DatetimeIndex
    lookback       : bars on each side of the pivot candidate (default 3)
    atr_period     : ATR smoothing window (default 14)
    atr_multiplier : minimum protrusion as multiples of ATR (default 0.5)

    Returns
    -------
    df copy with added columns:
        swing_high (bool)
        swing_low  (bool)
        atr        (float)
    """
    _require_cols(df, "high", "low", "close")
    df = df.copy()

    _atr      = atr(df, period=atr_period)
    df["atr"] = _atr
    min_move  = _atr.values * atr_multiplier

    highs = df["high"].values
    lows  = df["low"].values
    n     = len(df)

    sh = np.zeros(n, dtype=bool)
    sl = np.zeros(n, dtype=bool)

    for i in range(lookback, n - lookback):
        w_hi = highs[i - lookback : i + lookback + 1]
        w_lo = lows [i - lookback : i + lookback + 1]

        if highs[i] == w_hi.max() and (highs[i] - w_hi.min()) > min_move[i]:
            sh[i] = True
        if lows[i]  == w_lo.min() and (w_lo.max() - lows[i])  > min_move[i]:
            sl[i] = True

    df["swing_high"] = sh
    df["swing_low"]  = sl
    return df


def get_swing_pivots(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only bars that are a confirmed swing high or swing low.
    Requires detect_swings() first.

    Returns
    -------
    Filtered df with added columns: pivot_type ("SH"|"SL"), pivot_price
    """
    _require_cols(df, "swing_high", "swing_low")
    mask = df["swing_high"] | df["swing_low"]
    out  = df[mask].copy()
    out["pivot_type"]  = np.where(out["swing_high"], "SH", "SL")
    out["pivot_price"] = np.where(out["swing_high"], out["high"], out["low"])
    return out


# ============================================================================
#  MARKET STRUCTURE  (HH / HL / LH / LL + TREND STATE)
# ============================================================================

def compute_market_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label each confirmed swing pivot as HH, HL, LH, or LL.
    Derive a rolling trend state from the last two confirmed pivot pairs.

    Structure logic
    ---------------
    Swing HIGH vs last confirmed swing HIGH:
        current > last  →  HH   (bullish continuation)
        current < last  →  LH   (bearish sign)

    Swing LOW vs last confirmed swing LOW:
        current > last  →  HL   (bullish sign)
        current < last  →  LL   (bearish continuation)

    Reference level updates AFTER label is written — no look-ahead.

    Trend state pairs
    -----------------
    HH + HL  →  "bullish"
    LH + LL  →  "bearish"
    HH + LL or LH + HL  →  "ranging"
    None yet  →  None

    Requires detect_swings() first.

    Returns
    -------
    df copy with added columns: structure, trend_state
    """
    _require_cols(df, "swing_high", "swing_low", "high", "low")
    df = df.copy()
    n  = len(df)

    structure = [None] * n
    last_sh: float | None = None
    last_sl: float | None = None

    for i in range(n):
        if df["swing_high"].iloc[i]:
            cur = df["high"].iloc[i]
            if last_sh is not None:
                structure[i] = "HH" if cur > last_sh else "LH"
            last_sh = cur

        elif df["swing_low"].iloc[i]:
            cur = df["low"].iloc[i]
            if last_sl is not None:
                structure[i] = "HL" if cur > last_sl else "LL"
            last_sl = cur

    df["structure"] = structure

    states: list[str | None] = [None] * n
    last_hl: str | None = None
    last_ll: str | None = None
    cur_state: str | None = None

    for i, s in enumerate(structure):
        if s in ("HH", "LH"): last_hl = s
        elif s in ("HL", "LL"): last_ll = s

        if last_hl and last_ll:
            if   last_hl == "HH" and last_ll == "HL": cur_state = "bullish"
            elif last_hl == "LH" and last_ll == "LL": cur_state = "bearish"
            else:                                       cur_state = "ranging"

        states[i] = cur_state

    df["trend_state"] = states
    return df


def get_structure_pivots(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only bars with a confirmed structure label.
    Requires compute_market_structure() first.

    Returns
    -------
    Filtered df with added column: pivot_price
    """
    _require_cols(df, "structure")
    out = df[df["structure"].notna()].copy()
    out["pivot_price"] = np.where(
        out["structure"].isin(["HH", "LH"]), out["high"], out["low"]
    )
    return out


# ============================================================================
#  BOS / CHOCH
# ============================================================================

def detect_bos_choch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect Break of Structure (BOS) and Change of Character (CHOCH).

    Definitions
    -----------
    BOS   — price closes beyond a prior swing level IN trend direction.
            Confirms the trend is continuing. Trade with it.

    CHOCH — price closes beyond a prior swing level AGAINST the trend.
            First warning of possible reversal. Look for entries in new direction.

    Signals
    -------
    Bullish trend:
        close > last confirmed swing HIGH  →  BOS_BULL    (continuation)
        close < last confirmed swing LOW   →  CHOCH_BEAR  (reversal warning)

    Bearish trend:
        close < last confirmed swing LOW   →  BOS_BEAR    (continuation)
        close > last confirmed swing HIGH  →  CHOCH_BULL  (reversal warning)

    Implementation notes
    --------------------
    • Candle CLOSE used (not wick) — filters liquidity sweeps that reverse
      immediately without confirming a real break.
    • Each level consumed after triggering (set to None) — no duplicates.
    • Trend flips immediately on CHOCH — next signals use the new direction.

    Requires detect_swings() and compute_market_structure() first.

    Returns
    -------
    df copy with added columns:
        signal       : "BOS_BULL" | "BOS_BEAR" | "CHOCH_BULL" | "CHOCH_BEAR" | None
        signal_level : float — the price level that was broken
        signal_type  : "BOS" | "CHOCH" | None
        signal_dir   : "bull" | "bear" | None
    """
    _require_cols(df, "swing_high", "swing_low", "structure", "high", "low", "close")
    df = df.copy()
    n  = len(df)

    signals      = [None]          * n
    signal_level = [float("nan")] * n

    trend:  str | None   = None
    ref_sh: float | None = None
    ref_sl: float | None = None

    highs   = df["high"].values
    lows    = df["low"].values
    closes  = df["close"].values
    structs = df["structure"].values
    s_highs = df["swing_high"].values
    s_lows  = df["swing_low"].values

    _BOS   = {"BOS_BULL",   "BOS_BEAR"}
    _CHOCH = {"CHOCH_BULL", "CHOCH_BEAR"}

    for i in range(n):
        s = structs[i]
        c = closes[i]

        # update reference levels when new pivots form
        if s_highs[i]:
            ref_sh = highs[i]
            if s == "HH":   trend = "bullish"
            elif s == "LH": trend = "bearish"

        if s_lows[i]:
            ref_sl = lows[i]
            if s == "HL":   trend = "bullish"
            elif s == "LL": trend = "bearish"

        # close-based break detection
        if trend == "bullish":
            if ref_sh is not None and c > ref_sh:
                signals[i]      = "BOS_BULL"
                signal_level[i] = ref_sh
                ref_sh          = None
            elif ref_sl is not None and c < ref_sl:
                signals[i]      = "CHOCH_BEAR"
                signal_level[i] = ref_sl
                trend           = "bearish"
                ref_sl          = None

        elif trend == "bearish":
            if ref_sl is not None and c < ref_sl:
                signals[i]      = "BOS_BEAR"
                signal_level[i] = ref_sl
                ref_sl          = None
            elif ref_sh is not None and c > ref_sh:
                signals[i]      = "CHOCH_BULL"
                signal_level[i] = ref_sh
                trend           = "bullish"
                ref_sh          = None

    df["signal"]       = signals
    df["signal_level"] = signal_level

    sig = pd.Series(signals, index=df.index)
    df["signal_type"] = sig.map(
        lambda x: "BOS" if x in _BOS else ("CHOCH" if x in _CHOCH else None)
    )
    df["signal_dir"] = sig.map(
        lambda x: "bull" if x in ("BOS_BULL",  "CHOCH_BULL") else
                 ("bear" if x in ("BOS_BEAR", "CHOCH_BEAR") else None)
    )
    return df


def get_bos(df: pd.DataFrame, direction: str | None = None) -> pd.DataFrame:
    """
    Filter bars where a BOS occurred.

    Parameters
    ----------
    direction : "bull" | "bear" | None (returns both)
    """
    _require_cols(df, "signal_type", "signal_dir")
    mask = df["signal_type"] == "BOS"
    if direction:
        mask &= df["signal_dir"] == direction
    return df[mask].copy()


def get_choch(df: pd.DataFrame, direction: str | None = None) -> pd.DataFrame:
    """
    Filter bars where a CHOCH occurred.

    Parameters
    ----------
    direction : "bull" | "bear" | None (returns both)
    """
    _require_cols(df, "signal_type", "signal_dir")
    mask = df["signal_type"] == "CHOCH"
    if direction:
        mask &= df["signal_dir"] == direction
    return df[mask].copy()


# ============================================================================
#  SESSION RANGES  (Asian / London / New York)
# ============================================================================

_SESSION_WINDOWS: dict[str, tuple[int, int]] = {
    "asian":  (0,  6),
    "london": (7, 16),
    "ny":     (13, 22),
}
_SESSION_PREFIX: dict[str, str] = {
    "asian":  "ASIAN",
    "london": "LONDON",
    "ny":     "NY",
}


def session_range(
    df: pd.DataFrame,
    session:    str        = "asian",
    start_hour: int | None = None,
    end_hour:   int | None = None,
) -> pd.DataFrame:
    """
    Compute high, low, mid, range, and bias for a named trading session.

    Session windows (UTC defaults):
        asian   → 00:00–06:00
        london  → 07:00–16:00
        ny      → 13:00–22:00

    MT5 broker offset note: if your broker runs on a non-UTC clock
    (e.g. UTC+2 / UTC+3 EET common for European brokers), adjust
    start_hour / end_hour accordingly. For UTC+3: Asian = start=21, end=3
    (previous day's 21:00 to 03:00 UTC+3).

    Added columns (PREFIX = ASIAN | LONDON | NY | CUSTOM):
        {PREFIX}_HIGH   — session high for that calendar day
        {PREFIX}_LOW    — session low
        {PREFIX}_MID    — (HIGH + LOW) / 2  — institutional equilibrium
        {PREFIX}_RANGE  — HIGH − LOW in price units
        {PREFIX}_BULL   — bool: session closed above mid (bullish bias)
        {PREFIX}_BEAR   — bool: session closed below mid (bearish bias)

    Values forward-filled within the day — every bar sees current session levels.

    Parameters
    ----------
    df         : OHLCV with tz-aware DatetimeIndex (UTC from MT5)
    session    : "asian" | "london" | "ny" | "custom"
    start_hour : UTC hour (inclusive) — overrides preset
    end_hour   : UTC hour (exclusive) — overrides preset

    Returns
    -------
    df copy with session columns appended
    """
    _require_tz(df)
    df = df.copy()

    if session == "custom":
        if start_hour is None or end_hour is None:
            raise ValueError("Provide start_hour and end_hour for custom session.")
        prefix = "CUSTOM"
    else:
        if session not in _SESSION_WINDOWS:
            raise ValueError(
                f"Unknown session '{session}'. "
                f"Choose from {list(_SESSION_WINDOWS)} or 'custom'."
            )
        default_start, default_end = _SESSION_WINDOWS[session]
        start_hour = start_hour if start_hour is not None else default_start
        end_hour   = end_hour   if end_hour   is not None else default_end
        prefix     = _SESSION_PREFIX[session]

    utc_df    = df.tz_convert("UTC")
    hours     = utc_df.index.hour
    in_window = (hours >= start_hour) & (hours < end_hour)
    sess_df   = utc_df[in_window]

    if sess_df.empty:
        for col in ("HIGH", "LOW", "MID", "RANGE", "BULL", "BEAR"):
            df[f"{prefix}_{col}"] = float("nan")
        return df

    daily = (
        sess_df["high"].resample("D").max().rename(f"{prefix}_HIGH").to_frame()
        .join(sess_df["low"].resample("D").min().rename(f"{prefix}_LOW"))
    )
    daily[f"{prefix}_MID"]   = (daily[f"{prefix}_HIGH"] + daily[f"{prefix}_LOW"]) / 2
    daily[f"{prefix}_RANGE"] = daily[f"{prefix}_HIGH"] - daily[f"{prefix}_LOW"]

    last_close = sess_df["close"].resample("D").last()
    daily[f"{prefix}_BULL"] = last_close > daily[f"{prefix}_MID"]
    daily[f"{prefix}_BEAR"] = last_close < daily[f"{prefix}_MID"]

    df = df.join(daily.reindex(df.index, method="ffill"))
    return df


def asian_range(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Compute Asian session (00:00–06:00 UTC) range. See session_range()."""
    return session_range(df, session="asian", **kwargs)


def london_range(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Compute London session (07:00–16:00 UTC) range. See session_range()."""
    return session_range(df, session="london", **kwargs)


def ny_range(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Compute New York session (13:00–22:00 UTC) range. See session_range()."""
    return session_range(df, session="ny", **kwargs)


def all_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Asian, London, and NY session ranges in one call."""
    df = asian_range(df)
    df = london_range(df)
    df = ny_range(df)
    return df


# ============================================================================
#  FULL PIPELINE
# ============================================================================

def build_features(
    df: pd.DataFrame,
    *,
    swing_lookback: int   = 3,
    atr_period:     int   = 14,
    atr_multiplier: float = 0.5,
    sessions: bool = True,
    levels:   bool = True,
) -> pd.DataFrame:
    """
    Full SMC / ICT feature pipeline in one call.

    Order of operations:
        1. Reference levels   (PDH/PDL, PWH/PWL, PMH/PML)
        2. Swing detection    (ATR-adaptive fractal pivots)
        3. Market structure   (HH/HL/LH/LL + trend_state)
        4. BOS / CHOCH        (close-based break detection)
        5. Session ranges     (Asian / London / NY)

    Parameters
    ----------
    df             : OHLCV DataFrame from fetch() / get_eurusd_1h() etc.
                     Required columns: open, high, low, close, volume
                     Index must be tz-aware (MT5 data is always UTC).
    swing_lookback : bars on each side for fractal pivot (default 3)
    atr_period     : ATR smoothing period (default 14)
    atr_multiplier : minimum pivot protrusion as multiples of ATR (default 0.5)
    sessions       : compute Asian / London / NY ranges (default True)
    levels         : compute PDH/PDL, PWH/PWL, PMH/PML (default True)

    Returns
    -------
    df copy with all SMC columns:

        PDH, PDL, PWH, PWL, PMH, PML           (levels=True)
        atr, swing_high, swing_low              (always)
        structure, trend_state                  (always)
        signal, signal_level, signal_type,
        signal_dir                              (always)
        ASIAN_HIGH/LOW/MID/RANGE/BULL/BEAR      (sessions=True)
        LONDON_HIGH/LOW/MID/RANGE/BULL/BEAR     (sessions=True)
        NY_HIGH/LOW/MID/RANGE/BULL/BEAR         (sessions=True)

    Example
    -------
        from smclib import connect, get_eurusd_1h, build_features, get_choch

        connect()
        eu     = get_eurusd_1h(300)
        result = build_features(eu)

        # all bearish reversal signals
        bears = get_choch(result, direction="bear")
        print(bears[["close", "signal", "signal_level", "trend_state"]])

        disconnect()
    """
    required = {"open", "high", "low", "close", "volume"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")
    _require_tz(df)

    df = df.copy()

    if levels:
        df = all_reference_levels(df)

    df = detect_swings(
        df,
        lookback       = swing_lookback,
        atr_period     = atr_period,
        atr_multiplier = atr_multiplier,
    )
    df = compute_market_structure(df)
    df = detect_bos_choch(df)

    if sessions:
        df = all_sessions(df)

    return df
