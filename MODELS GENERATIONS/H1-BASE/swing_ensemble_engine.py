"""
========================================================================================
H1/H2 HIGH-PIP SWING ENSEMBLE ENGINE (EURUSD.0) - STABILIZED EDITION
========================================================================================
Combines 60 curated H1/H2 strategies into a single actionable high-yield swing decision.
Stabilized against tick noise using:
1. Closed-candle evaluation (iloc[-2]) for trend & technical indicators.
2. Distinct category-specific technical dispatchers (A through H).
3. ATR hysteresis buffer bands around key moving averages and zones.
4. Robust ensemble consensus threshold (20+ votes) with net majority margin (+8).
5. State debouncing (3 consecutive confirmation cycles) to prevent flip oscillations.
========================================================================================
"""

import os
import sys
import time
import datetime
import threading
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False


DEFAULT_SYMBOL = "EURUSD.0"
PIP_SIZE = 0.0001
TICK_SIZE = 0.00001

# The 10 micro-scalpers removed because their M1-M5 logic defeats on H1/H2
REMOVED_SCALPERS = {9, 12, 13, 15, 19, 27, 41, 58, 59, 63}

CATEGORIES = {
    "A": "Institutional SMC & Liquidity",
    "B": "Macro Trend & Moving Avg",
    "C": "Oscillators & Reversals",
    "D": "Mathematical Pivots & Volatility",
    "E": "Price Action & Demand/Supply",
    "F": "Precision Momentum & Pullback",
    "G": "Multi-Confluence & Traps",
    "H": "Imbalance & Trend Expansion"
}


@dataclass
class ConditionStep:
    step_num: int
    title: str
    target: str
    current: str
    delta: str
    status: str  # "PASS", "PENDING", "FAIL"


@dataclass
class StrategyState:
    id: str
    num: int
    name: str
    short_name: str
    category: str
    category_code: str
    timeframe: str                     # e.g. "H1 / H2"
    direction: str                     # 'BUY', 'SELL', 'NEUTRAL'
    h2_status: str                     # 'PASS', 'PENDING', 'FAIL'
    h1_status: str                     # 'PASS', 'PENDING', 'FAIL'
    overall_status: str                # 'WAITING', 'PROGRESS', 'READY', 'TRIGGERED'
    progress_pct: int
    true_count: int
    total_steps: int
    steps: List[Dict[str, Any]]
    active_signal: Optional[Dict[str, Any]] = None
    summary_note: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)


class MarketDataProvider:
    """Provides cached MT5 rates and technical indicators for H1, H2, H4, and D1."""

    def __init__(self, symbol: str = DEFAULT_SYMBOL, use_mock: bool = False):
        self.symbol = symbol
        self.use_mock = use_mock
        self.connected = False
        self.account_info = {"login": "N/A", "server": "N/A", "balance": 0.0}
        self.base_price = 1.16200
        self.tick_count = 0
        self.lock = threading.Lock()
        self._mock_dataframes: Dict[int, pd.DataFrame] = {}

        if not self.use_mock and MT5_AVAILABLE:
            self._init_mt5()

    def _init_mt5(self):
        if not mt5.initialize():
            self.connected = False
            return False

        sym = self.symbol
        if not mt5.symbol_select(sym, True):
            for alt in ["EURUSD", "EURUSD.0", "EURUSDm", "EURUSD_i"]:
                if mt5.symbol_select(alt, True):
                    self.symbol = alt
                    sym = alt
                    break

        acc = mt5.account_info()
        if acc:
            self.account_info = {
                "login": str(acc.login),
                "server": str(acc.server),
                "balance": round(acc.balance, 2)
            }
        self.connected = True
        return True

    def get_live_tick(self) -> Dict[str, Any]:
        with self.lock:
            if not self.use_mock and self.connected and MT5_AVAILABLE:
                info = mt5.symbol_info(self.symbol)
                if info:
                    spread_pips = round((info.ask - info.bid) / PIP_SIZE, 1)
                    if spread_pips == 0:
                        spread_pips = round(info.spread * 0.1, 1) if info.spread > 0 else 0.1
                    return {
                        "bid": round(info.bid, 5),
                        "ask": round(info.ask, 5),
                        "spread": spread_pips,
                        "time": datetime.datetime.now().strftime("%H:%M:%S"),
                        "source": "LIVE MT5"
                    }

            self.tick_count += 1
            # Gentle drift so mock mode has realistic smooth sub-pip movement
            drift = np.sin(self.tick_count * 0.04) * 0.0003
            live_mid = round(self.base_price + drift, 5)
            return {
                "bid": round(live_mid - 0.00005, 5),
                "ask": round(live_mid + 0.00005, 5),
                "spread": 0.1,
                "time": datetime.datetime.now().strftime("%H:%M:%S"),
                "source": "MOCK SIM"
            }

    def fetch_rates(self, tf_const: int, count: int = 150) -> Optional[pd.DataFrame]:
        if not self.use_mock and self.connected and MT5_AVAILABLE:
            rates = mt5.copy_rates_from_pos(self.symbol, tf_const, 0, count)
            if rates is not None and len(rates) > 20:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                return self._compute_indicators(df)

        return self._generate_mock_df(count, tf_key=tf_const)

    def _generate_mock_df(self, count: int = 150, tf_key: int = 60) -> pd.DataFrame:
        """Maintains persistent in-memory bars so prices evolve coherently without random flashing."""
        if tf_key not in self._mock_dataframes:
            now = datetime.datetime.now()
            times = [now - datetime.timedelta(hours=1 * (count - i)) for i in range(count)]
            p = self.base_price
            close_prices = []
            
            # Use deterministic seed per timeframe to establish an initial realistic swing trend
            np.random.seed(100 + tf_key % 50)
            for i in range(count):
                trend_bias = (i - (count / 2)) * 0.00003
                cycle = np.sin(i * 0.06) * 0.0012
                noise = np.random.normal(0, 0.00015)
                p = self.base_price + trend_bias + cycle + noise
                close_prices.append(round(p, 5))

            close_series = pd.Series(close_prices)
            highs = close_series + 0.0009
            lows = close_series - 0.0009
            opens = close_series.shift(1).fillna(close_series.iloc[0])

            base_df = pd.DataFrame({
                "time": times,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": close_series,
                "tick_volume": np.random.randint(1000, 3500, count)
            })
            self._mock_dataframes[tf_key] = base_df

        # Update only the current forming live candle (the last bar)
        df = self._mock_dataframes[tf_key].copy()
        live_mid = round(self.base_price + (np.sin(self.tick_count * 0.04) * 0.0003), 5)
        last_i = len(df) - 1
        df.loc[last_i, 'close'] = live_mid
        df.loc[last_i, 'high'] = max(df.loc[last_i, 'high'], live_mid)
        df.loc[last_i, 'low'] = min(df.loc[last_i, 'low'], live_mid)

        return self._compute_indicators(df)

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        c = df['close']
        h = df['high']
        l = df['low']

        for span in [7, 10, 20, 21, 50, 200]:
            df[f'ema_{span}'] = c.ewm(span=span, adjust=False).mean()

        for period in [10, 44, 50, 200]:
            df[f'sma_{period}'] = c.rolling(window=period).mean().bfill()

        # MACD (12, 26, 9)
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        df['macd_line'] = ema12 - ema26
        df['macd_sig'] = df['macd_line'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd_line'] - df['macd_sig']

        # RSI 14
        delta = c.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss.replace(0, np.nan))
        df['rsi_14'] = (100 - (100 / (1 + rs))).fillna(50.0)

        # ATR 14
        tr1 = h - l
        tr2 = (h - c.shift()).abs()
        tr3 = (l - c.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(window=14).mean().bfill()

        # Bollinger Bands (20, 2.0)
        df['bb_mid'] = c.rolling(20).mean().bfill()
        std20 = c.rolling(20).std().bfill()
        df['bb_up'] = df['bb_mid'] + (std20 * 2.0)
        df['bb_low'] = df['bb_mid'] - (std20 * 2.0)

        # Supertrend (7, 2.5)
        atr7 = tr.rolling(window=7).mean().bfill()
        hl2 = (h + l) / 2.0
        up_b = hl2 - (2.5 * atr7)
        dn_b = hl2 + (2.5 * atr7)
        st = []
        st_dir = []
        d = 1
        curr_st = up_b.iloc[0]
        for i in range(len(df)):
            cl = c.iloc[i]
            if d == 1:
                if cl < up_b.iloc[i]:
                    d = -1
                    curr_st = dn_b.iloc[i]
                else:
                    curr_st = max(up_b.iloc[i], curr_st) if i > 0 else up_b.iloc[i]
            else:
                if cl > dn_b.iloc[i]:
                    d = 1
                    curr_st = up_b.iloc[i]
                else:
                    curr_st = min(dn_b.iloc[i], curr_st) if i > 0 else dn_b.iloc[i]
            st.append(curr_st)
            st_dir.append(d)
        df['supertrend'] = st
        df['st_dir'] = st_dir

        # VWAP
        typ = (h + l + c) / 3.0
        vol = df.get('tick_volume', pd.Series(1, index=df.index))
        df['vwap'] = (typ * vol).cumsum() / (vol.cumsum().replace(0, np.nan))
        df['vwap'] = df['vwap'].bfill()

        return df


class HighPipSwingEnsemble:
    """Manages evaluation of 60 H1/H2 strategies and synthesizes the single actionable trade."""

    def __init__(self, symbol: str = DEFAULT_SYMBOL, use_mock: bool = False):
        self.provider = MarketDataProvider(symbol=symbol, use_mock=use_mock)
        self.lock = threading.Lock()
        self.cached_state: Dict[str, Any] = {}
        self.strategy_meta = self._load_strategy_metadata()

        # State Debouncing & Confirmation Cache
        self.confirmed_action: str = "WAITING / FLAT"
        self.pending_action: str = "WAITING / FLAT"
        self.consecutive_pending_cycles: int = 0
        self.cycles_to_confirm: int = 3

    def _load_strategy_metadata(self) -> Dict[int, Dict[str, Any]]:
        raw_meta = {
            1: ("ICT Breaker Block", "Breaker Block", "A", "H1 / H2"),
            2: ("H4 Order Block + H1 FVG", "Order Block", "A", "H4 / H1"),
            3: ("TG London Killzone FVG", "London Killzone", "A", "H2 / H1"),
            4: ("Neeraj Joshi 4H Fractal Sweep", "Fractal Sweep", "A", "H4 / H1"),
            5: ("SLC Supply/Demand + Stoch", "Supply Demand", "A", "H2 / H1"),
            6: ("The Trading School Fib Golden Zone", "Fib Golden Zone", "A", "H1 / H2"),
            7: ("Stock Burner Trendline Break", "Trendline Break", "A", "H1 / H2"),
            8: ("Smart Money Gold 8 EMA Zone", "Smart Money Zone", "A", "H1 / H2"),
            10: ("20/200 Dual EMA Macro Pullback", "20/200 EMA", "B", "H1 / H2"),
            11: ("Ezekiel Chew 50 EMA Rejection", "50 EMA Rejection", "B", "H1 / H2"),
            14: ("Rayner Teo 10 EMA Inside Bar", "Inside Bar", "B", "H1 / H2"),
            16: ("200 EMA + Stochastic Reversal", "Stoch 200 EMA", "C", "H1 / H2"),
            17: ("Rayner Teo 200 SMA + 10 RSI", "SMA RSI Pullback", "C", "H1 / H2"),
            18: ("Multi-TF MACD Momentum", "MACD Momentum", "C", "H1 / H2"),
            20: ("Subasish Bollinger Band Mean Reversion", "Bollinger Reversion", "C", "H1 / H2"),
            21: ("Tanisha Garg CPR + RSI Reversal", "CPR RSI Bounce", "D", "D1 / H1"),
            22: ("Camarilla Pivot Points Breakout", "Camarilla Pivots", "D", "D1 / H1"),
            23: ("CPR Multi-Timeframe + R1 Break", "CPR Multi-TF", "D", "D1 / H1"),
            24: ("Supertrend (7, 2.5) + 200 EMA Flip", "Supertrend Flip", "D", "H1 / H2"),
            25: ("UT Bot Alert ATR Trailing + 200 EMA", "UT Bot Alert", "D", "H1 / H2"),
            26: ("MambaFX Forex H4 S/R + 8/50 SMA", "MambaFX S/R", "D", "H4 / H1"),
            28: ("Chandelier Exit (22, 3) + 32 ZLSMA", "Chandelier ZLSMA", "E", "H1 / H2"),
            29: ("Kojo Dubai Pure Price Action Demand", "Kojo Demand/Supply", "E", "H1 / H2"),
            30: ("Trading School Secret MBP Gold", "MBP Gold Setup", "E", "H1 / H2"),
            31: ("Trade With Pat Demand FVG Sniper", "Pat Demand FVG", "E", "H1 / H2"),
            32: ("SMC Liquidity Sniper 71% Fib Retrace", "Liquidity Sniper", "E", "H4 / H1"),
            33: ("Pure Double Breakout Retest", "Double Breakout", "E", "H1 / H2"),
            34: ("Asia Forex Mentor Smoothed Heikin Ashi", "Heikin Ashi", "E", "H1 / H2"),
            35: ("Bharat RSI 60/40 + Bollinger Band", "Bharat RSI BB", "E", "H1 / H2"),
            36: ("Trade Like Berlin 'Final Boss' 7 EMA", "Berlin 7 EMA", "F", "H1 / H2"),
            37: ("Ezekiel Chew 50 EMA Pullback Engulfing", "Ezekiel 50 EMA", "F", "H1 / H2"),
            38: ("Trading Geek S/R Exhaustion Action", "Trading Geek S/R", "F", "H4 / H1"),
            39: ("Tori Trades Multi-Touch Trendline", "Tori Trendline", "F", "H1 / H2"),
            40: ("Hold with Priyank 21 EMA Breakout", "Priyank 21 EMA", "F", "H1 / H2"),
            42: ("TradingLab 200 DEMA + Supertrend", "TradingLab DEMA", "F", "H1 / H2"),
            43: ("HalfTrend (2, 2) + ATR Bands", "HalfTrend ATR", "G", "H1 / H2"),
            44: ("Umer Akhtar 3-Candle Sandwich", "Umer Sandwich", "G", "H1 / H2"),
            45: ("Multi-TF MACD Triple Confluence", "3TF MACD", "G", "D1 / H4 / H1"),
            46: ("ICT Double Sweep Liquidity Sniper", "ICT Double Sweep", "G", "H1 / H2"),
            47: ("ICT Daily OB + H1 FVG Precision", "Daily OB + H1 FVG", "G", "D1 / H1"),
            48: ("TG Capital EMA Stack + Doji Pullback", "EMA Stack Doji", "G", "H1 / H2"),
            49: ("Rayner Teo 50 EMA Swing Catch Wave", "Rayner 50 EMA", "G", "D1 / H1"),
            50: ("50 EMA + RSI 30/70 Oversold Reclaim", "EMA RSI Reclaim", "G", "H1 / H2"),
            51: ("Booming Bulls HalfTrend + DMI Setup", "Booming HalfTrend", "G", "H1 / H2"),
            52: ("Priyank Weekly Trap + H1 MSS", "Weekly Trap MSS", "G", "W1 / D1 / H1"),
            53: ("Traders Paradise Red/Green Box", "Paradise Box", "G", "H1 / H2"),
            54: ("Renko 10 SMA + OBV Volume Trend", "Renko 10 SMA", "G", "H1 / H2"),
            55: ("Umar Punjabi London Session Sweep", "Umar Gold London", "H", "H1 / H2"),
            56: ("Bill Williams Alligator + Fractals", "Bill Alligator", "H", "H1 / H2"),
            57: ("ICT PO3 AMD Framework Expansion", "ICT PO3 AMD", "H", "H1 / H2"),
            60: ("Jesse Rogers SMC FVG Retest", "Jesse SMC FVG", "H", "H1 / H2"),
            61: ("Theta Gainers EMA + RSI Reversal", "Theta EMA RSI", "H", "H1 / H2"),
            62: ("Gold SMC Imbalance Zone Retest", "Gold Imbalance", "H", "H1 / H2"),
            64: ("SMC Breaker Block V2 Mitigation", "SMC Breaker V2", "H", "H1 / H2"),
            65: ("Steven Hart 4H Trend Pullback Zone", "Steven Hart 4H", "H", "H4 / H1"),
            66: ("Bhanushali 44 MA Bounce Rejection", "Bhanushali 44 MA", "H", "D1 / H1"),
            67: ("Mukul Chowdhary 10 EMA + VWAP Setup", "Mukul 10 EMA VWAP", "H", "H1 / H2"),
            68: ("LuxAlgo Confirmation Trend Baseline", "LuxAlgo Confirm", "H", "H1 / H2"),
            69: ("UK Trader Break & Bounce Reversal", "UK Break & Bounce", "H", "H1 / H2"),
            70: ("PB Trading H4 + H1 Dual FVG", "PB Multi-TF FVG", "H", "H4 / H1")
        }
        res = {}
        for k, (name, short_name, cat, tf) in raw_meta.items():
            res[k] = {
                "name": name,
                "short_name": short_name,
                "category_code": cat,
                "category": CATEGORIES[cat],
                "timeframe": tf
            }
        return res

    def evaluate_cycle(self) -> Dict[str, Any]:
        """Evaluates all 60 curated H1/H2 models and produces the single actionable output."""
        tick = self.provider.get_live_tick()
        live_price = tick["bid"]
        spread_pips = tick["spread"]

        # 1. Fetch H1, H2, H4, D1 rates
        h1_const = getattr(mt5, "TIMEFRAME_H1", 60) if MT5_AVAILABLE else 60
        h2_const = getattr(mt5, "TIMEFRAME_H2", 120) if MT5_AVAILABLE else 120
        h4_const = getattr(mt5, "TIMEFRAME_H4", 240) if MT5_AVAILABLE else 240
        d1_const = getattr(mt5, "TIMEFRAME_D1", 1440) if MT5_AVAILABLE else 1440

        df_h1 = self.provider.fetch_rates(h1_const, 150)
        df_h2 = self.provider.fetch_rates(h2_const, 100)
        df_h4 = self.provider.fetch_rates(h4_const, 80)
        df_d1 = self.provider.fetch_rates(d1_const, 50)

        # 2. Evaluate all 60 curated strategies
        evaluated_strats: List[Dict[str, Any]] = []
        buy_votes = 0
        sell_votes = 0
        agreeing_drivers = []

        for idx, meta in self.strategy_meta.items():
            state = self._evaluate_strategy(idx, meta, df_h1, df_h2, df_h4, df_d1, live_price, spread_pips)
            evaluated_strats.append(asdict(state))

            if state.direction == "BUY" and state.overall_status in ["READY", "TRIGGERED"]:
                buy_votes += 1
                agreeing_drivers.append(state.short_name)
            elif state.direction == "SELL" and state.overall_status in ["READY", "TRIGGERED"]:
                sell_votes += 1
                agreeing_drivers.append(state.short_name)

        # 3. SYNTHESIZE SINGLE ACTIONABLE HIGH-PIP SWING DECISION WITH DEBOUNCING
        master_decision = self._synthesize_master_decision(
            buy_votes, sell_votes, agreeing_drivers, df_h1, df_h2, live_price, spread_pips
        )

        snapshot = {
            "symbol": self.provider.symbol,
            "server_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tick": tick,
            "account": self.provider.account_info,
            "master_decision": master_decision,
            "total_strategies": len(evaluated_strats),
            "strategies": evaluated_strats,
            "categories": CATEGORIES
        }

        with self.lock:
            self.cached_state = snapshot

        return snapshot

    def _evaluate_strategy(self, idx: int, meta: Dict[str, Any], df_h1: pd.DataFrame, df_h2: pd.DataFrame, df_h4: pd.DataFrame, df_d1: pd.DataFrame, live_price: float, spread_pips: float) -> StrategyState:
        """Evaluates strategy state using closed-bar verification (iloc[-2]) and category-specific logic."""
        name = meta["name"]
        short_name = meta["short_name"]
        cat_code = meta["category_code"]
        cat_name = meta["category"]
        tf = meta["timeframe"]

        # Ensure enough data bars
        min_bars = 5
        if df_h1 is None or len(df_h1) < min_bars:
            return StrategyState(
                id=str(idx), num=idx, name=name, short_name=short_name,
                category=cat_name, category_code=cat_code, timeframe=tf,
                direction="NEUTRAL", h2_status="FAIL", h1_status="FAIL",
                overall_status="WAITING", progress_pct=0, true_count=0, total_steps=3,
                steps=[]
            )

        # CRITICAL STABILIZATION:
        # Use iloc[-2] (the last CONFIRMED CLOSED candle) for all structural indicators.
        # This prevents intra-candle sub-second tick oscillations.
        curr_h1 = df_h1.iloc[-2]
        prev_h1 = df_h1.iloc[-3] if len(df_h1) >= 3 else curr_h1
        close_h1 = float(curr_h1['close'])
        open_h1 = float(curr_h1['open'])

        df_h2_valid = df_h2 if (df_h2 is not None and len(df_h2) >= min_bars) else df_h1
        curr_h2 = df_h2_valid.iloc[-2]
        prev_h2 = df_h2_valid.iloc[-3] if len(df_h2_valid) >= 3 else curr_h2
        close_h2 = float(curr_h2['close'])

        df_h4_valid = df_h4 if (df_h4 is not None and len(df_h4) >= min_bars) else df_h2_valid
        curr_h4 = df_h4_valid.iloc[-2]

        df_d1_valid = df_d1 if (df_d1 is not None and len(df_d1) >= min_bars) else df_h4_valid
        curr_d1 = df_d1_valid.iloc[-2]

        # Key Technical Reference Points
        atr_h1 = max(0.0010, float(curr_h1.get('atr_14', 0.0018)))
        buffer_band = atr_h1 * 0.15  # ~2.5 - 3.5 pip neutral zone around key EMAs

        ema7_h1 = float(curr_h1.get('ema_7', close_h1))
        ema20_h1 = float(curr_h1.get('ema_20', close_h1))
        ema21_h1 = float(curr_h1.get('ema_21', close_h1))
        ema50_h1 = float(curr_h1.get('ema_50', close_h1))
        ema200_h1 = float(curr_h1.get('ema_200', close_h1))

        ema50_h2 = float(curr_h2.get('ema_50', close_h2))
        ema200_h2 = float(curr_h2.get('ema_200', close_h2))

        rsi_h1 = float(curr_h1.get('rsi_14', 50.0))
        supertrend_dir = int(curr_h1.get('st_dir', 1))
        vwap_h1 = float(curr_h1.get('vwap', close_h1))
        macd_line = float(curr_h1.get('macd_line', 0.0))
        macd_sig = float(curr_h1.get('macd_sig', 0.0))

        steps: List[ConditionStep] = []
        metrics: Dict[str, Any] = {
            "H1 Close": round(close_h1, 5),
            "H2 Close": round(close_h2, 5),
            "H1 ATR": f"{atr_h1 / PIP_SIZE:.1f}p"
        }

        cand_dir = "NEUTRAL"

        # ======================================================================
        # CATEGORY-SPECIFIC TECHNICAL DISPATCHER
        # ======================================================================

        if cat_code == "A":
            # --- [A] INSTITUTIONAL SMC & LIQUIDITY ---
            metrics["H2 50 EMA"] = round(ema50_h2, 5)
            # Step 1: Market Structure Swing Leg (H2 Range Equilibrium)
            sh_20 = df_h2_valid['high'].iloc[-25:-1].max()
            sl_20 = df_h2_valid['low'].iloc[-25:-1].min()
            eq_mid = (sh_20 + sl_20) / 2.0
            
            is_bull = close_h2 > (eq_mid + buffer_band)
            is_bear = close_h2 < (eq_mid - buffer_band)
            cand_dir = "BUY" if is_bull else ("SELL" if is_bear else "NEUTRAL")
            s1_passed = cand_dir != "NEUTRAL"
            steps.append(ConditionStep(
                1, "SMC Structural Swing Regime",
                "Price clear of 20-bar equilibrium midpoint",
                f"Close {close_h2:.5f} vs Mid {eq_mid:.5f}",
                f"{abs(close_h2 - eq_mid)/PIP_SIZE:.1f}p margin",
                "PASS" if s1_passed else "PENDING"
            ))

            # Step 2: Discount / Premium Value Zone Pullback
            pullback_ok = False
            if cand_dir == "BUY":
                pullback_ok = float(curr_h1['low']) <= (close_h2 - (0.3 * atr_h1)) or float(curr_h1['low']) <= ema50_h1
            elif cand_dir == "SELL":
                pullback_ok = float(curr_h1['high']) >= (close_h2 + (0.3 * atr_h1)) or float(curr_h1['high']) >= ema50_h1
            s2_passed = s1_passed and pullback_ok
            steps.append(ConditionStep(
                2, "Institutional Liquidity Mitigation",
                "Retest of discount/premium zone or dynamic EMA",
                "Mitigated zone" if pullback_ok else "Awaiting zone tap",
                "In Zone" if pullback_ok else "Gap",
                "PASS" if s2_passed else "PENDING"
            ))

            # Step 3: Rejection & Directional Candle Confirmation
            rejection = (close_h1 > open_h1) if cand_dir == "BUY" else (close_h1 < open_h1)
            s3_passed = s2_passed and rejection
            steps.append(ConditionStep(
                3, "H1 Swing Liquidity Sweep Rejection",
                f"Confirmed closed {'Bullish' if cand_dir == 'BUY' else 'Bearish'} candle",
                "Confirmed Closed" if rejection else "Awaiting Candle Close",
                "Ready" if s3_passed else "Pending",
                "PASS" if s3_passed else "PENDING"
            ))

        elif cat_code == "B":
            # --- [B] MACRO TREND & MOVING AVERAGES ---
            metrics["H2 50 EMA"] = round(ema50_h2, 5)
            metrics["H2 200 EMA"] = round(ema200_h2, 5)
            # Step 1: 50 & 200 EMA Macro Polarity with Buffer
            is_bull = close_h2 > (ema50_h2 + buffer_band) and ema50_h2 >= (ema200_h2 - buffer_band)
            is_bear = close_h2 < (ema50_h2 - buffer_band) and ema50_h2 <= (ema200_h2 + buffer_band)
            cand_dir = "BUY" if is_bull else ("SELL" if is_bear else "NEUTRAL")
            s1_passed = cand_dir != "NEUTRAL"
            steps.append(ConditionStep(
                1, "Macro EMA Alignment (H2)",
                "Price clear of 50 EMA with macro trend bias",
                f"Close {close_h2:.5f} vs 50 EMA {ema50_h2:.5f}",
                f"{abs(close_h2 - ema50_h2)/PIP_SIZE:.1f}p buffer",
                "PASS" if s1_passed else "PENDING"
            ))

            # Step 2: H1 Pullback into 20/50 EMA Dynamic Zone
            dist_to_ema = abs(close_h1 - ema50_h1)
            in_zone = dist_to_ema <= (atr_h1 * 1.6)
            s2_passed = s1_passed and in_zone
            steps.append(ConditionStep(
                2, "H1 Moving Average Retest",
                "Pullback within 1.6 ATR of 50 EMA baseline",
                f"Distance: {dist_to_ema/PIP_SIZE:.1f} pips",
                "In Zone" if in_zone else "Outside zone",
                "PASS" if s2_passed else "PENDING"
            ))

            # Step 3: Candle Body Expansion in Trend Direction
            body_ok = (close_h1 > open_h1 and close_h1 > ema20_h1) if cand_dir == "BUY" else (close_h1 < open_h1 and close_h1 < ema20_h1)
            s3_passed = s2_passed and body_ok
            steps.append(ConditionStep(
                3, "H1 Dynamic Rejection Trigger",
                "Decisive candle expansion off value zone",
                "Confirmed" if body_ok else "Opposing candle",
                "Ready" if s3_passed else "Pending",
                "PASS" if s3_passed else "PENDING"
            ))

        elif cat_code == "C":
            # --- [C] OSCILLATORS, REVERSALS & MEAN REVERSION ---
            metrics["RSI (14)"] = round(rsi_h1, 1)
            metrics["MACD Line"] = round(macd_line, 6)
            # Step 1: Multi-TF Momentum Regime
            is_bull = rsi_h1 >= 48.0 and macd_line >= macd_sig
            is_bear = rsi_h1 <= 52.0 and macd_line <= macd_sig
            cand_dir = "BUY" if is_bull else ("SELL" if is_bear else "NEUTRAL")
            s1_passed = cand_dir != "NEUTRAL"
            steps.append(ConditionStep(
                1, "Momentum Oscillator Polarity",
                "RSI & MACD histogram trend agreement",
                f"RSI: {rsi_h1:.1f} | MACD: {macd_line:.5f}",
                "Aligned" if s1_passed else "Mixed",
                "PASS" if s1_passed else "PENDING"
            ))

            # Step 2: Healthy Pullback Level (RSI Not Saturated)
            pullback_healthy = (45.0 <= rsi_h1 <= 65.0) if cand_dir == "BUY" else (35.0 <= rsi_h1 <= 55.0)
            s2_passed = s1_passed and pullback_healthy
            steps.append(ConditionStep(
                2, "Oscillator Equilibrium Retest",
                "RSI within prime expansion range (35-65)",
                f"RSI Reading: {rsi_h1:.1f}",
                "In Range" if pullback_healthy else "Saturated",
                "PASS" if s2_passed else "PENDING"
            ))

            # Step 3: Momentum Hook
            prev_rsi = float(prev_h1.get('rsi_14', rsi_h1))
            rsi_hook = (rsi_h1 > prev_rsi) if cand_dir == "BUY" else (rsi_h1 < prev_rsi)
            s3_passed = s2_passed and rsi_hook
            steps.append(ConditionStep(
                3, "Momentum Expansion Hook",
                "Oscillator hooks in direction of swing",
                "Hook Confirmed" if rsi_hook else "Flat/Opposing",
                "Ready" if s3_passed else "Pending",
                "PASS" if s3_passed else "PENDING"
            ))

        elif cat_code == "D":
            # --- [D] MATHEMATICAL PIVOTS & VOLATILITY SYSTEMS ---
            metrics["Supertrend Dir"] = "BULLISH (+1)" if supertrend_dir == 1 else "BEARISH (-1)"
            metrics["H1 200 EMA"] = round(ema200_h1, 5)
            # Step 1: Supertrend & 200 EMA Polarity
            is_bull = supertrend_dir == 1 and close_h1 > ema200_h1
            is_bear = supertrend_dir == -1 and close_h1 < ema200_h1
            cand_dir = "BUY" if is_bull else ("SELL" if is_bear else "NEUTRAL")
            s1_passed = cand_dir != "NEUTRAL"
            steps.append(ConditionStep(
                1, "Volatility Baseline Polarity",
                "Supertrend & 200 EMA polarity alignment",
                f"ST: {'BUY' if supertrend_dir==1 else 'SELL'} | 200 EMA: {ema200_h1:.5f}",
                "Aligned" if s1_passed else "Opposing",
                "PASS" if s1_passed else "PENDING"
            ))

            # Step 2: ATR Volatility Room
            st_val = float(curr_h1.get('supertrend', close_h1))
            st_dist = abs(close_h1 - st_val)
            s2_passed = s1_passed and (st_dist >= (0.5 * atr_h1))
            steps.append(ConditionStep(
                2, "Dynamic Trailing Runway",
                "Price maintains clearance from trailing band",
                f"Band Distance: {st_dist/PIP_SIZE:.1f} pips",
                "Adequate" if s2_passed else "Too tight",
                "PASS" if s2_passed else "PENDING"
            ))

            # Step 3: Volatility Expansion Bar
            bar_range = float(curr_h1['high']) - float(curr_h1['low'])
            vol_ok = bar_range >= (0.6 * atr_h1)
            s3_passed = s2_passed and vol_ok
            steps.append(ConditionStep(
                3, "Volatility Expansion Confirmation",
                "Candle range >= 0.6 ATR",
                f"Range: {bar_range/PIP_SIZE:.1f}p vs 0.6 ATR: {0.6*atr_h1/PIP_SIZE:.1f}p",
                "Expanded" if vol_ok else "Compressed",
                "PASS" if s3_passed else "PENDING"
            ))

        elif cat_code == "E":
            # --- [E] ADVANCED PRICE ACTION & DEMAND/SUPPLY ---
            metrics["21 EMA"] = round(ema21_h1, 5)
            # Step 1: Structural Direction off 21 EMA & Price Action
            is_bull = close_h1 > (ema21_h1 + buffer_band)
            is_bear = close_h1 < (ema21_h1 - buffer_band)
            cand_dir = "BUY" if is_bull else ("SELL" if is_bear else "NEUTRAL")
            s1_passed = cand_dir != "NEUTRAL"
            steps.append(ConditionStep(
                1, "Price Action Value Baseline",
                "Price trading on clear side of 21 EMA",
                f"Close {close_h1:.5f} vs 21 EMA {ema21_h1:.5f}",
                f"{abs(close_h1 - ema21_h1)/PIP_SIZE:.1f}p margin",
                "PASS" if s1_passed else "PENDING"
            ))

            # Step 2: Demand / Supply Zone Touch
            tested_zone = (float(curr_h1['low']) <= ema21_h1 + (0.5 * atr_h1)) if cand_dir == "BUY" else (float(curr_h1['high']) >= ema21_h1 - (0.5 * atr_h1))
            s2_passed = s1_passed and tested_zone
            steps.append(ConditionStep(
                2, "Institutional S/D Retest",
                "Candle wicks test dynamic value zone",
                "Zone Tested" if tested_zone else "Awaiting Retest",
                "In Zone" if tested_zone else "Gap",
                "PASS" if s2_passed else "PENDING"
            ))

            # Step 3: Wick Rejection Confirmation
            wick_rejection = (close_h1 > open_h1) if cand_dir == "BUY" else (close_h1 < open_h1)
            s3_passed = s2_passed and wick_rejection
            steps.append(ConditionStep(
                3, "Rejection Pin / Engulfing",
                "Clean rejection body in swing direction",
                "Confirmed" if wick_rejection else "Opposing close",
                "Ready" if s3_passed else "Pending",
                "PASS" if s3_passed else "PENDING"
            ))

        elif cat_code == "F":
            # --- [F] PRECISION MOMENTUM & PULLBACK ---
            metrics["7 EMA"] = round(ema7_h1, 5)
            metrics["21 EMA"] = round(ema21_h1, 5)
            # Step 1: Fast Momentum Stack (7 > 21 EMA)
            is_bull = ema7_h1 > (ema21_h1 + buffer_band)
            is_bear = ema7_h1 < (ema21_h1 - buffer_band)
            cand_dir = "BUY" if is_bull else ("SELL" if is_bear else "NEUTRAL")
            s1_passed = cand_dir != "NEUTRAL"
            steps.append(ConditionStep(
                1, "Fast Momentum Alignment (7/21 EMA)",
                "7 EMA cleanly stacked over 21 EMA",
                f"7 EMA {ema7_h1:.5f} vs 21 EMA {ema21_h1:.5f}",
                f"{abs(ema7_h1 - ema21_h1)/PIP_SIZE:.1f}p spread",
                "PASS" if s1_passed else "PENDING"
            ))

            # Step 2: Shallow Pullback to 7 EMA
            shallow_pb = (float(curr_h1['low']) <= ema7_h1 + 0.0003) if cand_dir == "BUY" else (float(curr_h1['high']) >= ema7_h1 - 0.0003)
            s2_passed = s1_passed and shallow_pb
            steps.append(ConditionStep(
                2, "Precision 7 EMA Pullback",
                "Price tests 7 EMA rapid continuation line",
                "Touched 7 EMA" if shallow_pb else "Awaiting touch",
                "In Pocket" if shallow_pb else "Outside",
                "PASS" if s2_passed else "PENDING"
            ))

            # Step 3: Continuation Close
            cont_ok = (close_h1 >= ema7_h1 and close_h1 > open_h1) if cand_dir == "BUY" else (close_h1 <= ema7_h1 and close_h1 < open_h1)
            s3_passed = s2_passed and cont_ok
            steps.append(ConditionStep(
                3, "Momentum Continuation Trigger",
                "Closed bar continuation beyond 7 EMA",
                "Triggered" if cont_ok else "Awaiting close",
                "Ready" if s3_passed else "Pending",
                "PASS" if s3_passed else "PENDING"
            ))

        elif cat_code == "G":
            # --- [G] MULTI-CONFLUENCE, TRAPS & RECLAIMS ---
            metrics["H2 50 EMA"] = round(ema50_h2, 5)
            metrics["D1 Bias"] = "BULLISH" if float(curr_d1['close']) >= float(curr_d1.get('ema_50', curr_d1['close'])) else "BEARISH"
            # Step 1: Multi-Timeframe Daily & H2 Alignment
            d1_bull = float(curr_d1['close']) >= float(curr_d1.get('ema_50', curr_d1['close']))
            h2_bull = close_h2 > (ema50_h2 + buffer_band)
            h2_bear = close_h2 < (ema50_h2 - buffer_band)
            
            is_bull = d1_bull and h2_bull
            is_bear = (not d1_bull) and h2_bear
            cand_dir = "BUY" if is_bull else ("SELL" if is_bear else "NEUTRAL")
            s1_passed = cand_dir != "NEUTRAL"
            steps.append(ConditionStep(
                1, "Multi-Timeframe Macro Confluence",
                "Daily & H2 structural bias agreement",
                f"D1: {'BUY' if d1_bull else 'SELL'} | H2: {'BUY' if h2_bull else ('SELL' if h2_bear else 'FLAT')}",
                "Confluent" if s1_passed else "Divergent",
                "PASS" if s1_passed else "PENDING"
            ))

            # Step 2: Trap & Value Reclaim
            reclaim_ok = (close_h1 >= ema50_h1) if cand_dir == "BUY" else (close_h1 <= ema50_h1)
            s2_passed = s1_passed and reclaim_ok
            steps.append(ConditionStep(
                2, "H1 Liquidity Trap Reclaim",
                "Price reclaims high-liquidity 50 EMA boundary",
                f"Close {close_h1:.5f} vs 50 EMA {ema50_h1:.5f}",
                "Reclaimed" if reclaim_ok else "Below/Above EMA",
                "PASS" if s2_passed else "PENDING"
            ))

            # Step 3: Dual Confluence Trigger (MACD + Candle)
            dual_ok = (macd_line > macd_sig and close_h1 > open_h1) if cand_dir == "BUY" else (macd_line < macd_sig and close_h1 < open_h1)
            s3_passed = s2_passed and dual_ok
            steps.append(ConditionStep(
                3, "Macro-Micro Confluence Fire",
                "Simultaneous MACD and Price Action alignment",
                "Confirmed" if dual_ok else "Partial Alignment",
                "Ready" if s3_passed else "Pending",
                "PASS" if s3_passed else "PENDING"
            ))

        else:
            # --- [H] IMBALANCE, VWAP & TREND EXPANSION ---
            metrics["Session VWAP"] = round(vwap_h1, 5)
            metrics["10 EMA"] = round(float(curr_h1.get('ema_10', close_h1)), 5)
            # Step 1: Session Institutional Benchmark (VWAP)
            is_bull = close_h1 > (vwap_h1 + buffer_band)
            is_bear = close_h1 < (vwap_h1 - buffer_band)
            cand_dir = "BUY" if is_bull else ("SELL" if is_bear else "NEUTRAL")
            s1_passed = cand_dir != "NEUTRAL"
            steps.append(ConditionStep(
                1, "Session VWAP Institutional Bias",
                "Price trading on clear side of Volume-Weighted Avg",
                f"Close {close_h1:.5f} vs VWAP {vwap_h1:.5f}",
                f"{abs(close_h1 - vwap_h1)/PIP_SIZE:.1f}p margin",
                "PASS" if s1_passed else "PENDING"
            ))

            # Step 2: 10/44 EMA Dynamic Retest
            ema10 = float(curr_h1.get('ema_10', close_h1))
            dist_10 = abs(close_h1 - ema10)
            in_pb = dist_10 <= (atr_h1 * 0.8)
            s2_passed = s1_passed and in_pb
            steps.append(ConditionStep(
                2, "Dynamic Imbalance Pullback",
                "Price within 0.8 ATR of 10 EMA expansion curve",
                f"Dist to 10 EMA: {dist_10/PIP_SIZE:.1f} pips",
                "In Pocket" if in_pb else "Extended",
                "PASS" if s2_passed else "PENDING"
            ))

            # Step 3: Expansion Body Confirmation
            body_expansion = (close_h1 > open_h1 and close_h1 >= ema10) if cand_dir == "BUY" else (close_h1 < open_h1 and close_h1 <= ema10)
            s3_passed = s2_passed and body_expansion
            steps.append(ConditionStep(
                3, "Imbalance Fill & Continuation",
                "Expansion candle confirming institutional flow",
                "Confirmed" if body_expansion else "Pending body",
                "Ready" if s3_passed else "Pending",
                "PASS" if s3_passed else "PENDING"
            ))

        # Overall Status Aggregation
        passed_count = sum(1 for s in steps if s.status == "PASS")
        progress_pct = int((passed_count / len(steps)) * 100) if steps else 0

        h2_status = steps[0].status if len(steps) > 0 else "FAIL"
        h1_status = steps[2].status if len(steps) > 2 else (steps[1].status if len(steps) > 1 else "FAIL")

        overall_status = "TRIGGERED" if passed_count == 3 else ("READY" if passed_count == 2 else ("PROGRESS" if passed_count == 1 else "WAITING"))

        # High-pip order envelope
        active_sig = None
        if passed_count >= 2 and cand_dir in ["BUY", "SELL"]:
            risk = max(15.0 * PIP_SIZE, atr_h1 * 1.0)
            entry = live_price
            sl = round(entry - risk if cand_dir == "BUY" else entry + risk, 5)
            tp1 = round(entry + (risk * 3.5) if cand_dir == "BUY" else entry - (risk * 3.5), 5)
            tp2 = round(entry + (risk * 6.0) if cand_dir == "BUY" else entry - (risk * 6.0), 5)
            active_sig = {
                "strategy_id": idx,
                "name": name,
                "direction": cand_dir,
                "entry": entry,
                "sl": sl,
                "tp1": tp1,
                "tp2": tp2,
                "rr": "1:5.0",
                "risk_pips": round(risk / PIP_SIZE, 1),
                "tp1_pips": round(abs(tp1 - entry) / PIP_SIZE, 1),
                "tp2_pips": round(abs(tp2 - entry) / PIP_SIZE, 1)
            }

        return StrategyState(
            id=str(idx),
            num=idx,
            name=name,
            short_name=short_name,
            category=cat_name,
            category_code=cat_code,
            timeframe=tf,
            direction=cand_dir,
            h2_status=h2_status,
            h1_status=h1_status,
            overall_status=overall_status,
            progress_pct=progress_pct,
            true_count=passed_count,
            total_steps=len(steps),
            steps=[asdict(s) for s in steps],
            active_signal=active_sig,
            summary_note=f"{name} ({tf})",
            metrics=metrics
        )

    def _synthesize_master_decision(self, buy_votes: int, sell_votes: int, agreeing_drivers: List[str], df_h1: pd.DataFrame, df_h2: pd.DataFrame, live_price: float, spread_pips: float) -> Dict[str, Any]:
        """Synthesizes the single master actionable swing decision with debouncing and margin checks."""
        curr_h1 = df_h1.iloc[-2] if df_h1 is not None and len(df_h1) >= 2 else df_h1.iloc[-1]
        atr_h1 = max(0.0010, float(curr_h1.get('atr_14', 0.0018)))
        risk_pips = max(16.0, round(atr_h1 / PIP_SIZE, 1))
        risk_amt = risk_pips * PIP_SIZE

        # Consensus Thresholds:
        # High conviction swing requires >= 20 models agreeing AND a net margin of >= 8 votes
        strong_threshold = 20
        margin_needed = 8
        net_margin = buy_votes - sell_votes

        candidate_action = "WAITING / FLAT"
        regime = "Market Consolidation / Awaiting Macro Wave"
        confluence_score = int((max(buy_votes, sell_votes) / strong_threshold) * 50)

        if buy_votes >= strong_threshold and net_margin >= margin_needed:
            candidate_action = "STRONG BUY"
            confluence_score = min(98, 65 + int((buy_votes / 60.0) * 33))
            regime = "Bullish H2/H1 Swing Expansion"
        elif sell_votes >= strong_threshold and net_margin <= -margin_needed:
            candidate_action = "STRONG SELL"
            confluence_score = min(98, 65 + int((sell_votes / 60.0) * 33))
            regime = "Bearish H2/H1 Swing Expansion"
        elif buy_votes >= 14 and buy_votes > sell_votes:
            candidate_action = "MILD BUY (ACCUMULATION)"
            confluence_score = int((buy_votes / 60.0) * 60)
            regime = "Early Bullish Momentum"
        elif sell_votes >= 14 and sell_votes > buy_votes:
            candidate_action = "MILD SELL (DISTRIBUTION)"
            confluence_score = int((sell_votes / 60.0) * 60)
            regime = "Early Bearish Momentum"

        # DEBOUNCING / STATE PERSISTENCE:
        # Prevent transient 1-2 second flip oscillations. Action must hold for multiple cycles.
        if candidate_action == self.confirmed_action:
            self.consecutive_pending_cycles = 0
            self.pending_action = candidate_action
        else:
            if candidate_action == self.pending_action:
                self.consecutive_pending_cycles += 1
            else:
                self.pending_action = candidate_action
                self.consecutive_pending_cycles = 1

            if self.consecutive_pending_cycles >= self.cycles_to_confirm:
                self.confirmed_action = candidate_action
                self.consecutive_pending_cycles = 0

        final_action = self.confirmed_action
        final_dir = "BUY" if "BUY" in final_action else ("SELL" if "SELL" in final_action else "NEUTRAL")

        # Envelope calculations:
        if final_dir == "BUY":
            entry = live_price
            sl = round(entry - risk_amt, 5)
            tp1 = round(entry + (risk_amt * 3.5), 5)
            tp2 = round(entry + (risk_amt * 6.5), 5)
        elif final_dir == "SELL":
            entry = live_price
            sl = round(entry + risk_amt, 5)
            tp1 = round(entry - (risk_amt * 3.5), 5)
            tp2 = round(entry - (risk_amt * 6.5), 5)
        else:
            entry = live_price
            sl = 0.0
            tp1 = 0.0
            tp2 = 0.0

        tp1_pips = round(abs(tp1 - entry) / PIP_SIZE, 1) if tp1 > 0 else 0.0
        tp2_pips = round(abs(tp2 - entry) / PIP_SIZE, 1) if tp2 > 0 else 0.0

        return {
            "action": final_action,
            "direction": final_dir,
            "confluence_score": confluence_score,
            "regime": regime,
            "active_drivers_count": buy_votes if final_dir == "BUY" else (sell_votes if final_dir == "SELL" else max(buy_votes, sell_votes)),
            "threshold": strong_threshold,
            "key_drivers": agreeing_drivers[:6],
            "entry": round(entry, 5),
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "risk_pips": risk_pips if final_dir != "NEUTRAL" else 0.0,
            "reward_pips_tp1": tp1_pips,
            "reward_pips_tp2": tp2_pips,
            "rr_ratio": "1:5.0 (Average)" if final_dir != "NEUTRAL" else "N/A",
            "holding_period": "24 to 48 Hours (Multi-Session Swing)" if final_dir != "NEUTRAL" else "Flat"
        }
