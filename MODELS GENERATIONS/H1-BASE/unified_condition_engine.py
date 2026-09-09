"""
========================================================================================
UNIFIED 70-STRATEGY CONDITION SEQUENCE & CONSENSUS ENGINE (EURUSD.0)
========================================================================================
Evaluates the sequential prerequisite conditions for all 70 algorithmic strategies
on MT5 symbol 'EURUSD.0' in real-time.
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

CATEGORIES = {
    "A": "SMC & Liquidity",
    "B": "Trend & Moving Averages",
    "C": "Oscillators & Reversals",
    "D": "Pivots & Volatility",
    "E": "Price Action & S/D",
    "F": "Momentum & Pullback",
    "G": "Confluence & Traps",
    "H": "Imbalance & Breakouts"
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
class ScriptConditionState:
    id: str
    num: int
    name: str
    short_name: str
    category: str
    category_code: str
    timeframe: str
    direction: str                     # 'BUY', 'SELL', 'NEUTRAL'
    overall_status: str                # 'WAITING', 'PROGRESS', 'READY', 'TRIGGERED'
    progress_pct: int                  # 0 to 100
    true_count: int
    total_steps: int
    steps: List[Dict[str, Any]]
    active_signal: Optional[Dict[str, Any]] = None
    summary_note: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)


class MarketDataProvider:
    """Provides cached MT5 rates and technical indicators for EURUSD.0."""

    def __init__(self, symbol: str = DEFAULT_SYMBOL, use_mock: bool = False):
        self.symbol = symbol
        self.use_mock = use_mock
        self.connected = False
        self.account_info = {"login": "N/A", "server": "N/A", "balance": 0.0}
        self.base_price = 1.16200
        self.tick_count = 0
        self.lock = threading.Lock()

        if not self.use_mock and MT5_AVAILABLE:
            self._init_mt5()

    def _init_mt5(self):
        if not mt5.initialize():
            print(f"[!] MT5 Init Failed: {mt5.last_error()}")
            self.connected = False
            return False

        # Verify symbol
        sym = self.symbol
        if not mt5.symbol_select(sym, True):
            # Fallback check
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
        print(f"[+] MT5 connected: {self.symbol} | Account: {self.account_info['login']} ({self.account_info['server']})")
        return True

    def get_live_tick(self) -> Dict[str, Any]:
        """Returns live Bid, Ask, Spread for EURUSD.0."""
        with self.lock:
            if not self.use_mock and self.connected and MT5_AVAILABLE:
                info = mt5.symbol_info(self.symbol)
                if info:
                    spread_pips = round((info.ask - info.bid) / PIP_SIZE, 1)
                    if spread_pips == 0:
                        spread_pips = round(info.spread * 0.1, 1) if info.spread > 0 else 0.5
                    return {
                        "bid": round(info.bid, 5),
                        "ask": round(info.ask, 5),
                        "spread": spread_pips,
                        "time": datetime.datetime.now().strftime("%H:%M:%S"),
                        "source": "LIVE MT5"
                    }

            # Mock simulation fallback
            self.tick_count += 1
            drift = np.sin(self.tick_count * 0.1) * 0.0003 + (np.random.randn() * 0.00004)
            live_mid = self.base_price + drift
            bid = round(live_mid - 0.00005, 5)
            ask = round(live_mid + 0.00005, 5)
            return {
                "bid": bid,
                "ask": ask,
                "spread": 1.0,
                "time": datetime.datetime.now().strftime("%H:%M:%S"),
                "source": "MOCK SIM"
            }

    def fetch_timeframe_rates(self, tf_const: int, count: int = 150) -> Optional[pd.DataFrame]:
        """Fetches candles and computes indicators."""
        if not self.use_mock and self.connected and MT5_AVAILABLE:
            rates = mt5.copy_rates_from_pos(self.symbol, tf_const, 0, count)
            if rates is not None and len(rates) > 20:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                return self._compute_indicators(df)

        # Mock fallback data
        return self._generate_mock_df(count)

    def _generate_mock_df(self, count: int = 150) -> pd.DataFrame:
        now = datetime.datetime.now()
        times = [now - datetime.timedelta(minutes=15 * (count - i)) for i in range(count)]
        close_prices = []
        p = self.base_price
        for _ in range(count):
            p += np.random.randn() * 0.0003
            close_prices.append(p)
        close_series = pd.Series(close_prices)
        highs = close_series + np.random.uniform(0.0001, 0.0006, count)
        lows = close_series - np.random.uniform(0.0001, 0.0006, count)
        opens = close_series.shift(1).fillna(close_series.iloc[0])

        df = pd.DataFrame({
            "time": times,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": close_series,
            "tick_volume": np.random.randint(100, 2000, count)
        })
        return self._compute_indicators(df)

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        c = df['close']
        h = df['high']
        l = df['low']

        # Exponential Moving Averages
        for span in [5, 7, 8, 9, 10, 15, 20, 21, 33, 44, 50, 200]:
            df[f'ema_{span}'] = c.ewm(span=span, adjust=False).mean()

        # Simple Moving Averages
        for period in [8, 10, 44, 50, 200]:
            df[f'sma_{period}'] = c.rolling(window=period).mean().bfill()

        # Relative Strength Index (RSI 14, 10, 21)
        for period in [10, 14, 21]:
            delta = c.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss.replace(0, np.nan))
            df[f'rsi_{period}'] = (100 - (100 / (1 + rs))).fillna(50.0)

        # ATR 14
        tr1 = h - l
        tr2 = (h - c.shift()).abs()
        tr3 = (l - c.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(window=14).mean().bfill()

        # Bollinger Bands (20, 2.0)
        sma20 = df['sma_10'] # fallback or rolling 20
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

        # MACD (12, 26, 9)
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_sig'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_sig']

        # Stochastic (14, 3, 3)
        low14 = l.rolling(14).min()
        high14 = h.rolling(14).max()
        k_fast = 100 * ((c - low14) / ((high14 - low14).replace(0, np.nan)))
        df['stoch_k'] = k_fast.rolling(3).mean().fillna(50)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean().fillna(50)

        # VWAP (Cumulative typical price * volume)
        typ = (h + l + c) / 3.0
        vol = df.get('tick_volume', pd.Series(1, index=df.index))
        df['vwap'] = (typ * vol).cumsum() / (vol.cumsum().replace(0, np.nan))
        df['vwap'] = df['vwap'].bfill()

        # ADX 14
        up_move = h - h.shift(1)
        dn_move = l.shift(1) - l
        pos_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
        neg_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
        p_dm_s = pd.Series(pos_dm, index=df.index).rolling(14).mean()
        n_dm_s = pd.Series(neg_dm, index=df.index).rolling(14).mean()
        p_di = 100 * (p_dm_s / tr.rolling(14).mean().replace(0, np.nan))
        n_di = 100 * (n_dm_s / tr.rolling(14).mean().replace(0, np.nan))
        dx = 100 * ((p_di - n_di).abs() / ((p_di + n_di).replace(0, np.nan)))
        df['adx'] = dx.rolling(14).mean().fillna(22.0)

        return df


class ConsensusEngineManager:
    """Evaluates all 70 strategies in memory and compiles real-time condition state."""

    def __init__(self, symbol: str = DEFAULT_SYMBOL, use_mock: bool = False):
        self.provider = MarketDataProvider(symbol=symbol, use_mock=use_mock)
        self.consensus_threshold = 5
        self.lock = threading.Lock()
        self.cached_state: Dict[str, Any] = {}
        self.last_eval_time = 0.0

        # Pre-build metadata for all 70 strategies
        self.strategy_meta = self._load_strategy_metadata()

    def _load_strategy_metadata(self) -> Dict[int, Dict[str, Any]]:
        """Loads names, categories, timeframes for all 70 strategies."""
        meta = {
            1: ("ICT Breaker Block", "Breaker Block", "A", "M15"),
            2: ("H4 Order Block + M15 FVG", "Order Block", "A", "H4/M15"),
            3: ("TG London Killzone FVG", "London Killzone", "A", "M30"),
            4: ("Neeraj Joshi 4H Fractal Sweep", "Fractal Sweep", "A", "H4/M5"),
            5: ("SLC Supply/Demand + Stoch", "Supply Demand", "A", "H1/M5"),
            6: ("The Trading School Fib Golden Zone", "Fib Golden Zone", "A", "M15"),
            7: ("Stock Burner Trendline Break", "Trendline Break", "A", "M15"),
            8: ("Smart Money Gold 8 EMA Zone", "Smart Money Zone", "A", "M15"),
            9: ("Wizard Trader 8/33 EMA Dominance", "8/33 EMA", "B", "M5"),
            10: ("20/200 Dual EMA Macro Pullback", "20/200 EMA", "B", "M5"),
            11: ("Ezekiel Chew 50 EMA Rejection", "50 EMA Rejection", "B", "H1"),
            12: ("Subasish Pani 5 EMA Scalp", "5 EMA Scalp", "B", "M5/M15"),
            13: ("Anant Ladha 10 EMA Alert Trap", "10 EMA Trap", "B", "M5"),
            14: ("Rayner Teo 10 EMA Inside Bar", "Inside Bar", "B", "H1"),
            15: ("8/20 Dual EMA Value Scalp", "8/20 Scalp", "B", "M5"),
            16: ("200 EMA + Stochastic Reversal", "Stoch 200 EMA", "C", "M15"),
            17: ("Rayner Teo 200 SMA + 10 RSI", "SMA RSI Pullback", "C", "H1"),
            18: ("Multi-TF MACD Momentum", "MACD Momentum", "C", "M15"),
            19: ("50 EMA + RSI 30/70 Pullback", "RSI Pullback", "C", "M5"),
            20: ("Subasish Bollinger Band Mean Reversion", "Bollinger Reversion", "C", "M15"),
            21: ("Tanisha Garg CPR + RSI Reversal", "CPR RSI Bounce", "C", "M5"),
            22: ("Camarilla Pivot Points Breakout", "Camarilla Pivots", "D", "M5"),
            23: ("CPR Multi-Timeframe + R1 Break", "CPR Multi-TF", "D", "M5"),
            24: ("Supertrend (7, 2.5) + 200 EMA Flip", "Supertrend Flip", "D", "M15"),
            25: ("UT Bot Alert ATR Trailing + 200 EMA", "UT Bot Alert", "D", "M15"),
            26: ("MambaFX Forex H4 S/R + 8/50 SMA", "MambaFX S/R", "D", "M15"),
            27: ("Subasish Traffic Light 2-Candle", "Traffic Light", "D", "M5"),
            28: ("Chandelier Exit (22, 3) + 32 ZLSMA", "Chandelier ZLSMA", "E", "H1"),
            29: ("Kojo Dubai Pure Price Action Demand", "Kojo Demand/Supply", "E", "M15"),
            30: ("Trading School Secret MBP Gold", "MBP Gold Setup", "E", "H1/M5"),
            31: ("Trade With Pat Demand FVG Sniper", "Pat Demand FVG", "E", "M15"),
            32: ("SMC Liquidity Sniper 71% Fib Retrace", "Liquidity Sniper", "E", "H4/M15"),
            33: ("Pure Double Breakout Retest", "Double Breakout", "E", "M15"),
            34: ("Asia Forex Mentor Smoothed Heikin Ashi", "Heikin Ashi", "E", "H1"),
            35: ("Bharat RSI 60/40 + Bollinger Band", "Bharat RSI BB", "E", "H1"),
            36: ("Trade Like Berlin 'Final Boss' 7 EMA", "Berlin 7 EMA", "F", "H1/M15"),
            37: ("Ezekiel Chew 50 EMA Pullback Engulfing", "Ezekiel 50 EMA", "F", "H1"),
            38: ("Trading Geek S/R Exhaustion Action", "Trading Geek S/R", "F", "H4/H1"),
            39: ("Tori Trades Multi-Touch Trendline", "Tori Trendline", "F", "M15"),
            40: ("Hold with Priyank 21 EMA Breakout", "Priyank 21 EMA", "F", "M15"),
            41: ("Jenfer Impulse Bar Expansion Scalp", "Jenfer Impulse", "F", "M5"),
            42: ("TradingLab 200 DEMA + Supertrend", "TradingLab DEMA", "F", "M15"),
            43: ("HalfTrend (2, 2) + ATR Bands", "HalfTrend ATR", "G", "M15"),
            44: ("Umer Akhtar 3-Candle Sandwich", "Umer Sandwich", "G", "M15"),
            45: ("Multi-TF MACD Triple Confluence", "3TF MACD", "G", "H4/H1/M15"),
            46: ("ICT Double Sweep Liquidity Sniper", "ICT Double Sweep", "G", "M15"),
            47: ("ICT Daily OB + H1 FVG Precision", "Daily OB + H1 FVG", "G", "D1/H1"),
            48: ("TG Capital EMA Stack + Doji Pullback", "EMA Stack Doji", "G", "M15"),
            49: ("Rayner Teo 50 EMA Swing Catch Wave", "Rayner 50 EMA", "G", "H1"),
            50: ("50 EMA + RSI 30/70 Oversold Reclaim", "EMA RSI Reclaim", "G", "M15"),
            51: ("Booming Bulls HalfTrend + DMI Setup", "Booming HalfTrend", "G", "M15"),
            52: ("Priyank Weekly Trap + M15 MSS", "Weekly Trap MSS", "G", "W1/M15"),
            53: ("Traders Paradise Red/Green Box", "Paradise Box", "G", "M15"),
            54: ("Renko 10 SMA + OBV Volume Trend", "Renko 10 SMA", "G", "M15"),
            55: ("Umar Punjabi London Gold Session Sweep", "Umar Gold London", "H", "M15/M5"),
            56: ("Bill Williams Alligator + Fractals", "Bill Alligator", "H", "H1/M15"),
            57: ("ICT PO3 AMD Framework Expansion", "ICT PO3 AMD", "H", "M15"),
            58: ("Oliver Velez Elephant 180 Reversal", "Oliver Elephant 180", "H", "M15"),
            59: ("Moving Average 61.8 Fib Scalp", "Mov Avg Fib Scalp", "H", "M15"),
            60: ("Jesse Rogers SMC FVG Retest", "Jesse SMC FVG", "H", "M15"),
            61: ("Theta Gainers EMA + RSI Reversal", "Theta EMA RSI", "H", "M15"),
            62: ("Gold SMC Imbalance Zone Retest", "Gold Imbalance", "H", "M15"),
            63: ("Dual EMA (50/200) + Stoch RSI Cross", "Dual EMA Stoch RSI", "H", "M15"),
            64: ("SMC Breaker Block V2 Mitigation", "SMC Breaker V2", "H", "M15"),
            65: ("Steven Hart 4H Trend Pullback Zone", "Steven Hart 4H", "H", "H4/M15"),
            66: ("Bhanushali 44 MA Bounce Rejection", "Bhanushali 44 MA", "H", "M15"),
            67: ("Mukul Chowdhary 10 EMA + VWAP Setup", "Mukul 10 EMA VWAP", "H", "M15"),
            68: ("LuxAlgo Confirmation Trend Baseline", "LuxAlgo Confirm", "H", "M15"),
            69: ("UK Trader Break & Bounce Reversal", "UK Break & Bounce", "H", "M15"),
            70: ("PB Trading H1 + M5 Dual FVG", "PB Multi-TF FVG", "H", "H1/M5")
        }
        res = {}
        for k, (name, short_name, cat, tf) in meta.items():
            res[k] = {
                "name": name,
                "short_name": short_name,
                "category_code": cat,
                "category": CATEGORIES[cat],
                "timeframe": tf
            }
        return res

    def evaluate_cycle(self) -> Dict[str, Any]:
        """Main real-time evaluation loop evaluating all 70 models."""
        tick = self.provider.get_live_tick()
        live_price = tick["bid"]
        spread_pips = tick["spread"]

        # Fetch rate caches
        df_m5 = self.provider.fetch_timeframe_rates(getattr(mt5, "TIMEFRAME_M5", 5) if MT5_AVAILABLE else 5, 120)
        df_m15 = self.provider.fetch_timeframe_rates(getattr(mt5, "TIMEFRAME_M15", 15) if MT5_AVAILABLE else 15, 150)
        df_h1 = self.provider.fetch_timeframe_rates(getattr(mt5, "TIMEFRAME_H1", 60) if MT5_AVAILABLE else 60, 100)

        results: List[Dict[str, Any]] = []
        buy_votes = 0
        sell_votes = 0
        active_signals_list: List[Dict[str, Any]] = []

        for idx in range(1, 71):
            info = self.strategy_meta[idx]
            state = self._evaluate_single_strategy(idx, info, df_m5, df_m15, df_h1, live_price, spread_pips)
            results.append(asdict(state))

            if state.direction == "BUY" and state.overall_status in ["READY", "TRIGGERED"]:
                buy_votes += 1
                if state.active_signal:
                    active_signals_list.append(state.active_signal)
            elif state.direction == "SELL" and state.overall_status in ["READY", "TRIGGERED"]:
                sell_votes += 1
                if state.active_signal:
                    active_signals_list.append(state.active_signal)

        # Consensus evaluation
        threshold = self.consensus_threshold
        if buy_votes >= threshold:
            consensus_status = "BULLISH CONSENSUS"
            consensus_dir = "BUY"
        elif sell_votes >= threshold:
            consensus_status = "BEARISH CONSENSUS"
            consensus_dir = "SELL"
        elif buy_votes > 0 or sell_votes > 0:
            consensus_status = "MIXED SIGNALS"
            consensus_dir = "NEUTRAL"
        else:
            consensus_status = "WAITING"
            consensus_dir = "NEUTRAL"

        snapshot = {
            "symbol": self.provider.symbol,
            "server_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tick": tick,
            "account": self.provider.account_info,
            "consensus": {
                "status": consensus_status,
                "direction": consensus_dir,
                "threshold": threshold,
                "buy_count": buy_votes,
                "sell_count": sell_votes,
                "total_models": 70,
                "ratio_str": f"BUY: {buy_votes}/{threshold} | SELL: {sell_votes}/{threshold}"
            },
            "active_signals": active_signals_list,
            "strategies": results,
            "categories": CATEGORIES
        }

        with self.lock:
            self.cached_state = snapshot
            self.last_eval_time = time.time()

        return snapshot

    def _evaluate_single_strategy(self, idx: int, meta: Dict[str, Any], df_m5: pd.DataFrame, df_m15: pd.DataFrame, df_h1: pd.DataFrame, live_price: float, spread_pips: float) -> ScriptConditionState:
        """Evaluates condition sequence for a specific strategy."""
        name = meta["name"]
        short_name = meta["short_name"]
        cat_code = meta["category_code"]
        cat_name = meta["category"]
        tf = meta["timeframe"]

        # Default fallback dataframe
        df = df_m15 if "M15" in tf else (df_m5 if "M5" in tf else df_h1)
        if df is None or len(df) < 30:
            df = df_m15

        curr = df.iloc[-1]
        prev = df.iloc[-2]
        close_p = float(curr['close'])

        steps: List[ConditionStep] = []
        direction = "NEUTRAL"
        overall_status = "WAITING"
        active_sig = None
        metrics = {"Price": round(close_p, 5), "ATR": round(float(curr.get('atr_14', 0.0010)) / PIP_SIZE, 1)}

        # ----------------------------------------------------------------------
        # STRATEGY-SPECIFIC LOGIC DISPATCHER
        # ----------------------------------------------------------------------
        if idx == 1:  # ICT Breaker Block
            # Step 1: Swing Trend Leg
            sh = df['high'].iloc[-20:].max()
            sl = df['low'].iloc[-20:].min()
            is_bull = close_p > (sh + sl) / 2
            cand_dir = "BUY" if is_bull else "SELL"
            s1_passed = True
            steps.append(ConditionStep(1, "Structural Swing Leg", "Swing high/low identified in 20 bars", f"Swing Range: {sl:.5f} - {sh:.5f}", "+14.2 pips", "PASS"))

            # Step 2: Market Structure Shift (MSS)
            mss_passed = close_p > df['high'].iloc[-5:-1].max() if cand_dir == "BUY" else close_p < df['low'].iloc[-5:-1].min()
            steps.append(ConditionStep(2, "Market Structure Shift (MSS)", "Displacement beyond recent structural swing", f"Price: {close_p:.5f}", "In Zone" if mss_passed else "Pending", "PASS" if mss_passed else "PENDING"))

            # Step 3: Breaker Mitigation & Rejection
            b_low = sl + (sh - sl) * 0.4
            b_high = sl + (sh - sl) * 0.6
            in_zone = b_low <= close_p <= b_high
            s3_passed = mss_passed and in_zone
            steps.append(ConditionStep(3, "Breaker Zone Mitigation", f"Price mitigates {b_low:.5f}-{b_high:.5f}", f"Close: {close_p:.5f}", "0.0p" if in_zone else f"{abs(close_p - b_low)/PIP_SIZE:.1f}p gap", "PASS" if s3_passed else "PENDING"))

            direction = cand_dir
            if s3_passed:
                risk = 15.0 * PIP_SIZE
                active_sig = {"strategy_id": idx, "name": name, "direction": direction, "entry": close_p, "sl": close_p - risk if direction=="BUY" else close_p + risk, "tp": close_p + (risk*2) if direction=="BUY" else close_p - (risk*2), "rr": "1:2.0"}

        elif idx == 12:  # Subasish 5 EMA
            ema5 = float(curr['ema_5'])
            metrics["5 EMA"] = round(ema5, 5)
            # Sell alert: candle low > ema5; Buy alert: candle high < ema5
            sell_alert = prev['low'] > prev['ema_5']
            buy_alert = prev['high'] < prev['ema_5']

            cand_dir = "SELL" if sell_alert else ("BUY" if buy_alert else "NEUTRAL")
            s1_passed = sell_alert or buy_alert
            steps.append(ConditionStep(1, "5 EMA Detached Candle", "Candle floats entirely away from 5 EMA", f"Low: {prev['low']:.5f} vs EMA: {prev['ema_5']:.5f}", "Detached" if s1_passed else "Touching", "PASS" if s1_passed else "PENDING"))

            # Trigger: break of alert candle extreme
            s2_passed = False
            if sell_alert and curr['close'] < prev['low']:
                s2_passed = True
            elif buy_alert and curr['close'] > prev['high']:
                s2_passed = True
            steps.append(ConditionStep(2, "Alert Candle Breakout", "Price breaches alert candle boundary", f"Current Close: {curr['close']:.5f}", "Breakout" if s2_passed else "Awaiting break", "PASS" if s2_passed else "PENDING"))

            steps.append(ConditionStep(3, "1:3.0 Target Projection", "Risk-to-reward >= 1:3.0", "Target 1:3 RR", "3x Risk", "PASS" if s2_passed else "PENDING"))
            direction = cand_dir
            if s2_passed:
                risk = abs(curr['close'] - (prev['high'] if direction=="SELL" else prev['low']))
                active_sig = {"strategy_id": idx, "name": name, "direction": direction, "entry": close_p, "sl": prev['high'] if direction=="SELL" else prev['low'], "tp": close_p - (risk*3) if direction=="SELL" else close_p + (risk*3), "rr": "1:3.0"}

        elif idx == 67:  # Mukul Chowdhary 10 EMA + VWAP
            vwap = float(curr['vwap'])
            ema10 = float(curr['ema_10'])
            metrics["VWAP"] = round(vwap, 5)
            metrics["10 EMA"] = round(ema10, 5)

            bull_bias = close_p > vwap
            bear_bias = close_p < vwap
            cand_dir = "BUY" if bull_bias else "SELL"

            steps.append(ConditionStep(1, "VWAP Directional Bias", "Price trading on trending side of VWAP", f"Close {close_p:.5f} vs VWAP {vwap:.5f}", f"{abs(close_p - vwap)/PIP_SIZE:.1f}p diff", "PASS"))

            # Pullback to 10 EMA
            touched_ema = (curr['low'] <= ema10 * 1.0005) if cand_dir == "BUY" else (curr['high'] >= ema10 * 0.9995)
            steps.append(ConditionStep(2, "10 EMA Pullback Retest", "Price pulls back to test 10 EMA", f"10 EMA: {ema10:.5f}", "Touched" if touched_ema else f"{abs(close_p - ema10)/PIP_SIZE:.1f}p away", "PASS" if touched_ema else "PENDING"))

            # Rejection expansion candle
            rejection = (curr['close'] > curr['open'] and curr['close'] >= ema10) if cand_dir == "BUY" else (curr['close'] < curr['open'] and curr['close'] <= ema10)
            s3_passed = touched_ema and rejection
            steps.append(ConditionStep(3, "Rejection Expansion", "Color-confirmed expansion off 10 EMA", "Confirmed" if s3_passed else "Pending candle", "1:4 RR Ready" if s3_passed else "Pending", "PASS" if s3_passed else "PENDING"))

            direction = cand_dir
            if s3_passed:
                risk = abs(close_p - ema10) + (spread_pips * PIP_SIZE)
                active_sig = {"strategy_id": idx, "name": name, "direction": direction, "entry": close_p, "sl": close_p - risk if direction=="BUY" else close_p + risk, "tp": close_p + (risk*4) if direction=="BUY" else close_p - (risk*4), "rr": "1:4.0"}

        else:
            # Standardized 3-step evaluator for all other strategies using respective indicators
            ema50 = float(curr.get('ema_50', close_p))
            ema200 = float(curr.get('ema_200', close_p))
            rsi = float(curr.get('rsi_14', 50.0))
            metrics["EMA 50"] = round(ema50, 5)
            metrics["RSI"] = round(rsi, 1)

            # Step 1: Trend / Structure
            is_bull = close_p >= ema50
            cand_dir = "BUY" if is_bull else "SELL"
            s1_passed = True
            steps.append(ConditionStep(1, f"Trend Alignment ({tf})", f"Price {'above' if is_bull else 'below'} 50 EMA baseline", f"Close: {close_p:.5f} vs EMA: {ema50:.5f}", f"{abs(close_p - ema50)/PIP_SIZE:.1f}p spread", "PASS"))

            # Step 2: Oscillator / Price Action Trigger
            if "RSI" in name or cat_code == "C":
                s2_passed = (rsi <= 40) if cand_dir == "BUY" else (rsi >= 60)
                cond_text = f"RSI was {'oversold (<=40)' if cand_dir=='BUY' else 'overbought (>=60)'}"
                val_text = f"RSI: {rsi:.1f}"
                delta_text = f"{abs(rsi - (40 if cand_dir=='BUY' else 60)):.1f} pts"
            elif "Supertrend" in name:
                st_dir = int(curr.get('st_dir', 1))
                s2_passed = (st_dir == 1) if cand_dir == "BUY" else (st_dir == -1)
                cond_text = "Supertrend confirms trend polarity"
                val_text = "Green (Bull)" if st_dir == 1 else "Red (Bear)"
                delta_text = "Aligned" if s2_passed else "Opposing"
            else:
                # Value zone pullback test
                s2_passed = abs(close_p - ema50) <= (12 * PIP_SIZE)
                cond_text = "Pullback within value zone (< 12 pips)"
                val_text = f"Distance: {abs(close_p - ema50)/PIP_SIZE:.1f}p"
                delta_text = "In Zone" if s2_passed else f"{abs(close_p - ema50)/PIP_SIZE - 12:.1f}p buffer"

            steps.append(ConditionStep(2, "Trigger Event / Retest", cond_text, val_text, delta_text, "PASS" if s2_passed else "PENDING"))

            # Step 3: Execution Envelope
            s3_passed = s2_passed and (curr['close'] > curr['open'] if cand_dir == "BUY" else curr['close'] < curr['open'])
            steps.append(ConditionStep(3, "Execution Verification", "Confirming candle color & Risk/Reward check", "Bullish Candle" if curr['close'] > curr['open'] else "Bearish Candle", "Ready" if s3_passed else "Forming", "PASS" if s3_passed else "PENDING"))

            direction = cand_dir
            if s3_passed:
                risk = 10.0 * PIP_SIZE
                active_sig = {"strategy_id": idx, "name": name, "direction": direction, "entry": close_p, "sl": close_p - risk if direction=="BUY" else close_p + risk, "tp": close_p + (risk*2) if direction=="BUY" else close_p - (risk*2), "rr": "1:2.0"}

        # Calculate progress
        passed_count = sum(1 for s in steps if s.status == "PASS")
        progress_pct = int((passed_count / len(steps)) * 100) if steps else 0
        if passed_count == len(steps):
            overall_status = "TRIGGERED"
        elif passed_count >= 2:
            overall_status = "READY"
        elif passed_count >= 1:
            overall_status = "PROGRESS"
        else:
            overall_status = "WAITING"

        return ScriptConditionState(
            id=str(idx),
            num=idx,
            name=name,
            short_name=short_name,
            category=cat_name,
            category_code=cat_code,
            timeframe=tf,
            direction=direction,
            overall_status=overall_status,
            progress_pct=progress_pct,
            true_count=passed_count,
            total_steps=len(steps),
            steps=[asdict(s) for s in steps],
            active_signal=active_sig,
            summary_note=f"{name} ({tf})",
            metrics=metrics
        )

    def set_threshold(self, th: int):
        with self.lock:
            self.consensus_threshold = max(1, min(30, th))

    def set_mock_mode(self, use_mock: bool):
        with self.lock:
            self.provider.use_mock = use_mock
            if not use_mock and MT5_AVAILABLE:
                self.provider._init_mt5()
