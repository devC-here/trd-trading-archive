"""
========================================================================================
H1-ONLINE PIPELINE CLIENT SDK
========================================================================================
Direct programmatic bridge between your local computer and the cloud engine.
Fetch live pandas DataFrames, pre-calculated indicator matrices, master swing decisions,
and demo trade journal records with 1 line of Python.
========================================================================================
"""

import json
import urllib.request
import urllib.parse
from typing import Dict, Any, Optional, List
import pandas as pd


class H1OnlinePipeline:
    """Internal client pipeline for local quantitative research and analysis."""

    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url.rstrip("/")

    def _get_json(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        if params:
            url += f"?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "H1PipelineSDK/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def check_health(self) -> bool:
        """Checks if the cloud/local engine is live and healthy."""
        try:
            res = self._get_json("/healthz")
            return res.get("status") == "healthy"
        except Exception:
            return False

    def get_rates_df(self, timeframe: str = "H1", count: int = 150) -> pd.DataFrame:
        """
        Fetches historical candles and pre-calculated technical indicators as a pandas DataFrame.
        Supported timeframes: 'H1', 'H2', 'H4', 'D1'.
        """
        data = self._get_json("/api/pipeline/rates", {"tf": timeframe, "count": count})
        if data.get("success"):
            df = pd.DataFrame(data.get("rates", []))
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"])
            return df
        return pd.DataFrame()

    def get_master_decision(self) -> Dict[str, Any]:
        """
        Fetches the current master actionable swing decision, confluence score,
        and Fresh Inflection timing status.
        """
        snap = self._get_json("/api/status")
        return snap.get("master_decision", {})

    def get_demo_portfolio(self) -> Dict[str, Any]:
        """Fetches the automated demo trade ledger, win rate %, and active virtual positions."""
        return self._get_json("/api/demo-trades")

    def get_demo_trades_df(self) -> pd.DataFrame:
        """Fetches all closed demo swing trades as a clean pandas DataFrame."""
        summary = self.get_demo_portfolio()
        closed = summary.get("recent_closed_trades", [])
        return pd.DataFrame(closed)

    def get_strategies_df(self) -> pd.DataFrame:
        """Returns all 60 curated H1/H2 models and their current trigger state."""
        snap = self._get_json("/api/status")
        strats = snap.get("strategies", [])
        clean = []
        for s in strats:
            clean.append({
                "id": s.get("num"),
                "name": s.get("name"),
                "category": s.get("category"),
                "timeframe": s.get("timeframe"),
                "direction": s.get("direction"),
                "status": s.get("overall_status"),
                "progress_pct": s.get("progress_pct"),
                "steps_passed": f"{s.get('true_count')}/{s.get('total_steps')}"
            })
        return pd.DataFrame(clean)


if __name__ == "__main__":
    print("Testing H1OnlinePipeline client...")
    client = H1OnlinePipeline("http://127.0.0.1:5000")
    if client.check_health():
        print("[+] Connected to H1-ONLINE server!")
        df = client.get_rates_df("H2", 5)
        print("H2 Rates shape:", df.shape)
        decision = client.get_master_decision()
        print("Decision:", decision.get("action"), "| Confluence:", decision.get("confluence_score"))
        port = client.get_demo_portfolio()
        print(f"Portfolio: {port.get('win_rate_pct')}% Win Rate, {port.get('net_pips_banked')} Net Pips Banked.")
    else:
        print("[!] Server not responding on localhost:5000 (start gui_server.py first).")
