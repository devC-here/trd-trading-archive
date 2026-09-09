from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests


API_BASE_URL = "https://api.coindcx.com"
PUBLIC_BASE_URL = "https://public.coindcx.com"
CANDLE_COLUMNS = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
TIMEFRAME_SECONDS = {
    "1": 60,
    "5": 300,
    "15": 900,
    "30": 1800,
    "60": 3600,
    "120": 7200,
    "240": 14400,
    "1D": 86400,
}


class CoinDCXError(RuntimeError):
    """Base exception for CoinDCX connector failures."""


class CoinDCXCredentialsError(CoinDCXError):
    """Raised when a signed endpoint is requested without API credentials."""


class CoinDCXHTTPError(CoinDCXError):
    """Raised when CoinDCX returns a non-successful HTTP response."""


class CoinDCXLiveTradingDisabled(CoinDCXError):
    """Raised when a live order is requested while dry-run mode is enabled."""


def _clean_env(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_bool(name: str, default: bool) -> bool:
    value = _clean_env(os.getenv(name))
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def normalize_timeframe(timeframe: Any) -> str:
    """Normalize MT5-like and CoinDCX-like timeframe values to CoinDCX futures resolution."""
    if timeframe == "1d":
        return "1D"
    value = str(timeframe).strip()
    interval_map = {
        "1m": "1",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "2h": "120",
        "4h": "240",
        "1d": "1D",
    }
    if value in interval_map:
        return interval_map[value]
    if value in TIMEFRAME_SECONDS:
        return value
    mt5_name_map = {
        "TIMEFRAME_M1": "1",
        "TIMEFRAME_M5": "5",
        "TIMEFRAME_M15": "15",
        "TIMEFRAME_M30": "30",
        "TIMEFRAME_H1": "60",
        "TIMEFRAME_H2": "120",
        "TIMEFRAME_H4": "240",
        "TIMEFRAME_D1": "1D",
        "16385": "60",
        "16386": "120",
        "16388": "240",
        "16408": "1D",
    }
    if value in mt5_name_map:
        return mt5_name_map[value]
    raise ValueError(f"Unsupported CoinDCX timeframe: {timeframe!r}")


def decision_to_side(decision: str) -> str | None:
    normalized = decision.strip().lower().replace("_", " ")
    if normalized in {"buy", "long"}:
        return "buy"
    if normalized in {"sell", "short"}:
        return "sell"
    if normalized in {"hold", "no trade", "none", "neutral"}:
        return None
    raise ValueError(f"Unsupported trade decision: {decision!r}")


def position_side(position: dict[str, Any]) -> str | None:
    active_pos = _safe_float(position.get("active_pos"))
    if active_pos > 0:
        return "buy"
    if active_pos < 0:
        return "sell"
    return None


@dataclass(frozen=True)
class CoinDCXConfig:
    api_key: str | None = None
    api_secret: str | None = None
    api_base_url: str = API_BASE_URL
    public_base_url: str = PUBLIC_BASE_URL
    request_timeout: float = 10.0
    dry_run: bool = True
    default_pair: str = "B-BTC_USDT"
    default_timeframe: str = "15"
    default_quantity: str | None = None
    default_leverage: int | None = None
    margin_currency_short_name: str = "USDT"
    notification: str = "no_notification"
    position_margin_type: str | None = None

    @classmethod
    def from_env(cls) -> "CoinDCXConfig":
        leverage = _clean_env(os.getenv("COINDCX_LEVERAGE"))
        timeout = _clean_env(os.getenv("COINDCX_REQUEST_TIMEOUT"))
        return cls(
            api_key=_clean_env(os.getenv("COINDCX_API_KEY")),
            api_secret=_clean_env(os.getenv("COINDCX_API_SECRET")),
            api_base_url=_clean_env(os.getenv("COINDCX_API_BASE_URL")) or API_BASE_URL,
            public_base_url=_clean_env(os.getenv("COINDCX_PUBLIC_BASE_URL")) or PUBLIC_BASE_URL,
            request_timeout=float(timeout) if timeout else 10.0,
            dry_run=not _env_bool("COINDCX_LIVE_TRADING", False),
            default_pair=_clean_env(os.getenv("COINDCX_SYMBOL")) or "B-BTC_USDT",
            default_timeframe=normalize_timeframe(_clean_env(os.getenv("COINDCX_TIMEFRAME")) or "15"),
            default_quantity=_clean_env(os.getenv("COINDCX_QUANTITY")),
            default_leverage=int(leverage) if leverage else None,
            margin_currency_short_name=_clean_env(os.getenv("COINDCX_MARGIN_CURRENCY")) or "USDT",
            notification=_clean_env(os.getenv("COINDCX_NOTIFICATION")) or "no_notification",
            position_margin_type=_clean_env(os.getenv("COINDCX_POSITION_MARGIN_TYPE")),
        )

    @property
    def has_credentials(self) -> bool:
        return bool(self.api_key and self.api_secret)


class CoinDCXClient:
    def __init__(self, config: CoinDCXConfig | None = None, session: requests.Session | None = None):
        self.config = config or CoinDCXConfig.from_env()
        self.session = session or requests.Session()

    def public_get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        response = self.session.get(
            self.config.public_base_url + path,
            params=params,
            timeout=self.config.request_timeout,
        )
        return self._parse_response(response)

    def signed_post(self, path: str, body: dict[str, Any] | None = None) -> Any:
        if not self.config.has_credentials:
            raise CoinDCXCredentialsError("COINDCX_API_KEY and COINDCX_API_SECRET are required.")

        payload = dict(body or {})
        payload.setdefault("timestamp", int(round(time.time() * 1000)))
        json_body = json.dumps(payload, separators=(",", ":"))
        secret_bytes = self.config.api_secret.encode("utf-8")
        signature = hmac.new(secret_bytes, json_body.encode("utf-8"), hashlib.sha256).hexdigest()
        headers = {
            "Content-Type": "application/json",
            "X-AUTH-APIKEY": self.config.api_key or "",
            "X-AUTH-SIGNATURE": signature,
        }

        response = self.session.post(
            self.config.api_base_url + path,
            data=json_body,
            headers=headers,
            timeout=self.config.request_timeout,
        )
        return self._parse_response(response)

    @staticmethod
    def _parse_response(response: requests.Response) -> Any:
        if not response.ok:
            raise CoinDCXHTTPError(f"CoinDCX HTTP {response.status_code}: {response.text[:500]}")
        try:
            return response.json()
        except ValueError as exc:
            raise CoinDCXHTTPError(f"CoinDCX returned non-JSON response: {response.text[:500]}") from exc

    def fetch_futures_candles(
        self,
        symbol: str | None = None,
        timeframe: Any = "15",
        start_index: int = 1,
        num_candles: int = 100,
    ) -> pd.DataFrame:
        resolution = normalize_timeframe(timeframe)
        tf_seconds = TIMEFRAME_SECONDS[resolution]
        pair = symbol or self.config.default_pair

        skip_completed = max(int(start_index) - 1, 0)
        to_ts = int(time.time()) - 10 - (skip_completed * tf_seconds)
        from_ts = to_ts - (int(num_candles) * tf_seconds)
        payload = self.public_get(
            "/market_data/candlesticks",
            params={"pair": pair, "from": from_ts, "to": to_ts, "resolution": resolution, "pcode": "f"},
        )

        raw_candles = payload.get("data", []) if isinstance(payload, dict) else payload
        if not raw_candles:
            return pd.DataFrame(columns=CANDLE_COLUMNS)

        rows = []
        for candle in raw_candles:
            open_price = _safe_float(candle.get("open"))
            close_price = _safe_float(candle.get("close"))
            timestamp = int(candle["time"])
            if timestamp > 10_000_000_000:
                timestamp = timestamp // 1000
            rows.append(
                (
                    timestamp,
                    open_price,
                    _safe_float(candle.get("high")),
                    _safe_float(candle.get("low")),
                    close_price,
                    _safe_float(candle.get("volume")),
                    abs(open_price - close_price),
                    0,
                )
            )

        df = pd.DataFrame(rows, columns=CANDLE_COLUMNS)
        return df.sort_values("time").drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)

    def list_futures_positions(
        self,
        pairs: str | None = None,
        page: str = "1",
        size: str = "10",
        margin_currency_short_name: str | None = None,
    ) -> list[dict[str, Any]]:
        body: dict[str, Any] = {
            "page": page,
            "size": size,
            "margin_currency_short_name": [margin_currency_short_name or self.config.margin_currency_short_name],
        }
        if pairs:
            body["pairs"] = pairs
        response = self.signed_post("/exchange/v1/derivatives/futures/positions", body)
        return response if isinstance(response, list) else []

    def create_futures_order(
        self,
        side: str,
        pair: str | None = None,
        total_quantity: str | float | int | None = None,
        order_type: str = "market_order",
        price: str | float | int | None = None,
        stop_price: str | float | int | None = None,
        leverage: int | None = None,
        take_profit_price: str | float | int | None = None,
        stop_loss_price: str | float | int | None = None,
    ) -> Any:
        if total_quantity is None:
            total_quantity = self.config.default_quantity
        if total_quantity is None:
            raise ValueError("total_quantity is required. Set COINDCX_QUANTITY or pass quantity explicitly.")

        order: dict[str, Any] = {
            "side": side,
            "pair": pair or self.config.default_pair,
            "order_type": order_type,
            "total_quantity": total_quantity,
            "notification": self.config.notification,
            "margin_currency_short_name": self.config.margin_currency_short_name,
            "hidden": False,
            "post_only": False,
        }
        if order_type == "market_order":
            order["price"] = None
            order["stop_price"] = None
        if leverage or self.config.default_leverage:
            order["leverage"] = leverage or self.config.default_leverage
        if self.config.position_margin_type:
            order["position_margin_type"] = self.config.position_margin_type
        if price is not None:
            order["price"] = price
        if stop_price is not None:
            order["stop_price"] = stop_price
        if take_profit_price is not None:
            order["take_profit_price"] = take_profit_price
        if stop_loss_price is not None:
            order["stop_loss_price"] = stop_loss_price

        return self.signed_post("/exchange/v1/derivatives/futures/orders/create", {"order": order})

    def exit_futures_position(self, position_id: str) -> Any:
        return self.signed_post("/exchange/v1/derivatives/futures/positions/exit", {"id": position_id})


class CoinDCXFuturesPlugin:
    """Thin execution plugin that maps strategy decisions to CoinDCX futures REST calls."""

    def __init__(self, config: CoinDCXConfig | None = None, client: CoinDCXClient | None = None):
        self.config = config or CoinDCXConfig.from_env()
        self.client = client or CoinDCXClient(self.config)

    def fetch_candles(self, symbol: str | None = None, timeframe: Any = "15", start_index: int = 1, num_candles: int = 100) -> pd.DataFrame:
        return self.client.fetch_futures_candles(symbol, timeframe, start_index, num_candles)

    def get_positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        return self.client.list_futures_positions(pairs=symbol or self.config.default_pair)

    def exit_position(self, position: dict[str, Any]) -> Any:
        position_id = position.get("id")
        if not position_id:
            raise ValueError("Position does not include an id.")
        if self.config.dry_run:
            return {"dry_run": True, "action": "exit_position", "position_id": position_id, "position": position}
        return self.client.exit_futures_position(str(position_id))

    def execute_trade(
        self,
        symbol: str | None,
        decision: str,
        quantity: str | float | int | None = None,
        sl: str | float | int | None = None,
        tp: str | float | int | None = None,
    ) -> Any:
        side = decision_to_side(decision)
        if side is None:
            return {"skipped": True, "reason": "hold_or_no_trade", "decision": decision}

        order_preview = {
            "side": side,
            "pair": symbol or self.config.default_pair,
            "total_quantity": quantity or self.config.default_quantity,
            "order_type": "market_order",
            "take_profit_price": tp,
            "stop_loss_price": sl,
        }
        if self.config.dry_run:
            return {"dry_run": True, "action": "create_futures_order", "order": order_preview}

        if not self.config.has_credentials:
            raise CoinDCXCredentialsError("Live trading requires COINDCX_API_KEY and COINDCX_API_SECRET.")

        return self.client.create_futures_order(
            side=side,
            pair=symbol,
            total_quantity=quantity,
            order_type="market_order",
            take_profit_price=tp,
            stop_loss_price=sl,
        )


def fetch_candles(symbol: str = "B-BTC_USDT", timeframe: Any = "15", start_index: int = 1, num_candles: int = 100) -> pd.DataFrame:
    """Compatibility wrapper used by the existing predictor modules."""
    return CoinDCXClient(CoinDCXConfig.from_env()).fetch_futures_candles(symbol, timeframe, start_index, num_candles)
