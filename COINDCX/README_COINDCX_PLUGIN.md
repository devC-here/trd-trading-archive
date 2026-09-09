# CoinDCX Plugin

This workspace now has a small CoinDCX futures connector that can reuse the existing model/fusion stack without requiring MT5 for execution.

## Files

- `coindcx_plugin.py` - market data, signed CoinDCX REST calls, and a dry-run-first execution plugin.
- `coindcx_runner.py` - a CoinDCX-only runner that loads your current predictors and places futures orders through the plugin.
- `cdx.py` - compatibility wrapper used by the existing predictor modules.

## Safe Default

Live trading is disabled by default. If you run the CoinDCX runner without setting `COINDCX_LIVE_TRADING=true`, it only logs a dry-run order preview.

## Environment

Set these before running live:

```powershell
$env:COINDCX_API_KEY="your_api_key"
$env:COINDCX_API_SECRET="your_api_secret"
$env:COINDCX_SYMBOL="B-BTC_USDT"
$env:COINDCX_TIMEFRAME="15"
$env:COINDCX_QUANTITY="0.001"
$env:COINDCX_MARGIN_CURRENCY="USDT"
$env:COINDCX_LEVERAGE="1"
$env:COINDCX_LIVE_TRADING="true"
```

For dry-run testing, leave `COINDCX_LIVE_TRADING` unset or set it to `false`.

## Run

```powershell
python coindcx_runner.py
```

## Use As A Library

```python
from coindcx_plugin import CoinDCXFuturesPlugin

plugin = CoinDCXFuturesPlugin()
candles = plugin.fetch_candles("B-BTC_USDT", "15", num_candles=100)
preview = plugin.execute_trade("B-BTC_USDT", "Buy", quantity="0.001")
print(candles.tail())
print(preview)
```

## Notes

- The connector targets CoinDCX futures because your strategy uses both buy and sell direction like an MT5 system.
- The plugin signs authenticated requests with HMAC-SHA256 and sends the CoinDCX timestamp in milliseconds.
- Market orders intentionally omit `time_in_force`, matching CoinDCX futures guidance.
