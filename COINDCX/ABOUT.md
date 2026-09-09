# COINDCX: Crypto Futures Execution & Model Fusion Hub

## Overview
This directory houses the live CoinDCX exchange runner, model deployers, and prediction fusion orchestrator.

## Key Files & Modules
- **`coindcx_runner.py`**: Main live loop fetching candles, invoking predictors, calculating fusion scores, and triggering orders.
- **`coindcx_plugin.py`**: Signed CoinDCX REST futures client with dry-run safety gates and HMAC-SHA256 authentication.
- **`fusion.py`**: Multi-model consensus combiner aggregating signals from Momentum, Breakout, Microstructure, VWAP, and Candlestick sequence models.
- **`override_engine.py`**: 700-line deterministic risk guardrail engine checking for microstructure traps, liquidity sweeps, and trend chop.
- **`breakoutdeploy.py`, `momentumdeploy.py`, `VWdeploy.py`, `candledeploy.py`, `microstructure.py`**: Individual predictor inference wrappers loading PyTorch models (`.pth`) and scalers (`.pkl`).

## Purpose & Strategy
Bridges machine learning predictions into live or dry-run execution on crypto futures pairs (e.g., `B-BTC_USDT`).
