# HELPING SCRIPTS: Algorithmic Trading Utilities

## Overview
A collection of support utilities handling automated data fetching, position management, and dynamic stop-loss / take-profit calculations.

## Key Files
- **`DATA FETCH.py`**: Automated script for downloading and caching clean OHLCV candle datasets from broker APIs.
- **`sltp-V2.py`**: Dynamic risk/reward calculator calculating ATR-based Stop Loss and Take Profit levels.
- **`manage.py`**: Trade state manager monitoring active positions and managing trailing stop ratchets.

## Purpose
Provides foundational helper functions used across backtests, model training pipelines, and live runners.
