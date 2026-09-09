# 5M-BACKTEST: Multi-Timeframe Strategy Simulation Engine

## Overview
This folder contains historical backtesting frameworks and trade simulation engines designed to evaluate trading edges across the 5-minute (M5), 15-minute (M15), and 1-hour (H1) timeframes.

## Subdirectories & Components
- **`METATRADER-BACKTEST/`**: Backtesting harness connecting strategy rules directly to MT5 historical tick and bar feeds.
- **`METATRADER-V3-BACKTEST/`**: Enhanced V3 backtester with trailing stop simulation, spread modeling, and multi-asset metrics.
- **`15M-H1-BACKTEST/`**: Multi-timeframe trend alignment backtest simulating higher-timeframe H1 bias with M15/M5 precision entries.

## Purpose & Strategy
Validates model predictions, win rates, maximum drawdown (MDD), profit factor, and Sharpe ratios against historical candle data prior to paper or live execution.
