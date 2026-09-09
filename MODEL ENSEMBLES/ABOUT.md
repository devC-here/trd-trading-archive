# MODEL ENSEMBLES: Multi-Model Fusion Frameworks

## Overview
Comprehensive research and deployment architectures testing various combinations of neural network models to achieve consensus.

## Subdirectories
- **`SINGLE/`**: Baseline performance of single standalone models (Momentum only, Breakout only, etc.).
- **`DUAL/`**: Pairwise consensus models (e.g., Momentum + Breakout).
- **`TRIPLE/`**: Tri-model ensembles (Breakout + Momentum + Microstructure).
- **`QUAD/`**: Four-model voting ensembles (adding VWAP / Volume flow).
- **`PENT/`**: Five-model architectures incorporating Candlestick Sequence analysis.
- **`HEXA/` & `SEPT/`**: High-dimensional voting ensembles incorporating intermarket and multi-timeframe filters.

## Purpose & Strategy
Tests whether combining uncorrelated sub-models improves the Sharpe ratio and reduces maximum drawdown by requiring a supermajority vote before entering a trade.
