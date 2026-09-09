# DATA SYNTHESIS AND MANIPULATION: Synthetic Market Generator

## Overview
A data generation suite designed to synthesize realistic financial price action, volatility regimes, and order book distributions to train neural networks without memorizing historical market noise.

## Key Files
- **`syntheticdata.py`**: Generates synthetic OHLCV sequences with customizable drift, volatility spikes, and liquidity compression squeezes.
- **`composition.py` & `sepration.py`**: Splits, balances, and composes balanced training/validation datasets across Long, Short, and No-Trade classes.
- **`converter.py`**: Normalizes and formats raw historical ticks into model-compatible feature matrices.

## Purpose & Strategy
Addresses the fundamental problem of financial ML: limited historical data leading to overfitting. By training models on thousands of synthetically generated regimes, models learn invariant market mechanics rather than specific historical dates.
