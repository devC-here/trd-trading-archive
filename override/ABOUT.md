# override: Standalone Risk Filter Iteration

## Overview
A dedicated development workspace focused specifically on developing and testing deterministic override guardrails.

## Key Files
- **`3candles.py`**: Triple-candle sequence evaluation logic.
- **`override_engine.py`**: Rule-based condition checker mapping multi-model states to execution permissions.
- **Model checkpoints & scalers**: Local model weights used during rule calibration.

## Purpose
Ensures that no trade is taken during unfavorable market microstructures, even if an individual neural network emits a buy or sell signal.
