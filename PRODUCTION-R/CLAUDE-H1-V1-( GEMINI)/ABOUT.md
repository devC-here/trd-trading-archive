# CLAUDE-H1-V1 (GEMINI): H1 Market Physics Production Runner

## Overview
A live production deployment suite driven by 1-hour (H1) models and market physics indicators.

## Models & Deployers
- **`breakout_h1_best.pth` & `breakout_h1_scaler.pkl`**: H1 volatility expansion model.
- **`candle_h1_best.pth` & `candle_h1_scaler.pkl`**: H1 candle pattern dominance model.
- **`microstructure_h1_best.pth` & `microstructure_h1_scaler.pkl`**: H1 trap and absorption detection.
- **`momentum_h1_best.pth` & `momentum_h1_scaler.pkl`**: H1 directional trend momentum model.
- **`vwap_h1_best.pth` & `vwap_h1_scaler.pkl`**: H1 volume-weighted average price pressure model.

## Operational Scripts
- **`market_physics_h1.py`**: Computes continuous market physics indicators (velocity, acceleration, spread pressure).
- **`aggregator.py`**: Multi-model signal voting and consensus engine.
- **`placer.py` & `sltp.py`**: Order submission and dynamic ATR stop-loss / take-profit manager.
