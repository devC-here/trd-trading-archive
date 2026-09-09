# METATRADER5 (V5): Current Stable Production Release

## Overview
The latest and most refined production build of the 5-model neural MetaTrader 5 trading system.

## Complete Model Ensemble
- **Breakout Predictor**: `breakout_model.pth` & `breakout_scaler.pkl`
- **Candlestick Ratios**: `candle_ratios_model.pth`
- **Sequence Classifier**: `candle_seq_model.pth` & `candle_seq_scaler.pkl`
- **Microstructure Filter**: `microstructure_model.pth` & `microstructure_scaler.pkl`
- **Momentum Predictor**: `momentum_model.pth` & `momentum_scaler.pkl`
- **VWAP Volume Flow**: `vwap_volume_model.pth` & `vwap_volume_scaler.pkl`

## Operational Pipeline
- **`MAIN.py`**: Primary live loop coordinating data ingestion, inference, risk checking, and execution.
- **`override_engine.py`**: Full 40+ condition deterministic guardrail engine.
- **`sltp.py`**: Volatility-adjusted Stop Loss / Take Profit targeting.
- **`placer.py`**: Direct MetaTrader 5 order dispatcher.
