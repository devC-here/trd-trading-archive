# METATRADER5 (V1): Foundational MetaTrader Production Release

## Overview
The baseline production-grade MetaTrader 5 live execution package.

## Included Models
- `breakout_model.pth` (Breakout expansion)
- `candle_ratios_model.pth` (Candle wick & body geometry)
- `candle_seq_model.pth` (Candlestick sequence continuation/reversal)
- `microstructure_model.pth` (Market microstructure trap detection)
- `momentum_model.pth` (Directional velocity)
- `vwap_volume_model.pth` (Volume flow relative to VWAP)

## Architecture
- **`MAIN.py`**: Live polling loop reading MT5 chart bars and executing signals.
- **`fusion.py`**: Model output scoring and synthesis.
- **`override_engine.py`**: Initial deterministic risk filter set.
- **`placer.py`**: Broker order execution.
