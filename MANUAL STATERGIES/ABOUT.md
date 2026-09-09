# MANUAL STATERGIES: Candlestick Geometry & Discretionary Rule Sets

## Overview
Focuses on candlestick anatomy, ratio analysis, and discretionary trading strategy codification.

## Key Files
- **`classifier.py`**: Classifies individual candle geometries (wicks, body ratios, absorptions, engulfing patterns).
- **`candle_ratios_model_best.pth` & `candle_ratios_scaler.pkl`**: Pre-trained model scoring candle quality and directional dominance.
- **`MAIN.py`**: Standalone execution harness applying candlestick classification rules to live feeds.

## Purpose & Strategy
Quantifies human discretionary price-action concepts (rejections, exhaustion bars, wick absorptions) into mathematical features readable by automated trading systems.
