# CORELATION-TRD: Cross-Asset Intermarket Pressure Scanner

## Overview
Analyzes real-time intermarket correlations between EUR/USD and the US Dollar Index (DXY) to detect institutional capital flow leads and divergences.

## Key Files
- **`dollar-index.py`**: Live monitor tracking DXY price action and momentum slope.
- **`MAIN1.py`**: Intermarket correlation scanner comparing tick pressure across paired assets.
- **`tick_pressure.json` & `tick_pressure_log.jsonl`**: Real-time tick frequency and delta imbalance logs.

## Purpose & Strategy
Exploits lead-lag relationships where DXY dollar strength leads currency pair reversals (EURUSD, GBPUSD), confirming genuine directional breakouts and filtering false moves.
