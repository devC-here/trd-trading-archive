# CLAUDE-V4 (15M-H1): Multi-Timeframe Production Suite

## Overview
A multi-timeframe deployment system that aligns higher-timeframe H1 directional trends with lower-timeframe 15-minute (M15) execution timing.

## Core Features
- **`ensemble_h1.py`**: Multi-model consensus aggregator combining 5 neural models.
- **`run_all_models.py`**: Automated orchestrator executing parallel inference on all 5 H1 models.
- **Deployers**:
  - `deploy_breakout_h1.py`
  - `deploy_candle_h1.py`
  - `deploy_microstructure_h1.py`
  - `deploy_momentum_h1.py`
  - `deploy_vwap_h1.py`
- **Execution**: `placer.py` with `sltp.py` risk calculation.
