# METATRADER5 (V3): Telemetry & Prediction Logging Suite

## Overview
Third generation of the MetaTrader 5 live execution package, introducing comprehensive prediction logging, telemetry, and dry-run output auditing.

## Key Additions over V2
- **`predictions.json`**: Live rolling telemetry log capturing raw model outputs, probabilities, and fusion scores for post-trade analysis.
- **`modeloutputonly.py` & `test-output.py`**: Observation-only modes for forward-testing model accuracy without placing broker orders.
- **`code1.py`**: Performance-optimized utility pipeline for fast feature calculation.
