# METATRADER5 (V2): Risk-Enhanced Release

## Overview
Second generation of the MetaTrader 5 live execution package, adding dynamic risk management and connectivity testing.

## Key Additions over V1
- **`sltp.py`**: Integrated dynamic Stop Loss and Take Profit engine based on Average True Range (ATR).
- **`test.py`**: Automated pre-flight health check verifying broker connection and spread conditions before opening live loops.
- **Model Checkpoints**: Calibrated model weights for higher accuracy on EUR/USD and major Forex pairs.
