# METATRADER5 (V4): Low-Latency Execution & Override Refinement

## Overview
Fourth iteration of the MetaTrader 5 execution suite, optimized for execution speed and lower slippage.

## Key Features
- **Refined `override_engine.py`**: Streamlined condition checking with improved trap recognition logic.
- **Optimized Model Scalers**: Compact serializations for sub-10ms feature normalization.
- **Modular Pipeline**: Clean separation between broker bridge (`placer.py`), risk guardrails (`sltp.py`), and model inference.
