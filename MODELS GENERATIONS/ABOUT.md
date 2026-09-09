# MODELS GENERATIONS: Neural Network Evolution & Training Checkpoints

## Overview
The primary model research laboratory tracking the iterative evolution of neural network architectures, training experiments, and loss function tuning across multiple development generations.

## Key Generations & Subdirectories
- **`H1-BASE/`**: **ACTIVE WORKING SETUP & PROPRIETARY EDGE**. Features Forex ICT Daily Order Block (OB) and H1 Fair Value Gap (FVG) scanners, strategy transcripts, and multi-timeframe execution.
- **`CLAUDE-BUILDV1` through `CLAUDE-V4 (15M-H1)` & `(H1-H4)`**: Iterations of neural networks trained on 15M and 1-hour timeframes with custom loss weightings.
- **`5-MODEL-BUILD/`**: Pent-model training scripts and PyTorch weights.
- **`ARCHIVE ( 4H candle pattern)/`**: 4-hour higher-timeframe candle pattern models.
- **`TICK-TRADE ( eurusd - dxyusd)/`**: High-frequency tick models trained on intermarket currency data.

## Purpose
Acts as the central repository of trained weights (`.pth`), feature normalizers (`.pkl`), and training logs across the lifetime of the project.
