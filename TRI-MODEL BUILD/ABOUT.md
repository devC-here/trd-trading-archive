# TRI-MODEL BUILD: Composite Multi-Head Neural Architecture

## Overview
The dedicated training and deployment ground for the three-pillar neural architecture combining Breakout, Momentum, and Synthetic Data.

## Key Files
- **`neural-breakout.py`**: PyTorch Residual Multi-Head network predicting breakout regime, bias, and probability.
- **`neural-momentum.py`**: Multi-head momentum network predicting trend slope, momentum quality, and continuation likelihood.
- **`syntheticdata-breakout.py` & `syntheticdata-momentum.py`**: Domain-specific synthetic data generation engines.
- **`breakoutdeploy.py` & `momentumdeploy.py`**: Production inference modules with RobustScaler transformations.

## Purpose & Strategy
Demonstrates a production-grade multi-task learning system designed to eliminate financial market curve-fitting.
