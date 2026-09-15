# 🏛️ TRD: The Quantitative Trading & Neural Research Archive

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MetaTrader 5](https://img.shields.io/badge/MetaTrader-5-2563EB?logo=metatrader5&logoColor=white)](https://mql5.com)
[![Status](https://img.shields.io/badge/Status-Research%20Archive%20%26%20Monorepo-10b981)](https://github.com/devC-here)

Welcome to the **TRD Quantitative Research Vault**. This monorepo preserves the comprehensive research, backtesting, model evolution checkpoints, and execution systems engineered across MetaTrader 5 and CoinDCX.

---

## 🌐 Specialized Standalone Ecosystem Repositories

To ensure modularity, high performance, and clarity, core production engines have been decoupled from this research monorepo into dedicated, specialized repositories:

| Flagship Repository | Domain | Core Capabilities & Architecture |
|---|:---:|---|
| **[`h1-online-trading`](https://github.com/devC-here/h1-online-trading)** | **Forex** | **70-Strategy Swing Ensemble & Azure 24/7 Cloud Host**<br>• Real-time multi-model voting with asymmetric R:R (2.0x–4.0x).<br>• Low-RAM SQLite WAL persistence engine (< 50MB RAM).<br>• Dedicated Private Data Bridge API (Port 5001) & modern dark web GUI. |
| **[`neural-market-prediction`](https://github.com/devC-here/neural-market-prediction)** | **Deep Learning** | **Multi-Head PyTorch Neural Predictors & Risk Engine**<br>• Simultaneous Volatility Regime, Directional Bias & Strength prediction.<br>• 700-line deterministic rule-based override guardrails against chop.<br>• Microstructure, Momentum, and Candlestick Sequence sub-models. |
| **[`coindcx-mt5-bridge`](https://github.com/devC-here/coindcx-mt5-bridge)** | **Cross-Market** | **1:1 Drop-in Python MT5 Bridge to CoinDCX Crypto Futures**<br>• Zero code overhaul: swap import to run MT5 EAs on CoinDCX.<br>• Native NumPy structured array schemas matching `copy_rates_from_pos`.<br>• High-fidelity simulation ledger with live mark-to-market PnL tracking. |
| **[`signalbridge-webhook-gateway`](https://github.com/devC-here/signalbridge-webhook-gateway)** | **Execution** | **Sub-40ms Webhook Gateway & Obsidian Dark Terminal**<br>• Ultra-fast TradingView alert & MT5 indicator execution on CoinDCX.<br>• Multi-tiered circuit breakers (order notional cap, daily loss limit).<br>• Real-time web control terminal and Pine Script strategy templates. |
| **[`DATA-SYNTHESIS`](https://github.com/devC-here/DATA-SYNTHESIS)** | **Simulation** | **Synthetic Microstructure & Financial Order Flow Generator**<br>• 32-feature extraction across FX M15 auction physics.<br>• 6-head multi-task PyTorch residual network training pipeline.<br>• Eliminates overfitting to historical market noise. |

---

## 🔬 Monorepo Research Architecture & Lifecycle

This repository organizes the research pipeline across four chronological stages:

```
[ STAGE 1: DATA & INTERMARKET RESEARCH ]
 ├── DATA SYNTHESIS AND MANIPULATION/ ──> Stochastic market regime generation & converters
 ├── CORELATION-TRD/                  ──> DXY vs EURUSD tick pressure correlation models
 └── RESEARCH/                        ──> FFT cycle discovery, Parkinson volatility, 50% impulse

[ STAGE 2: NEURAL ARCHITECTURE R&D ]
 ├── MODELS GENERATIONS/              ──> Model checkpoints from Gen 1 to Claude V4 & H1-BASE
 ├── TRI-MODEL BUILD/                 ──> Composite Breakout + Momentum multi-head architectures
 └── MODEL ENSEMBLES/                 ──> Dual, Triple, Quad, Pent, and Sept-model fusion logic

[ STAGE 3: DETERMINISTIC RISK GUARDRAILS ]
 ├── override/                        ──> 700-line condition checker, trap filters, consensus rules
 ├── HELPING SCRIPTS/                 ──> Position managers, SL/TP calculators, live rate fetchers
 └── MANUAL STATERGIES/               ──> Candlestick geometry classifiers & discretionary models

[ STAGE 4: PRODUCTION & EXECUTION SUITES ]
 ├── PRODUCTION-R/                    ──> Turnkey live production rigs (MetaTrader 5 V1–V5)
 ├── COINDCX/                         ──> Native CoinDCX spot & futures runners
 ├── MQL5-SCRIPTS/                    ──> Native MetaTrader 5 Expert Advisors (`estimater.mq5`)
 └── ORDER PUNCHER/                   ──> Low-latency direct socket order punchers
```

---

## 📁 Repository Directory Index

| Directory | Core Purpose & Contents |
| :--- | :--- |
| **[`5M-BACKTEST/`](./5M-BACKTEST/ABOUT.md)** | Multi-timeframe backtest engine (`15M-H1`, `METATRADER-V3`). |
| **[`COINDCX/`](./COINDCX/ABOUT.md)** | Direct CoinDCX futures runner, predictors, and fusion logic. |
| **[`CORELATION-TRD/`](./CORELATION-TRD/ABOUT.md)** | DXY / EURUSD tick pressure intermarket correlation engine. |
| **[`DATA SYNTHESIS AND MANIPULATION/`](./DATA%20SYNTHESIS%20AND%20MANIPULATION/ABOUT.md)** | Market regime simulation, synthetic data generators, and converters. |
| **[`HELPING SCRIPTS/`](./HELPING%20SCRIPTS/ABOUT.md)** | Data fetching, position management, and SL/TP calculators. |
| **[`MANUAL STATERGIES/`](./MANUAL%20STATERGIES/ABOUT.md)** | Candlestick geometry classifiers and discretionary price action models. |
| **[`MODEL ENSEMBLES/`](./MODEL%20ENSEMBLES/ABOUT.md)** | Dual, Triple, Quad, Pent, Hexa, and Sept model fusion frameworks. |
| **[`MODELS GENERATIONS/`](./MODELS%20GENERATIONS/ABOUT.md)** | All model iterations from Gen 1 to Claude V4 and **`H1-BASE`** (Proprietary). |
| **[`MQL5-SCRIPTS/`](./MQL5-SCRIPTS/ABOUT.md)** | MetaTrader 5 Expert Advisor (`estimater.mq5` & `.ex5`). |
| **[`ORDER PUNCHER/`](./ORDER%20PUNCHER/ABOUT.md)** | Low-latency order submission scripts. |
| **[`override/`](./override/ABOUT.md)** | Condition checkers, risk filters, and market state guardrails. |
| **[`PRODUCTION-R/`](./PRODUCTION-R/ABOUT.md)** | Turnkey live production deployment packages (`METATRADER5*`, `.bat` runners). |
| **[`RESEARCH/`](./RESEARCH/ABOUT.md)** | FFT cycle analysis, Parkinson volatility, and 50% impulse touch tests. |
| **[`RESRH-BACKUP1/`](./RESRH-BACKUP1/ABOUT.md)** | Early model checkpoint archive. |
| **[`TEST CODES/`](./TEST%20CODES/ABOUT.md)** | Integration and broker connection testing scripts. |
| **[`TRI-MODEL BUILD/`](./TRI-MODEL%20BUILD/ABOUT.md)** | Composite breakout + momentum multi-head neural models. |

---

## 🔒 Confidentiality & Research Notice
All strategies, neural network weights, and architectures in this repository represent proprietary intellectual property.
