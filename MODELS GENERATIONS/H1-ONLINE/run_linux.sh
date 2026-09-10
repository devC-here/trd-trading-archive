#!/bin/bash
echo "=================================================================="
echo "    EURUSD.0 H1/H2 SWING ENSEMBLE (100% CLOUD NATIVE / LINUX)"
echo "=================================================================="
export PORT=${PORT:-5000}
export HOST="0.0.0.0"
python3 gui_server.py
