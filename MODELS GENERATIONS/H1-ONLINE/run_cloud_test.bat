@echo off
title EURUSD.0 Swing Ensemble - 100% Cloud Native Mode (Zero MT5)
echo ====================================================================
echo    EURUSD.0 H1/H2 SWING ENSEMBLE (100% CLOUD NATIVE MODE)
echo ====================================================================
echo [*] No MetaTrader 5 app needed or opened.
echo [*] Multi-timeframe H1/H2/H4/D1 engine running in memory.
echo [*] Starting Flask server on http://localhost:5000...
echo ====================================================================
echo.
python gui_server.py
pause
