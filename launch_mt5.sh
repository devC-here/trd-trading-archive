#!/usr/bin/env bash
# ==============================================================================
# Helper script to launch or install MetaTrader 5 inside Wine
# ==============================================================================
set -e

export DISPLAY="${DISPLAY:-:1}"
export WINEPREFIX="${WINEPREFIX:-/home/vscode/.wine}"
export WINEDEBUG="-all"

MT5_INSTALLED_PATH="$WINEPREFIX/drive_c/Program Files/MetaTrader 5/terminal64.exe"
LOCAL_WORKSPACE_TERMINAL="$(pwd)/terminal64.exe"
SETUP_PATH="/home/vscode/mt5setup.exe"

echo "========================================================================"
echo "   📊 METATRADER 5 LAUNCHER (CODESPACES LINUX)"
echo "========================================================================"

if [ -f "$MT5_INSTALLED_PATH" ]; then
    echo "[+] Found installed MT5 terminal: $MT5_INSTALLED_PATH"
    echo "[+] Launching MetaTrader 5 on display $DISPLAY..."
    echo "[👉] Switch to Port 6080 in your browser to view and trade on MT5!"
    wine "$MT5_INSTALLED_PATH" /portable &
    exit 0
fi

if [ -f "$LOCAL_WORKSPACE_TERMINAL" ]; then
    echo "[+] Found portable terminal64.exe in workspace."
    mkdir -p "$(dirname "$MT5_INSTALLED_PATH")"
    cp -f "$LOCAL_WORKSPACE_TERMINAL" "$MT5_INSTALLED_PATH"
    chmod +x "$MT5_INSTALLED_PATH"
    echo "[+] Copied to $MT5_INSTALLED_PATH"
    echo "[+] Launching MetaTrader 5 on display $DISPLAY..."
    echo "[👉] Switch to Port 6080 in your browser to view MT5!"
    wine "$MT5_INSTALLED_PATH" /portable &
    exit 0
fi

if [ -f "$SETUP_PATH" ]; then
    echo "[+] MetaTrader 5 not yet installed. Running official installer..."
    echo "[+] Executing: wine mt5setup.exe /auto"
    echo "[👉] Switch to Port 6080 in your browser to observe installation progress!"
    wine "$SETUP_PATH" /auto &
    echo "[+] Installer initiated in background."
    echo "[+] Once installation finishes, run 'launch_mt5' again to start terminal."
else
    echo "[!] Error: Neither terminal64.exe nor mt5setup.exe found."
    echo "[!] Please download mt5setup.exe or place terminal64.exe in this folder."
    exit 1
fi
