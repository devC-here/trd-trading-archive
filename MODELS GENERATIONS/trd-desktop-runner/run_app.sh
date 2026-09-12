#!/usr/bin/env bash
# ==============================================================================
# TRD Desktop Runner - Launch Script for Codespaces GUI
# ==============================================================================
set -e

# Default display in GitHub Codespaces desktop-lite feature is :1
export DISPLAY="${DISPLAY:-:1}"

echo "========================================================"
echo "🖥️  TRD Desktop Runner - Codespaces GUI"
echo "========================================================"
echo "Target Display: $DISPLAY"

# Check if an X server is responding on $DISPLAY
if ! command -v xset >/dev/null 2>&1 || ! xset q >/dev/null 2>&1; then
    echo "⚠️  Note: X server on $DISPLAY is not yet reachable or xset is not installed."
    echo "   If using Codespaces desktop-lite, ensure the desktop service is running."
    echo "   You can check the noVNC web interface on forwarded port 6080."
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

APP_TARGET="${1:-app.py}"

echo "🚀 Launching desktop application: $APP_TARGET"
echo "👉 Open the 'Ports' panel in VS Code/Codespaces and open Port 6080 to view the GUI."
echo "========================================================"

python3 "$APP_TARGET"
