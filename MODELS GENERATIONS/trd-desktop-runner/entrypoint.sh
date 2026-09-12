#!/usr/bin/env bash
# ==============================================================================
# Entrypoint for MT5 Codespaces Linux Container
# Starts Xvfb, Fluxbox, x11vnc, and noVNC (Port 6080)
# ==============================================================================
set -e

export DISPLAY="${DISPLAY:-:1}"
export WINEPREFIX="${WINEPREFIX:-/home/vscode/.wine}"
export WINEDEBUG="-all"
export WINEDLLOVERRIDES="mscoree,mshtml="

echo "========================================================================"
echo "   🚀 STARTING TRD METATRADER 5 CODESPACE GUI ENVIRONMENT"
echo "========================================================================"
echo "[+] Virtual Display: $DISPLAY (1280x800)"
echo "[+] Wine Prefix:     $WINEPREFIX"
echo "[+] noVNC Web GUI:   http://localhost:6080"
echo "[+] Direct VNC:      localhost:5901"
echo "========================================================================"

# Ensure noVNC root displays vnc.html automatically
if [ -d "/usr/share/novnc" ] && [ ! -f "/usr/share/novnc/index.html" ]; then
    ln -sf /usr/share/novnc/vnc.html /usr/share/novnc/index.html 2>/dev/null || true
fi

# 1. Start Xvfb Virtual Framebuffer
if ! pgrep -f "Xvfb $DISPLAY" > /dev/null; then
    echo "[+] Launching Xvfb display on $DISPLAY..."
    Xvfb "$DISPLAY" -screen 0 1280x800x24 -nolisten tcp &
    sleep 1
fi

# 2. Start Fluxbox Window Manager
if ! pgrep -f "fluxbox" > /dev/null; then
    echo "[+] Launching Fluxbox window manager..."
    fluxbox &
    sleep 1
fi

# 3. Start x11vnc server
if ! pgrep -f "x11vnc" > /dev/null; then
    echo "[+] Launching x11vnc on port 5901..."
    x11vnc -display "$DISPLAY" -forever -shared -nopw -rfbport 5901 -quiet &
    sleep 1
fi

# 4. Start Websockify + noVNC on port 6080
if ! pgrep -f "websockify" > /dev/null; then
    echo "[+] Launching noVNC Web GUI on port 6080..."
    websockify --web=/usr/share/novnc 6080 localhost:5901 >/dev/null 2>&1 &
    sleep 1
fi

echo "========================================================================"
echo "   ✅ GUI DISPLAY PIPELINE ACTIVE!"
echo "   👉 Open Port 6080 in Codespaces to see the desktop."
echo "   👉 Run 'launch_mt5' to start MetaTrader 5."
echo "   👉 Run 'wine C:\\\\Python39\\\\python.exe test_mt5_bridge.py' for Python MT5."
echo "========================================================================"

# If command is passed, execute it; otherwise keep container interactive
if [ "$#" -gt 0 ]; then
    exec "$@"
else
    exec bash
fi
