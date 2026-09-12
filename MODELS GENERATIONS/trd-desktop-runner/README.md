# TRD MetaTrader 5 on Codespaces Linux

A production-grade environment to run **MetaTrader 5 (MT5)** directly inside **GitHub Codespaces Linux** with interactive web browser desktop access (noVNC) and native Python `MetaTrader5` algorithmic trading support via Wine.

---

## 🌟 Key Features

- 🖥️ **Full Interactive GUI**: Access the MT5 terminal directly in your web browser on **Port 6080** via noVNC & Fluxbox.
- 🍷 **Pre-configured Wine 9/10**: 64-bit Windows 10 environment ready to run Windows trading terminals.
- 🐍 **Wine Windows Python 3.9**: Pre-installed with the official `MetaTrader5`, `pandas`, and `numpy` packages for algorithmic execution and EA scripts.
- 🚀 **Turnkey Dockerfile**: Self-contained devcontainer configuration for GitHub Codespaces.

---

## 🚀 Getting Started in GitHub Codespaces

### 1. Launch Codespace
Open this repository in GitHub Codespaces. Codespaces will build the container using [`.devcontainer/devcontainer.json`](file:///.devcontainer/devcontainer.json) and [`Dockerfile`](file:///Dockerfile).

### 2. View the Desktop GUI
1. Open the **Ports** tab in the bottom panel of VS Code / Codespaces.
2. Locate port **`6080`** (`MT5 Desktop GUI (noVNC)`).
3. Click the **Globe icon** ("Open in Browser") or right-click to **Preview in Editor**.
4. You will see the virtual desktop ready to display MT5 windows.

---

## 📈 Running MetaTrader 5

### Launch MT5 Terminal
In the terminal, run:
```bash
launch_mt5
```
or:
```bash
./launch_mt5.sh
```
- If MT5 is already installed, it immediately starts `terminal64.exe` in `/portable` mode.
- If not yet installed, it starts the official installer (`mt5setup.exe /auto`). Switch to your browser tab on Port 6080 to watch the progress.

---

## 🤖 Running Algorithmic Python Scripts with MT5

Because MetaTrader 5's Python library requires Windows binaries, execute your Python trading scripts through Wine's Python:

```bash
# Test connection to MT5
wine "C:\\Python39\\python.exe" test_mt5_bridge.py
```

### Example Python Trading Script
```python
import MetaTrader5 as mt5

if not mt5.initialize():
    print("Failed to connect to MT5:", mt5.last_error())
    quit()

print("Connected to MT5:", mt5.terminal_info().name)
# Retrieve live quotes:
eurusd_rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M1, 0, 10)
print(eurusd_rates)

mt5.shutdown()
```

---

## 📁 File Manifest

| File | Purpose |
| :--- | :--- |
| **[`Dockerfile`](file:///Dockerfile)** | Container build definition (Ubuntu 22.04 + Wine 9/10 + Windows Python + noVNC + MT5). |
| **[`.devcontainer/devcontainer.json`](file:///.devcontainer/devcontainer.json)** | GitHub Codespaces integration and port forward configuration. |
| **[`entrypoint.sh`](file:///entrypoint.sh)** | Container startup script managing Xvfb display, Fluxbox, and noVNC on port 6080. |
| **[`launch_mt5.sh`](file:///launch_mt5.sh)** | Shell script to start or install MT5. |
| **[`test_mt5_bridge.py`](file:///test_mt5_bridge.py)** | Script to verify Python `MetaTrader5` connectivity. |
| **[`app.py`](file:///app.py)** | Lightweight fallback GUI test app. |
