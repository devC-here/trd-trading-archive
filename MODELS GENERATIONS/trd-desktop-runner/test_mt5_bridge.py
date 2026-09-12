#!/usr/bin/env python3
"""
Test Python MetaTrader 5 Bridge inside Codespaces Wine environment.
Run via:
    wine "C:\\Python39\\python.exe" test_mt5_bridge.py
"""

import sys
import os

try:
    import MetaTrader5 as mt5
    print("[+] Successfully imported MetaTrader5 Python package!")
    print(f"[+] Package Version: {mt5.__version__}")
    print(f"[+] Author: {mt5.__author__}")
except ImportError as e:
    print(f"[!] Failed to import MetaTrader5: {e}")
    sys.exit(1)

print("\n[+] Attempting to initialize connection to MT5 terminal...")
# Attempt initialization
initialized = mt5.initialize()

if not initialized:
    print(f"[!] mt5.initialize() returned False.")
    print(f"[!] Last Error: {mt5.last_error()}")
    print("[i] If MT5 is not yet running or logged in, launch it with: launch_mt5")
else:
    print("[✅] MT5 terminal connection established successfully!")
    version = mt5.version()
    print(f"[+] Terminal Version: {version}")
    terminal_info = mt5.terminal_info()
    if terminal_info:
        print(f"[+] Terminal Name: {terminal_info.name}")
        print(f"[+] Data Path: {terminal_info.data_path}")
        print(f"[+] Connected: {terminal_info.connected}")
    mt5.shutdown()
