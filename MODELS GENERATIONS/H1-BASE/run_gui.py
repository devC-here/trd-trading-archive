"""
========================================================================================
LAUNCHER: EURUSD.0 70-STRATEGY CONDITION MONITOR GUI
========================================================================================
Launches the real-time web server and automatically opens Google Chrome.
========================================================================================
"""

import os
import sys
import time
import socket
import webbrowser
import threading
import subprocess

PORT = 5000
HOST = "127.0.0.1"


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((HOST, port)) == 0


def open_chrome_browser(url: str):
    """Attempts to launch Google Chrome specifically, falling back to default browser."""
    time.sleep(1.5)  # Wait for server to bind
    chrome_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expanduser(r"~\AppData\Local\Google\Chrome\Application\chrome.exe")
    ]
    opened = False
    for path in chrome_paths:
        if os.path.exists(path):
            try:
                subprocess.Popen([path, url])
                print(f"[+] Opened Google Chrome at {url}")
                opened = True
                break
            except Exception as e:
                print(f"[!] Error launching Chrome executable: {e}")

    if not opened:
        print(f"[+] Opening default browser at {url}")
        webbrowser.open(url)


def main():
    global PORT
    print("=" * 80)
    print("      EURUSD.0 70-STRATEGY REAL-TIME CONDITION MONITOR (CHROME GUI)")
    print("=" * 80)
    print(f" Symbol:      EURUSD.0 (Deriv-Demo Account 6288307)")
    print(f" Strategies:  All 70 Verified MT5 Models")
    print(f" Port:        {PORT}")
    print("=" * 80)

    # Check port availability
    if is_port_in_use(PORT):
        print(f"[!] Port {PORT} is already in use. Selecting port {PORT + 5}...")
        PORT += 5

    url = f"http://{HOST}:{PORT}/"

    # Start browser opener in thread
    threading.Thread(target=open_chrome_browser, args=(url,), daemon=True).start()

    # Start Flask server
    from gui_server import start_server
    try:
        start_server(host=HOST, port=PORT, debug=False)
    except KeyboardInterrupt:
        print("\n[STOP] Shutting down GUI monitor server.")


if __name__ == "__main__":
    main()
