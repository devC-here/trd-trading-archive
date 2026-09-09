"""
========================================================================================
EURUSD.0 H1/H2 HIGH-PIP SWING ENSEMBLE SERVER
========================================================================================
Flask server streaming the master high-yield swing decision and 60 H1/H2 strategy states.
========================================================================================
"""

import os
import sys
import time
import threading
from typing import Any, Dict
from flask import Flask, jsonify, render_template, request

from swing_ensemble_engine import HighPipSwingEnsemble, MT5_AVAILABLE

app = Flask(__name__, static_folder="static", template_folder="templates")

# Initialize swing ensemble engine for EURUSD.0
initial_mock = not MT5_AVAILABLE
engine = HighPipSwingEnsemble(symbol="EURUSD.0", use_mock=initial_mock)
is_running = True


def background_engine_loop():
    """Continuously evaluates 60 H1/H2 strategies and updates the master decision."""
    global is_running
    print("[+] Background H1/H2 Swing Ensemble Engine started for EURUSD.0.")
    while is_running:
        try:
            engine.evaluate_cycle()
        except Exception as e:
            print(f"[!] Error in evaluation cycle: {e}")
        time.sleep(2.0)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def get_status():
    if not engine.cached_state:
        snap = engine.evaluate_cycle()
        return jsonify(snap)
    return jsonify(engine.cached_state)


@app.route("/api/settings", methods=["POST"])
def update_settings():
    data = request.get_json() or {}
    if "mock_mode" in data:
        new_mock = bool(data["mock_mode"])
        engine.provider.use_mock = new_mock
        if not new_mock and MT5_AVAILABLE:
            engine.provider._init_mt5()
        print(f"[*] Engine mode switched to: {'MOCK SIMULATION' if new_mock else 'LIVE MT5'}")

    snap = engine.evaluate_cycle()
    return jsonify({"success": True, "state": snap})


def start_server(host: str = "127.0.0.1", port: int = 5000, debug: bool = False):
    t = threading.Thread(target=background_engine_loop, daemon=True)
    t.start()
    print(f"[*] Starting Swing Ensemble Server on http://{host}:{port}...")
    app.run(host=host, port=port, debug=debug, use_reloader=False)


if __name__ == "__main__":
    start_server()
