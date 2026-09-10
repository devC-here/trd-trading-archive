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
import datetime
import threading
from typing import Any, Dict
from flask import Flask, jsonify, render_template, request

from swing_ensemble_engine import HighPipSwingEnsemble, MT5_AVAILABLE

import urllib.request

app = Flask(__name__, static_folder="static", template_folder="templates")

# Initialize swing ensemble engine for EURUSD.0
initial_mock = not MT5_AVAILABLE
engine = HighPipSwingEnsemble(symbol="EURUSD.0", use_mock=initial_mock)
is_running = True


def background_engine_loop():
    """Continuously evaluates 60 H1/H2 strategies and updates the master decision."""
    global is_running
    print("[+] Background H1/H2 Swing Ensemble Engine started for EURUSD.0.", flush=True)
    while is_running:
        try:
            engine.evaluate_cycle()
        except Exception as e:
            print(f"[!] Error in evaluation cycle: {e}", flush=True)
        time.sleep(2.0)


def keep_alive_pinger_loop(port: int):
    """Pings the server every 9 minutes to prevent Render/Cloud free tiers from sleeping."""
    time.sleep(15.0)  # Wait for initial server boot
    print("[+] 24/7 Keep-Alive Pinger active. Server will not spin down on inactivity.", flush=True)
    external_url = os.environ.get("RENDER_EXTERNAL_URL", os.environ.get("APP_URL", "")).rstrip("/")

    while is_running:
        target_url = f"{external_url}/healthz" if external_url else f"http://127.0.0.1:{port}/healthz"
        try:
            req = urllib.request.Request(target_url, headers={"User-Agent": "KeepAlivePinger/2.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [KEEP-ALIVE] Heartbeat ping successful -> {target_url}", flush=True)
        except Exception as e:
            print(f"[!] Keep-alive ping notice: {e}", flush=True)
        time.sleep(540.0)  # Ping every 9 minutes (well before 15m idle threshold)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/healthz")
def healthz():
    return jsonify({
        "status": "healthy",
        "service": "eurusd-swing-ensemble",
        "mode": "cloud-native",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })


@app.route("/api/status", methods=["GET"])
def get_status():
    if not engine.cached_state:
        snap = engine.evaluate_cycle()
        return jsonify(snap)
    return jsonify(engine.cached_state)


@app.route("/api/demo-trades", methods=["GET"])
def get_demo_trades():
    return jsonify(engine.trade_manager.get_summary())


@app.route("/api/pipeline/rates", methods=["GET"])
def get_pipeline_rates():
    tf = request.args.get("tf", "H1").upper()
    count = int(request.args.get("count", 150))
    tf_map = {"H1": 60, "H2": 120, "H4": 240, "D1": 1440}
    tf_const = tf_map.get(tf, 60)
    df = engine.provider.fetch_rates(tf_const, count)
    if df is not None:
        records = df.to_dict(orient="records")
        for r in records:
            if "time" in r and hasattr(r["time"], "strftime"):
                r["time"] = r["time"].strftime("%Y-%m-%d %H:%M:%S")
        return jsonify({"success": True, "timeframe": tf, "count": len(records), "rates": records})
    return jsonify({"success": False, "error": "Unable to fetch rates"}), 500


@app.route("/api/settings", methods=["POST"])
def update_settings():
    data = request.get_json() or {}
    if "mock_mode" in data:
        new_mock = bool(data["mock_mode"])
        engine.provider.use_mock = new_mock
        if not new_mock and MT5_AVAILABLE:
            engine.provider._init_mt5()
        print(f"[*] Engine mode switched to: {'MOCK SIMULATION' if new_mock else 'LIVE MT5'}", flush=True)

    snap = engine.evaluate_cycle()
    return jsonify({"success": True, "state": snap})


def start_server(host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
    # 1. Background strategy evaluation loop
    t_engine = threading.Thread(target=background_engine_loop, daemon=True)
    t_engine.start()

    # 2. 24/7 Keep-alive self-pinger loop (runs like a VPS)
    t_pinger = threading.Thread(target=keep_alive_pinger_loop, args=(port,), daemon=True)
    t_pinger.start()

    print("==================================================================", flush=True)
    print("[*] H1/H2 SWING ENSEMBLE SERVER RUNNING (24/7 AUTONOMOUS VPS MODE)", flush=True)
    print(f"[*] Local access:       http://127.0.0.1:{port}", flush=True)
    print(f"[*] Network/Cloud URL:  http://{host}:{port}", flush=True)
    print("==================================================================", flush=True)
    app.run(host=host, port=port, debug=debug, use_reloader=False)


if __name__ == "__main__":
    configured_port = int(os.environ.get("PORT", 5000))
    configured_host = os.environ.get("HOST", "0.0.0.0")
    start_server(host=configured_host, port=configured_port)


