import json
import re


# ============================================================
# CONFIG
# ============================================================

CODES_FILE = "condition_codes.txt"
REFERENCE_FILE = "eurusd_15m_600files_reference.txt"
OUTPUT_FILE = "trading_conditions.json"


# ============================================================
# SIGNAL → BIAS MAPPING
# ============================================================

def determine_bias(signal_text):

    s = signal_text.lower()

    if "now-bear" in s or "act-bear" in s or "avoid" in s:
        return "sell", 0.9, "high"

    if "momentum=go" in s or "breakout=now" in s or "vwap=act" in s or "micro=act" in s:
        return "buy", 0.9, "high"

    if "exhaust" in s or "trap" in s:
        return "no_trade", 0.95, "low"

    if "range" in s or "compression" in s:
        return "no_trade", 0.8, "low"

    return "no_trade", 0.5, "low"


# ============================================================
# LOAD CONDITION CODES
# ============================================================

def load_condition_codes():

    codes = {}

    with open(CODES_FILE, "r") as f:

        for line in f:

            line = line.strip()

            if "=" in line:

                name, code = line.split("=")

                codes[name.strip()] = code.strip()

    return codes


# ============================================================
# LOAD REFERENCE SIGNALS
# ============================================================

def load_reference_signals():

    signals = {}

    with open(REFERENCE_FILE, "r") as f:

        text = f.read()

    pattern = r'FILE GROUP : (.*?)\\.csv.*?SIGNAL\\s*: (.*?)\\n'

    matches = re.findall(pattern, text, re.DOTALL)

    for name, signal in matches:

        key = name.split("_001")[0]

        signals[key] = signal.strip()

    return signals


# ============================================================
# BUILD CONDITIONS
# ============================================================

def build_conditions():

    codes = load_condition_codes()
    signals = load_reference_signals()

    conditions = {}

    condition_id = 1

    for name, code in codes.items():

        base = name.rsplit("_", 1)[0]

        signal = signals.get(base, "")

        bias, confidence, viability = determine_bias(signal)

        meaning = signal if signal else "Unknown condition"

        conditions[code] = {

            "condition_id": str(condition_id),

            "final_bias": bias,

            "confidence": confidence,

            "entry_viability": viability,

            "meaning": meaning

        }

        condition_id += 1

    return conditions


# ============================================================
# SAVE FILE
# ============================================================

def main():

    conditions = build_conditions()

    with open(OUTPUT_FILE, "w") as f:

        json.dump(conditions, f, indent=4)

    print("Saved:", OUTPUT_FILE)


# ============================================================

if __name__ == "__main__":
    main()