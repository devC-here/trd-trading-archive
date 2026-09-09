
# json_to_condition_code.txt
# Converts prediction JSON into condition code automatically
# You can rename to .py later if you want to run it directly.

import json
import os

# ============================================================
# ORDER DEFINITIONS (CRITICAL — DO NOT CHANGE)
# ============================================================

MOMENTUM_ORDER = [
    "momentum_state",
    "momentum_slope",
    "regime",
    "volatility_state",
    "structure_bias",
    "entry_quality"
]

BREAKOUT_ORDER = [
    "squeeze_state",
    "breakout_direction",
    "breakout_quality",
    "volatility_phase",
    "band_structure",
    "entry_timing"
]

VWAP_ORDER = [
    "vwap_zone",
    "volume_character",
    "inst_pressure",
    "vwap_interaction",
    "volume_trend",
    "vwap_signal"
]

MICRO_ORDER = [
    "flow_dominance",
    "bar_intent",
    "cluster_state",
    "rejection_structure",
    "absorption_state",
    "micro_signal"
]

CANDLE_ORDER = [
    "pattern_type",
    "seq_dominance",
    "conviction",
    "wick_pressure",
    "seq_phase",
    "candle_signal"
]


# ============================================================
# CORE FUNCTION
# ============================================================

def extract_digit(section, key):
    """
    Handles both formats:
    momentum: {pred: X}
    breakout/vwap/micro/candle: {class: X}
    """
    if "pred" in section[key]:
        return str(section[key]["pred"])
    elif "class" in section[key]:
        return str(section[key]["class"])
    else:
        raise Exception(f"Cannot find pred/class in {key}")


def encode_section(section_data, order_list):

    digits = ""

    for key in order_list:
        digits += extract_digit(section_data, key)

    return digits


def generate_condition_code(json_data):

    momentum = encode_section(json_data["momentum"], MOMENTUM_ORDER)
    breakout = encode_section(json_data["breakout"], BREAKOUT_ORDER)
    vwap     = encode_section(json_data["vwap"], VWAP_ORDER)
    micro    = encode_section(json_data["microstructure"], MICRO_ORDER)
    candle   = encode_section(json_data["candle"], CANDLE_ORDER)

    code = f"M-{momentum}-B-{breakout}-V-{vwap}-X-{micro}-C-{candle}"

    return code


# ============================================================
# FILE INTERFACE
# ============================================================

def convert_file(filepath):

    with open(filepath, "r") as f:
        data = json.load(f)

    code = generate_condition_code(data)

    return code


# ============================================================
# BATCH MODE
# ============================================================

def convert_folder(folder):

    results = {}

    for file in os.listdir(folder):

        if file.endswith(".json"):

            path = os.path.join(folder, file)

            code = convert_file(path)

            name = os.path.splitext(file)[0]

            results[name] = code

    return results


# ============================================================
# SAVE TO TXT FILE
# ============================================================

def save_codes_to_txt(folder, output_file="condition_codes.txt"):

    codes = convert_folder(folder)

    with open(output_file, "w") as f:

        for name, code in codes.items():
            f.write(f"{name} = {code}\n")

    print(f"Saved condition codes to {output_file}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    folder = "predictions"

    save_codes_to_txt(folder)
