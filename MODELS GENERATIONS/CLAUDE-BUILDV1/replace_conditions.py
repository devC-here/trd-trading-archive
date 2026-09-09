import json
import re

# INPUT FILES
OLD_CODE_FILE = "code1.py"
NEW_CONDITIONS_FILE = "trading_conditions.json"

# OUTPUT FILE
OUTPUT_FILE = "code1_updated.py"


def load_new_conditions():
    with open(NEW_CONDITIONS_FILE, "r") as f:
        return json.load(f)


def build_python_dictionary(conditions):

    lines = []
    lines.append("trading_conditions = {")

    for pattern, meta in conditions.items():

        entry = f'''
    "{pattern}": {{
        "condition_id": {meta["condition_id"]},
        "final_bias": "{meta["final_bias"]}",
        "confidence": {meta["confidence"]},
        "entry_viability": "{meta["entry_viability"]}",
        "meaning": "{meta["meaning"]}"
    }},'''

        lines.append(entry)

    lines.append("\n}\n")

    return "\n".join(lines)


def replace_conditions():

    # Load files
    with open(OLD_CODE_FILE, "r") as f:
        old_code = f.read()

    new_conditions = load_new_conditions()

    new_dict_code = build_python_dictionary(new_conditions)

    # Replace dictionary using regex
    updated_code = re.sub(
        r"trading_conditions\s*=\s*\{.*?\n\}",
        new_dict_code,
        old_code,
        flags=re.DOTALL
    )

    # Save new file
    with open(OUTPUT_FILE, "w") as f:
        f.write(updated_code)

    print("Done.")
    print("New file saved as:", OUTPUT_FILE)


if __name__ == "__main__":
    replace_conditions()