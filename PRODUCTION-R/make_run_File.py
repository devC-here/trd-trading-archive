import re
from pathlib import Path

ROOT = Path.cwd()
RUN_DIR = ROOT / "run files"
RUN_DIR.mkdir(exist_ok=True)

# Matches:
# main.py
# main(1).py
# main (1).py
# MAIN    (25).py
pattern = re.compile(
    r"^main(?:\s*\(\d+\))?\.py$",
    re.IGNORECASE
)

for py_file in ROOT.rglob("*"):
    if not py_file.is_file():
        continue

    if not pattern.match(py_file.name):
        continue

    rel_path = py_file.relative_to(ROOT)

    # Create unique BAT filename
    bat_name = "_".join(rel_path.with_suffix("").parts) + ".bat"

    # Clean invalid filename chars
    for ch in r'<>:"/\|?*':
        bat_name = bat_name.replace(ch, "_")

    bat_path = RUN_DIR / bat_name

    bat_content = f"""@echo off
cd /d "{py_file.parent}"
python "{py_file.name}"
pause
"""

    bat_path.write_text(bat_content, encoding="utf-8")

    print(f"Created: {bat_path.name}")

print(f"\nDone. Created BAT files in:\n{RUN_DIR}")