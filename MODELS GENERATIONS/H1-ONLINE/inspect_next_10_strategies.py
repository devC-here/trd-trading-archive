import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os

vids = [
    "-0oWb3Uv9FM", # Chandelier Exit
    "0OE0jCRge94", # Ichimoku
    "0n4sMsqTZdw", # Tori Trades
    "WRczqfxjR4g", # Half Trend
    "t9VzgJhqgHk", # Sandwich Pattern
    "UyDCSGoVmVk", # Heikin Ashi
    "4gvWmhCzE30", # Bharat Jhunjhunwala
    "rXoB4mIhLjc", # Nadaraya Watson
    "7XtF2FH92MM", # Gold 56.8%
    "9soHiKD3P1s", # Smallest SL EMA
]

for v in vids:
    fpath = os.path.join("transcripts", f"{v}.txt")
    if os.path.exists(fpath):
        with open(fpath, "r", encoding="utf-8") as fp:
            txt = fp.read()
        print("=" * 80)
        print(f"VIDEO: {v}")
        print(txt)
        print()
