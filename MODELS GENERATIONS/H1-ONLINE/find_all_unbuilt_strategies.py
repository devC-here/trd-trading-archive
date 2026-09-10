import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import glob
import re
import json

# List of already built scripts / identifiers
with open("run_all_forex_scanners.py", "r", encoding="utf-8") as f:
    runner_code = f.read()

# Load catalog
with open("strategies_data.json", "r", encoding="utf-8") as f:
    catalog = json.load(f)

catalog_by_id = {s["video_id"]: s for s in catalog if "video_id" in s}

transcript_files = glob.glob("transcripts/*.txt")

non_trading_words = ["credit card", "loan", "insurance", "fraud", "cyber", "inflation", "pi network", "death", "policy", "itr", "rent agreement", "tax", "fmcg", "idfc"]

candidate_strategies = []

for fpath in transcript_files:
    vid = os.path.splitext(os.path.basename(fpath))[0]
    
    # Check if this video is already in runner
    if vid in runner_code or f"_{vid}" in runner_code:
        continue

    with open(fpath, "r", encoding="utf-8") as fp:
        transcript = fp.read().strip()

    meta = catalog_by_id.get(vid, {})
    title = meta.get("title", "")
    desc = meta.get("raw_sources", {}).get("description", "")
    full_text = f"{title} {desc} {transcript}".lower()

    if any(nw in full_text for nw in non_trading_words) and not any(kw in full_text for kw in ["strategy", "indicator", "backtest", "ema", "rsi", "fvg"]):
        continue

    # Look for strategy indicators / rules
    indicators = []
    if "ema" in full_text or "ईएमए" in full_text: indicators.append("EMA")
    if "sma" in full_text or "एसएमए" in full_text: indicators.append("SMA")
    if "rsi" in full_text or "आरएसआई" in full_text: indicators.append("RSI")
    if "macd" in full_text or "एमएसीडी" in full_text: indicators.append("MACD")
    if "bollinger" in full_text or "बॉलिंजर" in full_text: indicators.append("Bollinger Bands")
    if "supertrend" in full_text or "सुपरट्रेंड" in full_text: indicators.append("Supertrend")
    if "fvg" in full_text or "फेयर वैल्यू" in full_text: indicators.append("FVG")
    if "order block" in full_text or "ऑर्डर ब्लॉक" in full_text: indicators.append("Order Block")
    if "heikin" in full_text or "हेकिन" in full_text: indicators.append("Heikin Ashi")
    if "renko" in full_text or "रेन्को" in full_text: indicators.append("Renko")
    if "vwap" in full_text or "वीवप" in full_text: indicators.append("VWAP")
    if "pivot" in full_text or "पिवट" in full_text or "cpr" in full_text or "सीपीआर" in full_text: indicators.append("Pivots")
    if "stochastic" in full_text or "स्टोकेस्टिक" in full_text: indicators.append("Stochastic")
    if "atr" in full_text or "एटीआर" in full_text: indicators.append("ATR")
    if "half trend" in full_text or "हाफ ट्रेंड" in full_text: indicators.append("Half Trend")
    if "nadaraya" in full_text or "नडारया" in full_text: indicators.append("Nadaraya Watson")
    if "sandwich" in full_text or "सैंडविच" in full_text: indicators.append("Sandwich Pattern")
    if "fractal" in full_text or "फ्रैक्टल" in full_text: indicators.append("Fractal")
    if "trendline" in full_text or "ट्रेंडलाइन" in full_text: indicators.append("Trendline")
    if "liquidity" in full_text or "लिक्विडिटी" in full_text or "sweep" in full_text: indicators.append("Liquidity Sweep")

    ret_match = re.search(r"(\d+(\.\d+)?)%\s*(का\s*)?(रिटर्न|return|प्रॉफिट|profit)", full_text)
    wr_match = re.search(r"(\d+(\.\d+)?)%\s*(का\s*)?(विन रेट|win rate|एक्यूरेसी|accuracy)", full_text)

    ret_val = ret_match.group(1) if ret_match else meta.get("backtest_metrics", {}).get("return_pct")
    wr_val = wr_match.group(1) if wr_match else meta.get("backtest_metrics", {}).get("win_rate_pct")

    # If it has any trading logic
    if indicators or "स्ट्रेटजी" in full_text or "strategy" in full_text or "backtest" in full_text or "ट्रेडिंग" in full_text:
        candidate_strategies.append({
            "video_id": vid,
            "title": title,
            "creator": meta.get("creator_attribution", "Trader in Video"),
            "indicators": indicators,
            "return_pct": ret_val,
            "win_rate": wr_val,
            "transcript": transcript
        })

print(f"Total new candidate trading strategies found: {len(candidate_strategies)}")
print("=" * 80)
for idx, c in enumerate(candidate_strategies[:35]):
    ind_s = ", ".join(c["indicators"]) if c["indicators"] else "Price Action"
    ret_s = f"{c['return_pct']}%" if c['return_pct'] else "N/A"
    wr_s = f"{c['win_rate']}%" if c['win_rate'] else "N/A"
    print(f"{idx+1:2d}. [{c['video_id']}] {c['title'][:48]:48} | Inds: {ind_s[:22]:22} | Ret: {ret_s:>6} WR: {wr_s:>5}")
