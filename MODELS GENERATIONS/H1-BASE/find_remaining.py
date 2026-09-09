import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
import glob
import json
import re

with open("run_all_forex_scanners.py", "r", encoding="utf-8") as f:
    runner_code = f.read()

with open("strategies_data.json", "r", encoding="utf-8") as f:
    catalog = json.load(f)
catalog_by_id = {s.get("video_id"): s for s in catalog if "video_id" in s}

non_trading = [
    "credit card", "loan", "insurance", "fraud", "cyber", "inflation", 
    "pi network", "death", "policy", "itr", "rent agreement", "tax", 
    "fmcg", "idfc", "dividend", "mutual fund", "scam", "budget", "salary", "sip"
]

remaining_strategies = []

for fpath in glob.glob("transcripts/*.txt"):
    vid = os.path.splitext(os.path.basename(fpath))[0]
    if vid in runner_code or f"_{vid}" in runner_code:
        continue
    
    with open(fpath, "r", encoding="utf-8", errors="ignore") as fp:
        transcript = fp.read()
        
    meta = catalog_by_id.get(vid, {})
    title = meta.get("title", "")
    desc = meta.get("raw_sources", {}).get("description", "")
    creator = meta.get("creator_attribution", "Trader")
    full_text = f"{title} {desc} {transcript}".lower()
    
    if any(nw in full_text for nw in non_trading) and not any(kw in full_text for kw in ["strategy", "indicator", "backtest", "ema", "rsi", "fvg", "कैंडल", "ट्रेड"]):
        continue
        
    has_trading_intent = any(w in full_text for w in [
        "strategy", "स्ट्रेटजी", "indicator", "इंडिकेटर", "कैंडल", "candle", 
        "entry", "एंट्री", "स्टॉप लॉस", "stop loss", "backtest", "बैक टेस्ट", 
        "ट्रेडिंग", "trading", "रिटर्न", "win rate"
    ])
    
    if has_trading_intent:
        ret_m = re.search(r"(\d+(\.\d+)?)%\s*(return|profit|प्रॉफिट|रिटर्न)", full_text)
        wr_m = re.search(r"(\d+(\.\d+)?)%\s*(win\s*rate|accuracy|एक्यूरेसी)", full_text)
        ret_val = float(ret_m.group(1)) if ret_m else meta.get("backtest_metrics", {}).get("return_pct", 0)
        wr_val = float(wr_m.group(1)) if wr_m else meta.get("backtest_metrics", {}).get("win_rate_pct", 0)
        
        # Check specific indicators
        inds = []
        if "ema" in full_text or "ईएमए" in full_text: inds.append("EMA")
        if "sma" in full_text or "एसएमए" in full_text: inds.append("SMA")
        if "rsi" in full_text or "आरएसआई" in full_text: inds.append("RSI")
        if "macd" in full_text or "एमएसीडी" in full_text: inds.append("MACD")
        if "supertrend" in full_text or "सुपरट्रेंड" in full_text: inds.append("Supertrend")
        if "bollinger" in full_text or "बॉलिंजर" in full_text: inds.append("Bollinger")
        if "fvg" in full_text or "फेयर वैल्यू" in full_text: inds.append("FVG")
        if "order block" in full_text or "ऑर्डर ब्लॉक" in full_text: inds.append("Order Block")
        if "vwap" in full_text or "वीवप" in full_text: inds.append("VWAP")
        if "heikin" in full_text or "हेकिन" in full_text: inds.append("Heikin Ashi")
        if "fibonacci" in full_text or "फिबोनाची" in full_text or "fib" in full_text: inds.append("Fibonacci")
        if "trendline" in full_text or "ट्रेंडलाइन" in full_text: inds.append("Trendline")
        if "nadaraya" in full_text or "नडारया" in full_text: inds.append("Nadaraya Watson")
        if "renko" in full_text or "रेन्को" in full_text: inds.append("Renko")
        if "ichimoku" in full_text or "इचिमोकू" in full_text or "इची" in full_text: inds.append("Ichimoku")
        if "sandwich" in full_text or "सैंडविच" in full_text: inds.append("Sandwich")
        if "breakout" in full_text or "ब्रेकआउट" in full_text: inds.append("Breakout")
        if "reversal" in full_text or "रिवर्सल" in full_text: inds.append("Reversal")
        
        remaining_strategies.append({
            "vid": vid,
            "title": title,
            "creator": creator,
            "ret": ret_val,
            "wr": wr_val,
            "indicators": inds,
            "transcript": transcript
        })

remaining_strategies.sort(key=lambda x: (x["ret"] or 0, x["wr"] or 0), reverse=True)

print(f"Total remaining candidate trading strategies: {len(remaining_strategies)}")
print("=" * 95)
for idx, s in enumerate(remaining_strategies[:50]):
    ret_str = f"{s['ret']}%" if s['ret'] else "N/A"
    wr_str = f"{s['wr']}%" if s['wr'] else "N/A"
    inds_str = ", ".join(s["indicators"][:3]) if s["indicators"] else "Price Action"
    print(f"{idx+1:2d}. [{s['vid']}] Ret: {ret_str:>7} | WR: {wr_str:>6} | {inds_str[:20]:20} | {s['creator'][:15]:15} | {s['title'][:38]}")

