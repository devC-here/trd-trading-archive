import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import re
import glob

def analyze():
    with open("strategies_data.json", "r", encoding="utf-8") as f:
        catalog = json.load(f)

    catalog_by_id = {s["video_id"]: s for s in catalog if "video_id" in s}

    transcript_files = glob.glob("transcripts/*.txt")
    print(f"Total transcripts to analyze: {len(transcript_files)}")

    trading_strategies = []
    skipped_non_trading = 0

    non_trading_keywords = [
        "credit card", "loan", "insurance", "fraud", "cyber", "inflation", 
        "pi network", "death", "policy", "scam"
    ]

    for fpath in transcript_files:
        vid = os.path.splitext(os.path.basename(fpath))[0]
        with open(fpath, "r", encoding="utf-8") as fp:
            transcript = fp.read().strip()

        meta = catalog_by_id.get(vid, {})
        title = meta.get("title", "")
        desc = meta.get("raw_sources", {}).get("description", "")
        text = f"{title} {desc} {transcript}".lower()

        # Check if purely non-trading finance topic
        if any(nt in text for nt in non_trading_keywords) and not any(tk in text for tk in ["strategy", "indicator", "backtest", "ema", "fvg", "rsi"]):
            skipped_non_trading += 1
            continue

        # Look for indicators & concepts in Hindi/English
        indicators_found = []
        if re.search(r"\b(ema|ईएमए|मूविंग एवरेज)\b", text):
            indicators_found.append("EMA")
        if re.search(r"\b(sma|एसएमए|सिंपल मूविंग)\b", text):
            indicators_found.append("SMA")
        if re.search(r"\b(rsi|आरएसआई)\b", text):
            indicators_found.append("RSI")
        if re.search(r"\b(macd|एमएसीडी)\b", text):
            indicators_found.append("MACD")
        if re.search(r"\b(bollinger|बॉलिंजर|बोलिंजर)\b", text):
            indicators_found.append("Bollinger Bands")
        if re.search(r"\b(supertrend|सुपरट्रेंड)\b", text):
            indicators_found.append("Supertrend")
        if re.search(r"\b(vwap|वीवप)\b", text):
            indicators_found.append("VWAP")
        if re.search(r"\b(fvg|फेयर वैल्यू गैप|fair value gap)\b", text):
            indicators_found.append("Fair Value Gap")
        if re.search(r"\b(order block|ऑर्डर ब्लॉक)\b", text):
            indicators_found.append("Order Block")
        if re.search(r"\b(breaker block|ब्रेकर ब्लॉक)\b", text):
            indicators_found.append("Breaker Block")
        if re.search(r"\b(fractal|फ्रैक्टल|विलियम)\b", text):
            indicators_found.append("William Fractal")
        if re.search(r"\b(fibonacci|फिबोनाची|गोल्डन)\b", text):
            indicators_found.append("Fibonacci")
        if re.search(r"\b(cpr|सीपीआर|पिवट|pivot|camarilla|कैमारिला)\b", text):
            indicators_found.append("Pivots/CPR")
        if re.search(r"\b(stochastic|स्टोकेस्टिक)\b", text):
            indicators_found.append("Stochastic")
        if re.search(r"\b(trendline|ट्रेंड लाइन|ट्रेंडलाइन)\b", text):
            indicators_found.append("Trendline")
        if re.search(r"\b(atr|एटीआर)\b", text):
            indicators_found.append("ATR")
        if re.search(r"\b(renko|रेन्को)\b", text):
            indicators_found.append("Renko")

        # Check return / winrate
        ret_match = re.search(r"(\d+(\.\d+)?)%\s*(का\s*)?(रिटर्न|return|प्रॉफिट|profit)", text)
        wr_match = re.search(r"(\d+(\.\d+)?)%\s*(का\s*)?(विन रेट|win rate|एक्यूरेसी|accuracy)", text)

        ret_val = ret_match.group(1) if ret_match else meta.get("backtest_metrics", {}).get("return_pct")
        wr_val = wr_match.group(1) if wr_match else meta.get("backtest_metrics", {}).get("win_rate_pct")

        # Check if strictly stock market opening (9:15 AM)
        is_opening_orb = bool(re.search(r"(9:15|9\.15|ओपनिंग रेंज|orb|opening range)", text))

        trading_strategies.append({
            "video_id": vid,
            "title": title,
            "clean_name": meta.get("clean_strategy_name", title),
            "creator": meta.get("creator_attribution", "Trader in Video"),
            "indicators": indicators_found,
            "return_pct": ret_val,
            "win_rate": wr_val,
            "is_orb": is_opening_orb,
            "transcript_snippet": transcript[:150]
        })

    print(f"[+] Total trading strategy videos identified: {len(trading_strategies)}")
    print(f"[-] Non-trading informational videos skipped: {skipped_non_trading}")

    # Group by indicators / actionable setups
    print("\n" + "=" * 80)
    print("ACTIONABLE STRATEGIES FROM FULL TRANSCRIPT CORPUS:")
    print("=" * 80)
    
    unique_setups = []
    seen_titles = set()

    for idx, s in enumerate(trading_strategies):
        ret_str = f"{s['return_pct']}%" if s['return_pct'] else "N/A"
        wr_str = f"{s['win_rate']}%" if s['win_rate'] else "N/A"
        inds = ", ".join(s['indicators']) if s['indicators'] else "Price Action / SMC"
        orb_tag = "[STOCK-ORB]" if s['is_orb'] else "[UNIVERSAL/MT5]"
        
        # Deduplicate identical setups
        base_name = s['clean_name'][:25]
        if base_name in seen_titles:
            continue
        seen_titles.add(base_name)
        unique_setups.append(s)

        print(f"{len(unique_setups):2d}. [{s['video_id']}] {orb_tag} {s['title'][:45]:45} | Inds: {inds[:25]:25} | Ret: {ret_str:>6} WR: {wr_str:>5}")

    with open("parsed_all_strategies.json", "w", encoding="utf-8") as f:
        json.dump(unique_setups, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Saved {len(unique_setups)} unique parsed strategies to parsed_all_strategies.json")

if __name__ == "__main__":
    analyze()
