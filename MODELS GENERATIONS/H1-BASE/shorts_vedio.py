"""
========================================================================================
STRATEGY INTELLIGENCE PIPELINE & YOUTUBE SHORTS SCRAPER
========================================================================================
Extracts, parses, and structures all algorithmic trading strategy data from short videos:
- Video metadata (Title, Link, ID, Duration, Description)
- Automatic speech-to-text transcript scraping via yt-dlp (Hindi/English audio captions)
- Strategy logic extraction: Indicators, Parameters, Timeframes, Setup Steps, Entry/Exit Rules
- Risk management (Stop Loss, Take Profit, Risk-to-Reward, Trailing SL)
- Backtest performance metrics (Return, Win Rate, Trades, Drawdown, Profit Factor)
- Machine-readable JSON output (strategies_data.json) & Human-readable Markdown specs (strategies_specs.md)
- Automated scanner template generator (generates Python/MT5 scanner files for ready strategies)
========================================================================================
"""

import os
import sys
import re
import json
import time
import glob
import argparse
import isodate
from typing import Dict, List, Optional, Any, Tuple
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import yt_dlp

# Set UTF-8 encoding for standard output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ================= Configuration =================
API_KEY = "AIzaSyCrRVSA-N30S-qpf6RTWVxMLtdZy8Cht7Y"
CHANNEL_IDENTIFIER = "@UnfilteredFinanceWithSahil"

MAX_SHORT_DURATION = 300  # seconds

# Output file paths
LINKS_FILE = "shorts_links.txt"
DESCRIPTIONS_FILE = "shorts_descriptions.txt"
TRANSCRIPTS_DIR = "transcripts"
TRANSCRIPTS_FILE = "shorts_transcripts.txt"
STRATEGIES_JSON = "strategies_data.json"
STRATEGIES_SPECS_MD = "strategies_specs.md"
GENERATED_SCANNERS_DIR = "generated_scanners"
# =================================================


# ========================================================================================
# 1. YOUTUBE API METADATA FETCHING
# ========================================================================================

def get_uploads_playlist_id(youtube, identifier: str) -> str:
    """Finds the 'Uploads' playlist ID using Channel ID or Handle."""
    if identifier.startswith("@"):
        resp = youtube.channels().list(
            part="contentDetails",
            forHandle=identifier
        ).execute()
    else:
        resp = youtube.channels().list(
            part="contentDetails",
            id=identifier
        ).execute()

    items = resp.get("items", [])
    if not items:
        raise ValueError(f"Channel not found for identifier: {identifier}")

    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]


def fetch_all_shorts_api(api_key: str, channel_identifier: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Fetches video snippets and content details from YouTube Data API v3."""
    youtube = build("youtube", "v3", developerKey=api_key)

    print(f"[*] Resolving channel: {channel_identifier}...")
    uploads_playlist_id = get_uploads_playlist_id(youtube, channel_identifier)
    print(f"[+] Found uploads playlist: {uploads_playlist_id}")

    shorts_list = []
    next_page_token = None
    total_scanned = 0

    print("[*] Fetching videos from YouTube API...")
    while True:
        playlist_request = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=uploads_playlist_id,
            maxResults=50,
            pageToken=next_page_token
        )
        playlist_response = playlist_request.execute()

        video_ids = [
            item["contentDetails"]["videoId"]
            for item in playlist_response.get("items", [])
        ]

        if not video_ids:
            break

        total_scanned += len(video_ids)
        print(f"    Scanned {total_scanned} uploads so far...")

        # Fetch snippet and duration in batch
        video_request = youtube.videos().list(
            part="snippet,contentDetails",
            id=",".join(video_ids)
        )
        video_response = video_request.execute()

        for item in video_response.get("items", []):
            duration_str = item["contentDetails"]["duration"]
            try:
                duration_seconds = isodate.parse_duration(duration_str).total_seconds()
            except Exception:
                duration_seconds = 0

            if 0 < duration_seconds <= MAX_SHORT_DURATION:
                video_id = item["id"]
                title = item["snippet"]["title"].strip()
                description = item["snippet"]["description"].strip()
                link = f"https://www.youtube.com/shorts/{video_id}"

                shorts_list.append({
                    "id": video_id,
                    "title": title,
                    "link": link,
                    "duration": duration_seconds,
                    "description": description
                })

                if limit and len(shorts_list) >= limit:
                    break

        if limit and len(shorts_list) >= limit:
            break

        next_page_token = playlist_response.get("nextPageToken")
        if not next_page_token:
            break

    print(f"[+] Completed! Found {len(shorts_list)} Shorts out of {total_scanned} total uploads.")
    return shorts_list


# ========================================================================================
# 2. LOCAL CACHE PARSER (OFFLINE MODE)
# ========================================================================================

def load_shorts_from_local_cache(descriptions_file: str = DESCRIPTIONS_FILE) -> List[Dict[str, Any]]:
    """Loads existing video entries from shorts_descriptions.txt without API hits."""
    if not os.path.exists(descriptions_file):
        print(f"[!] Cache file {descriptions_file} not found.")
        return []

    with open(descriptions_file, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = content.split("------------------------------------------------------------")
    shorts_list = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        title_match = re.search(r"===\s*\[(\d+)\]\s*(.*?)\s*===", block)
        link_match = re.search(r"Link:\s*(https?://[^\s]+)", block)

        if not title_match or not link_match:
            continue

        idx = int(title_match.group(1))
        title = title_match.group(2).strip()
        link = link_match.group(1).strip()
        video_id = link.split("/")[-1].split("?")[0]

        # Everything after the link line is the description
        lines = block.splitlines()
        desc_lines = []
        found_link = False
        for line in lines:
            if found_link:
                desc_lines.append(line)
            elif "Link:" in line:
                found_link = True

        description = "\n".join(desc_lines).strip()

        shorts_list.append({
            "index": idx,
            "id": video_id,
            "title": title,
            "link": link,
            "duration": 60,
            "description": description
        })

    print(f"[+] Loaded {len(shorts_list)} shorts from local cache: {descriptions_file}")
    return shorts_list


# ========================================================================================
# 3. VIDEO TRANSCRIPT SCRAPER (YT-DLP)
# ========================================================================================

def clean_vtt_subtitles(vtt_text: str) -> str:
    """Cleans VTT/timedtext timestamps, cue settings, and duplicate lines into clean text."""
    lines = vtt_text.splitlines()
    cleaned_tokens = []
    seen_lines = set()

    for line in lines:
        line = line.strip()
        if not line or "WEBVTT" in line or "Kind:" in line or "Language:" in line or "-->" in line:
            continue
        # Remove inner tags like <00:00:01.200><c>word</c>
        clean_line = re.sub(r"<[^>]+>", "", line).strip()
        # Remove bracketed artifacts like [संगीत], [Music]
        clean_line = re.sub(r"\[.*?\]", "", clean_line).strip()

        if clean_line and clean_line not in seen_lines:
            cleaned_tokens.append(clean_line)
            seen_lines.add(clean_line)

    return " ".join(cleaned_tokens).strip()


def fetch_video_transcript(video_id: str, cache_dir: str = TRANSCRIPTS_DIR, delay: float = 1.0, allow_network: bool = True) -> str:
    """
    Downloads and caches speech-to-text transcript (Hindi / English) for a video.
    Returns clean transcript string.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cached_file = os.path.join(cache_dir, f"{video_id}.txt")

    # Return cached transcript if already downloaded
    if os.path.exists(cached_file):
        with open(cached_file, "r", encoding="utf-8") as f:
            return f.read().strip()

    if not allow_network:
        return ""

    url = f"https://www.youtube.com/shorts/{video_id}"
    temp_prefix = os.path.join(cache_dir, f"temp_{video_id}")

    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["hi", "en"],
        "subtitlesformat": "vtt",
        "ignoreerrors": True,
        "quiet": True,
        "no_warnings": True,
        "outtmpl": f"{temp_prefix}_%(id)s"
    }

    transcript_text = ""
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Look for downloaded vtt files
        vtt_files = glob.glob(f"{temp_prefix}*.vtt")
        chosen_vtt = None
        for fpath in vtt_files:
            if ".hi." in fpath:
                chosen_vtt = fpath
                break
        if not chosen_vtt and vtt_files:
            chosen_vtt = vtt_files[0]

        if chosen_vtt and os.path.exists(chosen_vtt):
            with open(chosen_vtt, "r", encoding="utf-8") as fp:
                transcript_text = clean_vtt_subtitles(fp.read())

        # Clean up temporary vtt files
        for fpath in glob.glob(f"{temp_prefix}*"):
            try:
                os.remove(fpath)
            except Exception:
                pass

        # Cache clean transcript
        if transcript_text:
            with open(cached_file, "w", encoding="utf-8") as f:
                f.write(transcript_text)

        if delay > 0:
            time.sleep(delay)

    except Exception as e:
        print(f"    [!] Warning: Failed transcript fetch for {video_id}: {e}")

    return transcript_text


# ========================================================================================
# 4. STRATEGY INTELLIGENCE & RULE EXTRACTION ENGINE
# ========================================================================================

class StrategyIntelligenceParser:
    """Parses raw video descriptions and audio transcripts into structured algo strategies."""

    KNOWN_CREATORS = [
        ("Wizard Trader", r"Wizard\s*Trader"),
        ("Neeraj Joshi", r"Neeraj\s*Joshi"),
        ("Vishal Malkan", r"Vishal\s*Malkan"),
        ("Umar Punjabi", r"Umar\s*Punjabi"),
        ("Anish Thakur / Booming Bulls", r"Booming\s*Bulls|Anish\s*Thakur"),
        ("Stock Burner", r"Stock\s*Burner"),
        ("PB Trading", r"PB\s*Trading"),
        ("Casper SMC", r"Casper|Jesee\s*Rodgers"),
        ("TG Capital", r"TG\s*Capital"),
        ("Harindra Sahu / King Research", r"King\s*Research|Harindra\s*Sahu"),
        ("Gomathi Shankar", r"Gomathi\s*Shankar"),
        ("Kushal Varshney", r"Kushal\s*Varshney"),
        ("The Moving Average", r"The\s*Moving\s*Average"),
        ("Dhan Broker", r"Dhan\s*Broker"),
        ("Stock Learners", r"Stock\s*Learners"),
        ("Hold with Priyank", r"Hold\s*with\s*Priyank"),
        ("Topgtraders", r"Topgtraders"),
        ("ICT / Michael Huddleston", r"ICT|Inner\s*Circle\s*Trader")
    ]

    INDICATOR_PATTERNS = [
        ("EMA", r"\b(?:EMA|Exponential Moving Average)\b(?:\s*(\d+(?:\s*(?:aur|and|,|\/)\s*\d+)*))?", r"(\d+)\s*(?:EMA|ईएमए)"),
        ("SMA", r"\b(?:SMA|Simple Moving Average)\b(?:\s*(\d+))?", r"(\d+)\s*(?:SMA|एसएमए)"),
        ("RSI", r"\b(?:RSI|Relative Strength Index|आरएसआई)\b(?:\s*\(?(\d+)\)?)?", None),
        ("Supertrend", r"\b(?:Supertrend|Super\s*Trend|सुपर\s*ट्रेंड)\b", None),
        ("Fair Value Gap (FVG)", r"\b(?:FVG|Fair\s*Value\s*Gap|Inversion\s*FVG)\b", None),
        ("Order Block / Breaker Block", r"\b(?:Order\s*Block|Breaker\s*Block|Bullish\s*OB|Bearish\s*OB)\b", None),
        ("Fibonacci Retracement", r"\b(?:Fibonacci|Fib|फिबोनाची|Golden\s*Zone|Golden\s*Pocket)\b", None),
        ("William Fractal", r"\b(?:William\s*Fractal|Fractal|फ्रैक्टल)\b", None),
        ("Camarilla Pivot", r"\b(?:Camarilla|कैमरिला|L3|L4|H3|H4)\b", None),
        ("Central Pivot Range (CPR)", r"\b(?:CPR|Central\s*Pivot\s*Range)\b", None),
        ("MACD", r"\b(?:MACD)\b", None),
        ("Stochastic", r"\b(?:Stochastic|स्टोकेस्टिक)\b", None),
        ("Bollinger Bands", r"\b(?:Bollinger\s*Bands?|BB)\b", None),
        ("VWAP", r"\b(?:VWAP|Volume\s*Weighted\s*Average\s*Price)\b", None),
        ("ADX", r"\b(?:ADX)\b", None),
        ("ATR", r"\b(?:ATR|Average\s*True\s*Range|एटीआर)\b", None),
        ("Inside Bar", r"\b(?:Inside\s*Bar|Mother\s*Bar)\b", None),
        ("Volume Profile / FRVP", r"\b(?:FRVP|Volume\s*Profile|Fixed\s*Range\s*Volume)\b", None),
        ("Gann Levels", r"\b(?:Gann|Gann\s*Levels)\b", None),
        ("Trendline Breakout", r"\b(?:Trendline|Trend\s*Line|ट्रेंडलाइन)\b", None),
        ("Liquidity Sweep", r"\b(?:Liquidity|Sweep|लिक्विडिटी)\b", None)
    ]

    TIMEFRAME_PATTERNS = [
        ("1M", r"\b(?:1M|1\s*Min|1\s*Minute|1\s*मिनट)\b"),
        ("3M", r"\b(?:3M|3\s*Min|3\s*Minute|3\s*मिनट)\b"),
        ("5M", r"\b(?:5M|5\s*Min|5\s*Minute|5\s*मिनट)\b"),
        ("15M", r"\b(?:15M|15\s*Min|15\s*Minute|15\s*मिनट)\b"),
        ("30M", r"\b(?:30M|30\s*Min|30\s*Minute|30\s*मिनट)\b"),
        ("1H", r"\b(?:1H|1\s*Hour|1\s*hr|1\s*घंटा)\b"),
        ("4H", r"\b(?:4H|4\s*Hour|4\s*hr|4\s*घंटे)\b"),
        ("Daily", r"\b(?:Daily|1D|D1|दिन)\b")
    ]

    ASSET_PATTERNS = [
        ("Forex (EURUSD / Major Pairs)", r"\b(?:Forex|EURUSD|GBPUSD|USDJPY|फॉररेक्स)\b"),
        ("Gold (XAUUSD)", r"\b(?:Gold|XAUUSD|गोल्ड|सोना)\b"),
        ("Indian Indices (Nifty / BankNifty)", r"\b(?:Nifty|BankNifty|Bank\s*Nifty|निफ्टी|बैंकनिफ्टी)\b"),
        ("Crypto (BTC / ETH)", r"\b(?:Crypto|Bitcoin|BTC|ETH|USDT|क्रिप्टो)\b"),
        ("Options Trading", r"\b(?:Options?|Call|Put|ATM|OTM|ऑप्शंस?)\b"),
        ("Equities / Stocks", r"\b(?:Stocks|Shares|Equity|स्टॉक्स)\b")
    ]

    @classmethod
    def parse(cls, entry: Dict[str, Any], transcript: str = "") -> Dict[str, Any]:
        title = entry.get("title", "")
        description = entry.get("description", "")
        link = entry.get("link", "")
        video_id = entry.get("id", "")
        combined_text = f"{title}\n\n{description}\n\n{transcript}"

        # 1. Attributed Creator
        creator = "Independent / Sahil Algo Setup"
        for name, pattern in cls.KNOWN_CREATORS:
            if re.search(pattern, combined_text, re.I):
                creator = name
                break

        # 2. Strategy Name Clean Title
        clean_name = re.sub(r"—\s*Does.*Hold\?", "", title)
        clean_name = re.sub(r"—\s*I\s*Algo\s*Backtested.*", "", clean_name, flags=re.I)
        clean_name = re.sub(r"—\s*Full\s*Breakdown.*", "", clean_name, flags=re.I)
        clean_name = re.sub(r"Algo\s*Backtest.*", "", clean_name, flags=re.I)
        clean_name = clean_name.strip(" —-|:")

        # 3. Target Assets
        assets = []
        for asset_name, pat in cls.ASSET_PATTERNS:
            if re.search(pat, combined_text, re.I):
                assets.append(asset_name)
        if not assets:
            assets = ["Universal / Forex & Equities"]

        # 4. Timeframes
        timeframes = []
        for tf, pat in cls.TIMEFRAME_PATTERNS:
            if re.search(pat, combined_text, re.I):
                timeframes.append(tf)

        higher_tf = None
        lower_tf = None
        tf_order = ["1M", "3M", "5M", "15M", "30M", "1H", "4H", "Daily"]
        sorted_tfs = sorted(timeframes, key=lambda x: tf_order.index(x) if x in tf_order else 99)
        if len(sorted_tfs) >= 2:
            lower_tf = sorted_tfs[0]
            higher_tf = sorted_tfs[-1]
        elif len(sorted_tfs) == 1:
            lower_tf = sorted_tfs[0]

        # 5. Indicators & Parameters
        indicators_found = []
        for ind_name, pat1, pat2 in cls.INDICATOR_PATTERNS:
            match1 = re.search(pat1, combined_text, re.I) if pat1 else None
            match2 = re.search(pat2, combined_text, re.I) if pat2 else None
            if match1 or match2:
                param_str = ""
                if match1 and match1.lastindex and match1.group(1):
                    param_str = match1.group(1).strip()
                elif match2 and match2.lastindex and match2.group(1):
                    param_str = match2.group(1).strip()

                indicators_found.append({
                    "name": ind_name,
                    "parameters": param_str if param_str else "Default / Video Specified"
                })

        # Specific indicator parameter extraction
        # EMA specific periods
        ema_nums = re.findall(r"\b(\d+)\s*(?:EMA|ईएमए)\b", combined_text, re.I)
        ema_nums += re.findall(r"\bEMA\s*\(?(\d+)\)?\b", combined_text, re.I)
        if ema_nums:
            unique_emas = sorted(list(set([int(n) for n in ema_nums if int(n) < 500])))
            for ind in indicators_found:
                if ind["name"] == "EMA":
                    ind["parameters"] = f"Periods: {', '.join(map(str, unique_emas))}"

        # Supertrend specific params
        st_atr = re.search(r"(?:ATR\s*Period|एटीआर\s*पीरियड)\s*(?:is|:)?\s*(\d+)", combined_text, re.I)
        st_mult = re.search(r"(?:Multiplier|मल्टीप्लायर)\s*(?:is|:)?\s*(\d+(?:\.\d+)?)", combined_text, re.I)
        if st_atr or st_mult:
            for ind in indicators_found:
                if ind["name"] == "Supertrend":
                    atr_p = st_atr.group(1) if st_atr else "10"
                    mult_p = st_mult.group(1) if st_mult else "3.0"
                    ind["parameters"] = f"ATR Period: {atr_p}, Multiplier: {mult_p}"

        # RSI specific params
        rsi_bull = re.search(r"RSI\s*(?:cross|ऊपर|above)\s*(\d+)", combined_text, re.I)
        rsi_bear = re.search(r"RSI\s*(?:below|neeche|नीचे)\s*(\d+)", combined_text, re.I)
        if rsi_bull or rsi_bear:
            for ind in indicators_found:
                if ind["name"] == "RSI":
                    b_val = rsi_bull.group(1) if rsi_bull else "60"
                    s_val = rsi_bear.group(1) if rsi_bear else "40"
                    ind["parameters"] = f"Period: 14, Bull Threshold: {b_val}, Bear Threshold: {s_val}"

        # 6. Sequential Step-by-Step Logic
        steps = []
        # Try extracting bullet points from description
        desc_logic = re.search(r"Strategy ka logic:(.*?)(?:Algo Backtest Results|Equity curve|📌|\Z)", description, re.S | re.I)
        if desc_logic:
            for line in desc_logic.group(1).splitlines():
                line = line.strip()
                if line.startswith(("✅", "-", "*", "•")) or re.match(r"^\d+\.", line):
                    clean_l = re.sub(r"^(?:[✅\-*•\s]|\b\d+[\.\)]\s*)+", "", line).strip()
                    if clean_l:
                        steps.append(clean_l)

        # If no bullet steps in description, synthesize from transcript
        if not steps and transcript:
            sentences = re.split(r"[।\.\n]+", transcript)
            for s in sentences:
                s = s.strip()
                if any(k in s for k in ["रूल", "कंडीशन", "एंट्री", "स्टॉप लॉस", "टारगेट", "ईएमए के ऊपर", "क्रॉस करे"]):
                    if len(s) > 10:
                        steps.append(s)

        # 7. Entry Rules
        long_entry = None
        short_entry = None
        for step in steps:
            if re.search(r"\b(?:Buy|Long|बाय|लॉन्ग)\b", step, re.I) and not long_entry and "ulta" not in step.lower():
                long_entry = step
            if re.search(r"\b(?:Sell|Short|सेल|शॉर्ट)\b", step, re.I) and not short_entry and "ulta" not in step.lower():
                short_entry = step

        if not long_entry and any("buy" in s.lower() for s in steps):
            long_entry = next(s for s in steps if "buy" in s.lower())
        if not short_entry:
            short_entry = "Inverted logic of Buy setup (Downtrend alignment + Bearish trigger)"

        # 8. Risk Management (Stop Loss & Take Profit)
        stop_loss = "Below recent swing low / key structural support"
        take_profit = "1:2 Risk-to-Reward ratio"
        rr_ratio = "1:2"

        for step in steps:
            if re.search(r"\b(?:Stop-Loss|Stop\s*Loss|SL|स्टॉप\s*लॉस)\b", step, re.I):
                stop_loss = step
            if re.search(r"\b(?:Target|TP|टारगेट|Risk\s*Reward)\b", step, re.I):
                take_profit = step
                rr_m = re.search(r"1\s*:\s*(\d+(?:\.\d+)?)", step)
                if rr_m:
                    rr_ratio = f"1:{rr_m.group(1)}"

        # Search transcript if not found in steps
        if stop_loss == "Below recent swing low / key structural support":
            sl_m = re.search(r"(?:स्टॉप लॉस|Stop Loss)\s*([^\.।\n]+)", combined_text, re.I)
            if sl_m:
                stop_loss = sl_m.group(1).strip()

        if take_profit == "1:2 Risk-to-Reward ratio":
            tp_m = re.search(r"(?:टारगेट|Target)\s*([^\.।\n]+)", combined_text, re.I)
            if tp_m:
                take_profit = tp_m.group(1).strip()
                rr_m = re.search(r"1\s*:\s*(\d+(?:\.\d+)?)", take_profit)
                if rr_m:
                    rr_ratio = f"1:{rr_m.group(1)}"

        # 9. Backtest Metrics
        ret_m = re.search(r"(?:Total Return|Return|टोटल रिटर्न|रिटर्न):\s*([+-]?\d+(?:\.\d+)?%?)", combined_text, re.I)
        win_m = re.search(r"(?:Win Rate|Accuracy|विन रेट):\s*(\d+(?:\.\d+)?%?)", combined_text, re.I)
        trades_m = re.search(r"(?:Total Trades|Trades|टोटल ट्रेड्स):\s*(\d+)", combined_text, re.I)
        dd_m = re.search(r"(?:Max Drawdown|Drawdown|ड्रॉ डाउन):\s*([+-]?\d+(?:\.\d+)?%?)", combined_text, re.I)
        pf_m = re.search(r"(?:Profit Factor|प्रॉफिट फैक्टर):\s*(\d+(?:\.\d+)?)", combined_text, re.I)

        # 10. Strategy Category Classification
        title_lower = title.lower()
        if any(k in title_lower for k in ["ict", "smc", "breaker block", "order block", "liquidity", "fvg"]):
            category = "SMC_ICT"
        elif any(k in title_lower for k in ["ema", "sma", "moving average"]):
            category = "EMA_TREND_FOLLOWING"
        elif any(k in title_lower for k in ["rsi", "macd", "stochastic"]):
            category = "MOMENTUM_OSCILLATOR"
        elif any(k in title_lower for k in ["supertrend"]):
            category = "SUPERTREND_VOLATILITY"
        elif any(k in title_lower for k in ["fibonacci"]):
            category = "FIBONACCI_RETRACEMENT"
        elif any(k in title_lower for k in ["camarilla", "cpr", "pivot", "gann"]):
            category = "PIVOT_MATHEMATICAL"
        elif any(k in title_lower for k in ["option", "vishal malkan"]):
            category = "OPTIONS_DERIVATIVES"
        elif any(k in title_lower for k in ["scalping", "fractal", "vwap"]):
            category = "SCALPING_MICRO"
        elif any(k in title_lower for k in ["offer", "course", "giveaway", "independence", "surprise"]):
            category = "PROMOTIONAL_ANNOUNCEMENT"
        else:
            category = "PRICE_ACTION"

        is_strategy = category != "PROMOTIONAL_ANNOUNCEMENT" and (len(indicators_found) > 0 or win_m is not None or len(steps) > 0)

        # Readiness Score
        if len(steps) >= 3 and len(indicators_found) >= 1 and (win_m or ret_m):
            readiness = "READY_TO_BUILD"
        elif len(indicators_found) >= 1 and (steps or win_m):
            readiness = "PARTIAL_SPECS"
        elif is_strategy:
            readiness = "CONCEPT_ONLY"
        else:
            readiness = "NOT_A_STRATEGY"

        # Scanner Script Recommendation
        slug = re.sub(r"[^a-zA-Z0-9_]+", "_", clean_name.lower()).strip("_")
        suggested_scanner = f"eurusd_{slug[:25]}_scanner.py"

        return {
            "index": entry.get("index", 0),
            "video_id": video_id,
            "title": title,
            "clean_strategy_name": clean_name,
            "creator_attribution": creator,
            "video_url": link,
            "is_actionable_strategy": is_strategy,
            "category": category,
            "readiness_score": readiness,
            "suggested_scanner_file": suggested_scanner,
            "market_instruments": assets,
            "timeframes": {
                "higher_trend_tf": higher_tf,
                "execution_trigger_tf": lower_tf or "5M (Default Intraday)"
            },
            "indicators": indicators_found,
            "condition_sequence_steps": steps,
            "trade_rules": {
                "long_entry": long_entry or "Buy on confirmed bullish signal matching setup criteria",
                "short_entry": short_entry or "Sell on confirmed bearish signal matching setup criteria",
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_reward_ratio": rr_ratio
            },
            "backtest_metrics": {
                "win_rate": win_m.group(1) if win_m else None,
                "total_return": ret_m.group(1) if ret_m else None,
                "max_drawdown": dd_m.group(1) if dd_m else None,
                "total_trades": int(trades_m.group(1)) if trades_m else None,
                "profit_factor": float(pf_m.group(1)) if pf_m else None
            },
            "raw_sources": {
                "has_transcript": bool(transcript),
                "transcript_length": len(transcript),
                "transcript_preview": transcript[:300] if transcript else ""
            }
        }


# ========================================================================================
# 5. SCANNER CODE GENERATOR
# ========================================================================================

def generate_scanner_template(strategy: Dict[str, Any]) -> str:
    """Generates a standalone Python/MetaTrader5 scanner script for an extracted strategy."""
    name = strategy["clean_strategy_name"]
    category = strategy["category"]
    exec_tf = strategy["timeframes"]["execution_trigger_tf"]
    rr_str = strategy["trade_rules"]["risk_reward_ratio"]
    rr_num = 2.0
    m = re.search(r"1:(\d+(?:\.\d+)?)", rr_str)
    if m:
        try:
            rr_num = float(m.group(1))
        except Exception:
            rr_num = 2.0

    mt5_tf_map = {
        "1M": "mt5.TIMEFRAME_M1",
        "3M": "mt5.TIMEFRAME_M3",
        "5M": "mt5.TIMEFRAME_M5",
        "15M": "mt5.TIMEFRAME_M15",
        "30M": "mt5.TIMEFRAME_M30",
        "1H": "mt5.TIMEFRAME_H1",
        "4H": "mt5.TIMEFRAME_H4",
        "Daily": "mt5.TIMEFRAME_D1"
    }
    tf_const = mt5_tf_map.get(exec_tf, "mt5.TIMEFRAME_M5")

    indicators = strategy.get("indicators", [])
    ind_comments = "\n".join([f"# - {ind['name']} ({ind['parameters']})" for ind in indicators])
    steps_comments = "\n".join([f"# Step {i}: {s}" for i, s in enumerate(strategy.get("condition_sequence_steps", []), 1)])

    code = f'''"""
========================================================================================
STRATEGY SCANNER: {name}
========================================================================================
Category: {category}
Attributed Creator: {strategy.get("creator_attribution", "Unknown")}
Source Video: {strategy.get("video_url", "")}
Claimed Backtest: Return {strategy["backtest_metrics"].get("total_return")}, Win Rate {strategy["backtest_metrics"].get("win_rate")}
========================================================================================
RULES & LOGIC:
{steps_comments}
========================================================================================
"""

import time
import datetime
import pandas as pd
import numpy as np
import MetaTrader5 as mt5

# ==========================================
# 1. CONFIGURATION & STRATEGY SETTINGS
# ==========================================
SYMBOL = "EURUSD.0"
TIMEFRAME = {tf_const}
RISK_REWARD_RATIO = {rr_num}
POLL_INTERVAL = 5

{ind_comments}

# ==========================================
# 2. MT5 INITIALIZATION
# ==========================================
def initialize_mt5():
    if not mt5.initialize():
        print(f"❌ MT5 Initialization failed: {{mt5.last_error()}}")
        return False
    
    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is None or not symbol_info.visible:
        if not mt5.symbol_select(SYMBOL, True):
            print(f"❌ Failed to select symbol {{SYMBOL}}")
            return False
            
    print(f"✅ Connected to MT5. {name} Scanner active for {{SYMBOL}}...")
    return True

# ==========================================
# 3. DATA FETCH & INDICATORS
# ==========================================
def fetch_data():
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 150)
    if rates is None or len(rates) == 0:
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Calculate indicators
    df['ema_fast'] = df['close'].ewm(span=8, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=33, adjust=False).mean()
    
    # RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    return df

# ==========================================
# 4. MAIN SCANNER LOOP
# ==========================================
def run_scanner():
    if not initialize_mt5():
        return

    last_candle_time = None
    print("\\n" + "=" * 65)
    print("      {name.upper()} SCANNER ({{SYMBOL}})      ")
    print("=" * 65)

    try:
        while True:
            df = fetch_data()
            if df is None or len(df) < 50:
                time.sleep(POLL_INTERVAL)
                continue

            current_candle_time = df['time'].iloc[-1]
            closed_bar = df.iloc[-2]
            current_bar = df.iloc[-1]

            # Trigger condition logic
            if current_candle_time != last_candle_time:
                last_candle_time = current_candle_time
                print(f"[{{datetime.datetime.now().strftime('%H:%M:%S')}}] Checking bar {{current_candle_time}}...")

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\\n🛑 Scanner stopped by user.")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    run_scanner()
'''
    return code


# ========================================================================================
# 6. EXPORT & ARTIFACT GENERATION
# ========================================================================================

def save_outputs(
    shorts_raw: List[Dict[str, Any]],
    parsed_strategies: List[Dict[str, Any]],
    transcripts_map: Dict[str, str],
    is_subset: bool = False,
    generate_scanners: bool = False
):
    """Saves all outputs: links, descriptions, transcripts, JSON database, and Markdown specs."""
    # If this is not a limited subset, update the primary cache files
    if not is_subset:
        # 1. shorts_links.txt
        with open(LINKS_FILE, "w", encoding="utf-8") as f:
            for idx, item in enumerate(shorts_raw, 1):
                f.write(f"{idx}. {item['title']} - {item['link']}\n")
        print(f"[+] Saved links to: {LINKS_FILE}")

        # 2. shorts_descriptions.txt
        with open(DESCRIPTIONS_FILE, "w", encoding="utf-8") as f:
            for idx, item in enumerate(shorts_raw, 1):
                f.write(f"=== [{idx}] {item['title']} ===\n")
                f.write(f"Link: {item['link']}\n\n")
                f.write(f"{item.get('description', '[No description provided]')}\n\n")
                f.write("-" * 60 + "\n\n")
        print(f"[+] Saved descriptions to: {DESCRIPTIONS_FILE}")

        # 3. shorts_transcripts.txt
        with open(TRANSCRIPTS_FILE, "w", encoding="utf-8") as f:
            for idx, item in enumerate(shorts_raw, 1):
                vid = item.get("id", "")
                t_text = transcripts_map.get(vid, "[No transcript available]")
                f.write(f"=== [{idx}] {item['title']} ===\n")
                f.write(f"Link: {item['link']}\n\n")
                f.write(f"{t_text}\n\n")
                f.write("-" * 60 + "\n\n")
        print(f"[+] Saved combined transcripts to: {TRANSCRIPTS_FILE}")

    # Output file destinations
    json_path = STRATEGIES_JSON
    md_path = STRATEGIES_SPECS_MD
    if is_subset and len(parsed_strategies) == 1:
        vid_slug = parsed_strategies[0]["video_id"]
        json_path = f"strategy_{vid_slug}.json"
        md_path = f"strategy_{vid_slug}.md"

    # 4. JSON database
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(parsed_strategies, f, indent=2, ensure_ascii=False)
    print(f"[+] Saved structured strategies database to: {json_path}")

    # 5. Markdown specifications
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Algorithmic Trading Strategies Catalog\n\n")
        f.write("> **Source Channel**: `@UnfilteredFinanceWithSahil`\n")
        f.write(f"> **Total Videos Analyzed**: {len(parsed_strategies)}\n")
        actionable_count = sum(1 for s in parsed_strategies if s["is_actionable_strategy"])
        f.write(f"> **Actionable Strategies**: {actionable_count}\n\n")

        f.write("## Quick Index & Performance Summary\n\n")
        f.write("| # | Strategy Name | Category | Timeframe | Win Rate | Return | Drawdown | SL / TP | Readiness |\n")
        f.write("|---|---------------|----------|-----------|----------|--------|----------|---------|-----------|\n")

        for s in parsed_strategies:
            if not s["is_actionable_strategy"]:
                continue
            idx = s["index"]
            name = s["clean_strategy_name"][:32]
            cat = s["category"]
            tf = s["timeframes"]["execution_trigger_tf"]
            m = s["backtest_metrics"]
            wr = m.get("win_rate") or "N/A"
            ret = m.get("total_return") or "N/A"
            dd = m.get("max_drawdown") or "N/A"
            rr = s["trade_rules"]["risk_reward_ratio"]
            ready = s["readiness_score"]
            f.write(f"| {idx} | [{name}](#strategy-{idx}) | `{cat}` | {tf} | **{wr}** | **{ret}** | {dd} | {rr} | `{ready}` |\n")

        f.write("\n---\n\n")
        f.write("## Detailed Strategy Specifications\n\n")

        for s in parsed_strategies:
            idx = s["index"]
            title = s["title"]
            clean_name = s["clean_strategy_name"]
            url = s["video_url"]
            cat = s["category"]
            creator = s["creator_attribution"]
            ready = s["readiness_score"]
            scanner_file = s["suggested_scanner_file"]
            m = s["backtest_metrics"]

            f.write(f"<a id=\"strategy-{idx}\"></a>\n\n")
            f.write(f"### [{idx}] {clean_name}\n\n")
            f.write(f"- **Original Title**: {title}\n")
            f.write(f"- **Video URL**: [{url}]({url})\n")
            f.write(f"- **Attributed Creator**: {creator}\n")
            f.write(f"- **Category**: `{cat}` | **Readiness**: `{ready}`\n")
            f.write(f"- **Suggested Scanner File**: `{scanner_file}`\n\n")

            f.write("#### 1. Setup & Market Context\n")
            f.write(f"- **Target Instruments**: {', '.join(s['market_instruments'])}\n")
            f.write(f"- **Higher Context Timeframe**: {s['timeframes']['higher_trend_tf'] or 'N/A'}\n")
            f.write(f"- **Execution Timeframe**: {s['timeframes']['execution_trigger_tf']}\n\n")

            f.write("#### 2. Indicator Parameters\n")
            if s["indicators"]:
                f.write("| Indicator | Parameters / Thresholds |\n")
                f.write("|-----------|--------------------------|\n")
                for ind in s["indicators"]:
                    f.write(f"| **{ind['name']}** | {ind['parameters']} |\n")
            else:
                f.write("- Pure Price Action / Chart Pattern (No secondary indicators required)\n")
            f.write("\n")

            f.write("#### 3. Condition Sequence Steps\n")
            if s["condition_sequence_steps"]:
                for step_i, step_text in enumerate(s["condition_sequence_steps"], 1):
                    f.write(f"{step_i}. {step_text}\n")
            else:
                f.write("1. Follow market trend direction on primary timeframe.\n")
                f.write("2. Wait for confirmation trigger bar.\n")
            f.write("\n")

            f.write("#### 4. Execution & Risk Rules\n")
            f.write(f"- **Long (Buy)**: {s['trade_rules']['long_entry']}\n")
            f.write(f"- **Short (Sell)**: {s['trade_rules']['short_entry']}\n")
            f.write(f"- **Stop Loss**: {s['trade_rules']['stop_loss']}\n")
            f.write(f"- **Take Profit**: {s['trade_rules']['take_profit']}\n")
            f.write(f"- **Target RR**: `{s['trade_rules']['risk_reward_ratio']}`\n\n")

            f.write("#### 5. Backtest Claims\n")
            f.write(f"- **Win Rate**: {m.get('win_rate') or 'N/A'}\n")
            f.write(f"- **Total Return**: {m.get('total_return') or 'N/A'}\n")
            f.write(f"- **Max Drawdown**: {m.get('max_drawdown') or 'N/A'}\n")
            f.write(f"- **Total Trades**: {m.get('total_trades') or 'N/A'}\n")
            if m.get("profit_factor"):
                f.write(f"- **Profit Factor**: {m['profit_factor']}\n")
            f.write("\n---\n\n")

    print(f"[+] Saved Markdown specifications dossier to: {md_path}")

    # 6. Optional: Generate scanner templates
    if generate_scanners:
        os.makedirs(GENERATED_SCANNERS_DIR, exist_ok=True)
        gen_count = 0
        for s in parsed_strategies:
            if s["readiness_score"] in ["READY_TO_BUILD", "PARTIAL_SPECS"]:
                filename = s["suggested_scanner_file"]
                filepath = os.path.join(GENERATED_SCANNERS_DIR, filename)
                template_code = generate_scanner_template(s)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(template_code)
                gen_count += 1
        print(f"[+] Generated {gen_count} scanner script templates in '{GENERATED_SCANNERS_DIR}/'")


# ========================================================================================
# 7. MAIN EXECUTION PIPELINE
# ========================================================================================

def run_pipeline(
    from_cache: bool = False,
    video_target: Optional[str] = None,
    limit: Optional[int] = None,
    fetch_transcripts: bool = False,
    delay_between_transcripts: float = 1.0,
    generate_scanners: bool = False
):
    print("=" * 70)
    print("     STRATEGY INTELLIGENCE PIPELINE & YOUTUBE SHORTS SCRAPER")
    print("=" * 70)

    shorts_raw = []

    # Mode 1: Single target video
    if video_target:
        vid_id = video_target.split("/")[-1].split("?")[0]
        print(f"[*] Processing single target video: {vid_id}")
        
        # Check if already present in cached descriptions
        cached_all = load_shorts_from_local_cache(DESCRIPTIONS_FILE) if os.path.exists(DESCRIPTIONS_FILE) else []
        match = next((s for s in cached_all if s["id"] == vid_id), None)
        if match:
            shorts_raw = [match]
            print(f"[+] Found cached metadata for: {match['title']}")
        else:
            title = f"Video {vid_id}"
            description = ""
            try:
                youtube = build("youtube", "v3", developerKey=API_KEY)
                resp = youtube.videos().list(part="snippet,contentDetails", id=vid_id).execute()
                items = resp.get("items", [])
                if items:
                    title = items[0]["snippet"]["title"].strip()
                    description = items[0]["snippet"]["description"].strip()
            except Exception:
                pass
            shorts_raw = [{
                "index": 1,
                "id": vid_id,
                "title": title,
                "link": f"https://www.youtube.com/shorts/{vid_id}",
                "duration": 60,
                "description": description
            }]

    # Mode 2: From local cache (shorts_descriptions.txt)
    elif from_cache:
        print("[*] Running in OFFLINE/CACHE mode...")
        shorts_raw = load_shorts_from_local_cache(DESCRIPTIONS_FILE)

    # Mode 3: Fetch fresh from YouTube API
    else:
        try:
            print("[*] Connecting to YouTube Data API v3...")
            api_shorts = fetch_all_shorts_api(API_KEY, CHANNEL_IDENTIFIER, limit=limit)
            for idx, item in enumerate(api_shorts, 1):
                item["index"] = idx
                shorts_raw.append(item)
        except HttpError as err:
            print(f"[!] YouTube API Error: {err}")
            print("[*] Falling back to local cache...")
            shorts_raw = load_shorts_from_local_cache(DESCRIPTIONS_FILE)
        except Exception as e:
            print(f"[!] Error: {e}. Falling back to cache...")
            shorts_raw = load_shorts_from_local_cache(DESCRIPTIONS_FILE)

    if not shorts_raw:
        print("[!] No shorts found to process. Exiting.")
        return

    is_subset = bool(limit and len(shorts_raw) > limit) or bool(video_target)
    if limit and len(shorts_raw) > limit:
        shorts_raw = shorts_raw[:limit]

    print(f"\n[*] Processing {len(shorts_raw)} video entries...")

    # Fetch transcripts & parse strategies
    transcripts_map = {}
    parsed_strategies = []

    for i, item in enumerate(shorts_raw, 1):
        vid_id = item["id"]
        title = item["title"]

        # Pre-check if video is an actionable strategy before fetching transcript
        pre_parsed = StrategyIntelligenceParser.parse(item, "")
        is_actionable = pre_parsed["is_actionable_strategy"] or bool(video_target)

        # Fetch transcript: allow network if fetch_transcripts is True and video is actionable
        allow_net = (fetch_transcripts and is_actionable) or bool(video_target)
        transcript = fetch_video_transcript(
            vid_id,
            cache_dir=TRANSCRIPTS_DIR,
            delay=delay_between_transcripts,
            allow_network=allow_net
        )

        if transcript:
            transcripts_map[vid_id] = transcript
        
        parsed = StrategyIntelligenceParser.parse(item, transcript)
        parsed["index"] = item.get("index", i)
        parsed_strategies.append(parsed)

    # Save all output files
    print("\n[*] Generating output databases and specification files...")
    save_outputs(
        shorts_raw,
        parsed_strategies,
        transcripts_map,
        is_subset=is_subset,
        generate_scanners=generate_scanners
    )

    print("\n" + "=" * 70)
    print("✅ Pipeline execution completed successfully!")
    print(f"1. Structured JSON:      {STRATEGIES_JSON}")
    print(f"2. Markdown Dossier:     {STRATEGIES_SPECS_MD}")
    if not is_subset:
        print(f"3. All Transcripts:      {TRANSCRIPTS_FILE}")
        print(f"4. Links File:           {LINKS_FILE}")
        print(f"5. Descriptions File:    {DESCRIPTIONS_FILE}")
    print(f"6. Transcripts Cache:    {TRANSCRIPTS_DIR}/")
    if generate_scanners:
        print(f"7. Generated Scanners:   {GENERATED_SCANNERS_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strategy Intelligence Pipeline for YouTube Shorts")
    parser.add_argument("--from-cache", action="store_true", help="Process existing shorts_descriptions.txt without API call")
    parser.add_argument("--video", type=str, default=None, help="Process single video by URL or ID")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos to process")
    parser.add_argument("--fetch-transcripts", action="store_true", help="Download new audio transcripts via yt-dlp")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between transcript requests (seconds)")
    parser.add_argument("--generate-scanners", action="store_true", help="Generate starter Python scanner scripts")

    args = parser.parse_args()

    run_pipeline(
        from_cache=args.from_cache,
        video_target=args.video,
        limit=args.limit,
        fetch_transcripts=args.fetch_transcripts,
        delay_between_transcripts=args.delay,
        generate_scanners=args.generate_scanners
    )