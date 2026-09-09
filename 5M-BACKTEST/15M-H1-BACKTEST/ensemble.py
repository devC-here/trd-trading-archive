# ensemble.py
# 5-Specialist EUR/USD M15 Ensemble — Data-Driven Consensus
#
# WHAT THE BACKTEST DATA (3180 signals, 5000 H1 bars) PROVED:
#
#   1. M2 voting AGAINST the signal direction = stronger signal
#      M2=SELL + signal=BUY  → 56.2%  (price near resistance, buying anyway)
#      M2=BUY  + signal=BUY  → 47.1%  (price near support, obvious move = priced in)
#
#   2. Score ≥ 60 + good hours → 56.3% on 453 signals
#
#   3. Hours to avoid: 18, 23, 15, 06 (all below 45%)
#      Best hours: 20, 04, 10, 13, 03 (all above 56% with 3+ agree)
#
#   4. BUY signals systematically worse than SELL (1855 vs 981)
#      SELL at 16,17,13 → 59-62%
#
# PARADIGM:
#   M1 sets direction (weighted majority).
#   Score = consensus strength 0-100.
#   No score-based HOLD — only chop veto.
#   Hour filter applied in main.py / backtest, not here.
# ─────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────
# Weights — directional specialists only (M3 is regime, not direction)
# M4 weight reduced: it mirrors M1 99% of the time (from earlier data)
# M2 weight increased: its opposition signal is the most informative
# ─────────────────────────────────────────────────────────────────
WEIGHTS = {
    "m1": 0.35,
    "m2": 0.30,   # increased — opposition pattern is meaningful
    "m4": 0.15,   # reduced — mostly echoes M1
    "m5": 0.20,
}

# Only hard HOLD: genuine flat market
CHOP_VETO = 0.02

# Lot sizing
SCORE_HALF = 40
SCORE_FULL = 65

# Hours where the ensemble has NO edge — used by main.py
# (exported so main.py can import and use directly)
AVOID_HOURS = {6, 15, 18, 23}
BEST_HOURS  = {3, 4, 10, 13, 20, 21, 22}


# ─────────────────────────────────────────────────────────────────
# Vote extractors
# ─────────────────────────────────────────────────────────────────

def _vote_m1(r1: dict) -> tuple:
    p_bull = r1["direction"]["probs"].get("bullish", 0.0)
    p_bear = r1["direction"]["probs"].get("bearish", 0.0)
    if p_bull > p_bear:
        return ("BUY",  round(p_bull, 4))
    elif p_bear > p_bull:
        return ("SELL", round(p_bear, 4))
    return ("NEUTRAL", 0.5)


def _vote_m2(r2: dict) -> tuple:
    """
    Structure vote.
    NOTE: from backtest data, M2=SELL when signal=BUY gives 56% accuracy.
    M2 is intentionally allowed to vote against the crowd — its opposition
    is informative, not noise. Do not suppress NEUTRAL here.
    E_s < -0.10 → near support → BUY
    E_s > +0.10 → near resistance → SELL
    """
    es = r2["structural_position"]["E_s"]
    conviction = round(min(1.0, abs(es) / 0.5), 4)
    if es < -0.10:
        return ("BUY",  conviction)
    elif es > +0.10:
        return ("SELL", conviction)
    return ("NEUTRAL", conviction)


def _vote_m4(r4: dict) -> tuple:
    """Volume: money flow direction + I_c conviction."""
    ic  = r4["volume_intensity"]["I_c"]
    mfa = r4.get("features", {}).get("90_money_flow_acceleration", 0.0)
    conviction = round(min(1.0, ic / 3.0), 4)
    if mfa > 0:
        return ("BUY",  conviction)
    elif mfa < 0:
        return ("SELL", conviction)
    return ("NEUTRAL", conviction * 0.5)


def _vote_m5(r5: dict) -> tuple:
    """Geometry: wick polarity direction."""
    rp = r5["rejection_polarity"]["R_p"]
    conviction = round(min(1.0, abs(rp) / 0.5), 4)
    if rp < -0.05:
        return ("BUY",  conviction)
    elif rp > +0.05:
        return ("SELL", conviction)
    return ("NEUTRAL", conviction * 0.3)


def _m3_regime(r3: dict) -> tuple:
    p_brk = r3["regime"]["p_breakout"]
    thr   = r3["regime"].get("threshold", 0.12)
    return (p_brk >= thr, round(p_brk, 4))


# ─────────────────────────────────────────────────────────────────
# consensus_vote()
# ─────────────────────────────────────────────────────────────────

def consensus_vote(r1, r2, r3, r4, r5) -> dict:
    """
    Weighted majority vote across M1/M2/M4/M5.
    M3 provides regime context only.

    Score = dominance of winning side:
      (winner - loser) / (winner + loser) × 100
    100 = unanimous, 50 = slight edge, 0 = tied.

    Also tracks m2_opposes: True when M2 votes against the winner.
    This flag is the single most predictive feature from the backtest.
    """
    is_trend, p_brk = _m3_regime(r3)

    raw = {
        "m1": _vote_m1(r1),
        "m2": _vote_m2(r2),
        "m4": _vote_m4(r4),
        "m5": _vote_m5(r5),
    }

    buy_score  = 0.0
    sell_score = 0.0
    vote_count = {"BUY": 0, "SELL": 0, "NEUTRAL": 0}

    for key, (vote, conviction) in raw.items():
        w = WEIGHTS[key]
        vote_count[vote] = vote_count.get(vote, 0) + 1
        if vote == "BUY":
            buy_score  += w * conviction
        elif vote == "SELL":
            sell_score += w * conviction

    total = buy_score + sell_score
    if total < 1e-9:
        direction = "NEUTRAL"
        score = 0
    elif buy_score >= sell_score:
        direction = "BUY"
        score = int(round((buy_score - sell_score) / total * 100))
    else:
        direction = "SELL"
        score = int(round((sell_score - buy_score) / total * 100))

    score = max(0, min(100, score))
    if is_trend and direction != "NEUTRAL":
        score = min(100, score + 8)

    # Key flag from backtest analysis
    m2_vote = raw["m2"][0]
    m2_opposes = (m2_vote != "NEUTRAL" and m2_vote != direction)

    votes = {
        "m1": {"vote": raw["m1"][0], "conviction": raw["m1"][1],
               "raw": round(r1["direction"]["probs"].get("bullish",0) -
                            r1["direction"]["probs"].get("bearish",0), 4),
               "role": "momentum"},
        "m2": {"vote": raw["m2"][0], "conviction": raw["m2"][1],
               "raw": round(r2["structural_position"]["E_s"], 4),
               "role": "structure"},
        "m3": {"vote": "TRENDING" if is_trend else "CHOP",
               "conviction": round(min(1.0, p_brk), 4),
               "raw": p_brk, "role": "regime"},
        "m4": {"vote": raw["m4"][0], "conviction": raw["m4"][1],
               "raw": round(r4["volume_intensity"]["I_c"], 4),
               "role": "volume"},
        "m5": {"vote": raw["m5"][0], "conviction": raw["m5"][1],
               "raw": round(r5["rejection_polarity"]["R_p"], 4),
               "role": "geometry"},
    }

    return {
        "direction":    direction,
        "buy_score":    round(buy_score,  4),
        "sell_score":   round(sell_score, 4),
        "score":        score,
        "vote_count":   vote_count,
        "m2_opposes":   m2_opposes,      # key signal from backtest
        "is_trending":  is_trend,
        "p_breakout":   p_brk,
        "votes":        votes,
        # CSV compatibility
        "m1_conf":        raw["m1"][1],
        "m2_alignment":   raw["m2"][1] if raw["m2"][0]=="BUY" else -raw["m2"][1],
        "m5_alignment":   raw["m5"][1] if raw["m5"][0]=="BUY" else -raw["m5"][1],
        "m2_opposition":  raw["m2"][1] if raw["m2"][0]!= direction else 0.0,
        "m5_opposition":  raw["m5"][1] if raw["m5"][0]!= direction else 0.0,
        "raw_divergence": 0.0,
        "volume_norm":    round(min(1.0, r4["volume_intensity"]["I_c"]/3.0), 4),
    }


def lot_size_label(score: int) -> str:
    if score < SCORE_HALF: return "0.5× standard lot"
    if score < SCORE_FULL: return "1.0× standard lot"
    return "1.5× standard lot"


# ─────────────────────────────────────────────────────────────────
# run_ensemble_from_results()
# ─────────────────────────────────────────────────────────────────

def run_ensemble_from_results(r1, r2, r3, r4, r5) -> dict:
    """
    HOLD only on:
      1. p_breakout < CHOP_VETO — genuine flat noise
      2. Exact tie (buy_score == sell_score)
    Everything else fires. Hour filter lives in main.py.
    """
    def _hold(reason, cs=None):
        return {
            "signal": "HOLD", "reason": reason,
            "score": cs["score"] if cs else 0,
            "lot_size": "—",
            "direction": cs["direction"] if cs else "—",
            "consensus": cs,
            "model_1": r1, "model_2": r2, "model_3": r3,
            "model_4": r4, "model_5": r5,
        }

    p_brk = r3["regime"]["p_breakout"]
    if p_brk < CHOP_VETO:
        return _hold(f"CHOP_VETO · p_breakout={p_brk:.4f}")

    cs = consensus_vote(r1, r2, r3, r4, r5)

    if cs["direction"] == "NEUTRAL":
        return _hold("TIED · votes exactly equal", cs=cs)

    vc  = cs["vote_count"]
    dir = cs["direction"]
    return {
        "signal":    dir,
        "reason":    (f"CONSENSUS · {vc.get(dir,0)}/4 · "
                      f"score={cs['score']} · "
                      f"M2={'OPPOSES' if cs['m2_opposes'] else 'agrees'}"),
        "score":     cs["score"],
        "lot_size":  lot_size_label(cs["score"]),
        "direction": dir,
        "consensus": cs,
        "model_1": r1, "model_2": r2, "model_3": r3,
        "model_4": r4, "model_5": r5,
    }


def run_ensemble(df, m1, m2, m3, m4, m5) -> dict:
    r1 = m1.predict_from_df(df)
    r2 = m2.predict_from_df(df)
    r3 = m3.predict_from_df(df)
    r4 = m4.predict_from_df(df)
    r5 = m5.predict_from_df(df)
    return run_ensemble_from_results(r1, r2, r3, r4, r5)


def print_report(result: dict) -> None:
    cs  = result.get("consensus") or {}
    sep = "─" * 66
    print(f"\n{sep}")
    print(f"  ENSEMBLE  →  {result['signal']}   "
          f"score={result['score']}/100  {result['lot_size']}")
    if cs.get("m2_opposes"):
        print(f"  ★ M2 OPPOSES direction — historically stronger signal")
    print(sep)
    for key, label in [("m1","M1 Momentum"),("m2","M2 Structure"),
                        ("m3","M3 Volatility"),("m4","M4 Volume"),
                        ("m5","M5 Geometry")]:
        v = cs.get("votes",{}).get(key)
        if not v: continue
        bar = "█" * int(v["conviction"]*10)
        print(f"  {label:<18} {v['vote']:<8} {bar:<12} "
              f"conv={v['conviction']:.3f} raw={v['raw']:+.4f}")
    print(sep)
    vc = cs.get("vote_count",{})
    print(f"  BUY  {cs.get('buy_score',0):.4f}  ({vc.get('BUY',0)} votes)  |  "
          f"SELL {cs.get('sell_score',0):.4f}  ({vc.get('SELL',0)} votes)  |  "
          f"NEUTRAL {vc.get('NEUTRAL',0)} votes")
    print(f"  Regime: {'TRENDING' if cs.get('is_trending') else 'CHOP'}  "
          f"p_breakout={cs.get('p_breakout',0):.4f}")
    if result["signal"] != "HOLD":
        print(f"  → {result['reason']}")
    else:
        print(f"  → HOLD: {result['reason']}")
    print(sep + "\n")