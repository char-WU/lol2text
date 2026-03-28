import re
import numpy as np
from collections import defaultdict
from typing import List, Dict

# ── Keyword vocabularies ──────────────────────────────────────────────────────
KILL_KEYWORDS = [
    r"\bkill(?:s|ed|ing)?\b", r"\bslay(?:s|ed|ing)?\b",
    r"\bdefeat(?:s|ed|ing)?\b", r"\bshutdown\b", r"\bfirst blood\b",
    r"\bpunish(?:es|ed|ing)?\b", r"\boutplay(?:s|ed|ing)?\b",
    r"\btake(?:s|n)? down\b", r"\btakedown\b", r"\bgoes down\b",
    r"\bdrops\b", r"\bpick(?:s|ed)?\b", r"\beliminate(?:s|d)?\b",
    r"\bexecute(?:s|d)?\b", r"\bfinish(?:es|ed)?\b", r"\bbursts?\b",
    r"\bdeletes?\b", r"\bdeleted\b", r"\bmelts?\b", r"\bmelted\b",
    r"\bshreds?\b", r"\bone.shot\b", r"\binstakill\b", r"\bdies\b",
    r"\bdead\b", r"\bfalls?\b", r"\bfallen\b", r"\bace(?:s|d)?\b",
    r"\bclean(?:s|ed)? up\b", r"\bsecure(?:s|d)?\b", r"\bdive(?:s|d)?\b",
    r"\bfind(?:s)? the kill\b", r"\bcatch(?:es)?\b", r"\bcaught\b",
    r"\btrapped\b", r"\bno escape\b", r"\brun(?:s|ning)? down\b",
    r"\bclutch\b",
]

TOWER_KEYWORDS = [
    r"\btower(?:s)?\b", r"\bturret(?:s)?\b", r"\bt1\b", r"\bt2\b", r"\bt3\b",
    r"\bouter tower\b", r"\binner tower\b", r"\bnexus tower\b",
    r"\bfirst tower\b", r"\bplating\b", r"\bcrumbles\b",
    r"\btopple(?:s|d)?\b", r"\bsiege(?:s|d)?\b", r"\bpush(?:es|ed|ing)?\b",
]

INHIBITOR_KEYWORDS = [
    r"\binhib(?:itor)?(?:s)?\b", r"\bsuper\s*minion(?:s)?\b",
    r"\bsupers\b", r"\bopen(?:s|ed|ing)?\b", r"\bexposed\b",
]

NEXUS_KEYWORDS = [
    r"\bnexus\b", r"\bgame(?:s|d)? over\b", r"\bit(?:'s| is) over\b",
    r"\bgg\b", r"\bwins?\b", r"\bvictory\b", r"\bfinished\b",
    r"\bclose(?:s|d)? out\b", r"\bseal(?:s|ed)?\b",
]

DRAGON_KEYWORDS = [
    r"\bdragon(?:s)?\b", r"\bdrake(?:s)?\b", r"\binfernal\b",
    r"\bmountain\b", r"\bocean\b", r"\bcloud\b", r"\belder\b",
    r"\bsoul\b", r"\bchemtech\b", r"\bhextech\b",
]

BARON_KEYWORDS = [
    r"\bbaron\b", r"\bnashor\b", r"\bhand of baron\b",
    r"\bbaron buff\b", r"\bpower play\b",
]

HERALD_KEYWORDS = [
    r"\bherald\b", r"\brift herald\b", r"\bshelly\b",
]

ALL_EVENT_TYPES = {
    "CHAMPION_KILL":           KILL_KEYWORDS,
    "BUILDING_KILL_Tower":     TOWER_KEYWORDS,
    "BUILDING_KILL_Inhibitor": INHIBITOR_KEYWORDS,
    "BUILDING_KILL_Nexus":     NEXUS_KEYWORDS,
    "DRAGON":                  DRAGON_KEYWORDS,
    "BARON":                   BARON_KEYWORDS,
    "HERALD":                  HERALD_KEYWORDS,
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def keyword_hit(text: str, patterns: List[str]) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)

def parse_events(input_text: str) -> List[Dict]:
    events = []
    for line in input_text.split("\n"):
        if not line.startswith("[EVENT"):
            continue
        event = {}
        for field in ["type", "side", "structure", "killer", "victim", "actor"]:
            match = re.search(rf"{field}=(\S+)", line)
            if match:
                val = match.group(1)
                if "(" in val:
                    player, champion = val.rstrip(")").split("(")
                    event[f"{field}_player"] = player
                    event[f"{field}_champion"] = champion
                else:
                    event[field] = val
        events.append(event)
    return events

# ── Per-sample scoring ────────────────────────────────────────────────────────
def score_text_against_events(text: str, events: List[Dict], penalty_lambda: float = 0.5) -> Dict:
    text_lower = text.lower()
    event_scores = []
    details = []

    # get input event types
    input_event_types = set()
    for event in events:
        event_type = event.get("type", "")
        structure  = event.get("structure", "")
        if event_type == "BUILDING_KILL":
            vocab_key = f"BUILDING_KILL_{structure}" if f"BUILDING_KILL_{structure}" in ALL_EVENT_TYPES else "BUILDING_KILL_Tower"
        else:
            vocab_key = event_type
        input_event_types.add(vocab_key)

    # score each input event
    for event in events:
        event_type = event.get("type", "")
        structure  = event.get("structure", "")

        if event_type == "BUILDING_KILL":
            vocab_key = f"BUILDING_KILL_{structure}" if f"BUILDING_KILL_{structure}" in ALL_EVENT_TYPES else "BUILDING_KILL_Tower"
        else:
            vocab_key = event_type

        keywords = ALL_EVENT_TYPES.get(vocab_key, [])
        type_hit = keyword_hit(text_lower, keywords)

        name_hits = []
        for field in ["killer_player", "killer_champion", "victim_player",
                      "victim_champion", "actor_player", "actor_champion"]:
            name = event.get(field, "").lower()
            if name and name in text_lower:
                name_hits.append(name)

        score = (0.5 if type_hit else 0.0) + (0.5 if name_hits else 0.0)
        event_scores.append(score)
        details.append({
            "event_type": event_type,
            "type_hit":   type_hit,
            "name_hits":  name_hits,
            "score":      score,
        })

    # hallucination penalty
    non_input_types = set(ALL_EVENT_TYPES.keys()) - input_event_types
    hallucinated    = [t for t in non_input_types if keyword_hit(text_lower, ALL_EVENT_TYPES[t])]
    penalty         = penalty_lambda * (len(hallucinated) / len(non_input_types)) if non_input_types else 0.0

    base_score  = np.mean(event_scores) if event_scores else 0.0
    final_score = max(0.0, base_score - penalty)

    return {
        "faithfulness_score":           final_score,
        "faithfulness_base_score":      base_score,
        "faithfulness_pct":             final_score * 100,
        "penalty":                      penalty,
        "hallucinated_event_types":     hallucinated,
        "events_fully_covered":         sum(1 for s in event_scores if s == 1.0),
        "events_partially_covered":     sum(1 for s in event_scores if 0 < s < 1.0),
        "events_missed":                sum(1 for s in event_scores if s == 0.0),
        "total_events":                 len(event_scores),
        "details":                      details,
    }

# ── Corpus-level evaluation ───────────────────────────────────────────────────
def evaluate_faithfulness(predictions: List[Dict], penalty_lambda: float = 0.5) -> Dict:
    pred_scores   = []
    target_scores = []
    pred_fully    = []
    target_fully  = []
    pred_missed   = []
    target_missed = []

    pred_hallucination_counts   = defaultdict(int)
    target_hallucination_counts = defaultdict(int)

    for ex in predictions:
        events = parse_events(ex["input"])
        if not events:
            continue

        pred_result   = score_text_against_events(ex["prediction"], events, penalty_lambda)
        target_result = score_text_against_events(ex["target"],     events, penalty_lambda)

        pred_scores.append(pred_result["faithfulness_score"])
        target_scores.append(target_result["faithfulness_score"])

        pred_fully.append(pred_result["events_fully_covered"])
        target_fully.append(target_result["events_fully_covered"])

        pred_missed.append(pred_result["events_missed"])
        target_missed.append(target_result["events_missed"])

        for t in pred_result["hallucinated_event_types"]:
            pred_hallucination_counts[t] += 1
        for t in target_result["hallucinated_event_types"]:
            target_hallucination_counts[t] += 1

    total = len(predictions)

    pred_hallucination_rates   = {
        t: c / total for t, c in sorted(pred_hallucination_counts.items(),   key=lambda x: -x[1])
    }
    target_hallucination_rates = {
        t: c / total for t, c in sorted(target_hallucination_counts.items(), key=lambda x: -x[1])
    }

    return {
        "prediction": {
            "faithfulness_mean":        float(np.mean(pred_scores)),
            "faithfulness_pct":         float(np.mean(pred_scores)) * 100,
            "faithfulness_std":         float(np.std(pred_scores)),
            "events_fully_covered":     float(np.mean(pred_fully)),
            "events_missed":            float(np.mean(pred_missed)),
            "hallucination_counts":     dict(pred_hallucination_counts),
            "hallucination_rates":      pred_hallucination_rates,
        },
        "target": {
            "faithfulness_mean":        float(np.mean(target_scores)),
            "faithfulness_pct":         float(np.mean(target_scores)) * 100,
            "faithfulness_std":         float(np.std(target_scores)),
            "events_fully_covered":     float(np.mean(target_fully)),
            "events_missed":            float(np.mean(target_missed)),
            "hallucination_counts":     dict(target_hallucination_counts),
            "hallucination_rates":      target_hallucination_rates,
        },
        "delta": {
            "faithfulness_mean": float(np.mean(pred_scores)) - float(np.mean(target_scores)),
            "faithfulness_pct":  (float(np.mean(pred_scores)) - float(np.mean(target_scores))) * 100,
        }
    }