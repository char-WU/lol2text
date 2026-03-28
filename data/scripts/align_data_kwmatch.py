# scripts/align_data.py
# Align YouTube caption timestamps (video timeline) to gol.gg event timestamps (game timeline)
# by applying an offset: caption_game_time = caption_video_time - offset_seconds
#
# Then compute TWO coverage metrics:
#  1) time_coverage: does the event have ANY caption text within the window?
#  2) keyword_coverage: does the window text contain event-specific keywords
#     (e.g., kill/defeat/punish/etc for CHAMPION_KILL)
#
# Usage (from project root):
#   python scripts/align_data.py
#
# TO-DO: + an "offset_seconds" column to data/match_list.csv for per-match offsets.
# OFFSET_SECONDS computed as Youtube Video Time - In-game Clock in Youtube Video
# E.g. if first gameplay appears at video 69s when the in-game clock is 37s, offset = 32.
# 

import json
import time
import re
import unicodedata
from pathlib import Path
import pandas as pd
import argparse


WINDOW_SECONDS = 15           # +/- window around each event (in game seconds)
MIN_CAPTION_CHARS = 2         # ignore very short caption fragments
DEFAULT_OFFSET_SECONDS = 0  # fallback if match_list.csv has no offset_seconds column


# Keyword Coverage
def normalize_text(s: str) -> str:
    s = s or ""
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    # keep letters/numbers/spaces, turn everything else into spaces
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def contains_any(text: str, patterns: list[str]) -> bool:
    for p in patterns:
        if re.search(p, text):
            return True
    return False

# Keywords
KILL_KEYWORDS = [
    r"\bkill(?:s|ed|ing)?\b",
    r"\bdelet(?:e|ed|ing)\b",
    r"\bdefeat(?:s|ed|ing)?\b",
    r"\bslay(?:s|ed|ing)?\b",
    r"\btake(?:s|n)? down\b",     # take down / taken down
    r"\btakedown\b",
    r"\bgoes down\b",
    r"\bdrops\b",
    r"\bpick(?:s)?\b",            # "find a pick"
    r"\bshutdown\b",
    r"\bexecute(?:d|s)?\b",
    r"\bpunish(?:es|ed|ing)?\b",
    r"\boutplay(?:s|ed|ing)?\b",
    r"\bfirst blood\b",
]

DRAGON_KEYWORDS = [
    r"\bdragon\b",
    r"\bdrake\b",
    r"\binfernal\b",
    r"\bmountain\b",
    r"\bocean\b",
    r"\bcloud\b",
    r"\belder\b",
    r"\bsoul\b",
]

BARON_KEYWORDS = [
    r"\bbaron\b",
    r"\bnashor\b",
]

HERALD_KEYWORDS = [
    r"\bherald\b",
    r"\brift herald\b",
]

BUILDING_KEYWORDS = [
    r"\bturret\b",
    r"\btower\b",
    r"\binhib(?:itor)?\b",
    r"\bnexus\b",
]

def keyword_covered(event: dict, captions: list[dict]) -> bool:
    """
    captions: list of aligned caption dicts, each has 'text'
    returns True if the caption window contains relevant keywords for this event type.
    """
    joined = normalize_text(" ".join(c.get("text", "") for c in captions))

    etype = (event.get("type") or "").upper()
    subtype = (event.get("subtype") or "").upper()

    if etype == "CHAMPION_KILL":
        return contains_any(joined, KILL_KEYWORDS)

    if etype == "ELITE_MONSTER_KILL":
        if "BARON" in subtype:
            return contains_any(joined, BARON_KEYWORDS)
        if "HERALD" in subtype:
            return contains_any(joined, HERALD_KEYWORDS)
        if "DRAKE" in subtype or "DRAGON" in subtype:
            return contains_any(joined, DRAGON_KEYWORDS)
        # fallback
        return contains_any(joined, DRAGON_KEYWORDS + BARON_KEYWORDS + HERALD_KEYWORDS)

    if etype == "BUILDING_KILL":
        return contains_any(joined, BUILDING_KEYWORDS)

    return False


# -------------------------
# JSON IO
# -------------------------
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -------------------------
# Alignment helpers
# -------------------------
def normalize_captions(captions, offset_s: float):
    """
    Convert captions from video timeline -> game timeline by subtracting offset.

    Input caption items expected:
      { "start": <seconds>, "duration": <seconds>, "text": <str> }

    Output items:
      {
        "start_s": <game seconds>,
        "end_s": <game seconds>,
        "text": <str>,
        "start_video_s": <video seconds>   # kept for debugging
      }
    """
    out = []
    for c in captions:
        start_video = float(c.get("start", 0.0))
        dur = float(c.get("duration", 0.0))
        text = (c.get("text") or "").strip()
        if len(text) < MIN_CAPTION_CHARS:
            continue

        # shift into game timeline
        start_game = start_video - offset_s
        end_game = start_game + max(dur, 0.0)

        out.append({
            "start_s": start_game,
            "end_s": end_game,
            "text": text,
            "start_video_s": start_video
        })

    out.sort(key=lambda x: x["start_s"])
    return out

def event_time_seconds(event: dict) -> float:
    """
    Convert event timestamp to seconds.

    Your gol.gg scraper SHOULD store 'timestamp' in milliseconds (ms) from game start.
    This function converts ms -> seconds, and includes a guard for the common "ms * 1000" bug.
    """
    t = event.get("timestamp", 0)
    if t is None:
        return 0.0
    t = float(t)

    seconds = t / 1000.0

    # guard: if seconds looks wildly too large (> 1 hour), it might be ms*1000 -> divide again
    if seconds > 60 * 60:
        seconds = seconds / 1000.0

    return seconds

def captions_in_window(captions_norm, center_s: float, window_s: float):
    """
    Return captions overlapping [center-window, center+window] in GAME time.
    Captions are assumed sorted by start_s.
    """
    w0 = center_s - window_s
    w1 = center_s + window_s

    hits = []
    for c in captions_norm:
        if c["end_s"] < w0:
            continue
        if c["start_s"] > w1:
            break
        hits.append(c)
    return hits

def align_match(match_id: int, captions_path: Path, events_path: Path, window_s: float, offset_s: float):
    captions = load_json(captions_path)
    events = load_json(events_path)

    caps = normalize_captions(captions, offset_s)

    aligned = []
    time_covered = 0
    keyword_covered_count = 0
    commentary_lengths = []
    
    for e in events:
        t_s = event_time_seconds(e)
        hits = captions_in_window(caps, t_s, window_s)

        # Combine caption texts
        commentary_text = ' '.join([c['text'] for c in hits])
        commentary_word_count = len(commentary_text.split())
        commentary_lengths.append(commentary_word_count)

        # Coverage checks
        time_cov = bool(hits)
        kw_cov = keyword_covered(e, hits) if hits else False

        if time_cov:
            time_covered += 1
        if kw_cov:
            keyword_covered_count += 1

        aligned.append({
            "match_id": match_id,
            "offset_seconds": offset_s,
            "window_seconds": window_s,
            "event_time_s": t_s,
            "event": e,
            "commentary": commentary_text,
            "commentary_word_count": commentary_word_count,
            "time_coverage": time_cov,
            "time_coverage": time_cov,
            "keyword_coverage": kw_cov,
        })

    # Compute match-level statistics
    match_stats = {
        'total_events': len(events),
        'time_covered': time_covered,
        'keyword_covered': keyword_covered_count,
        'time_coverage_pct': (time_covered / len(events) * 100.0) if len(events) > 0 else 0.0,
        'keyword_coverage_pct': (keyword_covered_count / len(events) * 100.0) if len(events) > 0 else 0.0,
        'avg_commentary_words': sum(commentary_lengths) / len(commentary_lengths) if commentary_lengths else 0.0,
        'min_commentary_words': min(commentary_lengths) if commentary_lengths else 0,
        'max_commentary_words': max(commentary_lengths) if commentary_lengths else 0,
        'events_with_no_commentary': sum(1 for length in commentary_lengths if length == 0)
    }
    return aligned, match_stats

def compute_overall_statistics(all_match_stats):
    """
    Compute dataset-level statistics from individual match stats
    """
    if not all_match_stats: return {}
    total_events = sum(s['total_events'] for s in all_match_stats)
    total_time_covered = sum(s['time_covered'] for s in all_match_stats)
    total_keyword_covered = sum(s['keyword_covered'] for s in all_match_stats)
    
    all_avg_words = [s['avg_commentary_words'] for s in all_match_stats]
    
    return {
        'num_matches': len(all_match_stats),
        'total_events': total_events,
        'total_time_covered': total_time_covered,
        'total_keyword_covered': total_keyword_covered,
        'overall_time_coverage_pct': (total_time_covered / total_events * 100.0) if total_events > 0 else 0.0,
        'overall_keyword_coverage_pct': (total_keyword_covered / total_events * 100.0) if total_events > 0 else 0.0,
        'avg_commentary_words_per_event': sum(all_avg_words) / len(all_avg_words) if all_avg_words else 0.0,
        'per_match_stats': all_match_stats
    }

def get_tournament_dir(folder_name):
    """
    "MSI 2019" -> data/2019/MSI 2019
    """
    year_match = re.search(r'(\d{4})', folder_name)
    if year_match:
        year = year_match.group(1)
        return Path("data") / year / folder_name
    else:
        return Path("data") / folder_name

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description='Align game events with captions.')
    parser.add_argument(
        'tournament',
        type=str,
        help='Tournament folder name (e.g., "MSI 2019")'
    )
    args = parser.parse_args()

    tournament_dir = get_tournament_dir(args.tournament)
    MATCH_LIST_PATH = tournament_dir / "match_list.csv"
    CAPTIONS_DIR = tournament_dir / "raw" / "captions" / 'v2'
    EVENTS_DIR = tournament_dir / "raw" / "events" / 'v2'
    OUTPUT_DIR = tournament_dir / "aligned"

    if not MATCH_LIST_PATH.exists():
        raise FileNotFoundError(f"Could not find {MATCH_LIST_PATH}. Did you run this from the project root?")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    matches = pd.read_csv(MATCH_LIST_PATH)
    total_matches = len(matches)

    processed = 0
    skipped_missing = 0
    all_match_stats = []

    has_offset_col = "offset_seconds" in matches.columns

    print(f"Loaded {total_matches} matches from {MATCH_LIST_PATH}")
    print(f"Per-match offsets enabled: {'YES' if has_offset_col else 'NO (using DEFAULT_OFFSET_SECONDS)'}")
    print(f"Window: ±{WINDOW_SECONDS}s\n")
    print(f"Keyword matching: ENABLED\n")

    for i, row in matches.iterrows():
        match_id = int(row["match_id"])
        offset_s = float(row["offset_seconds"]) if has_offset_col and pd.notna(row["offset_seconds"]) else float(DEFAULT_OFFSET_SECONDS)

        captions_path = CAPTIONS_DIR / f"match_{match_id}.json"
        events_path = EVENTS_DIR / f"match_{match_id}.json"

        print(f"[{i+1}/{total_matches}] match_id={match_id} offset={offset_s:.2f}s")

        if not captions_path.exists():
            print(f"  !! Missing captions: {captions_path}")
            print()
            skipped_missing += 1
            continue
        if not events_path.exists():
            print(f"  !! Missing events:   {events_path}")
            print()
            skipped_missing += 1
            continue

        aligned, match_stats = align_match(
            match_id=match_id,
            captions_path=captions_path,
            events_path=events_path,
            window_s=float(WINDOW_SECONDS),
            offset_s=offset_s
        )

        out_path = OUTPUT_DIR / f"match_{match_id}.json"
        save_json(out_path, aligned)

        # Store statistics
        match_stats['match_id'] = match_id
        match_stats['teams'] = f"{row.get('team1', 'N/A')} vs {row.get('team2', 'N/A')}"
        match_stats['offset_seconds'] = offset_s
        all_match_stats.append(match_stats)

        # Print match summary
        print(f"  ✓ Saved: {out_path}")
        print(f"  ↳ Time coverage:    {match_stats['time_covered']}/{match_stats['total_events']} ({match_stats['time_coverage_pct']:.1f}%)")
        print(f"  ↳ Keyword coverage: {match_stats['keyword_covered']}/{match_stats['total_events']} ({match_stats['keyword_coverage_pct']:.1f}%)")
        print(f"  ↳ Avg commentary:   {match_stats['avg_commentary_words']:.1f} words/event")
        
        if match_stats['events_with_no_commentary'] > 0:
            print(f"    Events with no commentary: {match_stats['events_with_no_commentary']}")
        
        print()
        processed += 1

        time.sleep(0.05)

    overall_stats = compute_overall_statistics(all_match_stats)
    
    # Save statistics to JSON
    stats_path = OUTPUT_DIR / "alignment_stats_key.json"
    save_json(stats_path, overall_stats)

    print("=" * 60)
    print(f"Processed matches: {processed}/{total_matches}")
    if skipped_missing:
        print(f"Skipped (missing files): {skipped_missing}")

    if overall_stats:
        print(f"\nTotal events: {overall_stats['total_events']}")
        print(f"Time coverage: {overall_stats['total_time_covered']} ({overall_stats['overall_time_coverage_pct']:.1f}%)")
        print(f"Keyword coverage: {overall_stats['total_keyword_covered']} ({overall_stats['overall_keyword_coverage_pct']:.1f}%)")
        print(f"Avg commentary per event: {overall_stats['avg_commentary_words_per_event']:.1f} words")

    print(f"\n✓ Aligned data saved to: {OUTPUT_DIR}")
    print(f"✓ Statistics saved to: {stats_path}")

    if not has_offset_col:
        print("\nTip: add an 'offset_seconds' column to data/match_list.csv for per-match offsets.")
        print("Example: if gameplay appears at video 69s when the in-game clock is 37s, offset = 32.\n")


if __name__ == "__main__":
    main()
