import argparse
import json
import re
import unicodedata
from pathlib import Path


DEFAULT_ANCHOR_TYPES = {"CHAMPION_KILL", "ELITE_MONSTER_KILL", "BUILDING_KILL"}


def compact_value(value):
    """Convert values to a compact string for model input."""
    if value is None:
        return None

    if isinstance(value, list):
        if not value:
            return None

        formatted_items = []
        for x in value:
            if isinstance(x, dict):
                # Format dictionaries cleanly as Player(Champion)
                p = x.get("player", "").replace(" ", "_")
                c = x.get("champion", "").replace(" ", "_")
                if p and c:
                    formatted_items.append(f"{p}({c})")
                elif p:
                    formatted_items.append(p)
                elif c:
                    formatted_items.append(c)
            else:
                val_str = str(x).strip().replace(" ", "_")
                if val_str:
                    formatted_items.append(val_str)
                    
        return ",".join(formatted_items) if formatted_items else None
    
    if isinstance(value, dict):
        p = value.get("player", "").replace(" ", "_")
        c = value.get("champion", "").replace(" ", "_")
        if p and c: return f"{p}({c})"
        return p or c or None

    value = str(value).strip()
    if not value:
        return None
    
    return value.replace(" ", "_")


def normalize_event_for_model(event):
    """Map raw event fields into cleaner model-facing fields."""
    etype = (event.get("type") or "").upper()
    subtype = event.get("subtype")
    team = event.get("team")
    player = event.get("player")
    participants = event.get("participants")
    victim = event.get("victim")
    target = event.get("target")

    out = {"type": etype}

    if team:
        out["side"] = team

    if etype == "CHAMPION_KILL":
        if player:
            out["killer"] = player
        if participants:
            out["assists"] = participants
        if victim:
            out["victim"] = victim
        if subtype and str(subtype).strip().lower() not in {"kill"}:
            out["subtype"] = subtype
    elif etype == "ELITE_MONSTER_KILL":
        if subtype:
            out["objective"] = subtype
        if player:
            out["actor"] = player
        if participants:
            out["assists"] = participants
        if target:
            out["target"] = target
    elif etype == "BUILDING_KILL":
        if subtype:
            out["structure"] = subtype
        if player:
            out["actor"] = player
        if participants:
            out["assists"] = participants
        if target:
            out["target"] = target
    else:
        if subtype:
            out["subtype"] = subtype
        if player:
            out["player"] = player
        if participants:
            out["participants"] = participants
        if victim:
            out["victim"] = victim
        if target:
            out["target"] = target

    return out


def linearize_event(event, event_idx=None):
    """Convert one event into a clearer field=value format."""
    ev = normalize_event_for_model(event)
    preferred_order = [
        "type",
        "time",
        "side",
        "subtype",
        "objective",
        "structure",
        "killer",
        "actor",
        "assists",
        "victim",
        "target",
        "player",
        "participants",
    ]

    parts = []
    for key in preferred_order:
        if key not in ev:
            continue
        value = compact_value(ev[key])
        if value is None:
            continue
        parts.append(f"{key}={value}")

    prefix = f"[EVENT {event_idx}]" if event_idx is not None else "[EVENT]"
    return f"{prefix} " + " ".join(parts)


def linearize_events_for_t5(events):
    lines = ["generate_commentary:"]
    prev_time = None
    times = []

    for event in events:
        timestamp = event.get("timestamp")
        if timestamp is None:
            times.append(None)
            continue
        try:
            ts = float(timestamp)
            if ts > 10000:
                ts = ts / 1000.0
            times.append(ts)
        except Exception:
            times.append(None)

    for i, (event, t) in enumerate(zip(events, times), 1):
        ev = normalize_event_for_model(event)
        if prev_time is None or t is None:
            dt = 0
        else:
            dt = int(round(t - prev_time))

        if t is not None:
            prev_time = t
        ev["dt"] = dt

        parts = []
        for key, value in ev.items():
            compact = compact_value(value)
            if compact is not None:
                parts.append(f"{key}={compact}")

        lines.append(f"[EVENT {i}] " + " ".join(parts))

    return "\n".join(lines)


def get_commentary_text(item):
    """Extract commentary text from an aligned event item."""
    commentary = item.get("commentary")
    if isinstance(commentary, str):
        return commentary.strip()

    captions = item.get("captions")
    if isinstance(captions, list):
        texts = []
        for caption in captions:
            if isinstance(caption, dict):
                text = (caption.get("text") or "").strip()
            elif isinstance(caption, str):
                text = caption.strip()
            else:
                text = ""
            if text:
                texts.append(text)
        return " ".join(texts).strip()

    return ""


def normalize_text(text):
    text = unicodedata.normalize("NFKC", (text or "")).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_words(text):
    return re.findall(r"\S+", (text or "").strip())


def join_words(words):
    return " ".join(words).strip()


def split_commentary_into_chunks(text):
    """Split rolling commentary into smaller chunks so overlap removal works better."""
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return []

    parts = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    for part in parts:
        part = part.strip(" ,")
        if not part:
            continue
        words = split_words(part)
        if not words:
            continue
        if len(words) > 35 and "," in part:
            subparts = [p.strip(" ,") for p in re.split(r",\s+", part) if p.strip(" ,")]
            chunks.extend(subparts)
        else:
            chunks.append(part)
    return chunks


def trim_redundant_commentary_segments(commentaries, min_overlap_words=5):
    merged_segments = []

    for raw_text in commentaries:
        for text in split_commentary_into_chunks(raw_text):
            text = re.sub(r"\s+", " ", (text or "")).strip()
            if not text:
                continue

            if not merged_segments:
                merged_segments.append(text)
                continue

            prev_words = split_words(merged_segments[-1])
            curr_words = split_words(text)
            max_overlap = min(len(prev_words), len(curr_words))

            overlap = 0
            for n in range(max_overlap, min_overlap_words - 1, -1):
                if prev_words[-n:] == curr_words[:n]:
                    overlap = n
                    break

            if overlap > 0:
                novel_words = curr_words[overlap:]
                if novel_words:
                    merged_segments[-1] = join_words(prev_words + novel_words)
                continue

            prev_norm = normalize_text(merged_segments[-1])
            curr_norm = normalize_text(text)

            if curr_norm == prev_norm:
                continue
            if curr_norm and curr_norm in prev_norm:
                continue
            if prev_norm and prev_norm in curr_norm:
                merged_segments[-1] = text
                continue

            merged_segments.append(text)

    return merged_segments




def split_final_passages(text):
    """Split final commentary into sentence/phrase-like units for cleanup."""
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return []

    pieces = re.split(r"(?<=[.!?])\s+", text)
    out = []
    for piece in pieces:
        piece = piece.strip(' ,')
        if not piece:
            continue
        words = split_words(piece)
        if len(words) > 28 and ',' in piece:
            for sub in re.split(r",\s+", piece):
                sub = sub.strip(' ,')
                if sub:
                    out.append(sub)
        else:
            out.append(piece)
    return out


def dedupe_nearby_passages(passages, lookback=4):
    """Drop adjacent/nearby duplicate passages after the window is assembled."""
    kept = []
    kept_norms = []

    for passage in passages:
        norm = normalize_text(passage)
        if not norm:
            continue

        duplicate = False
        for prev_norm in kept_norms[-lookback:]:
            if norm == prev_norm:
                duplicate = True
                break
            if len(norm) >= 24 and norm in prev_norm:
                duplicate = True
                break
            if len(prev_norm) >= 24 and prev_norm in norm:
                duplicate = True
                break
        if duplicate:
            continue

        kept.append(passage)
        kept_norms.append(norm)

    return kept


def remove_immediate_token_repetition(text, min_repeat=8, max_repeat=40):
    """
    Remove immediate repeated token spans inside one long commentary blob.
    Example: A B C D A B C D E -> A B C D E
    """
    words = split_words(text)
    if not words:
        return ""

    output = []
    i = 0
    while i < len(words):
        output.append(words[i])
        i += 1

        changed = True
        while changed and i < len(words):
            changed = False
            max_n = min(max_repeat, len(output), len(words) - i)
            for n in range(max_n, min_repeat - 1, -1):
                if output[-n:] == words[i:i + n]:
                    i += n
                    changed = True
                    break

    return join_words(output)


def clean_final_commentary(text):
    """Last-pass cleanup for repeated rolling commentary in the final target."""
    passages = split_final_passages(text)
    passages = dedupe_nearby_passages(passages, lookback=8)
    cleaned = " ".join(passages).strip()
    cleaned = remove_immediate_token_repetition(cleaned, min_repeat=5, max_repeat=40)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(' ,')
    return cleaned

KILL_KEYWORDS = [
    r"\bkill(?:s|ed|ing)?\b",
    r"\bslay(?:s|ed|ing)?\b",
    r"\bdefeat(?:s|ed|ing)?\b",
    r"\bshutdown\b",
    r"\bfirst blood\b",
    r"\bpunish(?:es|ed|ing)?\b",
    r"\boutplay(?:s|ed|ing)?\b",
    r"\btake(?:s|n)? down\b",
    r"\btakedown\b",
    r"\bgoes down\b",
    r"\bdrops\b",
    r"\bpick\b",
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
BARON_KEYWORDS = [r"\bbaron\b", r"\bnashor\b"]
HERALD_KEYWORDS = [r"\bherald\b", r"\brift herald\b"]
BUILDING_KEYWORDS = [r"\btower\b", r"\bturret\b", r"\binhib(?:itor)?\b", r"\bnexus\b"]


def keywords_for_event(event):
    etype = (event.get("type") or "").upper()
    subtype = (event.get("subtype") or "").upper()

    if etype == "CHAMPION_KILL":
        return KILL_KEYWORDS
    if etype == "BUILDING_KILL":
        return BUILDING_KEYWORDS
    if etype == "ELITE_MONSTER_KILL":
        if "BARON" in subtype:
            return BARON_KEYWORDS
        if "HERALD" in subtype:
            return HERALD_KEYWORDS
        if "DRAKE" in subtype or "DRAGON" in subtype:
            return DRAGON_KEYWORDS
        return DRAGON_KEYWORDS + BARON_KEYWORDS + HERALD_KEYWORDS
    return []


def extract_caption_time(caption):
    """Get caption time in game seconds from common timestamp fields."""
    if not isinstance(caption, dict):
        return None

    for key in ("start_s", "start"):
        value = caption.get(key)
        if value is not None:
            try:
                return float(value)
            except Exception:
                pass

    value = caption.get("start_video_s")
    if value is not None:
        try:
            return float(value)
        except Exception:
            pass

    return None


def build_global_commentary_timeline(aligned_events):
    """Build a deduped, time-sorted list of commentary segments."""
    timeline = []
    seen = set()

    for item in aligned_events:
        captions = item.get("captions")
        if isinstance(captions, list) and captions:
            for caption in captions:
                if isinstance(caption, dict):
                    t = extract_caption_time(caption)
                    text = (caption.get("text") or "").strip()
                elif isinstance(caption, str):
                    t = float(item.get("event_time_s", 0.0))
                    text = caption.strip()
                else:
                    continue

                if t is None or not text:
                    continue

                key = (round(float(t), 2), text)
                if key in seen:
                    continue
                seen.add(key)
                timeline.append({"t": float(t), "text": text})
            continue

        text = get_commentary_text(item)
        if text:
            t = float(item.get("event_time_s", 0.0))
            key = (round(t, 2), text)
            if key not in seen:
                seen.add(key)
                timeline.append({"t": t, "text": text})

    timeline.sort(key=lambda x: x["t"])
    return timeline


def pick_best_keyword_hit(event, captions_timeline, t_event, search_pre, search_post):
    """Find the best keyword-matching caption near an event."""
    patterns = keywords_for_event(event)
    if not patterns:
        return None

    t0 = t_event - float(search_pre)
    t1 = t_event + float(search_post)
    best_time = None
    best_score = float("-inf")

    for caption in captions_timeline:
        t = caption["t"]
        if t < t0:
            continue
        if t > t1:
            break

        text_norm = normalize_text(caption["text"])
        hit_count = sum(1 for pattern in patterns if re.search(pattern, text_norm))
        if hit_count <= 0:
            continue

        dist = abs(t - t_event)
        score = hit_count - 0.05 * dist
        if score > best_score:
            best_score = score
            best_time = t

    return best_time


def group_events_keyword_windowed(
    aligned_events,
    window_size=60,
    search_pre=30,
    search_post=45,
    anchor_types=None,
    center_strategy="asymmetric",
    pre_s=None,
    post_s=None,
):
    """Create one keyword-windowed example per anchor event."""
    if not aligned_events:
        return {}

    anchor_types = {t.upper() for t in (anchor_types or DEFAULT_ANCHOR_TYPES)}
    aligned_sorted = sorted(aligned_events, key=lambda x: x["event_time_s"])
    captions_timeline = build_global_commentary_timeline(aligned_sorted)

    anchors = [
        item for item in aligned_sorted
        if ((item.get("event") or {}).get("type") or "").upper() in anchor_types
    ]

    if pre_s is not None and post_s is not None:
        final_pre = float(pre_s)
        final_post = float(post_s)
    elif center_strategy == "centered":
        final_pre = float(window_size) / 2.0
        final_post = float(window_size) / 2.0
    elif int(window_size) == 60:
        final_pre = 15.0
        final_post = 45.0
    else:
        final_pre = float(window_size) * 0.25
        final_post = float(window_size) * 0.75

    windows = {}

    for window_id, anchor_item in enumerate(anchors):
        event = anchor_item["event"]
        t_event = float(anchor_item["event_time_s"])

        t_hit = pick_best_keyword_hit(
            event=event,
            captions_timeline=captions_timeline,
            t_event=t_event,
            search_pre=search_pre,
            search_post=search_post,
        )
        center_t = float(t_hit) if t_hit is not None else t_event
        window_start = max(0.0, center_t - final_pre)
        window_end = max(0.0, center_t + final_post)

        events_in_window = []
        anchor_event_positions = []
        for item in aligned_sorted:
            t = float(item["event_time_s"])
            if t < window_start:
                continue
            if t > window_end:
                break

            events_in_window.append(item["event"])
            etype = ((item.get("event") or {}).get("type") or "").upper()
            if etype in anchor_types:
                anchor_event_positions.append(len(events_in_window) - 1)

        if not events_in_window:
            continue

        commentaries = []
        commentary_times = []
        for caption in captions_timeline:
            t = caption["t"]
            if t < window_start:
                continue
            if t > window_end:
                break
            commentaries.append(caption["text"])
            commentary_times.append(t)

        commentaries = trim_redundant_commentary_segments(commentaries)
        if not commentaries:
            continue

        windows[window_id] = {
            "match_id": anchor_item["match_id"],
            "window_start": window_start,
            "window_end": window_end,
            "events": events_in_window,
            "commentaries": commentaries,
            "anchor_time_s": t_event,
            "anchor_type": event.get("type") or "",
            "anchor_subtype": event.get("subtype") or "",
            "keyword_center_time_s": center_t,
            "keyword_center_found": t_hit is not None,
            "window_size_s": window_end - window_start,
            "num_events": len(events_in_window),
            "num_commentary_segments": len(commentaries),
            "anchor_event_positions": anchor_event_positions,
            "commentary_start_s": commentary_times[0],
            "commentary_end_s": commentary_times[-1],
        }

    return windows


def dedupe_windows(windows):
    """Deduplicate windows by match_id + ordered event sequence."""
    deduped = {}
    seen = set()
    removed = 0

    for window_data in windows.values():
        match_id = window_data["match_id"]
        event_signature = tuple(
            json.dumps(event, sort_keys=True, ensure_ascii=False)
            for event in window_data.get("events", [])
        )
        key = (match_id, event_signature)
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        deduped[len(deduped)] = window_data

    return deduped, removed


def create_training_examples(windows):
    """Turn grouped windows into final JSONL-ready examples."""
    examples = []

    for window_id, window_data in sorted(windows.items()):
        events = window_data["events"]
        commentaries = window_data["commentaries"]
        combined_commentary = clean_final_commentary(" ".join(commentaries).strip())
        model_input = linearize_events_for_t5(events)

        example = {
            "match_id": window_data["match_id"],
            "window_id": window_id,
            "window_start": window_data["window_start"],
            "window_end": window_data["window_end"],
            "events": events,
            "num_events": len(events),
            "input": model_input,
            "commentary_segments": commentaries,
            "target": combined_commentary,
            "input_tokens": len(model_input.split()),
            "target_words": len(combined_commentary.split()),
        }

        for key in (
            "anchor_time_s",
            "anchor_type",
            "anchor_subtype",
            "keyword_center_time_s",
            "keyword_center_found",
            "window_size_s",
            "num_commentary_segments",
            "anchor_event_positions",
            "commentary_start_s",
            "commentary_end_s",
        ):
            if key in window_data:
                example[key] = window_data[key]

        examples.append(example)

    return examples


def extract_match_id(filepath):
    match = re.search(r"match_(.+)$", filepath.stem)
    return match.group(1) if match else filepath.stem


def process_all_matches(aligned_dir, window_size, search_pre, search_post, anchor_types, pre_s=None, post_s=None):
    """Run keyword_windowed grouping over every aligned match file."""
    all_examples = []
    match_stats = []
    aligned_files = sorted(aligned_dir.glob("match_*.json"))

    if not aligned_files:
        print(f"  !! No aligned files found in {aligned_dir}")
        return [], []

    print(f"Found {len(aligned_files)} aligned match files\n")

    for i, filepath in enumerate(aligned_files, 1):
        match_id = extract_match_id(filepath)
        print(f"[{i}/{len(aligned_files)}] Processing match {match_id}...")

        with open(filepath, "r", encoding="utf-8") as f:
            aligned_events = json.load(f)

        if not aligned_events:
            print("  !! Empty file, skipping")
            continue

        windows = group_events_keyword_windowed(
            aligned_events=aligned_events,
            window_size=window_size,
            search_pre=search_pre,
            search_post=search_post,
            anchor_types=anchor_types,
            center_strategy="asymmetric",
            pre_s=pre_s,
            post_s=post_s,
        )

        windows, num_removed = dedupe_windows(windows)
        if num_removed:
            print(f"  - Removed {num_removed} duplicate windows (same event sequence)")

        examples = create_training_examples(windows)
        if not examples:
            print("  !! No valid windows after filtering")
            continue

        all_examples.extend(examples)
        match_stats.append(
            {
                "match_id": match_id,
                "num_windows": len(examples),
                "avg_events_per_window": sum(e["num_events"] for e in examples) / len(examples),
                "avg_words_per_window": sum(e["target_words"] for e in examples) / len(examples),
            }
        )
        print(f"  ✓ Created {len(examples)} training windows")

    return all_examples, match_stats


def compute_stats(examples, match_stats):
    if not examples:
        return {}

    input_lengths = [e["input_tokens"] for e in examples]
    target_lengths = [e["target_words"] for e in examples]
    events_per_window = [e["num_events"] for e in examples]

    return {
        "num_matches": len(match_stats),
        "total_examples": len(examples),
        "avg_examples_per_match": len(examples) / len(match_stats) if match_stats else 0,
        "input_stats": {
            "avg_tokens": sum(input_lengths) / len(input_lengths),
            "min_tokens": min(input_lengths),
            "max_tokens": max(input_lengths),
        },
        "target_stats": {
            "avg_words": sum(target_lengths) / len(target_lengths),
            "min_words": min(target_lengths),
            "max_words": max(target_lengths),
        },
        "events_stats": {
            "avg_per_window": sum(events_per_window) / len(events_per_window),
            "min_per_window": min(events_per_window),
            "max_per_window": max(events_per_window),
        },
        "per_match_stats": match_stats,
    }


def save_training_data(examples, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(f"\n✓ Saved {len(examples)} examples to {output_path}")


def print_examples_multi_events(examples, num=2, min_events=2):
    print("\n" + "=" * 60)
    print("SAMPLE TRAINING EXAMPLES")

    filtered = [e for e in examples if e.get("num_events", 0) >= min_events]
    if not filtered:
        print(f"\nNo examples found with num_events >= {min_events}")
        return

    for i, example in enumerate(filtered[:num], 1):
        print(f"\n--- Example {i} ---")
        print(f"Match ID: {example['match_id']}")
        print(f"Window: {example['window_start']:.0f}s - {example['window_end']:.0f}s")
        print(f"Events: {example['num_events']}")
        print(
            f"Anchor: {example.get('anchor_type')} @ {example.get('anchor_time_s', 0):.0f}s"
            f" | keyword_center={example.get('keyword_center_time_s', 0):.0f}s"
        )

        print("\nEVENT LIST:")
        for j, event in enumerate(example.get("events", []), 1):
            print(f"  {linearize_event(event, event_idx=j)}")

        print(f"\nINPUT ({example['input_tokens']} tokens):")
        print(example["input"][:500] + "..." if len(example["input"]) > 500 else example["input"])

        print(f"\nTARGET ({example['target_words']} words):")
        print(example["target"][:500] + "..." if len(example["target"]) > 500 else example["target"])

def get_tournament_dir(folder_name):
    """
    Extract year from folder name and build path dynamically.
    e.g., "MSI 2019" -> data/2019/MSI 2019
    """
    year_match = re.search(r'(\d{4})', folder_name)
    if year_match:
        year = year_match.group(1)
        return Path("data") / year / folder_name
    else:
        return Path("data") / folder_name

def main():
    parser = argparse.ArgumentParser(description="Linearize events using only keyword_windowed grouping.")
    parser.add_argument(
        'tournament',
        type=str,
        help='Tournament folder name (e.g., "WC 2019" or "MSI 2019")'
    )
    parser.add_argument("--window-size", type=int, default=60, help="Final keyword window size in seconds")
    parser.add_argument("--search-pre", type=int, default=30, help="Seconds before event time to search for keyword hit")
    parser.add_argument("--search-post", type=int, default=45, help="Seconds after event time to search for keyword hit")
    parser.add_argument("--pre", type=float, default=None, help="Optional explicit seconds before the keyword center")
    parser.add_argument("--post", type=float, default=None, help="Optional explicit seconds after the keyword center")
    parser.add_argument(
        "--anchor-types",
        type=str,
        default="CHAMPION_KILL,ELITE_MONSTER_KILL,BUILDING_KILL",
        help="Comma-separated event types to anchor on",
    )
    args = parser.parse_args()

    tournament_dir = get_tournament_dir(args.tournament)
    aligned_dir = tournament_dir / "aligned" / "v2"
    output_base_dir = tournament_dir / "processed"
    output_base_dir.mkdir(parents=True, exist_ok=True)

    anchor_types = [t.strip() for t in args.anchor_types.split(",") if t.strip()]
    output_path = output_base_dir / (
        f"training_data_keyword_windowed_w{args.window_size}s"
        f"_search{args.search_pre}-{args.search_post}.jsonl"
    )
    stats_path = output_base_dir / (
        f"linearization_stats_keyword_windowed_w{args.window_size}s"
        f"_search{args.search_pre}-{args.search_post}.json"
    )

    print("=" * 60)
    print("LINEARIZATION AND TRAINING DATA")
    print("Grouping: keyword_windowed")
    print(f"Anchor types: {anchor_types}")
    print(f"Window size: {args.window_size}s")
    print(f"Keyword search: -{args.search_pre}s / +{args.search_post}s around event time")
    if args.pre is not None and args.post is not None:
        print(f"Explicit final window split: -{args.pre}s / +{args.post}s")

    examples, match_stats = process_all_matches(
        aligned_dir=aligned_dir,
        window_size=args.window_size,
        search_pre=args.search_pre,
        search_post=args.search_post,
        anchor_types=anchor_types,
        pre_s=args.pre,
        post_s=args.post,
    )

    if not examples:
        print("\n  !! No training examples created!")
        return

    save_training_data(examples, output_path)
    stats = compute_stats(examples, match_stats)

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved statistics to {stats_path}")
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print(f"Matches processed: {stats['num_matches']}")
    print(f"Total examples: {stats['total_examples']}")
    print(f"Avg examples/match: {stats['avg_examples_per_match']:.1f}")
    print("\nInput tokens:")
    print(f"  Mean: {stats['input_stats']['avg_tokens']:.1f}")
    print(f"  Range: {stats['input_stats']['min_tokens']}-{stats['input_stats']['max_tokens']}")
    print("\nTarget words:")
    print(f"  Mean: {stats['target_stats']['avg_words']:.1f}")
    print(f"  Range: {stats['target_stats']['min_words']}-{stats['target_stats']['max_words']}")
    print("\nEvents per window:")
    print(f"  Mean: {stats['events_stats']['avg_per_window']:.1f}")
    print(f"  Range: {stats['events_stats']['min_per_window']}-{stats['events_stats']['max_per_window']}")

    print_examples_multi_events(examples, num=2, min_events=2)
    multi_event_examples = [e for e in examples if e["num_events"] >= 2]
    print(f"\nExamples with >=2 events: {len(multi_event_examples)} / {len(examples)}")


if __name__ == "__main__":
    main()

    # sample = """That is their point of defense right now. With the dragon spawning I think it will be dangerous for them to even attempt to go out into neutral territory and they should probably remain defensive at this time. They simply do not have the extra information to try and set up an offensive move. See if Team Liquid's defense can be good because G2 have been keeping the pressure on. The next drake has spawned very recently. We have five minutes until Baron but the mid lane is already there. That bastion of defense just being shut down. One kill picked up ready to expit me and look out for Jensen Rude. Shockwave is not going to be enough. They trade one fort at the end of it all and now Jankos able to walk away off that kill. See if Team Liquid's defense can be good because G2 have been keeping the pressure on. The next drake has spawned very recently. We have five minutes until Baron but the mid lane is already there. That bastion of defense just being shut down. One kill picked up ready to expit me and look out for Jensen Rude. Shockwave is not going to be enough. They trade one fort at the end of it all and now Jankos able to walk away off that kill. One kill picked up ready to expit me and look out for Jensen Rude. Shockwave is not going to be enough. They trade one fort at the end of it all and now Jankos able to walk away off that kill. One kill picked up ready to expit me and look out for Jensen Rude. Shockwave is not going to be enough. They trade one fort at the end of it all and now Jankos able to walk away off that kill. Barakhan combined with Kaps's Morgana just swept through Jensen who's just barely able to get something back for Team Liquid. Barakhan combined with Kaps's Morgana just swept through Jensen who's just barely able to get something back for Team Liquid. G2 making a statement in game number one here. The killer instinct alive in all of them going for the dive here. Miki does get both of them locking them up so they pick up the first kill."""
    # print(clean_final_commentary(sample))

