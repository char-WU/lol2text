import json
from pathlib import Path
from collections import defaultdict
import argparse
import re
import unicodedata


# Example commands for the script:
# - python scripts/linearize_events.py --grouping fixed --window-size 60
# - python scripts/linearize_events.py --grouping event_centered --pre 30 --post 30
# - python scripts/linearize_events.py --grouping gaps --gap 25 --pad 5
# - python scripts/linearize_events.py --grouping keyword_centered --pre 20 --post 20 --search-pre 15 --search-post 45


DEFAULT_ANCHOR_TYPES = {"CHAMPION_KILL", "ELITE_MONSTER_KILL", "BUILDING_KILL"}


# def linearize_event(event):
#     """
#     Convert single event dict to linearized text.

#     Output format:
#       "<value>|type <value>|subtype <value>|timestamp ..."
#     """
#     tokens = []

#     field_order = [
#         "type",
#         "subtype",
#         "timestamp",
#         "team",
#         "player",
#         "participants",
#         "victim",
#         "target",
#     ]

#     for key in field_order:
#         if key not in event:
#             continue

#         value = event[key]
#         if value is None or value == "":
#             continue

#         if isinstance(value, list):
#             if not value:
#                 continue
#             value = ",".join(map(str, value))

#         formatted = str(value).replace("_", "").replace(" ", "")
#         tokens.append(f"{formatted}|{key}")

#     return " ".join(tokens)


def compact_value(value):
    """
    Convert values to a compact string for model input.
    """
    if value is None:
        return None

    if isinstance(value, list):
        if not value:
            return None
        return ",".join(str(x).strip().replace(" ", "_") for x in value if str(x).strip())

    value = str(value).strip()
    if not value:
        return None

    return value.replace(" ", "_")


def normalize_event_for_model(event):
    """
    Map raw event fields into cleaner model-facing fields.

    Returns a dict with fields like:
      type, time, side, killer, assists, victim, objective, actor, target
    """
    etype = (event.get("type") or "").upper()
    subtype = event.get("subtype")
    timestamp = event.get("timestamp")
    team = event.get("team")
    player = event.get("player")
    participants = event.get("participants")
    victim = event.get("victim")
    target = event.get("target")

    out = {
        "type": etype,
    }

    # convert ms timestamps like 799000 -> 799 when appropriate
    if timestamp is not None:
        try:
            ts = float(timestamp)
            if ts > 10000:
                ts = int(round(ts / 1000.0))
            else:
                ts = int(round(ts))
            # out["time"] = ts #######################################
            # handled in sequence, not per-event
        except Exception:
            out["time"] = timestamp

    if team:
        out["side"] = team

    # event-specific role mapping
    if etype == "CHAMPION_KILL":
        if player:
            out["killer"] = player
        if participants:
            out["assists"] = participants
        if victim:
            out["victim"] = victim

        # keep subtype only if it adds information
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
        # fallback for unknown event types
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
    """
    Convert one event into a clearer field=value format.

    Example:
      [EVENT 1] type=CHAMPION_KILL time=799 side=blue killer=Doinb assists=Nautilus,Thresh victim=Caps
    """
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

    # get absolute times first
    times = []
    for e in events:
        t = e.get("timestamp")
        if t is None:
            times.append(None)
        else:
            try:
                t = float(t)
                if t > 10000:
                    t = t / 1000.0
                times.append(t)
            except:
                times.append(None)

    for i, (event, t) in enumerate(zip(events, times), 1):
        ev = normalize_event_for_model(event)

        # compute dt
        if prev_time is None or t is None:
            dt = 0
        else:
            dt = int(round(t - prev_time))

        prev_time = t if t is not None else prev_time

        ev["dt"] = dt

        # build string
        parts = []
        for k, v in ev.items():
            val = compact_value(v)
            if val is not None:
                parts.append(f"{k}={val}")

        lines.append(f"[EVENT {i}] " + " ".join(parts))

    return "\n".join(lines)


def get_commentary_text(item):
    """
    Extract commentary text from an aligned event item.

    Preferred:
      item["commentary"] -> str

    Fallbacks:
      item["captions"] -> list[dict] with {"text": ...}
      item["captions"] -> list[str]
    """
    commentary = item.get("commentary")
    if isinstance(commentary, str):
        return commentary.strip()

    caps = item.get("captions")
    if isinstance(caps, list):
        texts = []
        for c in caps:
            if isinstance(c, dict):
                t = (c.get("text") or "").strip()
                if t:
                    texts.append(t)
            elif isinstance(c, str):
                t = c.strip()
                if t:
                    texts.append(t)
        return " ".join(texts).strip()

    return ""


# -------------------------
# Text normalization + keyword rules
# -------------------------

def normalize_text(s: str) -> str:
    s = s or ""
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


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


def keywords_for_event(event: dict) -> list[str]:
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


def extract_caption_time(c: dict):
    """
    Get a caption time in game seconds from common fields.

    Priority:
      - start_s
      - start
      - start_video_s (fallback only)
    """
    if not isinstance(c, dict):
        return None

    for key in ("start_s", "start"):
        if key in c and c[key] is not None:
            try:
                return float(c[key])
            except Exception:
                pass

    if "start_video_s" in c and c["start_video_s"] is not None:
        try:
            return float(c["start_video_s"])
        except Exception:
            pass

    return None


def build_global_commentary_timeline(aligned_events):
    """
    Build a deduped, time-sorted list of commentary segments across the match.

    Preferred:
      use caption-level timestamps from item["captions"]

    Fallback:
      use event_time_s + merged commentary text

    Each item:
      {"t": float, "text": str}
    """
    timeline = []
    seen = set()

    for item in aligned_events:
        caps = item.get("captions")

        if isinstance(caps, list) and caps:
            for c in caps:
                if isinstance(c, dict):
                    t = extract_caption_time(c)
                    text = (c.get("text") or "").strip()
                elif isinstance(c, str):
                    t = float(item.get("event_time_s", 0.0))
                    text = c.strip()
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


# ***********************
# WINDOW GROUPING METHODS
# ***********************

def group_events_into_windows_fixed(aligned_events, window_size):
    """
    Group per-event aligned data into fixed-size time windows.
    """
    if not aligned_events:
        return {}

    windows = defaultdict(lambda: {"events": [], "commentaries": []})

    for item in aligned_events:
        event_time = float(item["event_time_s"])
        window_id = int(event_time // window_size)

        windows[window_id]["events"].append(item["event"])

        commentary_text = get_commentary_text(item)
        if commentary_text:
            windows[window_id]["commentaries"].append(commentary_text)

        if "match_id" not in windows[window_id]:
            windows[window_id]["match_id"] = item["match_id"]
            windows[window_id]["window_start"] = window_id * window_size
            windows[window_id]["window_end"] = (window_id + 1) * window_size

    return windows


def group_events_event_centered(aligned_events, pre_s=20, post_s=20, anchor_types=None):
    """
    Create one window per anchor event, centered around that event.
    """
    if not aligned_events:
        return {}

    anchor_types = set(anchor_types or DEFAULT_ANCHOR_TYPES)
    aligned_sorted = sorted(aligned_events, key=lambda x: x["event_time_s"])

    anchors = []
    for idx, item in enumerate(aligned_sorted):
        etype = (item.get("event") or {}).get("type")
        if etype in anchor_types:
            anchors.append((idx, item))

    windows = {}

    for window_id, (_anchor_idx, anchor_item) in enumerate(anchors):
        anchor_t = float(anchor_item["event_time_s"])
        t0 = max(0.0, anchor_t - float(pre_s))
        t1 = max(0.0, anchor_t + float(post_s))

        events = []
        commentaries = []

        for item in aligned_sorted:
            t = float(item["event_time_s"])
            if t < t0:
                continue
            if t > t1:
                break

            events.append(item["event"])

            commentary_text = get_commentary_text(item)
            if commentary_text:
                commentaries.append(commentary_text)

        if not events:
            continue

        windows[window_id] = {
            "match_id": anchor_item["match_id"],
            "window_start": t0,
            "window_end": t1,
            "events": events,
            "commentaries": commentaries,
            "anchor_time_s": anchor_t,
            "anchor_type": (anchor_item.get("event") or {}).get("type"),
            "anchor_subtype": (anchor_item.get("event") or {}).get("subtype"),
        }

    return windows


def group_events_by_gaps(aligned_events, gap_s=20, pad_s=0):
    """
    Group events into bursts based on time gaps.
    """
    if not aligned_events:
        return {}

    aligned_sorted = sorted(aligned_events, key=lambda x: x["event_time_s"])
    match_id = aligned_sorted[0]["match_id"]

    windows = {}
    window_id = 0

    cur_events = []
    cur_commentaries = []
    cur_start = None
    prev_t = None

    def flush():
        nonlocal window_id, cur_events, cur_commentaries, cur_start, prev_t

        if not cur_events:
            return

        w0 = max(0.0, float(cur_start) - float(pad_s))
        w1 = max(0.0, float(prev_t) + float(pad_s))

        windows[window_id] = {
            "match_id": match_id,
            "window_start": w0,
            "window_end": w1,
            "events": cur_events,
            "commentaries": cur_commentaries,
        }

        window_id += 1
        cur_events = []
        cur_commentaries = []
        cur_start = None
        prev_t = None

    for item in aligned_sorted:
        t = float(item["event_time_s"])

        if prev_t is None:
            cur_start = t
        elif (t - prev_t) > float(gap_s):
            flush()
            cur_start = t

        cur_events.append(item["event"])

        commentary_text = get_commentary_text(item)
        if commentary_text:
            cur_commentaries.append(commentary_text)

        prev_t = t

    flush()
    return windows


# -------------------------
# Grouping mode 4: keyword-centered windows
# -------------------------

def pick_best_keyword_hit(event: dict, captions_timeline: list[dict], t_event: float,
                          search_pre: float, search_post: float):
    """
    Find best keyword caption near event time within:
      [t_event - search_pre, t_event + search_post]

    Score:
      (#keyword hits) - 0.05 * abs(t - t_event)
    """
    patterns = keywords_for_event(event)
    if not patterns:
        return None

    t0 = t_event - float(search_pre)
    t1 = t_event + float(search_post)

    best = None
    best_score = float("-inf")

    for c in captions_timeline:
        t = c["t"]
        if t < t0:
            continue
        if t > t1:
            break

        text_norm = normalize_text(c["text"])
        hit_count = sum(1 for p in patterns if re.search(p, text_norm))
        if hit_count <= 0:
            continue

        dist = abs(t - t_event)
        score = hit_count - 0.05 * dist

        if score > best_score:
            best_score = score
            best = t

    return best


def group_events_keyword_centered(aligned_events, pre_s=20, post_s=20,
                                  search_pre=30, search_post=30,
                                  anchor_types=None):
    """
    For each anchor event:
      1) search for best keyword hit in caption timeline near the event time
      2) center window on that caption time (fallback to event time)
      3) include all events inside [center-pre_s, center+post_s]
      4) target commentary = caption texts inside that window
    """
    if not aligned_events:
        return {}

    anchor_types = set(anchor_types or DEFAULT_ANCHOR_TYPES)
    aligned_sorted = sorted(aligned_events, key=lambda x: x["event_time_s"])
    captions_timeline = build_global_commentary_timeline(aligned_sorted)

    anchors = [
        item for item in aligned_sorted
        if ((item.get("event") or {}).get("type") in anchor_types)
    ]

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

        w0 = max(0.0, center_t - float(pre_s))
        w1 = max(0.0, center_t + float(post_s))

        events_in_window = []
        for it in aligned_sorted:
            t = float(it["event_time_s"])
            if t < w0:
                continue
            if t > w1:
                break
            events_in_window.append(it["event"])

        if not events_in_window:
            continue

        commentaries = []
        for c in captions_timeline:
            t = c["t"]
            if t < w0:
                continue
            if t > w1:
                break
            commentaries.append(c["text"])

        windows[window_id] = {
            "match_id": anchor_item["match_id"],
            "window_start": w0,
            "window_end": w1,
            "events": events_in_window,
            "commentaries": commentaries,
            "anchor_time_s": t_event,
            "anchor_type": (event.get("type") or ""),
            "anchor_subtype": (event.get("subtype") or ""),
            "keyword_center_time_s": center_t,
            "keyword_center_found": (t_hit is not None),
        }

    return windows



def group_events_keyword_windowed(aligned_events, window_size=60,
                                  search_pre=30, search_post=45,
                                  anchor_types=None,
                                  center_strategy="asymmetric",
                                  pre_s=None, post_s=None):
    """
    For each anchor event:
      1) find nearest/best keyword hit in commentary near the event
      2) create a local fixed-length window around that keyword hit
      3) collect all events inside the window
      4) collect all commentary captions inside the window

    Output:
      one window per anchor event, where each window may contain multiple events

    Parameters
    ----------
    window_size : int
        Max total window length in seconds.
    center_strategy : str
        "centered"    -> equal split around anchor (window_size/2 each side)
        "asymmetric"  -> prefer more post-context; uses pre_s/post_s if given,
                         otherwise defaults to 15/45 for 60s window.
    pre_s, post_s : float | None
        Optional explicit pre/post durations. If both are given, they override
        center_strategy and window_size split.
    """
    if not aligned_events:
        return {}

    anchor_types = set(anchor_types or DEFAULT_ANCHOR_TYPES)
    aligned_sorted = sorted(aligned_events, key=lambda x: x["event_time_s"])
    captions_timeline = build_global_commentary_timeline(aligned_sorted)

    anchors = [
        item for item in aligned_sorted
        if ((item.get("event") or {}).get("type") in anchor_types)
    ]

    # decide final pre/post lengths
    if pre_s is not None and post_s is not None:
        final_pre = float(pre_s)
        final_post = float(post_s)
    else:
        if center_strategy == "centered":
            final_pre = float(window_size) / 2.0
            final_post = float(window_size) / 2.0
        else:
            # default asymmetric split
            if int(window_size) == 60:
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

        # fallback to event time if no keyword hit found
        center_t = float(t_hit) if t_hit is not None else t_event

        w0 = max(0.0, center_t - final_pre)
        w1 = max(0.0, center_t + final_post)

        events_in_window = []
        anchor_event_indices = []

        for idx, it in enumerate(aligned_sorted):
            t = float(it["event_time_s"])
            if t < w0:
                continue
            if t > w1:
                break

            events_in_window.append(it["event"])

            # track whether this event is itself an anchor type
            etype = ((it.get("event") or {}).get("type") or "").upper()
            if etype in anchor_types:
                anchor_event_indices.append(len(events_in_window) - 1)

        if not events_in_window:
            continue

        commentaries = []
        commentary_times = []

        for c in captions_timeline:
            t = c["t"]
            if t < w0:
                continue
            if t > w1:
                break
            commentaries.append(c["text"])
            commentary_times.append(t)

        if not commentaries:
            continue

        windows[window_id] = {
            "match_id": anchor_item["match_id"],
            "window_start": w0,
            "window_end": w1,
            "events": events_in_window,
            "commentaries": commentaries,

            # metadata
            "anchor_time_s": t_event,
            "anchor_type": (event.get("type") or ""),
            "anchor_subtype": (event.get("subtype") or ""),
            "keyword_center_time_s": center_t,
            "keyword_center_found": (t_hit is not None),
            "window_size_s": (w1 - w0),
            "num_events": len(events_in_window),
            "num_commentary_segments": len(commentaries),
            "anchor_event_positions": anchor_event_indices,
            "commentary_start_s": commentary_times[0] if commentary_times else None,
            "commentary_end_s": commentary_times[-1] if commentary_times else None,
        }

    return windows












# **************************
# Training examples creation
# **************************

# def create_training_examples(windows):
#     examples = []

#     for window_id, window_data in sorted(windows.items()):
#         events = window_data["events"]
#         commentaries = window_data["commentaries"]

#         combined_commentary = " ".join(commentaries).strip()
#         word_count = len(combined_commentary.split())

#         linearized_events = [linearize_event(e) for e in events]
#         linearized_input = " ".join(linearized_events)

#         ex = {
#             "match_id": window_data["match_id"],
#             "window_id": window_id,
#             "window_start": window_data["window_start"],
#             "window_end": window_data["window_end"],
#             "num_events": len(events),
#             "input": linearized_input,
#             "target": combined_commentary,
#             "input_tokens": len(linearized_input.split()),
#             "target_words": word_count,
#         }

#         for k in (
#             "anchor_time_s",
#             "anchor_type",
#             "anchor_subtype",
#             "keyword_center_time_s",
#             "keyword_center_found",
#         ):
#             if k in window_data:
#                 ex[k] = window_data[k]

#         examples.append(ex)

#     return examples


# def create_training_examples(windows):
#     examples = []

#     for window_id, window_data in sorted(windows.items()):
#         events = window_data["events"]
#         commentaries = window_data["commentaries"]

#         combined_commentary = " ".join(commentaries).strip()
#         word_count = len(combined_commentary.split())

#         linearized_events = [linearize_event(e) for e in events]
#         linearized_input = " ".join(linearized_events)

#         ex = {
#             "match_id": window_data["match_id"],
#             "window_id": window_id,
#             "window_start": window_data["window_start"],
#             "window_end": window_data["window_end"],

#             # explicit event list
#             "events": events,
#             "num_events": len(events),

#             # flattened model input
#             "input": linearized_input,

#             # explicit target span
#             "commentary_segments": commentaries,
#             "target": combined_commentary,

#             "input_tokens": len(linearized_input.split()),
#             "target_words": word_count,
#         }

#         for k in (
#             "anchor_time_s",
#             "anchor_type",
#             "anchor_subtype",
#             "keyword_center_time_s",
#             "keyword_center_found",
#             "window_size_s",
#             "num_commentary_segments",
#             "anchor_event_positions",
#             "commentary_start_s",
#             "commentary_end_s",
#         ):
#             if k in window_data:
#                 ex[k] = window_data[k]

#         examples.append(ex)

#     return examples


def dedupe_windows(windows):
    """
    Deduplicate windows using the ordered event sequence only.

    This keeps the first window for each unique event sequence within a match.
    """
    deduped = {}
    seen = set()
    new_id = 0
    removed = 0

    for _, window_data in sorted(windows.items()):
        match_id = window_data["match_id"]

        event_signature = tuple(
            json.dumps(ev, sort_keys=True, ensure_ascii=False)
            for ev in window_data.get("events", [])
        )

        key = (match_id, event_signature)

        if key in seen:
            removed += 1
            continue

        seen.add(key)
        deduped[new_id] = window_data
        new_id += 1

    return deduped, removed


def create_training_examples(windows):
    examples = []

    for window_id, window_data in sorted(windows.items()):
        events = window_data["events"]
        commentaries = window_data["commentaries"]

        combined_commentary = " ".join(commentaries).strip()
        word_count = len(combined_commentary.split())

        linearized_input = linearize_events_for_t5(events)

        ex = {
            "match_id": window_data["match_id"],
            "window_id": window_id,
            "window_start": window_data["window_start"],
            "window_end": window_data["window_end"],

            "events": events,
            "num_events": len(events),

            "input": linearized_input,

            "commentary_segments": commentaries,
            "target": combined_commentary,

            "input_tokens": len(linearized_input.split()),
            "target_words": word_count,
        }

        for k in (
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
            if k in window_data:
                ex[k] = window_data[k]

        examples.append(ex)

    return examples




def extract_match_id(filepath: Path) -> str:
    """
    More robust match ID extraction from filenames like:
      match_123.json
      match_123_anything.json
    """
    m = re.search(r"match_(.+)$", filepath.stem)
    return m.group(1) if m else filepath.stem


def process_all_matches(aligned_dir, grouping, window_size, pre, post, gap, pad,
                        anchor_types, search_pre, search_post):
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

        if grouping == "fixed":
            windows = group_events_into_windows_fixed(aligned_events, window_size)

        elif grouping == "event_centered":
            windows = group_events_event_centered(
                aligned_events,
                pre_s=pre,
                post_s=post,
                anchor_types=anchor_types,
            )

        elif grouping == "gaps":
            windows = group_events_by_gaps(
                aligned_events,
                gap_s=gap,
                pad_s=pad,
            )

        elif grouping == "keyword_centered":
            windows = group_events_keyword_centered(
                aligned_events,
                pre_s=pre,
                post_s=post,
                search_pre=search_pre,
                search_post=search_post,
                anchor_types=anchor_types,
            )

        elif grouping == "keyword_windowed":
            windows = group_events_keyword_windowed(
                aligned_events,
                window_size=window_size,
                search_pre=search_pre,
                search_post=search_post,
                anchor_types=anchor_types,
                center_strategy="asymmetric",
            )

        else:
            raise ValueError(f"Unknown grouping: {grouping}")

        # dedup windwos with identical match_id, time span, and event sequence to avoid redundant examples
        windows, num_removed = dedupe_windows(windows)
        if num_removed:
            print(f"  - Removed {num_removed} duplicate windows (same event sequence)")
        
        examples = create_training_examples(windows)

        if examples:
            all_examples.extend(examples)

            match_stats.append({
                "match_id": match_id,
                "num_windows": len(examples),
                "avg_events_per_window": sum(e["num_events"] for e in examples) / len(examples),
                "avg_words_per_window": sum(e["target_words"] for e in examples) / len(examples),
            })

            print(f"  ✓ Created {len(examples)} training windows")
        else:
            print("  !! No valid windows after filtering")

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





def print_examples(examples, num=2):
    print("\n" + "=" * 60)
    print("SAMPLE TRAINING EXAMPLES")

    for i, example in enumerate(examples[:num], 1):
        print(f"\n--- Example {i} ---")
        print(f"Match ID: {example['match_id']}")
        print(f"Window: {example['window_start']:.0f}s - {example['window_end']:.0f}s")
        print(f"Events: {example['num_events']}")

        if "anchor_time_s" in example:
            anchor_line = f"Anchor: {example.get('anchor_type')} @ {example.get('anchor_time_s'):.0f}s"
            if "keyword_center_time_s" in example:
                anchor_line += f" | keyword_center={example.get('keyword_center_time_s'):.0f}s"
            print(anchor_line)

        print(f"\nINPUT ({example['input_tokens']} tokens):")
        print(example["input"][:300] + "..." if len(example["input"]) > 300 else example["input"])

        print(f"\nTARGET ({example['target_words']} words):")
        print(example["target"][:300] + "..." if len(example["target"]) > 300 else example["target"])



def print_examples_multi_events(examples, num=2, min_events=1):
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

        if "anchor_time_s" in example:
            anchor_line = f"Anchor: {example.get('anchor_type')} @ {example.get('anchor_time_s'):.0f}s"
            if "keyword_center_time_s" in example:
                anchor_line += f" | keyword_center={example.get('keyword_center_time_s'):.0f}s"
            print(anchor_line)

        print("\nEVENT LIST:")
        for j, ev in enumerate(example.get("events", []), 1):
            print(f"  {linearize_event(ev, event_idx=j)}")

        print(f"\nINPUT ({example['input_tokens']} tokens):")
        print(example["input"][:500] + "..." if len(example["input"]) > 500 else example["input"])

        print(f"\nTARGET ({example['target_words']} words):")
        print(example["target"][:500] + "..." if len(example["target"]) > 500 else example["target"])



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--grouping",
        choices=["fixed", "event_centered", "gaps", "keyword_centered", "keyword_windowed"],
        default="fixed",
        help="How to group events into training windows",
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=60,
        help="Window size in seconds (fixed grouping only)",
    )

    parser.add_argument(
        "--pre",
        type=int,
        default=20,
        help="Seconds before center (event_centered / keyword_centered)",
    )
    parser.add_argument(
        "--post",
        type=int,
        default=20,
        help="Seconds after center (event_centered / keyword_centered)",
    )
    parser.add_argument(
        "--anchor-types",
        type=str,
        default="CHAMPION_KILL,ELITE_MONSTER_KILL,BUILDING_KILL",
        help="Comma-separated event types to anchor on (event_centered / keyword_centered)",
    )

    parser.add_argument(
        "--search-pre",
        type=int,
        default=30,
        help="Seconds BEFORE event time to search for keyword hit (keyword_centered)",
    )
    parser.add_argument(
        "--search-post",
        type=int,
        default=45,
        help="Seconds AFTER event time to search for keyword hit (keyword_centered)",
    )

    parser.add_argument(
        "--gap",
        type=int,
        default=20,
        help="Start a new window if time between events exceeds this (gaps grouping)",
    )
    parser.add_argument(
        "--pad",
        type=int,
        default=0,
        help="Pad burst windows by N seconds on both sides (gaps grouping)",
    )

    args = parser.parse_args()



    # aligned_dir = Path("data/processed/aligned") ##################################
    aligned_dir = Path("data/WCPI2019/aligned")
###########################################################################################



    anchor_types = [t.strip() for t in args.anchor_types.split(",") if t.strip()]

    if args.grouping == "fixed":
        output_path = Path("data/processed") / f"training_data_fixed_w{args.window_size}s.jsonl"
        stats_path = Path("data/processed") / f"linearization_stats_fixed_w{args.window_size}s.json"

    elif args.grouping == "event_centered":
        output_path = Path("data/processed") / f"training_data_event_centered_pre{args.pre}_post{args.post}.jsonl"
        stats_path = Path("data/processed") / f"linearization_stats_event_centered_pre{args.pre}_post{args.post}.json"

    elif args.grouping == "keyword_centered":
        output_path = Path("data/processed") / (
            f"training_data_keyword_centered_pre{args.pre}_post{args.post}"
            f"_search{args.search_pre}-{args.search_post}.jsonl"
        )
        stats_path = Path("data/processed") / (
            f"linearization_stats_keyword_centered_pre{args.pre}_post{args.post}"
            f"_search{args.search_pre}-{args.search_post}.json"
        )
    
    # OUTPUT PATH ###################################################################
    #################################################################################
    elif args.grouping == "keyword_windowed":
        output_path = Path("data/WCPI2019/processed") / (
            f"training_data_keyword_windowed_w{args.window_size}s"
            f"_search{args.search_pre}-{args.search_post}.jsonl"
        )
        stats_path = Path("data/WCPI2019/processed") / (
            f"linearization_stats_keyword_windowed_w{args.window_size}s"
            f"_search{args.search_pre}-{args.search_post}.json"
        )

    else:
        output_path = Path("data/WCPI2019/processed") / f"training_data_gaps_gap{args.gap}_pad{args.pad}.jsonl"
        stats_path = Path("data/WCPI2019/processed") / f"linearization_stats_gaps_gap{args.gap}_pad{args.pad}.json"

    print("=" * 60)
    print("LINEARIZATION AND TRAINING DATA")
    print(f"Grouping: {args.grouping}")

    if args.grouping == "fixed":
        print(f"Window size: {args.window_size}s")
    elif args.grouping == "event_centered":
        print(f"Anchor types: {anchor_types}")
        print(f"Context: -{args.pre}s / +{args.post}s")
    elif args.grouping == "keyword_centered":
        print(f"Anchor types: {anchor_types}")
        print(f"Final window: -{args.pre}s / +{args.post}s")
        print(f"Keyword search: -{args.search_pre}s / +{args.search_post}s around event time")
    elif args.grouping == "keyword_windowed":
        print(f"Anchor types: {anchor_types}")
        print(f"Window size: {args.window_size}s")
        print(f"Keyword search: -{args.search_pre}s / +{args.search_post}s around event time")
    else:
        print(f"Gap threshold: {args.gap}s")
        print(f"Pad: {args.pad}s")

    examples, match_stats = process_all_matches(
        aligned_dir=aligned_dir,
        grouping=args.grouping,
        window_size=args.window_size,
        pre=args.pre,
        post=args.post,
        gap=args.gap,
        pad=args.pad,
        anchor_types=anchor_types,
        search_pre=args.search_pre,
        search_post=args.search_post,
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

    # print_examples(examples, num=2)

    print_examples_multi_events(examples, num=2, min_events=2)

    multi_event_examples = [e for e in examples if e["num_events"] >= 2]
    print(f"\nExamples with >=2 events: {len(multi_event_examples)} / {len(examples)}")


if __name__ == "__main__":
    main()








