import json
import re
from pathlib import Path

# Change this to your file name inside data/
FILE_PATH = Path("data/full_data/combined_train.jsonl")

EVENT_PATTERN = re.compile(r"\[EVENT\s+\d+\]")

def main() -> None:
    total_examples = 0
    total_events = 0
    max_events = 0
    max_events_line = None

    with FILE_PATH.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                example = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON on line {line_num}: {e}")
                continue

            input_text = example.get("input", "")
            event_count = len(EVENT_PATTERN.findall(input_text))

            total_examples += 1
            total_events += event_count

            if event_count > max_events:
                max_events = event_count
                max_events_line = line_num

    if total_examples == 0:
        print("No valid examples found.")
        return

    avg_events = total_events / total_examples

    print(f"Processed examples: {total_examples}")
    print(f"Average events per example: {avg_events:.4f}")
    print(f"Maximum number of events in one example: {max_events}")
    print(f"Line with maximum events: {max_events_line}")

if __name__ == "__main__":
    main()