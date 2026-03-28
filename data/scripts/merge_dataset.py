import json
from pathlib import Path
import argparse


def load_jsonl_examples(filepath: Path):
    examples = []
    match_ids = set()

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  !! Skipping malformed JSON in {filepath} line {line_num}: {e}")
                continue

            if "input" not in obj or "target" not in obj:
                print(f"  !! Skipping example missing input/target in {filepath} line {line_num}")
                continue

            examples.append({
                "input": obj["input"],
                "target": obj["target"],
            })

            if "match_id" in obj:
                match_ids.add(obj["match_id"])

    return examples, match_ids


def find_training_files(data_dir: Path):
    """
    Find files matching:
      data/<year>/<tournament>/processed/train*.jsonl
    """
    matches = []

    for year_dir in sorted(data_dir.iterdir()):
        if not year_dir.is_dir():
            continue

        for tournament_dir in sorted(year_dir.iterdir()):
            if not tournament_dir.is_dir():
                continue

            processed_dir = tournament_dir / "processed"
            if not processed_dir.exists() or not processed_dir.is_dir():
                continue

            train_files = sorted(processed_dir.glob("train*.jsonl"))
            for filepath in train_files:
                dataset_name = f"{year_dir.name}/{tournament_dir.name}"
                matches.append((dataset_name, filepath))

    return matches


def merge_datasets(data_dir: Path, output_file: Path):
    found_files = find_training_files(data_dir)

    if not found_files:
        print(f"No train*.jsonl files found under: {data_dir}")
        return

    merged_examples = []
    summary = {}
    match_summary = {}

    print(f"Searching under: {data_dir}\n")

    for dataset_name, filepath in found_files:
        print(f"Processing: {filepath}")

        dataset_examples, dataset_matches = load_jsonl_examples(filepath)

        merged_examples.extend(dataset_examples)
        summary[dataset_name] = summary.get(dataset_name, 0) + len(dataset_examples)
        match_summary.setdefault(dataset_name, set()).update(dataset_matches)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for ex in merged_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print("MERGE SUMMARY")

    total_matches = set()

    for dataset_name in sorted(summary.keys()):
        num_examples = summary[dataset_name]
        dataset_match_ids = match_summary.get(dataset_name, set())
        num_matches = len(dataset_match_ids)

        total_matches.update((dataset_name, match_id) for match_id in dataset_match_ids)

        print(f"{dataset_name}:")
        print(f"  examples: {num_examples}")
        print(f"  matches:  {num_matches}")

    print("-" * 60)
    print(f"Total datasets: {len(summary)}")
    print(f"Total examples: {len(merged_examples)}")
    print(f"Total unique matches: {len(total_matches)}")
    print(f"Saved merged dataset to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge train*.jsonl files from data/<year>/<tournament>/processed into one combined JSONL."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Root data directory containing yearly tournament folders",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/merged/combined_train.jsonl",
        help="Output merged JSONL file",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_file = Path(args.output)

    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    merge_datasets(data_dir, output_file)


if __name__ == "__main__":
    main()
