import json
from pathlib import Path
import argparse


def process_and_stream_file(filepath: Path, out_file_handle):
    """Reads a single JSONL file and streams its valid examples directly to the output file."""
    match_ids = set()
    example_count = 0

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

            example = {
                "input": obj["input"],
                "target": obj["target"],
            }

            # Write directly to disk instead of saving to memory
            out_file_handle.write(json.dumps(example, ensure_ascii=False) + "\n")
            example_count += 1

    return example_count, match_ids


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

    summary = {}
    match_summary = {}
    total_examples_count = 0

    print(f"Searching under: {data_dir}\n")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Open the output file once, and stream data into it
    with open(output_file, "w", encoding="utf-8") as out_f:
        for dataset_name, filepath in found_files:
            print(f"Processing: {filepath}")

            # Stream directly from input file to output file
            count, dataset_matches = process_and_stream_file(filepath, out_f)

            summary[dataset_name] = summary.get(dataset_name, 0) + count
            match_summary.setdefault(dataset_name, set()).update(dataset_matches)
            total_examples_count += count

    # Print summaries
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
    print(f"Total examples: {total_examples_count}")
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
