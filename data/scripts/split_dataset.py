import json
import random
from pathlib import Path

INPUT_FILE = "data/combined_train.jsonl"
OUTPUT_DIR = "data"

TRAIN_RATIO = 0.90
VAL_RATIO = 0.05
TEST_RATIO = 0.05

SEED = 42


def main():
    random.seed(SEED)

    # Load all lines
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Total examples: {len(lines)}")

    # Shuffle
    random.shuffle(lines)

    n = len(lines)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_lines = lines[:n_train]
    val_lines = lines[n_train:n_train + n_val]
    test_lines = lines[n_train + n_val:]

    print(f"Train: {len(train_lines)}")
    print(f"Val:   {len(val_lines)}")
    print(f"Test:  {len(test_lines)}")

    # Write files
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    with open(f"{OUTPUT_DIR}/train.jsonl", "w", encoding="utf-8") as f:
        f.writelines(train_lines)

    with open(f"{OUTPUT_DIR}/val.jsonl", "w", encoding="utf-8") as f:
        f.writelines(val_lines)

    with open(f"{OUTPUT_DIR}/test.jsonl", "w", encoding="utf-8") as f:
        f.writelines(test_lines)

    print("Done.")


if __name__ == "__main__":
    main()
