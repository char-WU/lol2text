import json
import numpy as np
from transformers import AutoTokenizer

# 👇 change if you're using a different Qwen size
model_name = "Qwen/Qwen2.5-7B-Instruct"

# 👇 your file path
file_path = "data2/merged/combined_train.jsonl"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

input_lens = []
target_lens = []
total_lens = []

def format_example(inp, tgt):
    return (
        "### Task:\n"
        "You are generating live League of Legends caster commentary.\n\n"
        "### Input:\n"
        f"{inp}\n\n"
        "### Commentary:\n"
        f"{tgt}"
    )

with open(file_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        try:
            ex = json.loads(line)
        except:
            print(f"Skipping bad line {i}")
            continue

        inp = ex["input"]
        tgt = ex["target"]

        input_ids = tokenizer(inp).input_ids
        target_ids = tokenizer(tgt).input_ids
        total_ids = tokenizer(format_example(inp, tgt)).input_ids

        input_lens.append(len(input_ids))
        target_lens.append(len(target_ids))
        total_lens.append(len(total_ids))

def stats(arr, name):
    print(f"\n{name}:")
    print(f"  mean: {np.mean(arr):.1f}")
    print(f"  median: {np.median(arr):.1f}")
    print(f"  p90: {np.percentile(arr, 90):.1f}")
    print(f"  p95: {np.percentile(arr, 95):.1f}")
    print(f"  max: {np.max(arr)}")

stats(input_lens, "Input length")
stats(target_lens, "Target length")
stats(total_lens, "Total length")

# 👇 show a few longest examples
print("\n--- Longest samples ---")
idxs = np.argsort(total_lens)[-5:]

with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

for i in idxs:
    print(f"\nLength: {total_lens[i]}")
    print(lines[i][:500])  # preview first 500 chars


# sanity check for model limits
print(sum(x > 900 for x in total_lens)) # expect tiny 
print(sum(x > 1024 for x in total_lens)) # expect 0

