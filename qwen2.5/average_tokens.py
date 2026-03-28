import json
import numpy as np
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
FILE_PATH = "data/combined_train.jsonl"

RESPONSE_TEMPLATE = "### Response:\n"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

input_lens = []
target_lens = []
total_lens = []

def format_example(inp, tgt):
    return (
        "### Task:\n"
        "Generate live League of Legends shoutcaster commentary from the event log.\n\n"
        "### Input:\n"
        f"{inp}\n\n"
        f"{RESPONSE_TEMPLATE}"
        f"{tgt}"
    )

with open(FILE_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        try:
            ex = json.loads(line)
        except json.JSONDecodeError:
            print(f"Skipping bad line {i}")
            continue

        inp = ex["input"]
        tgt = ex["target"]

        input_ids  = tokenizer(inp).input_ids
        target_ids = tokenizer(tgt).input_ids
        total_ids  = tokenizer(format_example(inp, tgt)).input_ids

        input_lens.append(len(input_ids))
        target_lens.append(len(target_ids))
        total_lens.append(len(total_ids))

def stats(arr, name):
    print(f"\n{name}:")
    print(f"  mean:   {np.mean(arr):.1f}")
    print(f"  median: {np.median(arr):.1f}")
    print(f"  p90:    {np.percentile(arr, 90):.1f}")
    print(f"  p95:    {np.percentile(arr, 95):.1f}")
    print(f"  max:    {np.max(arr)}")

stats(input_lens,  "Input length")
stats(target_lens, "Target length")
stats(total_lens,  "Total length (formatted)")

print("\n--- Longest samples ---")
idxs = np.argsort(total_lens)[-5:]

with open(FILE_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()

for i in idxs:
    print(f"\nRank index: {i} | Total tokens: {total_lens[i]}")
    print(lines[i][:500])

print(f"\nSamples > 900 tokens:  {sum(x > 900  for x in total_lens)}")
print(f"Samples > 1024 tokens: {sum(x > 1024 for x in total_lens)}")
print(f"Total samples: {len(total_lens)}")
