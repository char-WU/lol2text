# CLEAN METRICS CELL

import json
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.append('/content')
from eval_metrics import (
    bleu, rouge, bertscore, bartscore,
    text_distance, pred_length,
    save_results, print_summary
)

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR       = Path("/content/drive/MyDrive/MLP/results/qwen7b_finetuned")
GENERATIONS_FILE = OUTPUT_DIR / "generations.jsonl"

# ── Reload saved generations ──────────────────────────────────────────────────
predictions = []
with open(GENERATIONS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        predictions.append(json.loads(line))

print(f"Loaded {len(predictions)} predictions")

# ── Compute metrics ───────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
metrics = {}

print("Computing BLEU...")
metrics.update(bleu(predictions))

print("Computing ROUGE...")
metrics.update(rouge(predictions))

print("Computing BERTScore...")
try:
    metrics.update(bertscore(predictions, device=device))
except Exception as e:
    print(f"BERTScore failed: {e}")

torch.cuda.empty_cache()

print("Computing text distance...")
try:
    metrics.update(text_distance(predictions))
except Exception as e:
    print(f"Text distance failed: {e}")

print("Computing BARTScore...")
try:
    metrics.update(bartscore(predictions, device=device))
except Exception as e:
    print(f"BARTScore failed: {e}")

torch.cuda.empty_cache()

print("Computing length stats...")
metrics["length_stats"] = pred_length(predictions)

# ── Save results and print summary ────────────────────────────────────────────
save_results(OUTPUT_DIR, metrics, predictions)
print_summary(metrics)

print(f"\nAll results saved to {OUTPUT_DIR}")