import json
import sys
from pathlib import Path

sys.path.append('.')  # looks for faithfulness.py in same directory
from faithfulness import evaluate_faithfulness

# ── Config ────────────────────────────────────────────────────────────────────
GENERATIONS_FILE = Path("./results/14B_1921/predictions.jsonl")
OUTPUT_FILE      = Path("./results/faithfulness.json")
PENALTY_LAMBDA   = 0.5

# ── Load generations ──────────────────────────────────────────────────────────
predictions = []
with open(GENERATIONS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        predictions.append(json.loads(line))

print(f"Loaded {len(predictions)} predictions")

# ── Run evaluation ────────────────────────────────────────────────────────────
print("Computing faithfulness...")
results = evaluate_faithfulness(predictions, penalty_lambda=PENALTY_LAMBDA)

# ── Print summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FAITHFULNESS EVALUATION SUMMARY")
print("=" * 60)

print(f"\nPrediction:")
print(f"  EFS mean:              {results['prediction']['faithfulness_mean']:.4f}")
print(f"  EFS (%):               {results['prediction']['faithfulness_pct']:.1f}%")
print(f"  EFS std:               {results['prediction']['faithfulness_std']:.4f}")
print(f"  Events fully covered:  {results['prediction']['events_fully_covered']:.2f}")
print(f"  Events missed:         {results['prediction']['events_missed']:.2f}")

print(f"\nTarget (reference):")
print(f"  EFS mean:              {results['target']['faithfulness_mean']:.4f}")
print(f"  EFS (%):               {results['target']['faithfulness_pct']:.1f}%")
print(f"  EFS std:               {results['target']['faithfulness_std']:.4f}")
print(f"  Events fully covered:  {results['target']['events_fully_covered']:.2f}")
print(f"  Events missed:         {results['target']['events_missed']:.2f}")

print(f"\nDelta (prediction - target):")
print(f"  EFS delta:             {results['delta']['faithfulness_pct']:+.1f}%")

print(f"\nPrediction hallucinated event types (out of {len(predictions)} samples):")
for t, rate in results['prediction']['hallucination_rates'].items():
    count = results['prediction']['hallucination_counts'][t]
    print(f"  {t:<30} {count:>5} samples  ({rate*100:.1f}%)")

print(f"\nTarget hallucinated event types (out of {len(predictions)} samples):")
for t, rate in results['target']['hallucination_rates'].items():
    count = results['target']['hallucination_counts'][t]
    print(f"  {t:<30} {count:>5} samples  ({rate*100:.1f}%)")

# ── Save ──────────────────────────────────────────────────────────────────────
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved faithfulness results to {OUTPUT_FILE}")
