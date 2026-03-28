# Qwen2.5 Fine-Tuning Pipeline

Fine-tuning and evaluation pipeline for automated League of Legends esports commentary generation using Qwen2.5-7B-Instruct with QLoRA.

---

## Table of Contents

- [Data Structure](#data-structure)
- [Scripts](#scripts)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Recommended Workflow](#recommended-workflow)
- [Notes](#notes)

---

## Data Structure

```bash
qwen/
├── data/
│   ├── train.jsonl               # Training split (80%)
│   ├── val.jsonl                 # Validation split (10%)
│   ├── test.jsonl                # Test split (10%)
│   └── combined_train.jsonl     # Full dataset before splitting
├── outputs/                      # Fine-tuned model checkpoints
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer.json
│   ├── checkpoint-100/
│   ├── checkpoint-200/
│   └── ...
└── results/
    ├── qwen7b_finetuned/
    │   ├── generations.jsonl     # Generated predictions for test set
    │   ├── metrics.json          # All evaluation metrics
    │   ├── predictions.jsonl     # Predictions in eval format
    │   ├── report.txt            # Human-readable report
    │   └── faithfulness.json     # EFS faithfulness results
    └── qwen14b_finetuned/
        └── ...
```

**File Meanings**

- Data Files
  - `train.jsonl` — training examples (80% of dataset)
  - `val.jsonl` — validation examples used during training (10%)
  - `test.jsonl` — held-out test set for final evaluation (10%)
  - `combined_train.jsonl` — full merged dataset before splitting

- Model Outputs
  - `adapter_model.safetensors` — LoRA adapter weights (best checkpoint)
  - `adapter_config.json` — LoRA configuration
  - `checkpoint-N/` — periodic training checkpoints
  - `tokenizer.json` — saved tokenizer

- Evaluation Outputs
  - `generations.jsonl` — raw model outputs for each test input
  - `metrics.json` — all automatic metrics in JSON
  - `predictions.jsonl` — predictions formatted for eval pipeline
  - `report.txt` — human-readable metric summary
  - `faithfulness.json` — Event Faithfulness Score (EFS) results

---

## Scripts

```bash
scripts/
├── split_dataset.py       # Split dataset into train/val/test (80/10/10)
├── average_tokens.py      # Analyse token length distribution
├── dataset_stats.py       # Compute event count statistics
├── finetune_qwen.py       # Fine-tune Qwen2.5-7B with QLoRA
├── batch_inference.py     # Batched test set inference
├── run_metrics.py         # Compute ROUGE, BERTScore, BARTScore, etc.
├── run_faithfulness.py    # Compute Event Faithfulness Score (EFS)
├── faithfulness.py        # EFS implementation
└── eval_metrics.py        # Evaluation metric implementations
```

---

## Dataset Preparation

### Split Dataset

Splits the combined dataset by example into train/val/test at 80/10/10:

```bash
python scripts/split_dataset.py
```

Edit `INPUT_FILE` inside the script to point to your combined dataset file. Output is written to `data/train.jsonl`, `data/val.jsonl`, and `data/test.jsonl`.

### Analyse Token Lengths

Before training, check the token length distribution to set `MAX_LENGTH` appropriately:

```bash
python scripts/average_tokens.py
```

Output example:
```
Input length:
  mean:   130.0
  p95:    327.0
  max:    504

Target length:
  mean:   240.0
  p95:    386.0
  max:    527
```

### Dataset Statistics

Check average number of events per training example:

```bash
python scripts/dataset_stats.py
```

---

## Training

Fine-tune Qwen2.5-7B-Instruct with 4-bit QLoRA on a single GPU:

```bash
python scripts/finetune_qwen.py
```

**Key configuration** (edit inside script):

| Parameter | Value | Description |
|---|---|---|
| `MODEL_NAME` | `Qwen/Qwen2.5-7B-Instruct` | Base model |
| `MAX_LENGTH` | `800` | Max sequence length (tokens) |
| `per_device_train_batch_size` | `16` | Batch size per GPU |
| `gradient_accumulation_steps` | `2` | Effective batch = 32 |
| `learning_rate` | `2e-4` | Peak learning rate |
| `num_train_epochs` | `5` | Training epochs |
| `lora_r` | `16` | LoRA rank |
| `lora_alpha` | `32` | LoRA scaling factor |
| `lora_dropout` | `0.1` | LoRA dropout |

**Hardware requirements:**

| Setup | Min VRAM | Notes |
|---|---|---|
| A100 40GB (Colab) | 40GB | Full batch, no gradient checkpointing |
| RTX 5080 16GB | 16GB | Reduce batch to 2, enable gradient checkpointing |
| G4 96GB | 96GB | Increase batch to 16–32 |

**Resume from checkpoint:**

Set `resume_from_checkpoint` in the train call:

```python
trainer.train(resume_from_checkpoint="/path/to/outputs/checkpoint-600")
```

**Monitor training:**

Training logs to Weights & Biases. Login before running:

```bash
wandb login
```

---

## Inference

Run batched inference on the test set:

```bash
python scripts/batch_inference.py
```

**Key configuration** (edit inside script):

| Parameter | Value | Description |
|---|---|---|
| `BASE_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | Base model |
| `ADAPTER_DIR` | path to outputs/ | Fine-tuned LoRA adapter |
| `TEST_FILE` | path to test.jsonl | Test set |
| `BATCH_SIZE` | `32` | Adjust based on VRAM |
| `max_new_tokens` | `400` | Max generation length |

The script automatically resumes if interrupted — it tracks which inputs have already been generated and skips them on restart.

**Generation parameters:**

| Parameter | Value |
|---|---|
| Temperature | 0.7 |
| Top-p | 0.9 |
| Repetition penalty | 1.1 |
| Padding side | Left (required for batched generation) |

---

## Evaluation

### Automatic Metrics

Compute ROUGE-L, BERTScore, BARTScore, and text distance:

```bash
python scripts/run_metrics.py
```

Edit `OUTPUT_DIR` and `GENERATIONS_FILE` to point to your results directory.

**Metrics reported:**

| Metric | Description |
|---|---|
| ROUGE-L | Longest common subsequence similarity |
| BERTScore | Semantic similarity via RoBERTa-large embeddings |
| BARTScore | Generation quality via BART log-likelihood (ref + src) |
| Text distance | Normalised Levenshtein edit distance |
| SacreBLEU | Corpus-level BLEU (reported but not primary metric) |

### Event Faithfulness Score (EFS)

Compute EFS to measure how faithfully generated commentary reflects input events:

```bash
python scripts/run_faithfulness.py
```

Edit `GENERATIONS_FILE` and `OUTPUT_FILE` inside the script.

**EFS output:**

```
Prediction:
  EFS mean:              0.4692
  EFS (%):               46.9%
  Events fully covered:  1.47
  Events missed:         0.55

Target (reference):
  EFS mean:              0.4321
  EFS (%):               43.2%

Delta (prediction - target):  +3.7%

Prediction hallucinated event types:
  BARON                          495 samples  (32.7%)
  DRAGON                         435 samples  (28.7%)
  ...
```

A positive delta indicates the model references input events more explicitly than human casters. Per-type hallucination rates are compared against the reference to distinguish model failures from domain-level behaviour.

---

## Recommended Workflow

```bash
# 1. Analyse dataset
python scripts/dataset_stats.py
python scripts/average_tokens.py

# 2. Split dataset
python scripts/split_dataset.py

# 3. Fine-tune
python scripts/finetune_qwen.py

# 4. Run inference on test set
python scripts/batch_inference.py

# 5. Evaluate
python scripts/run_metrics.py
python scripts/run_faithfulness.py
```

---

## Notes

- **Splitting**: Split is performed at the example level with a fixed seed of 42 for reproducibility
- **Masking**: Loss is computed on response tokens only — input prompt tokens are masked with `-100`
- **Checkpointing**: Best checkpoint (lowest validation loss) is saved to `outputs/` root via `load_best_model_at_end=True`
- **Quantisation**: QLoRA uses NF4 4-bit with double quantisation — base model weights remain fully frozen
- **Padding**: Use `padding_side="right"` during training and `padding_side="left"` during batched inference
- **Inference without quantisation**: `batch_inference.py` loads the model in full bfloat16 (no quantisation) for inference — this requires more VRAM but produces cleaner outputs
