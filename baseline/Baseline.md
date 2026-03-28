# Baseline Training

This branch implements the baseline training and evaluation pipeline for LoL esports commentary generation

**Table of Contents**

- [Data Structure](#data-structure)
- [Scripts](#scripts)
- [Dataset](#dataset-preparation)
- [Training](#training)
  - [Training Parameters](#training-parameters)
- [Evaluation](#evaluation)
- [Results](#results)
- [Recommended Workflow](#recommended-workflow)
- [Notes](#notes)

---

## Data Structure

```bash
baseline/
├── split/
│   ├── LoL19/
│   │   ├── LoL19_train.jsonl
│   │   ├── LoL19_val.jsonl
│   │   ├── LoL19_test.jsonl
│   │   └── LoL19_split_stats.json
│   └── LoL1921/
│       ├── LoL1921_train.jsonl
│       ├── LoL1921_val.jsonl
│       ├── LoL1921_test.jsonl
│       └── LoL1921_split_stats.json
├── models/                       # Trained model checkpoints
│   ├── t5-base_LoL19/
│   │   ├── best_model/
│   │   ├── checkpoint-epoch3-step5000/
│   │   └── runs/                 # TensorBoard logs
│   └── t5-base_LoL1921/
└── results/                      # Evaluation results
    ├── t5-base_LoL19/
    │   ├── faithfulness.json    
    │   ├── metrics.json
    │   ├── predictions.jsonl
    │   ├── samples.json
    │   └── report.txt
    └── t5-base_LoL1921/
```

**File Meanings**

- Split Files
  - `LoL19_train.jsonl` - Training set (80% of LoL19 matches)
  - `LoL19_val.jsonl` - Validation set (10% of LoL19 matches)
  - `LoL19_test.jsonl` - Test set (10% of LoL19 matches)
  - `LoL19_split_stats.json` - Dataset statistics and split metadata

- Model Checkpoints
  - `best_model/` - Best model based on validation loss
  - `checkpoint-epochN-stepM/` - Periodic training checkpoints
  - `runs/` - TensorBoard event files

- Evaluation Outputs
  - `metrics.json` - All evaluation metrics in JSON format
  - `predictions.jsonl` - Generated predictions for all test examples
  - `samples.json` - Diverse sample outputs for qualitative analysis
  - `report.txt` - Human-readable evaluation report
  - `faithfulness.json` - Faithfulness evaluation results measuring factual grounding to input events

## Scripts

```bash
scripts/
├── split.py                     # Split merged dataset by match_id
├── train_t5.py                  # Train seq2seq model (T5)
├── train_llama.py               # Train LLaMA 7B if you have LLaMA access/license
├── evaluate_t5.py               # Compute metrics on test set
├── run_faithfulness.py          # Compute Event Faithfulness Score (EFS) on test set
├── to_hf.py                     # Upload splits to HuggingFace Hub
└── requirements_training.txt    # Dependencies
```

**Script Purposes**

| Script | Input | Output | Purpose |
|--------|-------|--------|---------|
| `split.py` | `data/merged/*.jsonl` | `split/*/train/val/test.jsonl` | Split by match (80/10/10) |
| `train_t5.py` | `split/*/train.jsonl` | `models/*/best_model/` | Train seq2seq model |
| `evaluate_t5.py` | `models/*/best_model/` | `results/*/metrics.json` | Compute BLEU, ROUGE, etc. |
| `run_faithfulness.py` | `results/*/predictions.jsonl` | `results/*/faithfulness.json` | Compute EEFS for predictions against input events |
| `to_hf.py` | `split/*/` | HuggingFace Hub | Share datasets online |


## Dataset Preparation

### Split Dataset

Default: ratios 0.8 0.1 0.1

Split the merged dataset into train/val/test by match_id:

```bash
# Split LoL19 (2019 World Championship only)
python scripts/split.py LoL19

# Split LoL1921 (2019-2021 all tournaments)
python scripts/split.py LoL1921
```

**Output**
```
split/LoL19/
├── LoL19_train.jsonl          # 80% of matches
├── LoL19_val.jsonl            # 10% of matches
├── LoL19_test.jsonl           # 10% of matches
└── LoL19_split_stats.json     # Statistics
```

#### HuggingFace

For easier sharing and remote training, upload datasets to HuggingFace Hub:

**Step 1: Get HuggingFace token**
1. Go to `huggingface.co/settings/tokens`
2. Create a token with **Write** permission
3. Copy the token

**Step 2: Login**
```bash
pip install huggingface_hub
huggingface-cli login
# Paste your token when prompted
```

**Step 3: Upload datasets**
```bash
# Upload LoL19
python scripts/to_hf.py LoL19 --username your-username

# Upload LoL1921
python scripts/to_hf.py LoL1921 --username your-username
```

## Training

Train using local JSONL files:

```bash
# T5-base on LoL19
python scripts/train_t5.py \
  --model-name t5-base \
  --batch-size 8 \
  --learning-rate 5e-5 \
  --num-epochs 10 \
  --eval-steps 500 \
  --save-steps 1000
```

Train using datasets from HuggingFace Hub:

```bash
# T5-base from HuggingFace
python scripts/train_t5.py \
  --model-name t5-base \
  --name your-username/LoL19 \
  --batch-size 8 \
  --num-epochs 10
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model-name` | `t5-base` | Pretrained model: `t5-small`, `t5-base`, `t5-large`, `facebook/bart-base` |
| `--name` | None | HuggingFace dataset name (HF mode) |
| `--batch-size` | 8 | Training batch size |
| `--learning-rate` | 1e-3 | Learning rate |
| `--num-epochs` | 10 | Number of training epochs |
| `--max-input-length` | 512 | Maximum input sequence length |
| `--max-target-length` | 256 | Maximum target sequence length |
| `--eval-steps` | 500 | Evaluate every N steps |
| `--save-steps` | 1000 | Save checkpoint every N steps |
| `--resume-from` | None | Resume from checkpoint directory |

**Resume Training**

If training is interrupted:

```bash
python scripts/train_t5.py \
  --model-name t5-base \
  --resume-from models/t5-base_LoL19/checkpoint-epoch3-step5000
```

## Evaluation

Evaluate trained models on the test set:

```bash
# Evaluate on local test file
python scripts/evaluate_t5.py \
  --model models/t5-base_LoL19/best_model \
  --test-file split/LoL19/LoL19_test.jsonl

# Evaluate using HuggingFace dataset
python scripts/evaluate_t5.py \
  --model models/t5-base_LoL19/best_model \
  --dataset-name your-username/LoL19 \
  --output-dir results/t5-base_LoL19_hf

# Evaluate EFS
python scripts/run_faithfulness.py
```

### Evaluation Metrics

| Metric | Range | Interpretation |
|--------|-------|----------------|
| **BLEU-4** | 0-1 | N-gram overlap; 0.10-0.20 = acceptable, >0.30 = good |
| **ROUGE-L** | 0-1 | Longest common subsequence; 0.40-0.50 = good |
| **BERTScore** | 0-1 | Semantic similarity; 0.60-0.70 = baseline, >0.75 = good |
| **Text Similarity** | 0-1 | Character-level similarity; >0.70 = good |
| **BARTScore** | Negative | Generation quality; -2.0 to -1.0 = good (higher is better) |
| **Perplexity** | >0 | Model confidence; <20 = very good, 20-50 = acceptable |
| **Faithfulness** | 0-1 | Factual grounding to input events; >0.80 = good |


## Results

Evaluation outputs are saved in the `results/` directory:

```bash
results/t5-base_LoL19/
├── faithfulness.json      # EFS metrics in JSON format
├── metrics.json           # All metrics in JSON format
├── predictions.jsonl      # All test predictions
├── samples.json           # Diverse sample outputs
└── report.txt             # Human-readable report
```

**Example metrics.json**

```json
{
  "bleu_1": 0.3245,
  "bleu_4": 0.1142,
  "rouge_l": 0.4567,
  "bertscore_f1": 0.6789,
  "text_similarity": 0.7234,
  "bartscore_avg": -2.3456,
  "perplexity": 12.34
}
```

**Example report.txt excerpt**

```
================================================================================
EVALUATION SUMMARY
================================================================================

Core Metrics:
  BLEU-1:        0.3245
  BLEU-4:        0.1142
  ROUGE-L:       0.4567
  BERTScore:     0.6789
  Text Sim:      0.7234
  BARTScore:     -2.3456
  Perplexity:    12.34

--- Sample 1 ---
INPUT:  [EVENT 1] type=CHAMPION_KILL dt=0 side=blue killer=Doinb...
TARGET: Amazing kill by Doinb on Caps, the mid lane battle continues
PRED:   Doinb secures the kill on Caps in the mid lane
```


## Recommended Workflow

Complete baseline workflow for paper results:

```bash
# 1. Prepare datasets
python scripts/split.py LoL19
python scripts/split.py LoL1921

# 2. Upload to HuggingFace
python scripts/to_hf.py LoL19 --username your-username
python scripts/to_hf.py LoL1921 --username your-username

# 3. Train baseline models
# T5-base on LoL19
python scripts/train_t5.py \
  --model-name t5-base \
  --num-epochs 10

# T5-base on LoL1921 (more data)
python scripts/train_t5.py \
  --model-name t5-base \
  --num-epochs 10

# 4. Evaluate both models
python scripts/evaluate_t5.py \
  --model models/t5-base_LoL19/best_model \
  --test-file split/LoL19/LoL19_test.jsonl \

python scripts/evaluate_t5.py \
  --model models/t5-base_LoL1921/best_model \
  --test-file split/LoL1921/LoL1921_test.jsonl \
```


## Notes

- **Data Splitting**: Always split by `match_id` to prevent data leakage between train/val/test
- **Reproducibility**: Set `--seed` consistently across splits and training for reproducibility
- **Best Model**: The checkpoint with lowest validation loss is saved as `best_model/`