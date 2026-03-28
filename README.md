# lol2text

Neural data-to-text generation for esports commentary from structured gameplay logs.

## Overview
This project addresses limitations in neural esports commentary generation for League of Legends. Building on Wang & Yoshinaga (2024)'s foundational work, we make the following contributions:

**Key Contributions:**
- **Datasets**: LoL19 (~5,700 examples) and LoL19-21 (~12,000 examples) with complete construction pipeline handling video-to-match alignment, temporal offset extraction, and event linearization
- **Event Faithfulness Score (EFS)**: A hallucination-aware metric measuring event coverage and spurious mention detection in generated commentary
- **Model evaluation** of transformer architectures(T5-base, Qwen2.5-7/13B) across multiple temporal configurations: 60s baseline, 30s finer-grained, and filtered
- **Comprehensive metrics** including BLEU, ROUGE-L, BERTScore, BARTScore, and EFS

Built on transformer-based data-to-text generation, our approach aims to generate more coherent and strategically informative League of Legends commentary.

## Research Questions

**RQ1: Temporal Context Window**  
Does reducing temporal window size (60s → 30s) improve commentary coherence by reducing the event-commentary temporal mismatch, or does it fragment narrative context? How does window size interact with model capacity?

**RQ2: Event Filtering**  
Does filtering input to high-salience events reduce noise and improve generation quality, or does it remove essential context needed for strategic commentary?

## Repository Structure
 
```
lol2text/
├── data/           # Data collection, alignment, and linearization pipeline
├── baseline/       # T5-base seq2seq training and evaluation
├── qwen2.5/           # Qwen2.5-7/13B LoRA fine-tuning and evaluation
```

## Guides

### Data Collection and Preparation
For building the dataset from scratch — scraping gameplay events from gol.gg, transcribing YouTube broadcasts via Whisper, aligning commentary to events, and linearizing into training samples — see [`data/Dataset.md`](data/Dataset.md).

This covers:
- Match list generation (`automate_gol.py`, `automate_ytb.py`)
- Gameplay event scraping (`scrape_gol.py`)
- Caption collection and Whisper transcription (`transcribe_whisper.py`)
- Event-commentary alignment (`align_data_kwmatch.py`)
- Linearization into window-based training samples (`linearize_events.py`)
- Merging and uploading to HuggingFace (`merge_dataset.py`, `to_hf.py`)

### Baseline (T5-base)
For training and evaluating the T5-base seq2seq baseline — including dataset splitting, training, evaluation metrics, and faithfulness scoring — see [`baseline/Baseline.md`](baseline/Baseline.md).

This covers:
- Dataset splitting by match ID (`split.py`)
- Training T5-base on LoL19 / LoL1921 (`train_t5.py`)
- Evaluation: BLEU, ROUGE-L, BERTScore, BARTScore, Perplexity (`evaluate_t5.py`)
- Event Faithfulness Score (EFS) (`run_faithfulness.py`)

### Qwen2.5 (LoRA Fine-Tuning)
For fine-tuning Qwen2.5 with QLoRA on a single GPU, running batched inference, and computing evaluation metrics — see [`qwen2.5/Qwen.md`](qwen2.5/Qwen.md).

This covers:
- Dataset preparation and token length analysis
- QLoRA fine-tuning configuration (`finetune_qwen.py`)
- Batched test set inference (`batch_inference.py`)
- Evaluation metrics and EFS (`run_metrics.py`, `run_faithfulness.py`)


## Citation

Based on the baseline work by [Wang & Yoshinaga (2024)](https://aclanthology.org/2024.naacl-srw.28/)

---

**MLP 2025/26 Coursework - Group 055**