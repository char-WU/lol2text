"""
Evaluate trained model on test set.

Computes:
- BLEU (1-4)
- Text distance
- ROUGE-L
- BERTScore
- BARTScore
- Perplexity
"""
import json
import logging
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict
import argparse

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import numpy as np

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import sacrebleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from Levenshtein import distance as levenshtein_distance

# fix import path so bart_score.py is always found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from bart_score import BARTScorer

import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class Evaluation:
    def __init__(self, data_source, tokenizer, max_input_length=512, is_hf_dataset=False):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length

        if is_hf_dataset:
            self.examples = data_source
        else:
            self.examples = self._load_jsonl(data_source)

        logger.info(f"Loaded {len(self.examples)} test examples")

    def _load_jsonl(self, filepath):
        examples = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    example = json.loads(line)
                    if 'input' in example and 'target' in example:
                        examples.append(example)
                except json.JSONDecodeError:
                    continue
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        input_text = example['input']
        target_text = example['target']
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'target_text': target_text,
            'input_text': input_text,
            'metadata': {k: v for k, v in example.items() if k not in ['input', 'target']}
        }


def generate(model, dataloader, tokenizer, device, num_beams=5, max_length=256, num_samples=None):
    model.eval()
    predictions = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Generating")):
            if num_samples and i * dataloader.batch_size >= num_samples:
                break
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=10,
                early_stopping=True
            )
            generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for j in range(len(generated_texts)):
                predictions.append({
                    'input': batch['input_text'][j],
                    'target': batch['target_text'][j],
                    'prediction': generated_texts[j],
                    'metadata': {k: v[j] for k, v in batch['metadata'].items()} if ('metadata' in batch and batch['metadata']) else {}
                })
    return predictions


def bleu(predictions: List[Dict]) -> Dict:
    references = []
    hypotheses = []
    for pred in predictions:
        ref_tokens = pred['target'].split()
        hyp_tokens = pred['prediction'].split()
        references.append([ref_tokens])
        hypotheses.append(hyp_tokens)
    smoothing = SmoothingFunction().method1
    bleu_scores = {}
    for n in range(1, 5):
        weights = tuple([1.0/n] * n + [0.0] * (4-n))
        bleu_scores[f'bleu_{n}'] = corpus_bleu(
            references,
            hypotheses,
            weights=weights,
            smoothing_function=smoothing
        )
    refs_text = [[pred['target'] for pred in predictions]]
    hyps_text = [pred['prediction'] for pred in predictions]
    sacre_bleu = sacrebleu.corpus_bleu(hyps_text, refs_text)
    return {
        **bleu_scores,
        'sacrebleu': sacre_bleu.score,
        'sacrebleu_bp': sacre_bleu.bp,
    }


def rouge(predictions: List[Dict]) -> Dict:
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = []
    for pred in predictions:
        score = scorer.score(pred['target'], pred['prediction'])
        rouge_scores.append(score['rougeL'].fmeasure)
    return {
        'rouge_l': np.mean(rouge_scores),
        'rouge_l_std': np.std(rouge_scores)
    }


def text_distance(predictions: List[Dict]) -> Dict:
    edit_distances = []
    normalized_distances = []
    for pred in predictions:
        target = pred['target']
        prediction = pred['prediction']
        edit_dist = levenshtein_distance(target, prediction)
        edit_distances.append(edit_dist)
        max_len = max(len(target), len(prediction))
        norm_dist = edit_dist / max_len if max_len > 0 else 0.0
        normalized_distances.append(norm_dist)
    text_similarity_pct = (1.0 - np.mean(normalized_distances)) * 100
    return {
        'edit_distance_mean': np.mean(edit_distances),
        'edit_distance_std': np.std(edit_distances),
        'normalized_edit_distance': np.mean(normalized_distances),
        'text_similarity': 1.0 - np.mean(normalized_distances),
        'text_similarity_pct': text_similarity_pct
    }


def bertscore(predictions: List[Dict], device='cuda') -> Dict:
    references = [pred['target'] for pred in predictions]
    candidates = [pred['prediction'] for pred in predictions]
    P, R, F1 = bert_score(
        candidates,
        references,
        lang='en',
        model_type='roberta-large',
        device=device,
        verbose=False,
        batch_size=64,
    )
    return {
        'bertscore_precision': P.mean().item(),
        'bertscore_recall': R.mean().item(),
        'bertscore_f1': F1.mean().item()
    }


def bartscore(predictions: List[Dict], device='cuda') -> Dict:
    try:
        bart_scorer = BARTScorer(device=device, checkpoint='facebook/bart-large-cnn')
        inputs = [pred['input'] for pred in predictions]
        references = [pred['target'] for pred in predictions]
        candidates = [pred['prediction'] for pred in predictions]
        ref_scores = bart_scorer.score(candidates, references, batch_size=64)
        src_scores = bart_scorer.score(candidates, inputs, batch_size=64)
        avg_scores = [(ref + src) / 2 for ref, src in zip(ref_scores, src_scores)]
        return {
            'bartscore_ref': np.mean(ref_scores),
            'bartscore_ref_std': np.std(ref_scores),
            'bartscore_src': np.mean(src_scores),
            'bartscore_src_std': np.std(src_scores),
            'bartscore_avg': np.mean(avg_scores),
        }
    except Exception as e:
        logger.warning(f"BARTScore computation failed: {e}")
        return {
            'bartscore_ref': None,
            'bartscore_src': None,
            'bartscore_avg': None
        }


def perplexity(model, dataloader, tokenizer, device) -> float:
    model.eval()
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing perplexity"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_encoding = tokenizer(
                batch['target_text'],
                max_length=256,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            labels = target_encoding['input_ids'].to(device)
            labels[labels == tokenizer.pad_token_id] = -100
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            total_loss += outputs.loss.item()
            num_batches += 1
    avg_loss = total_loss / num_batches
    return np.exp(avg_loss)


def pred_length(predictions: List[Dict]) -> Dict:
    target_lengths = [len(pred['target'].split()) for pred in predictions]
    pred_lengths = [len(pred['prediction'].split()) for pred in predictions]
    return {
        'target_length': {
            'mean': float(np.mean(target_lengths)),
            'std': float(np.std(target_lengths)),
            'min': int(np.min(target_lengths)),
            'max': int(np.max(target_lengths))
        },
        'prediction_length': {
            'mean': float(np.mean(pred_lengths)),
            'std': float(np.std(pred_lengths)),
            'min': int(np.min(pred_lengths)),
            'max': int(np.max(pred_lengths))
        }
    }


def save_results(output_dir: Path, metrics: Dict, predictions: List[Dict]):
    output_dir.mkdir(parents=True, exist_ok=True)

    def numpy_finalizer(obj):
        if isinstance(obj, (np.int64, np.int32, np.integer)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    metrics_file = output_dir / 'metrics.json'
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=numpy_finalizer)
    logger.info(f"✓ Saved metrics to {metrics_file}")

    predictions_file = output_dir / 'predictions.jsonl'
    with open(predictions_file, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False, default=numpy_finalizer) + '\n')
    logger.info(f"✓ Saved {len(predictions)} predictions to {predictions_file}")

    report = output_dir / 'report.txt'
    with open(report, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("EVALUATION REPORT\n")
        f.write("METRICS:\n")
        f.write("-" * 80 + "\n")
        for key, value in metrics.items():
            if key == 'length_stats' and isinstance(value, dict):
                f.write(f"\n{key}:\n")
                for sub_key, sub_val in value.items():
                    f.write(f"  {sub_key}:\n")
                    for stat_name, stat_val in sub_val.items():
                        f.write(f"    {stat_name}: {stat_val:.4f}\n")
            elif isinstance(value, dict):
                f.write(f"\n{key}:\n")
                for k, v in value.items():
                    v = v if v is not None else 0.0
                    f.write(f"  {k}: {v:.4f}\n")
            else:
                f.write(f"{key}: {value:.4f}\n")
    logger.info(f"✓ Saved readable report to {report}")


def print_summary(metrics: Dict):
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("\nCore Metrics:")
    print(f"  BLEU-1: {metrics.get('bleu_1', 0):.4f}")
    print(f"  BLEU-2: {metrics.get('bleu_2', 0):.4f}")
    print(f"  BLEU-3: {metrics.get('bleu_3', 0):.4f}")
    print(f"  BLEU-4: {metrics.get('bleu_4', 0):.4f}")
    print(f"  Text distance: {metrics.get('normalized_edit_distance', 0):.4f}")
    print(f"  ROUGE-L: {metrics.get('rouge_l', 0):.4f}")
    print(f"  BERTScore: {metrics.get('bertscore_f1', 0):.4f}")
    print(f"  BARTScore (ref): {metrics.get('bartscore_ref', 0):.4f}")
    if 'length_stats' in metrics:
        print("\nLength Statistics:")
        print(f"  Target: {metrics['length_stats']['target_length']['mean']:.1f} words")
        print(f"  Prediction: {metrics['length_stats']['prediction_length']['mean']:.1f} words")