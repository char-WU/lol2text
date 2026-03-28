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

from scripts.modules.bart_score import BARTScorer

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
  
def generate(
    model,
    dataloader,
    tokenizer,
    device,
    num_beams=5,
    max_length=512,
    num_samples=None
):
  """
  Returns:
    List[Dict]: [{'input': ..., 'target': ..., 'prediction': ..., 'metadata': ...}]
  """
  model.eval()
  predictions = []
  
  with torch.no_grad():
    for i, batch in enumerate(tqdm(dataloader, desc="Generating")):
      if num_samples and i * dataloader.batch_size >= num_samples:
        break
      
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      
      # Generate
      outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_beams=num_beams,
        do_sample=False,
        max_new_tokens=max_length,
        min_length=10,
        early_stopping=True,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2
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
  
  # sacreBLEU
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
  """
  - Levenshtein distance (edit distance)
  - Normalized edit distance (0-1)
  - Character-level similarity
  """
  edit_distances = []
  normalized_distances = []
  
  for pred in predictions:
    target = pred['target']
    prediction = pred['prediction']
    
    # Levenshtein distance
    edit_dist = levenshtein_distance(target, prediction)
    edit_distances.append(edit_dist)
    
    # Normalized (0 = identical, 1 = completely different)
    max_len = max(len(target), len(prediction))
    if max_len > 0:
      norm_dist = edit_dist / max_len
    else:
      norm_dist = 0.0
    normalized_distances.append(norm_dist)
  text_similarity_pct = (1.0 - np.mean(normalized_distances)) * 100
  return {
    'edit_distance_mean': np.mean(edit_distances),
    'edit_distance_std': np.std(edit_distances),
    'normalized_edit_distance': np.mean(normalized_distances),  # lower is better
    'text_similarity': 1.0 - np.mean(normalized_distances),  # higher is better
    'text_similarity_pct': text_similarity_pct
  }

def bertscore(predictions: List[Dict], device='cuda') -> Dict:
  references = [pred['target'] for pred in predictions]
  candidates = [pred['prediction'] for pred in predictions]
  
  P, R, F1 = bert_score(
    candidates,
    references,
    lang='en',
    model_type='roberta-large',  #
    device=device,
    verbose=False
  )
  
  return {
    'bertscore_precision': P.mean().item(),
    'bertscore_recall': R.mean().item(),
    'bertscore_f1': F1.mean().item()
  }

def bartscore(predictions: List[Dict], device='cuda') -> Dict:
  """
  measures generation quality using BART model's log-likelihood,
  Higher scores indicate better quality
  
  Modes:
  - SRC: score(prediction | input)  - faithfulness to source
  - REF: score(prediction | target) - similarity to reference
  - AVG: average of both
  """
  try:
    # Use 'facebook/bart-large-cnn' for better performance
    bart_scorer = BARTScorer(device=device, checkpoint='facebook/bart-large-cnn')
    
    inputs = [pred['input'] for pred in predictions]
    references = [pred['target'] for pred in predictions]
    candidates = [pred['prediction'] for pred in predictions]
    
    # REF mode: how well prediction matches reference
    ref_scores = bart_scorer.score(candidates, references, batch_size=8)
    
    # SRC mode: how well prediction follows input
    src_scores = bart_scorer.score(candidates, inputs, batch_size=8)
    
    # Average mode
    avg_scores = [(ref + src) / 2 for ref, src in zip(ref_scores, src_scores)]
    
    return {
      'bartscore_ref': np.mean(ref_scores),   # higher is better
      'bartscore_ref_std': np.std(ref_scores),
      'bartscore_src': np.mean(src_scores),   # higher is better
      'bartscore_src_std': np.std(src_scores),
      'bartscore_avg': np.mean(avg_scores),   # higher is better
    }
  
  except Exception as e:
    logger.warning(f"BARTScore computation failed: {e}")
    logger.warning("Install with: pip install bart-score")
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
  perplexity = np.exp(avg_loss)
  
  return perplexity

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

def save_results(
    output_dir: Path,
    metrics: Dict,
    predictions: List[Dict]
):
  output_dir.mkdir(parents=True, exist_ok=True)

  def numpy_finalizer(obj):
    if isinstance(obj, (np.int64, np.int32, np.integer)):
      return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.floating)):
      return float(obj)
    elif isinstance(obj, np.ndarray):
      return obj.tolist()
    return obj
  
  # metrics
  metrics_file = output_dir / 'metrics.json'
  with open(metrics_file, 'w', encoding='utf-8') as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False, default=numpy_finalizer)
  logger.info(f"✓ Saved metrics to {metrics_file}")
  
  # predictions
  predictions_file = output_dir / 'predictions.jsonl'
  with open(predictions_file, 'w', encoding='utf-8') as f:
    for pred in predictions:
      f.write(json.dumps(pred, ensure_ascii=False, default=numpy_finalizer) + '\n')
  logger.info(f"✓ Saved {len(predictions)} predictions to {predictions_file}")
  
  # summary
  report = output_dir / 'report.txt'
  with open(report, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("EVALUATION REPORT\n")
    
    # Metrics
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

def print_summary(metrics: Dict,):
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
  print(f"  Perplexity: {metrics.get('perplexity', 0):.2f}")
  
  if 'length_stats' in metrics:
    print("\nLength Statistics:")
    print(f"  Target: {metrics['length_stats']['target_length']['mean']:.1f} words")
    print(f"  Prediction: {metrics['length_stats']['prediction_length']['mean']:.1f} words")

def generate_output_dir(model: str, name: Optional[str] = None) -> Path:
  """
  Examples:
  - name="LoL19" → results/LoL19/
  - models/baseline_t5_lol19/best_model → results/baseline_t5_lol19/
  - models/t5_large/checkpoint-epoch5-step1000 → results/t5_large_epoch5_step1000/
  """
  output_dir = Path('results') / 'evaluation' / name
  model = Path(model)
  parts = []
  if model.parent.name != 'models':
      parts.append(model.parent.name)
  
  if 'checkpoint' in model.name:
      # checkpoint-epoch5-step1000 → epoch5_step1000
      checkpoint_info = model.name.replace('checkpoint-', '').replace('-', '_')
      parts.append(checkpoint_info)
  elif model.name != 'best_model':
      parts.append(model.name)
  if parts:
      dir_name = '_'.join(parts)
  else:
      dir_name = 'evaluation'
  output_dir = output_dir / dir_name
  
  if output_dir.exists():
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if name:
      output_dir = Path('results') / 'evaluation' / f"{name}_{timestamp}"
    else:
      dir_name = output_dir.name
      output_dir = Path('results') / 'evaluation' / f"{dir_name}_{timestamp}"
  
  return output_dir

def main():
  parser = argparse.ArgumentParser(description="Evaluate trained model")
  parser.add_argument('--model', type=str, required=True,
                      help='Path to trained model checkpoint')
  
  # Data
  parser.add_argument('--name', type=str, default=None,
                      help='HuggingFace dataset name')
  
  # Generation params
  parser.add_argument('--batch-size', type=int, default=16,
                      help='Batch size for evaluation')
  parser.add_argument('--num-beams', type=int, default=5,
                      help='Number of beams for beam search')
  parser.add_argument('--max-length', type=int, default=256,
                      help='Maximum generation length')
  
  # Options
  parser.add_argument('--device', type=str, default=None,
                      help='Device (cuda/cpu)')
  
  args = parser.parse_args()
  
  # Validate inputs
  if not args.name:
    raise ValueError("Must provide either --name")
  
  # Device
  if args.device:
    device = torch.device(args.device)
  else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  logger.info("=" * 80)
  logger.info("EVALUATION CONFIGURATION")
  logger.info(f"Model: {args.model}")
  logger.info(f"Device: {device}")
  logger.info(f"Batch size: {args.batch_size}")
  logger.info(f"Num beams: {args.num_beams}")
  logger.info(f"Dataset: {args.name}")
  output_dir = generate_output_dir(args.model, args.name)
  logger.info(f"Output directory: {output_dir}")
  
  # Load model and tokenizer
  logger.info("\nLoading model and tokenizer...")
  tokenizer = AutoTokenizer.from_pretrained(args.model)
  model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
  model.to(device)
  model.eval()
  
  logger.info(f"✓ Loaded model from {args.model}")
  
  # Load dataset
  logger.info("\nLoading test dataset...")
  
  if args.name:
    from datasets import load_dataset
    test_hf = load_dataset(args.name, split='test')
    test_dataset = Evaluation(test_hf, tokenizer, is_hf_dataset=True)
  
  test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4
  )
  
  logger.info(f"✓ Loaded {len(test_dataset)} test examples")
  
  # Generate predictions
  logger.info("\n" + "=" * 80)
  logger.info("GENERATING PREDICTIONS")
  logger.info("=" * 80)
  
  predictions = generate(
    model,
    test_loader,
    tokenizer,
    device,
    num_beams=args.num_beams,
    max_length=args.max_length
  )
  
  logger.info(f"✓ Generated {len(predictions)} predictions")
  
  logger.info("\n" + "=" * 80)
  logger.info("COMPUTING METRICS")
  metrics = {}
  
  # BLEU
  logger.info("Computing BLEU...")
  bleu_scores = bleu(predictions)
  metrics.update(bleu_scores)
  
  # ROUGE
  logger.info("Computing ROUGE...")
  rouge_scores = rouge(predictions)
  metrics.update(rouge_scores)
  
  # BERTScore
  logger.info("Computing BERTScore (this may take a while)...")
  try:
    bertscore_metrics = bertscore(predictions, device=device)
    metrics.update(bertscore_metrics)
  except Exception as e:
    logger.warning(f"BERTScore computation failed: {e}")
  
  # Text Distance
  logger.info("Computing text distance metrics...")
  try:
    text_distance_metrics = text_distance(predictions)
    metrics.update(text_distance_metrics)
  except Exception as e:
    logger.warning(f"Text distance computation failed: {e}")
    
  # BARTScore
  logger.info("Computing BARTScore (this may take a while)...")
  try:
    bartscore_metrics = bartscore(predictions, device=device)
    metrics.update(bartscore_metrics)
  except Exception as e:
    logger.warning(f"BARTScore computation failed: {e}")
  
  # Perplexity
  logger.info("Computing perplexity...")
  ppl = perplexity(model, test_loader, tokenizer, device)
  metrics['perplexity'] = ppl
  
  # Length statistics
  logger.info("Computing length statistics...")
  stats = pred_length(predictions)
  metrics['length_stats'] = stats
  
  # Save results
  logger.info("\n" + "=" * 80)
  logger.info("SAVING RESULTS")
  
  save_results(output_dir, metrics, predictions)
  print_summary(metrics)
  
  logger.info(f"\n✓ Evaluation complete! Results saved to {output_dir}")


if __name__ == '__main__':
  main()