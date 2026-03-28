import json
import random
from pathlib import Path
from collections import defaultdict
import argparse

from datasets import load_dataset
def load_from_hf_by_match(hf_repo: str, split_name: str = "train"):
  match_examples = defaultdict(list)
  total_examples = 0
  
  print(f"Loading dataset from Hugging Face: {hf_repo} (split: {split_name})")
  try:
    # Load the dataset directly from Hugging Face
    dataset = load_dataset(hf_repo, split=split_name)
  except Exception as e:
    raise RuntimeError(f"Failed to load dataset from Hugging Face: {e}")
  
  for line_num, example in enumerate(dataset, 1):
    if 'input' not in example or 'target' not in example:
      print(f"  ⚠️  Skipping example missing input/target at index {line_num}")
      continue
    match_id = example.get('match_id', f'unknown_{line_num}')
    match_examples[match_id].append(example)
    total_examples += 1
  print(f"✓ Loaded {total_examples} examples from {len(match_examples)} matches")
  return match_examples

def load_exp_by_match(path: Path):
  """
  Load examples orgainized by match_id
  Returns:
    dict: {match_id: [example1, example2, ...]}
  """
  match_examples = defaultdict(list)
  total_examples = 0
  
  print(f"Loading examples from: {path}")
  with open(path, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
      line = line.strip()
      if not line:
        continue
      
      try:
        example = json.loads(line)
      except json.JSONDecodeError as e:
        print(f"  ⚠️  Skipping malformed JSON at line {line_num}: {e}")
        continue
      
      if 'input' not in example or 'target' not in example:
        print(f"  ⚠️  Skipping example missing input/target at line {line_num}")
        continue
      
      match_id = example.get('match_id', f'unknown_{line_num}')
      match_examples[match_id].append(example)
      total_examples += 1
  
  print(f"✓ Loaded {total_examples} examples from {len(match_examples)} matches")
  return match_examples

def split_by_match(match_examples, ratios=(0.8, 0.1, 0.1), seed=42):
  """    
  Args:
    match_examples: {match_id: [examples]}
    ratios: (train_ratio, val_ratio, test_ratio)
    seed: 42
  
  Returns:
    train_examples, val_examples, test_examples
  """
  random.seed(seed)
  
  match_ids = list(match_examples.keys())
  random.shuffle(match_ids)
  
  total_matches = len(match_ids)
  train_ratio, val_ratio, test_ratio = ratios
  
  ratio_sum = sum(ratios)
  if abs(ratio_sum - 1.0) > 0.001:
      train_ratio /= ratio_sum
      val_ratio /= ratio_sum
      test_ratio /= ratio_sum
  
  train_split = int(total_matches * train_ratio)
  val_split = int(total_matches * (train_ratio + val_ratio))
  
  train_match_ids = match_ids[:train_split]
  val_match_ids = match_ids[train_split:val_split]
  test_match_ids = match_ids[val_split:]
  
  train_examples = []
  val_examples = []
  test_examples = []
  
  for match_id in train_match_ids:
    train_examples.extend(match_examples[match_id])
  
  for match_id in val_match_ids:
    val_examples.extend(match_examples[match_id])
  
  for match_id in test_match_ids:
    test_examples.extend(match_examples[match_id])
  
  print("\n" + "=" * 60)
  print("SPLIT SUMMARY")
  print(f"Train: {len(train_match_ids)} matches, {len(train_examples)} examples ({len(train_examples)/len(train_examples+val_examples+test_examples)*100:.1f}%)")
  print(f"Val:   {len(val_match_ids)} matches, {len(val_examples)} examples ({len(val_examples)/len(train_examples+val_examples+test_examples)*100:.1f}%)")
  print(f"Test:  {len(test_match_ids)} matches, {len(test_examples)} examples ({len(test_examples)/len(train_examples+val_examples+test_examples)*100:.1f}%)")
  print(f"Total: {total_matches} matches, {len(train_examples) + len(val_examples) + len(test_examples)} examples")
  
  return train_examples, val_examples, test_examples
 
def save_split(examples, output_path: Path):
  output_path.parent.mkdir(parents=True, exist_ok=True)

  with open(output_path, 'w', encoding='utf-8') as f:
    for example in examples:
      f.write(json.dumps(example, ensure_ascii=False) + '\n')
  
  print(f"✓ Saved {len(examples)} examples to {output_path}")
 
def statistics(examples):
  if not examples:
    return {}
    
  input_lengths = [len(ex['input'].split()) for ex in examples]
  target_lengths = [len(ex['target'].split()) for ex in examples]
  
  return {
    'num_examples': len(examples),
    'input_tokens': {
      'mean': sum(input_lengths) / len(input_lengths),
      'min': min(input_lengths),
      'max': max(input_lengths),
    },
    'target_words': {
      'mean': sum(target_lengths) / len(target_lengths),
      'min': min(target_lengths),
      'max': max(target_lengths),
    }
  }
 
def main():
  parser = argparse.ArgumentParser(
    description="Split dataset by match_id to prevent data leakage",
    epilog="""
      Examples:
      python scripts/split.py LoL19
      python scripts/split.py LoL1921 --ratios 0.8 0.1 0.1
      python scripts/split.py LoL19 --output-dir my_splits
      """
    )
  
  parser.add_argument(
    'name',
    type=str,
    help='Dataset name (e.g., "username/LoL19")'
  )
    
  parser.add_argument(
    '--ratios',
    nargs=3,
    type=float,
    default=[0.8, 0.1, 0.1],
    help='Train/val/test ratios (default: 0.8 0.1 0.1)'
  )
  
  parser.add_argument(
    '--seed',
    type=int,
    default=42,
    help='Random seed for reproducibility'
  )
  
  args = parser.parse_args()

  dataset_name = args.name.split('/')[-1]
  output_dir = Path("split/") / dataset_name
  output_dir.mkdir(parents=True, exist_ok=True)
  
  prefix = f"{dataset_name}_"
  
  match_examples = load_from_hf_by_match(args.name)
    
  train_examples, val_examples, test_examples = split_by_match(
    match_examples,
    ratios=tuple(args.ratios),
    seed=args.seed
  )
  
  save_split(train_examples, output_dir / f"{prefix}train.jsonl")
  save_split(val_examples, output_dir / f"{prefix}val.jsonl")
  save_split(test_examples, output_dir / f"{prefix}test.jsonl")
  
  print("\n" + "=" * 60)
  print("DATASET STATISTICS")
  
  for split_name, examples in [("Train", train_examples), ("Val", val_examples), ("Test", test_examples)]:
    stats = statistics(examples)
    print(f"\n{split_name}:")
    print(f"  Examples: {stats['num_examples']}")
    print(f"  Input tokens:  {stats['input_tokens']['mean']:.1f} (range: {stats['input_tokens']['min']}-{stats['input_tokens']['max']})")
    print(f"  Target words:  {stats['target_words']['mean']:.1f} (range: {stats['target_words']['min']}-{stats['target_words']['max']})")
  
  stats_output = output_dir / f"{prefix}split_stats.json"
  with open(stats_output, 'w', encoding='utf-8') as f:
    json.dump({
      'input_file': args.name,
      'ratios': args.ratios,
      'seed': args.seed,
      'train': statistics(train_examples),
      'val': statistics(val_examples),
      'test': statistics(test_examples),
    }, f, indent=2)
  
  print(f"\n✓ Statistics saved to {stats_output}")
  print(f"\n✓ All splits saved to {output_dir}/")
 
 
if __name__ == '__main__':
  main()