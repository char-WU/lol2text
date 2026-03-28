from datasets import Dataset, DatasetDict
import json
import argparse

def load_jsonl(filepath):
  data = []
  with open(filepath, 'r') as f:
    for line in f:
      data.append(json.loads(line.strip()))
  return data

def main():
  parser = argparse.ArgumentParser(description='Push dataset to HuggingFace Hub')
  parser.add_argument(
    'dataset',
    type=str,
    help='Dataset to push: LoL19 or LoL1921'
  )
  parser.add_argument(
    '--username',
    type=str,
    default='your-username',
    help='HuggingFace username'
  )
  args = parser.parse_args()
  dataset_name = args.dataset
  base_path = f'split/{dataset_name}'
  
  train_file = f'{base_path}/{dataset_name}_train.jsonl'
  val_file = f'{base_path}/{dataset_name}_val.jsonl'
  test_file = f'{base_path}/{dataset_name}_test.jsonl'
  print(f"Loading {dataset_name} dataset...")
  print(f"  Train: {train_file}")
  print(f"  Val:   {val_file}")
  print(f"  Test:  {test_file}")
  
  train_data = load_jsonl(train_file)
  val_data = load_jsonl(val_file)
  test_data = load_jsonl(test_file)
  print(f"\nDataset sizes:")
  print(f"  Train: {len(train_data)} examples")
  print(f"  Val:   {len(val_data)} examples")
  print(f"  Test:  {len(test_data)} examples")
  
  train_dataset = Dataset.from_list(train_data)
  val_dataset = Dataset.from_list(val_data)
  test_dataset = Dataset.from_list(test_data)
  dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
    'test': test_dataset
  })
  
  # Push to hub
  hub_name = f"{args.username}/{dataset_name}"
  print(f"\nPushing to HuggingFace Hub: {hub_name}")
  dataset_dict.push_to_hub(hub_name)
  print(f"✓ Successfully pushed to {hub_name}")


if __name__ == '__main__':
  main()