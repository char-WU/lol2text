import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse
import json
from pathlib import Path

def extract_tensorboard_metrics(event_file, output_dir=None, output_format='csv'):
  """
  Extract all metrics from TensorBoard event file and save to standalone files
  """
  ea = EventAccumulator(event_file)
  ea.Reload()
  available_metrics = ea.Tags()['scalars']
  print(f"Found {len(available_metrics)} metrics:")
  for metric in available_metrics:
    print(f"  - {metric}")
  if output_dir is None:
    output_dir = Path(event_file).parent / 'extracted_metrics'
  else:
    output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  all_metrics = {}
  for metric_name in available_metrics:
    print(f"\nExtracting: {metric_name}")
    metric_data = ea.Scalars(metric_name)
    df = pd.DataFrame(metric_data)
    df = df[['step', 'value']]
    print(f"  Steps: {len(df)}, Range: {df['value'].min():.4f} - {df['value'].max():.4f}")
    all_metrics[metric_name] = df
    safe_name = metric_name.replace('/', '_')
    if output_format in ['csv', 'both']:
      csv_file = output_dir / f"{safe_name}.csv"
      df.to_csv(csv_file, index=False)
      print(f"  ✓ Saved to {csv_file}")
    if output_format in ['json', 'both']:
      json_file = output_dir / f"{safe_name}.json"
      df.to_json(json_file, orient='records', indent=2)
      print(f"  ✓ Saved to {json_file}")
  
  summary = {
    'event_file': str(event_file),
    'metrics': {}
  }
  for metric_name, df in all_metrics.items():
    summary['metrics'][metric_name] = {
      'num_steps': len(df),
      'final_value': float(df['value'].iloc[-1]),
      'min_value': float(df['value'].min()),
      'max_value': float(df['value'].max()),
      'mean_value': float(df['value'].mean())
    }
  
  summary_file = output_dir / 'summary.json'
  with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)
  print(f"\n✓ Summary saved to {summary_file}")
  
  # Create combined CSV with all metrics
  if len(all_metrics) > 0:
    base_metric = max(all_metrics.items(), key=lambda x: len(x[1]))[1]
    combined_df = pd.DataFrame({'step': base_metric['step']})
    for metric_name, df in all_metrics.items():
      safe_name = metric_name.replace('/', '_')
      combined_df = combined_df.merge(
        df.rename(columns={'value': safe_name}),
        on='step',
        how='left'
      )
    combined_file = output_dir / 'all_metrics.csv'
    combined_df.to_csv(combined_file, index=False)
    print(f"✓ Combined metrics saved to {combined_file}")
  
  print(f"\n{'='*60}")
  print(f"All files saved to: {output_dir}")
  print(f"{'='*60}")
  
  return all_metrics


def main():
  parser = argparse.ArgumentParser(description='Extract metrics from TensorBoard event file')
  parser.add_argument(
    'event_file',
    type=str,
    help='Path to TensorBoard event file'
  )
  parser.add_argument(
    '--output-dir',
    type=str,
    default=None,
    help='Output directory (default: <event_file_dir>/extracted_metrics)'
  )
  parser.add_argument(
    '--format',
    type=str,
    choices=['csv', 'json', 'both'],
    default='csv',
    help='Output format (default: csv)'
  )
  args = parser.parse_args()
  
  extract_tensorboard_metrics(
    args.event_file,
    output_dir=args.output_dir,
    output_format=args.format
  )


if __name__ == '__main__':
  main()