from pathlib import Path
import pandas as pd
import argparse
import re

def get_tournament_dir(folder_name):
  """
  Extract year from folder name and build path
  "WC 2020" -> data/2020/WC 2020/
  "MSI 2021" -> data/2021/MSI 2021/
  """
  year_match = re.search(r'(\d{4})', folder_name)
  if year_match:
    year = year_match.group(1)
    return Path("data") / year / folder_name
  else:
    return Path("data") / folder_name

def merge_offsets(folder):
  tournament_dir = get_tournament_dir(folder)
  INPUT_CSV = tournament_dir / "match_list_with_youtube.csv"
  TIMESTAMPS = tournament_dir / "timestamps.csv"
  OUTPUT_CSV = tournament_dir / "match_list.csv"
  
  df_original = pd.read_csv(INPUT_CSV)
  df_offsets = pd.read_csv(TIMESTAMPS)
  
  df_offsets = df_offsets[['youtube_url', 'difference_seconds']].dropna()
  df_offsets['difference_seconds'] = df_offsets['difference_seconds'].astype(int)
  
  offset_map = dict(zip(df_offsets['youtube_url'], df_offsets['difference_seconds']))
  
  df_original['offset_seconds'] = df_original['youtube_url'].map(offset_map).fillna(df_original['offset_seconds']).astype(int)
  
  df_original.to_csv(OUTPUT_CSV, index=False)
  print(f"✓ Final CSV saved to: {OUTPUT_CSV}")
  
  print("\n" + "="*60)
  print("STATISTICS")  
  total = len(df_original)
  extracted = len(offset_map)
  failed = total - extracted
  print(f"Total videos:       {total}")
  print(f"Offsets extracted:  {extracted} ({extracted/total*100:.1f}%)")
  print(f"Failed:             {failed} ({failed/total*100:.1f}%)")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('folder', type=str, help='Tournament folder')
  args = parser.parse_args()
  merge_offsets(args.folder)