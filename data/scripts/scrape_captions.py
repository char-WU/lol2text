import yt_dlp
import pandas as pd
import json
import time
from pathlib import Path
import argparse

def download_subtitle(youtube_url, output_path, match_id):
  """
  Download English subtitles from YouTube.

  Args:
    youtube_url: The YouTube video URL
    output_path: Path to save the subtitles (JSON format)
    match_id: Match ID (used for naming temporary files)

  Returns:
    captions: A list of dicts with keys 'start', 'duration', and 'text'.
  """
  #set yt_dlp
  ydl_opts = {
    'writesubtitles': True,
    'writeautomaticsub': True,
    'subtitleslangs': ['en'],
    'skip_download': True,
    'subtitlesformat': 'json3',
    'outtmpl': f'temp_sub_{match_id}',
    'quiet': True,
    'no_warnings': True,
  }

  try:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
      ydl.download([youtube_url])

      # yt_dlp save as: temp_sub_{match_id}.en.json3
      subtitle_file = f"temp_sub_{match_id}.en.json3"

      if not Path(subtitle_file).exists():
        print(f"  Subtitle file not found: {subtitle_file}")
        return None
      
      with open(subtitle_file, 'r', encoding='utf-8') as f:
        subtitle_data = json.load(f)

      captions = []

      if 'events' in subtitle_data:
        for event in subtitle_data['events']:
          if 'segs' not in event: continue

          start = event.get('tStartMs', 0) / 1000
          duration = event.get('dDurationMs', 0) / 1000

          #concatenate
          text = ''.join(
            [seg.get('utf8', '') for seg in event['segs']]
          ).strip()

          if text:
            captions.append(
              {
                'start': start,
                'duration': duration,
                'text': text
              }
            )
      
      #delelte temp file
      Path(subtitle_file).unlink(missing_ok=True)

      with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(captions, f, indent=2, ensure_ascii=False)
      
      return captions
  except Exception as e:
    print(f"  Error: {e}")
    Path(f"temp_sub_{match_id}.en.json3").unlink(missing_ok=True)
    return None

def main():
  parser = argparse.ArgumentParser(description='Scrape YouTube captions for matches')
  parser.add_argument(
    'tournament',
    type=str,
    nargs='?',
    help='Tournament folder name (default: "WC 2019")'
  )
  args = parser.parse_args()
  
  tournament_dir = Path('data') / args.tournament
  input_csv = tournament_dir / 'match_list_with_subs.csv'
  output_dir = tournament_dir / 'raw' / 'captions' /'v1'

  if not input_csv.exists():
    print(f"❌ Input file not found: {input_csv}")
    return
  
  output_dir.mkdir(parents=True, exist_ok=True)
  matches = pd.read_csv(input_csv)
  total = len(matches)
  successful = 0

  for idx, match in matches.iterrows():
    print(f"\n[{idx+1}/{total}] Downloading captions: {match['team1']} vs {match['team2']}")
    print(f"  URL: {match['youtube_url']}")

    output_path = output_dir / f"match_{match['match_id']}.json"

    captions = download_subtitle(
      match['youtube_url'], 
      output_path, 
      match['match_id']
    )

    if captions:
      print(f"  ✓ Saved {len(captions)} caption segments")
      successful += 1
    else:
      print(f"  !! Failed to download")
    
    time.sleep(2)
  
  print(f"\n{'='*50}")
  print(f"✓ Completed: {successful}/{total} captions downloaded")

if __name__ == "__main__":
  main()
