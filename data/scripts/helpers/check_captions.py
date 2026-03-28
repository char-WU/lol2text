import yt_dlp
import pandas as pd
import time
import argparse
from pathlib import Path

def check_subtitle_availability(youtube_url):
  ydl_opts = {
    'skip_download': True,
    'quiet': True,
    'no_warnings': True,
  }

  try:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
      info = ydl.extract_info(youtube_url, download=False)
      
      has_manual = 'subtitles' in info and 'en' in info.get('subtitles', {})
      has_auto = 'automatic_captions' in info and 'en' in info.get('automatic_captions', {})
      
      if has_manual:
        return True, "Manual subtitles"
      elif has_auto:
        return True, "Auto-generated"
      else:
        return False, "No English subtitles"
              
  except Exception as e:
      return False, f"Error: {str(e)[:50]}"
  
def main():
    parser = argparse.ArgumentParser(description='Check subtitle availability for YouTube videos')
    parser.add_argument(
      'tournament',
      type=str,
      nargs='?',
      help='Tournament folder name (default: "WC 2019")'
    )
    args = parser.parse_args()

    tournament_dir = Path("data") / args.tournament
    input_csv = tournament_dir / "match_list.csv"
    output_dir = tournament_dir
      
    matches = pd.read_csv(input_csv)
    
    print(f"Reading from: {input_csv}")
    print(f"Output directory: {output_dir}")
    print("Checking subtitle availability for all matches...\n")
    
    available = []
    unavailable = []
    
    for idx, match in matches.iterrows():
      print(f"[{idx+1}/{len(matches)}] {match['team1']} vs {match['team2']}: ", end='')
      
      has_sub, reason = check_subtitle_availability(match['youtube_url'])

      
      if has_sub:
        print(f"✓ {reason}")
        available.append(match.to_dict())
      else:
        print(f"✗ {reason}")
        unavailable.append(match.to_dict())

      time.sleep(1)
    
    print(f"\n{'='*50}")
    print(f"Results: {len(available)} with subtitles, {len(unavailable)} without")

    if available:
      available_df = pd.DataFrame(available)
      output_path = output_dir / 'match_list_with_subs.csv'
      available_df.to_csv(output_path, index=False)
      print(f"✓ Saved {len(available)} matches with subtitles to: {output_path}")

    
    if unavailable:
      unavailable_df = pd.DataFrame(unavailable)
      output_path = output_dir / 'match_list_no_subs.csv'
      unavailable_df.to_csv(output_path, index=False)
      print(f"✓ Saved {len(unavailable)} matches without subtitles to: {output_path}")

if __name__ == "__main__":
  main()