import whisper
import yt_dlp
import time
import pandas as pd
import json
from pathlib import Path
import tempfile
import argparse
import re
import subprocess

def validate_audio(audio_path):
  try:
    result = subprocess.run(
      ['ffmpeg', '-v', 'error', '-i', str(audio_path), '-f', 'null', '-'],
      capture_output=True,
      text=True,
      timeout=10
    )
    return result.returncode == 0
  except:
    return False

#CAPTIONS_DIR = Path("data/raw/captions")
#AUDIO_CACHE_DIR = Path("data/cache/audio")
#MATCH_LIST_PATH = Path("data/match_list_no_subs.csv")

#WHISPER_MODEL = "base"

def download_audio(url, output_path, max_retries=3):
  """
  Returns: fail/pass
  """
  for attempt in range(max_retries):
    ydl_ops = {
      'format': 'bestaudio/best',
      'sleep_interval': 3,
      'max_sleep_interval': 6,
      'postprocessors': [
        {
          'key': 'FFmpegExtractAudio',
          'preferredcodec': 'mp3',
          'preferredquality': '192',
        }
      ],
      'outtmpl': str(output_path.with_suffix('')),
      'quiet': False,
      'no_warnings': True,
    }

    try:
      with yt_dlp.YoutubeDL(ydl_ops) as ydl:
        ydl.download([url])

      audio_file = output_path.with_suffix('.mp3')
      
      if audio_file.exists() and validate_audio(audio_file):
        print(f"   ✓ Audio validated (attempt {attempt+1})")
        return True
      else:
        print(f"   ⚠️  Invalid audio, retrying ({attempt+1}/{max_retries})...")
        if audio_file.exists():
          audio_file.unlink()
          
    except Exception as e:
      print(f"   ⚠️  Download failed ({attempt+1}/{max_retries}): {e}")
      if output_path.with_suffix('.mp3').exists():
        output_path.with_suffix('.mp3').unlink()
    
    time.sleep(2)
  
  return False

def transcribe(audio_path, model_name="base", language="en"):
  """
  Args:
    audio_path: audio file
    model_name: Whisper model
    language: en=English
  
  Returns:
    dict: text with timestamps
  """
  print(f"   Loading Whisper model '{model_name}'...")
  model = whisper.load_model(model_name)

  print(f"   Transcribing audio (this may take a while)...")
  result = model.transcribe(
    str(audio_path),
    language=language,
    task="transcribe",
    verbose=False,
    word_timestamps=False
  )

  return result

def to_caption(result):
  """
  turn Whisper output to YTB caption format:
    [
      {'start': 0.5, 'duration': 2.7, 'text': 'xxx'},
      ...
    ]
  """
  captions = []

  for seg in result.get('segments', []):
    start = seg['start']
    end = seg['end']
    duration = end - start
    text = seg['text'].strip()

    if text:
      captions.append({
        'start': start,
        'duration': duration,
        'text': text
      })
    
  return captions

def process_video(url, caption_path, match_id, model_name="base", audio_cache_dir=None):
  """
  Pipeline: download autio -> Whisper transcribe -> save caption
  Returns:
    int: # caption (0=fail)
  """
  audio_cache_dir.mkdir(parents=True, exist_ok=True)
  temp_audio = audio_cache_dir/f"match_{match_id}_audio.mp3"

  try:
    # Step 1: download
    print(f"   Downloading audio...")
    if not download_audio(url, temp_audio):
      return 0

    # Step 2: transcribe
    result = transcribe(temp_audio, model_name=model_name)
    
    # Step 3: save
    captions = to_caption(result)

    if not captions:
      print (f"    Transcription returned no segments")
      return 0
    
    # Step 4: to JSON
    with open(caption_path, 'w', encoding='utf-8') as f:
      json.dump(captions, f, indent=2, ensure_ascii=False)
    
    return len(captions)
  
  except Exception as e:
    print(f"   !!! Processing error: {e}")
    return 0
  
  #finally:
  #  if temp_audio.exists():
  #    temp_audio.unlink()

def get_model_size(model):
  sizes = {
    'tiny': '~75MB',
    'base': '~150MB',
    'small': '~500MB',
    'medium': '~1.5GB',
    'large': '~3GB'
  }
  return sizes.get(model, 'unknown')

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

def main():
  parser = argparse.ArgumentParser(description='Transcribe videos without subtitles using Whisper')
  parser.add_argument(
    'folder',
    type=str,
    nargs='?'
  )
  parser.add_argument(
    '--model', '-m',
    type=str,
    default='base',
    choices=['tiny', 'base', 'small', 'medium', 'large'],
    help='Whisper model size (default: base)'
  )
  args = parser.parse_args()

  folder = get_tournament_dir(args.folder)
  base_folder = folder

  AUDIO_CACHE_DIR = base_folder / 'cache' / 'audio'
  MATCH_LIST_PATH = base_folder / 'match_list.csv'
  CAPTIONS_DIR = base_folder / "raw" / "captions" / "v2"

  WHISPER_MODEL = args.model

  if not MATCH_LIST_PATH.exists():
    print(f"!!! {MATCH_LIST_PATH} not found")
    return
  
  matches = pd.read_csv(MATCH_LIST_PATH)
  total = len(matches)

  CAPTIONS_DIR.mkdir(parents=True, exist_ok=True)

  successful = 0
  skipped = 0
  failed = 0
  print(f"Starting Whisper transcription for {total} matches")
  print(f"Using Whisper model: {WHISPER_MODEL}")
  print(f"Note: First run will download the model (~{get_model_size(WHISPER_MODEL)})\n")

  for i, match in matches.iterrows():
    match_id = int(match['match_id'])
    youtube_url = match['youtube_url']
    print(f"[{i+1}/{total}] Match {match_id}: {match.get('team1', 'N/A')} vs {match.get('team2', 'N/A')}")
        
    output_path = CAPTIONS_DIR / f"match_{match_id}.json"

    if output_path.exists():
      skipped += 1
      continue

    num = process_video(
      youtube_url,
      output_path,
      match_id,
      model_name=WHISPER_MODEL,
      audio_cache_dir=AUDIO_CACHE_DIR
    )

    if num > 0:
      print(f"   ✓ Transcribed {num} segments")
      successful += 1
    else:
      print(f"   !! Failed")
      failed += 1
    
    time.sleep(1)

  print("\n" + "="*60)
  print("WHISPER TRANSCRIPTION SUMMARY")
  print(f"Total matches: {total}")
  print(f"Successfully transcribed: {successful}")
  print(f"Skipped (already exist): {skipped}")
  print(f"Failed: {failed}")
  print("="*60)
    
if __name__ == "__main__":
  main()