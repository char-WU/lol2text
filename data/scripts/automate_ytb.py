"""
Retrieve youtube video links from channel: EpicSkillshot - LoL VOD Library
"""
from pathlib import Path
import pandas as pd
import yt_dlp
import time
import re
import argparse

CHANNEL_URL = "https://www.youtube.com/@EpicSkillshot"

TEAM_ABBREV = {
  'Funplus Phoenix': 'FPX',
  'G2 Esports': 'G2',
  'SK Telecom T1': 'SKT',
  'Invictus Gaming': 'IG',
  'DAMWON Gaming': 'DWG',
  'Griffin': 'GRF',
  'Fnatic': 'FNC',
  'Splyce': 'SPY',
  'Team Liquid': 'TL',
  'Cloud9': 'C9',
  'Royal Never Give Up': 'RNG',
  'ahq e-Sports Club': 'AHQ',
  'Hong Kong Attitude': 'HKA',
  'Clutch Gaming': 'CG',
  'J Team': 'JT',
  'GAM Esports': 'GAM',
  'DRX': 'DRX',
  'Suning': 'SN',
  'Top Esports': 'TES',
  'JD Gaming': 'JDG',
  'Gen.G': 'GEN',
  'TSM': 'TSM',
  'Machi Esports': 'MCX',
  'Unicorns Of Love': 'UOL',
  'PSG Talon': 'PSG',
  'LGD Gaming': 'LGD',
  'Rogue': 'RGE',
}

STAGE_NAMES = {
  'FINALS': 'Grand Finals',
  'SF': 'Semi Finals',
  'QF': 'Quarter Finals',
  'TIEBREAKERS': 'Tiebreakers',
  'QUAL.ROUND': 'Qualification Round',
  'ELIM.ROUND': 'Elimination Round',
  'DAY1': 'Day 1',
  'DAY2': 'Day 2',
  'DAY3': 'Day 3',
  'DAY4': 'Day 4',
  'DAY5': 'Day 5',
  'DAY6': 'Day 6',
  'DAY7': 'Day 7',
  'DAY8': 'Day 8',
}

def get_team(team_name):
  if team_name.isupper() and len(team_name) <= 5:
    return team_name
  
  if team_name in TEAM_ABBREV:
    return TEAM_ABBREV[team_name]
  
  words = team_name.replace('-', ' ').split()
  abbrev = ''.join(word[0].upper() for word in words if word)
  return abbrev

def get_season(year):
  """
  2011: S1, ..., 2019: S9,...
  """
  return year - 2010

def extract_stage(stage):
  """
  "FunplusPhoenixvsG2Esports_FINALS" -> "FINALS"
  """
  if '_' in stage:
    return stage.split('_')[-1]
  return stage

def get_name(stage):
  return STAGE_NAMES.get(stage, stage)

def build_title(team1, team2, game_num, stage, year, tournament):
  """
  Format: 
    Main: "{T1} vs {T2} - Game {N} | {Stage} S{Season} LoL Worlds {Year}"
    Play-In: "{T1} vs {T2} - Game {N} | Play-Ins S{Season} LoL Worlds {Year}"
  """
  team1 = get_team(team1)
  team2 = get_team(team2)
  stage = get_name(stage)
  season = get_season(year)

  is_playin = 'PlayIn' in tournament or 'Play-In' in tournament or 'play-in' in tournament.lower()

  part1 = f"{team1} vs {team2} - Game {game_num}"
  if is_playin:
    part2 = f"Play-Ins S{season} LoL Worlds {year}"
  else:
    part2 = f"{stage} S{season} LoL Worlds {year}"
  part3 = f"{team1} vs {team2} G-{game_num}"

  title = f"{part1} | {part2} | {part3}"
  return title

def search_channel(query, max_results=5):
  ydl_opts = {
    'quiet': True,
    'no_warnings': True,
    'extract_flat': True,
    'skip_download': True,
  }
  search_url = f"ytsearch{max_results}:{query}"

  with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    result = ydl.extract_info(search_url, download=False)
    videos = []
    for entry in result['entries']:
      if not entry: continue
      video_id = entry.get('id', '')
      videos.append({
        'title': entry.get('title', ''),
        'url': f"https://youtube.com/watch?v={video_id}",
        'channel': entry.get('uploader', ''),
      })

    return videos

def find_url(row):
  team1 = row['team1']
  team2 = row['team2']
  game_num = row['game_num']
  series = str(row['series'])
  tournament = str(row['tournament'])
  year_match = re.search(r'(\d{4})', tournament)
  year = int(year_match.group(1))
  stage = extract_stage(series)

  team1 = get_team(team1)
  team2 = get_team(team2)

  is_playin = 'PlayIn' in tournament or 'Play-In' in tournament
  is_group_stage = series.startswith('DAY') or series == 'TIEBREAKERS' or series == 'KNOCKOUTSTAGE'
  is_knockout = series == 'KNOCKOUTSTAGE'

  day_num = series.replace('DAY', '') if series.startswith('DAY') else None
  playin_keyword = 'Play-Ins' if is_playin else ''

  if is_knockout:
    search_query = f'{team2} vs {team1} Game {game_num} Knockouts {playin_keyword} Worlds {year} EpicSkillshot'
  elif is_group_stage:
    if series == 'TIEBREAKERS':
      search_query = f'{team2} vs {team1} tie breaker {playin_keyword} Worlds {year} EpicSkillshot'
    else:
      search_query = f'{team2} vs {team1} {playin_keyword} Day {day_num} Worlds {year} EpicSkillshot'
  else:
    stage_name = get_name(stage)
    search_query = f'{team2} vs {team1} {playin_keyword} {stage_name} Game {game_num} Worlds {year} EpicSkillshot'
    #search_query = f'EpicSkillshot Worlds {year} {team1} {team2} Game {game_num}'
  
  print(f"    Searching: {search_query}")
  videos = search_channel(search_query)
    
  if not videos:
    print(f"    ⚠️  No videos found")
    return ''
  
  EXCLUDE = ["recap", "highlight", "teaser", "preview", "re-broadcast"]
  
  for video in videos:
    if 'epicskillshot' in video['channel'].lower():
      title = video['title']
      title_lower = title.lower()

      if any(word in title for word in EXCLUDE):
        continue

      has_teams = team1.lower() in title.lower() and team2.lower() in title.lower()

      if is_playin:
        has_playin_keyword = 'play-in' in title_lower or 'play-ins' in title_lower
      else:
        has_playin_keyword = True

      stage_name = get_name(stage).lower()
      has_game = f"game {game_num}" in title_lower or f"g{game_num}" in title_lower or f"g-{game_num}" in title_lower

      if is_knockout:
          has_context = ('knockout' in title_lower)
      elif is_group_stage:
        if series == 'TIEBREAKERS':
          has_context = ('tiebreaker' in title_lower or 
                         'tie breaker' in title_lower or 
                         'tie-breaker' in title_lower)
        else:
          has_context = f"Day {day_num}" in title
      else:
        has_context = has_game

      if has_teams and has_context and has_playin_keyword:
        print(f"   ✓ Found: {title[:70]}...")
        return video['url']
  
  print(f"    !!! No EpicSkillshot video found")
  return ''

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
  parser = argparse.ArgumentParser(description='Scrape YouTube captions for matches')
  parser.add_argument(
    'tournament',
    type=str,
    nargs='?',
    help='Tournament folder name (default: "WC 2019")'
  )
  
  args = parser.parse_args()

  tournament_dir = get_tournament_dir(args.tournament)
  INPUT_CSV = tournament_dir / 'match_list_generated.csv'
  OUTPUT_CSV = tournament_dir / 'match_list_with_youtube.csv'

  df = pd.read_csv(INPUT_CSV)
  df['youtube_url'] = df['youtube_url'].astype(str)
  required_cols = ['team1', 'team2', 'game_num', 'series', 'tournament']
  for col in required_cols:
    if col not in df.columns:
      print(f"Missing column: {col}")
      return
  
  if 'youtube_url' not in df.columns:
    df['youtube_url'] = ''
  
  total = len(df)
  updated = 0
  skipped = 0
  failed = 0

  print("="*60)
  print("EPICSKILLSHOT YOUTUBE URL FINDER")
  print(f"Channel: {CHANNEL_URL}")
  print(f"Processing {total} games\n")

  for i, row in df.iterrows():
    match_id = row['match_id']
    team1 = row['team1']
    team2 = row['team2']
    game_num = row['game_num']
    print(f"[{i+1}/{total}] Match {match_id}: {team1} vs {team2} - Game {game_num}")
    
    if pd.notna(row.get('youtube_url')) and row['youtube_url'].strip():
      print(f"  Already has URL")
      skipped += 1
      continue

    url = find_url(row)
    if url:
      df.at[i, 'youtube_url'] = url
      updated += 1
    else:
      failed += 1
    
    time.sleep(1)
  
  OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
  df.to_csv(OUTPUT_CSV, index=False)
  print(f"\n{'='*60}")
  print("SUMMARY")
  print(f"Total games: {total}")
  print(f"URLs found: {updated}")
  print(f"Already had URLs: {skipped}")
  print(f"Failed: {failed}")
  print(f"\n✓ Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
  main()