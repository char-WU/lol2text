import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
import re
import urllib.parse
import argparse

# 2019
TOURNAMENT_NAME = None

def build_tournament_url(name):
  """
   "World Championship 2019" ->  "https://gol.gg/tournament/tournament-matchlist/World%20Championship%202019/"
    "World Championship Play-In 2019" -> ".../tournament-matchlist/World%20Championship%20Play-In%202019/"
  """
  encoded = urllib.parse.quote(name)
  return f"https://gol.gg/tournament/tournament-matchlist/{encoded}/"

def scrape_matches(url):
  """
  Returns:
    list of dicts: include match_id, team 1, 2, score, stage, date
  """
  headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
  }
  try:
    print(f"Fetching tournament page: {url}")
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    tables = soup.find_all('table')

    match_table = None
    for table in tables:
      headers = table.find_all('th')
      header_texts = [h.text.strip() for h in headers]
      if 'Game' in header_texts and 'Score' in header_texts:
          match_table = table
          break
    
    if not match_table:
      print("!! Could not find match table")
      return []
    
    matches = []
    rows = match_table.find_all('tr')[1:]

    for row in rows:
      cells = row.find_all('td')
      if len(cells) < 6: continue

      game_cell = cells[0] #Game column
      winner_cell = cells[1] #Winner
      score_cell = cells[2] #Score
      loser_cell = cells[3] #Loser
      stage_cell = cells[4] #Stage (FINALS, SF, QF, etc.)
      patch_cell = cells[5] #Patch
      date_cell = cells[6] #Date

      link = game_cell.find('a')
      if not link: continue

      href = link.get('href', '')
      match_id_pattern = r'/game/stats/(\d+)/'
      match_id_match = re.search(match_id_pattern, href)

      if not match_id_match: continue
      match_id = int(match_id_match.group(1))

      matchup_text = link.text.strip()
      teams = matchup_text.split(' vs ')
      if len(teams) != 2: continue

      team1 = teams[0].strip()
      team2 = teams[1].strip()
      score = score_cell.text.strip() #"3 - 0"
      stage = stage_cell.text.strip() #"FINALS"
      date = date_cell.text.strip() #"2019-11-10"

      gol_url = f"https://gol.gg/game/stats/{match_id}/page-timeline/"
      matches.append({
        'match_id': match_id,
        'team1': team2,
        'team2': team1,
        'score': score,
        'stage': stage,
        'date': date,
        'gol_url': gol_url,
        'youtube_url': '',
        'offset_seconds': 0
      })
  
    return matches
  
  except Exception as e:
    print(f"!!! Error scraping tournament page: {e}")
    return []

def expand_to_series(matches):
  """
  BO3: match_id, match_id+1, match_id+2
  """
  expand = []
  for match in matches:
    score = match['score']
    stage = match['stage']

    try:
      parts = score.split('-')
      if len(parts) == 2:
        score1 = int(parts[0].strip())
        score2 = int(parts[1].strip())
        total_games = score1 + score2
      else:
        total_games = 1
    except:
      total_games = 1

    base_match_id = match['match_id']
    for game_num in range(total_games):
      game_match = match.copy()
      game_match['series'] = stage

      if total_games > 1:
        game_match['match_id'] = base_match_id + game_num
        game_match['game_num'] = game_num + 1
        game_match['series_id'] = base_match_id
      else:
        #BO1
        game_match['game_num'] = 1
        game_match['series_id'] = base_match_id
      
      game_match['gol_url'] = f"https://gol.gg/game/stats/{game_match['match_id']}/page-timeline/"
      expand.append(game_match)
  
  return expand

def tournament_to_folder(name):
  """
  "World Championship 2020" -> ("2020", "WC 2020")
  "World Championship Play-In 2020" -> ("2020", "WC Play-In 2020")
  "MSI 2021" -> ("2021", "MSI 2021")
  """
  year_match = re.search(r'(\d{4})', name)
  year = year_match.group(1) if year_match else "unknown"
  
  if "Mid-Season Invitational" in name or name.startswith("MSI"):
    folder = f"MSI {year}"
  elif "Mid-Season Cup" in name:
    folder = f"MSC {year}"
  elif "World Championship Play-In" in name:
    folder = f"WC Play-In {year}"
  elif "World Championship" in name:
    folder = f"WC {year}"
  else:
    folder = name
  
  return year, folder


def main():
  parser = argparse.ArgumentParser(
    description='Build match list from gol.gg World Championship tournament',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
    Examples:
      python automate_gol.py World Championship 2019
    """
  )
  parser.add_argument(
    'name',
    type=str,
    help='Tournament name (e.g., "World Championship Play-In 2019", "World Championship 2019")'
  )

  args = parser.parse_args()
  tournament_url = build_tournament_url(args.name)
  tournament_name = f"{args.name}"

  year, folder_name = tournament_to_folder(args.name)
  tournament_dir = Path("data") / year / folder_name
  tournament_dir.mkdir(parents=True, exist_ok=True)
  OUTPUT_CSV = tournament_dir / "match_list_generated.csv"

  print("="*60)
  print("GOL.GG MATCH LIST")
  print(f"Tournament URL: {tournament_url}")
  print(f"Output file: {OUTPUT_CSV}")

  #1: tournament
  print("Step 1: Scraping tournament page...")
  matches = scrape_matches(tournament_url)

  if not matches:
    print("!!! No matches found!")
    return
  print(f"✓ Found {len(matches)} series")

  #2: BOx
  print("\nStep 2: Expanding BO series to individual games...")
  expanded_matches = expand_to_series(matches)
  print(f"✓ Expanded to {len(expanded_matches)} individual games")
  for i, match in enumerate(expanded_matches, start=1):
    match['tournament'] = tournament_name
    match['match_id'] = i

  print(f"\nStep 3: Saving to {OUTPUT_CSV}...")

  OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
  df = pd.DataFrame(expanded_matches)
  col_order = [
    'match_id',
    'tournament',
    'date',
    'series',
    'game_num',
    'team1',
    'team2',
    'gol_url',
    'youtube_url',
    'offset_seconds'
  ]
  col_order = [col for col in col_order if col in df.columns]
  df = df[col_order]
  df.to_csv(OUTPUT_CSV, index=False)
  print(f"✓ Saved {len(df)} games")

if __name__ == "__main__":
  main()