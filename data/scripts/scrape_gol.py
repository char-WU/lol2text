import requests
from bs4 import BeautifulSoup
import json
import time
import pandas as pd
import random
from pathlib import Path
import argparse
import re

HEADERS_LIST = [
    {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'},
    {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36'},
    {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'}
]

def parse_timestamp(time_str):
  """
  eg. "3:11" -> 191s
  """
  parts = time_str.strip().split(":")
  if len(parts) == 2:
    minutes, seconds = parts
    return (int(minutes) * 60 + int(seconds)) * 1000
  return 0

def determine_event_type(row):
  """
  determine event type based on icons
  """
  cells = row.find_all('td')
  if len(cells) < 5:
    return "UNKNOWN", ""
  
  action_cell = cells[4]
  img = action_cell.find('img')
  if not img:
    return "UNKNOWN", ""
  
  src = img.get('src', '').lower()

  if 'kill-icon' in src:
    return "CHAMPION_KILL", "Kill"
  elif 'tower-icon' in src:
    return "BUILDING_KILL", "Tower"
  elif 'inhib-icon' in src:
    return "BUILDING_KILL", "Inhibitor"
  elif 'nexus-icon' in src:
    return "BUILDING_KILL", "Nexus"
  elif 'cloud-dragon' in src:
    return "ELITE_MONSTER_KILL", "Cloud Drake"
  elif 'ocean-dragon' in src:
    return "ELITE_MONSTER_KILL", "Ocean Drake"
  elif 'fire-dragon' in src or 'infernal-dragon' in src:
    return "ELITE_MONSTER_KILL", "Infernal Drake"
  elif 'mountain-dragon' in src:
    return "ELITE_MONSTER_KILL", "Mountain Drake"
  elif 'elder-dragon' in src:
    return "ELITE_MONSTER_KILL", "Elder Dragon"
  elif 'herald-icon' in src:
    return "ELITE_MONSTER_KILL", "Rift Herald"
  elif 'nashor-icon' in src or 'baron' in src:
    return "ELITE_MONSTER_KILL", "Baron Nashor"
  else:
    return "UNKNOWN", ""

def extract_champions(cell):
  """
  Extract the list of champion icons from the cells
  Return the corresponding list of champion names
  """
  if not cell:
    return []
  
  imgs = cell.find_all('img')
  champions = []

  for img in imgs:
    src = img.get('src', '')
    if 'champions_icon' in src:
      champion_name = src.split('/')[-1].replace('.png', '')
      champions.append(champion_name)
  
  return champions

def get_player_champion_mapping(gol_url):
  fullstats_url = gol_url.replace('page-timeline', 'page-fullstats')
  mapping = {}
  rev_mapping = {}
  try:
    response = requests.get(
      fullstats_url,
      headers=random.choice(HEADERS_LIST),
      timeout=15
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    rows = soup.find_all('tr')
    champ_cells = []
    player_cells = []

    for i, row in enumerate(rows):
      cells = row.find_all(['th', 'td'])
      if cells and cells[0].text.strip() == 'Player':
        player_cells = cells[1:]
        if i > 0:
          champ_cells = rows[i-1].find_all(['th', 'td'])[1:]
        break
    if player_cells and champ_cells:
      for p_cell, c_cell in zip(player_cells, champ_cells):
        player = p_cell.text.strip()
        img = c_cell.find('img')
        if img and 'champions_icon' in img.get('src', ''):
          champ = img.get('src').split('/')[-1].replace('.png', '')
          mapping[player] = champ
          rev_mapping[champ] = player
  except Exception as e:
    print(f"    !! Failed to fetch mapping from {fullstats_url}: {e}")
  
  return mapping, rev_mapping

def format_entities(names, mapping, rev_mapping):
  if not names:
    return []
  
  if isinstance(names, str):
    names = [names]
      
  result = []
  for name in names:
    if not name:
      continue
    if name in mapping:
      result.append({"player": name, "champion": mapping[name]})
    elif name in rev_mapping:
      result.append({"player": rev_mapping[name], "champion": name})
    else:
      result.append(name)
          
  return result


def scrape_game_timeline(gol_url):
  """
  Scrape the timeline data for a single match from gol.gg
    Args:
      gol_url: URL of the match page
    Returns:
      events: List[dict], where each dict represents an in-game event
  """
  mapping, rev_mapping = get_player_champion_mapping(gol_url)
  if not mapping:
    print("    Warning: Could not fetch player mapping. Resorting to flat names.")

  max_retries = 3
  for attempt in range(max_retries):
    try:
      print(f"   Fetching {gol_url} (Attempt {attempt + 1})")
      # Randomize header and increase timeout to 30s
      response = requests.get(
        gol_url, 
        headers=random.choice(HEADERS_LIST), 
        timeout=30
      )
      response.raise_for_status()

      soup = BeautifulSoup(response.content, 'html.parser')
      timeline_table = soup.find('table', class_='timeline')

      if not timeline_table:
        print(f"Warning: Timeline table not found in {gol_url}")
        return []
      
      events = []
      rows = timeline_table.find_all('tr')
      for row in rows:
        if row.get('id') == 'lineheader': continue
        cells = row.find_all('td')
        if len(cells) < 5: continue

        timestamp_str = cells[0].text.strip() #3:11
        if not timestamp_str or ':' not in timestamp_str: continue
        timestamp = parse_timestamp(timestamp_str)

        team_icon = cells[1].find('img')
        if team_icon and 'blueside' in team_icon.get('src', ''):
          team = 'blue'
        else:
          team = 'red'

        #col 2
        player_str = cells[2].text.strip()

        #col 3
        participants_cell = cells[3] if len(cells) > 3 else None
        participants_list = extract_champions(participants_cell)

        #col 4
        event_type, event_subtype = determine_event_type(row)

        #col5, 6      
        victim_cell = cells[5] if len(cells) > 5 else None
        victim_champions = extract_champions(victim_cell)
        victim_str = victim_champions[0] if victim_champions else None
              
        #col 6
        target_desc = cells[6].text.strip() if len(cells) > 6 else ""

        event = {
          'timestamp': timestamp,
          'type': event_type,
          'subtype': event_subtype,
          'team': team,
          'player': format_entities(player_str, mapping, rev_mapping),
          'participants': format_entities(participants_list, mapping, rev_mapping),
          'victim': format_entities(victim_str, mapping, rev_mapping),
          'target': format_entities(target_desc, mapping, rev_mapping)
        }
        events.append(event)

      return events

    except (requests.Timeout, requests.RequestException) as e:
      print(f"   !! Attempt {attempt + 1} failed: {e}")
      if attempt < max_retries - 1:
        wait_time = (attempt + 1) * 10 # Wait 10s, then 20s
        print(f"  Waiting {wait_time}s before retrying...")
        time.sleep(wait_time)
      else:
        print(f"  !! All retries failed for {gol_url}")
        return []

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
  input_csv = tournament_dir / 'match_list.csv'
  output_dir = tournament_dir / 'raw' / 'events' / 'v2'

  if not input_csv.exists():
    print(f"❌ Input file not found: {input_csv}")
    return

  output_dir.mkdir(parents=True, exist_ok=True)
  matches = pd.read_csv(input_csv)
  total_matches = len(matches)
  successful = 0

  for idx, match in matches.iterrows():
    print(f"\n[{idx+1}/{total_matches}] Processing: {match.get('team1', 'T1')} vs {match.get('team2', 'T2')}")

    events = scrape_game_timeline(match['gol_url'])
    if not events:
      print(f"Warning: No events found, skipping...")
      continue
      
    output_path = output_dir / f"match_{match['match_id']}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
      json.dump(events, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Saved {len(events)} events to {output_path}")
    successful += 1

    time.sleep(2)
  
  print(f"\n{'='*50}")
  print("\n✓ All matches processed!")

if __name__ == "__main__":
  main()