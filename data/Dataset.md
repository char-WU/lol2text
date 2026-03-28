# Dataset

The workflow builds 2 base datasets for training:
- `LoL19`
- `LoL1921`

**Table of Contents**

- [Data Layout](#data-layout)
  - [File Meanings](#file-meanings)
- [Scripts](#scripts)
- [Data Collection](#data-collection)
- [Alignment](#alignment)
- [Linearization](#linearization)
- [Datasets](#datasets)
- [Recommended Workflow](#recommended-workflow)
  - [Pipeline Overview](#pipeline-overview)
  - [Workflow](#workflow)
- [Notes](#notes)

## Data Layout

```bash
data/
├── 2019/
│   ├── MSI 2019              
│   ├── MSI Play-In 2019
│   ├── RR NA-EU 2019
│   ├── WC 2019               
│   └── WC Play-In 2019
├── 2020/
│   ├── MSC 2020              
│   ├── WC 2020
│   └── WC Play-In 2020
└── 2021/
    ├── MSI 2021
    ├── WC 2021
    └── WC Play-In 2021
```

- `WC`: World Championship
- `MSI`: Mid-Season Invitational
- `MSC`: Mid-Season Cup
- `RR`: Rift Rivals

**Match Counts**
| Year | Tournament | Matches |
|------|------------|--------:|
| 2019 | MSI 2019 | 41 |
|  | MSI Play-In 2019 | 37 |
|  | RR NA-EU 2019 | 13 |
|  | WC 2019 | 77 |
|  | WC Play-In 2019 | 43 |
|  | **Total** | **211** |
| 2020 | MSC 2020 | 10 |
|  | WC 2020 | 76 |
|  | WC Play-In 2020 | 38 |
|  | **Total** | **124** |
| 2021 | MSI 2021 | 81 |
|  | WC 2021 | 83 |
|  | WC Play-In 2021 | 38 |
|  | **Total** | **202** |
| **2019–2021** | **Grand Total** | **537** |

Each tournament folder follows the same structure:

```bash
WC 2019/
├── aligned/                  # Alignment with player-champion mapping
├── processed/                # Linearized training data
├── raw/
│   ├── captions/
│   │   └── v2/               # Whisper transcriptions
│   └── events/               # Gameplay events scraped from gol.gg
├── match_list.csv
├── match_list_generated.csv
├── match_list_with_youtube.csv
└── timestamps.csv
```

**File Meanings**

- `match_list_generated.csv`  
  Match list generated from the gol.gg tournament page

- `match_list_with_youtube.csv`  
  Match list after attaching YouTube VOD links

- `timestamps.csv`  
  Extracted video-to-game offsets

- `match_list.csv`  
  Final merged match list used by later collection and alignment scripts

- `raw/events/`  
  Timestamped gameplay event JSON files

- `raw/captions/v1`  
  Caption JSON files collected from YouTube

- `raw/captions/v2/`  
  Higher-quality Whisper transcription outputs

- `aligned/`  
  Updated aligned outputs with player-champion mapping added to event fields

- `processed/`
Linearized training samples derived from aligned data

## Scripts

```bash
scripts/
├── align_data_kwmatch.py       # Alignment using keyword matching
├── automate_gol.py             # Step 1: Generate match list from gol.gg tournament
├── automate_ytb.py             # Step 2: Find YouTube URLs (EpicSkillshot channel)
├── btach_linearize.py
├── linearize_events.py         # Convert aligned event-level data into window-based training samples
├── merge_offset.py             # Merge offset results into match_list.csv
├── merge_dataset.py            # 🌟 Merge processed JSONL files into a single training dataset (in-memory)
├── merge_OOM.py                # OOM-safe version of merge_dataset.py that streams directly to disk
├── offset_extractor.py         # Step 3: Extract video-to-game time offsets
├── requirements.txt
├── scrape_captions.py          # Download YouTube subtitles
├── scrape_gol.py               # Scrape gameplay events from gol.gg
├── to_hf.py                    # 🌟 Upload the final merged training datasets to Hugging Face
└── transcribe_whisper.py       # Whisper transcription for missing or low-quality captions
```

## Data Collection

Once `match_list.csv` is ready, collect the raw data.

### Scrape Gameplay Events from gol.gg

```bash
python scripts/scrape_gol.py "WC 2019"
```

### Collect Captions

For faster but lower-quality caption collection, use YouTube subtitle scraping:

```bash
python scripts/scrape_captions.py "WC 2019"
# Check whether collected captions are complete and usable:
python scripts/helpers/check_captions.py "WC 2019"
```

For higher-quality captions, use Whisper transcription:

```bash
python scripts/transcribe_whisper.py "WC 2019"
```

The `v2` Whisper route is preferred


## Alignment

After collecting both events and captions, align commentary to gameplay events:

```bash
python scripts/align_data_kwmatch.py "WC 2019"
# → Output: data/2019/WC 2019/aligned/v2/
#   Adds: player-champion pairs
```

Alternative simpler alignment is also available:

```bash
python scripts/helpers/align_data.py "WC 2019"
# → Output: data/2019/WC 2019/aligned/v1/
#   champion-only fields
```

The alignment stage combines:
- raw event JSONs
- caption JSONs
- video-to-game offsets from `match_list.csv`

## Linearization

Convert aligned event-level data into training samples:

```bash
# Process a single tournament
python scripts/linearize_events.py "WC 2019"

# Or process all tournaments at once
python scripts/batch_linearize.py
```

This transforms aligned events into window-based input-output pairs

## Datasets

Under `data/merged/`, there are two merged datasets prepared for later training

For more flexible usage and easier sharing, the merged datasets can also be uploaded to Hugging Face

### Step 1: Get your Hugging Face token

1. Go to `huggingface.co` and create an account if you do not already have one.
2. Open **Settings > Access Tokens**.
3. Create a new token with **Write** permission, since upload access is required.
4. Copy the token.

### Step 2: Log in to Hugging Face on your machine

Install the required libraries if they are not already available:

```bash
pip install datasets huggingface_hub
```

Then log in from the terminal:
```bash
huggingface-cli login
```
You will be prompted to paste the Hugging Face token created in Step 1.

### Step 3: Push the dataset to the Hub
Run:
```bash
python scripts/to_hf.py
```

## Recommended Workflow

### Pipeline Overview
```bash
# Step 1: Generate match list from gol.gg
# Use the official tournament name as input:
python scripts/automate_gol.py "World Championship 2019"
# → Output: data/2019/WC 2019/match_list_generated.csv
#   Contains: match_id, tournament, gol_match_id, teams, stage, date, etc.

# Step 2: Find YouTube URLs (EpicSkillshot channel)
Use the tournament folder name:
python scripts/automate_ytb.py "WC 2019"
# → Output: data/2019/WC 2019/match_list_with_youtube.csv
#   Adds: youtube_url column

# Step 3: Extract video offsets (OCR game clock)
python scripts/offset_extractor.py "WC 2019"
# → Output: data/2019/WC 2019/timestamps.csv
#   Contains: video_time, game_time, offset

# Step 4: Merge offsets into final match list
python scripts/merge_offset.py "WC 2019"
# → Output: data/2019/WC 2019/match_list.csv (FINAL)
#   Ready for data collection pipeline
```

**Example Full Setup**
```bash
python scripts/automate_gol.py "World Championship 2019"
python scripts/automate_ytb.py "WC 2019"
python scripts/offset_extractor.py "WC 2019"
python scripts/merge_offset.py "WC 2019"
```

### Workflow
For a new tournament folder, the usual order is:

```bash
# 1. Build match list
python scripts/automate_gol.py "World Championship 2019"
python scripts/automate_ytb.py "WC 2019"
python scripts/offset_extractor.py "WC 2019"
python scripts/merge_offset.py "WC 2019"

# 2. Collect raw data
python scripts/scrape_gol.py "WC 2019"
python scripts/transcribe_whisper.py "WC 2019"

# 3. Align
python scripts/align_data_kwmatch.py "WC 2019"

# 4. Create training data
python scripts/batch_linearize.py

# 5. Build full dataset
python scripts/merge_dataset.py

# 6. Push to huggingface
python scripts/to_hf.py
```

## Notes

- `automate_gol.py` expects the full tournament name
- `automate_ytb.py`, `offset_extractor.py`, and `merge_offset.py` expect the folder name
- Whisper transcription is **slower**, but produces better commentary text than direct subtitle scraping