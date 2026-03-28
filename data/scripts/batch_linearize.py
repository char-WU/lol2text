import subprocess
from pathlib import Path

# ---- CONFIG ----
BASE_DATA_DIR = Path("data")
SCRIPT_PATH = Path("scripts/linearize_events.py")

WINDOW_SIZE = 60
SEARCH_PRE = 30
SEARCH_POST = 45
# ----------------


def run_for_tournament(tournament_path: Path):
    aligned_dir = tournament_path / "aligned" / "v2"
    output_dir = tournament_path / "processed"

    if not aligned_dir.exists():
        print(f"[SKIP] No aligned/v2 folder in {tournament_path}")
        return

    output_dir.mkdir(exist_ok=True)

    # tournament folder name, e.g. "WC 2019"
    tournament_name = tournament_path.name

    print(f"\n🚀 Processing: {tournament_path.relative_to(BASE_DATA_DIR)}")

    cmd = [
        "python",
        str(SCRIPT_PATH),
        tournament_name,
        "--window-size", str(WINDOW_SIZE),
        "--search-pre", str(SEARCH_PRE),
        "--search-post", str(SEARCH_POST),
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Done: {tournament_path.relative_to(BASE_DATA_DIR)}")
    except subprocess.CalledProcessError:
        print(f"❌ Failed: {tournament_path.relative_to(BASE_DATA_DIR)}")


def main():
    if not BASE_DATA_DIR.exists():
        print(f"Base data dir not found: {BASE_DATA_DIR}")
        return

    tournaments = []

    for year_dir in sorted(BASE_DATA_DIR.iterdir()):
        if not year_dir.is_dir():
            continue

        for tournament_dir in sorted(year_dir.iterdir()):
            if tournament_dir.is_dir():
                tournaments.append(tournament_dir)

    print(f"Found {len(tournaments)} tournaments")

    for t in tournaments:
        run_for_tournament(t)


if __name__ == "__main__":
    main()
