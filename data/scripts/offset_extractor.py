#!/usr/bin/env python3
"""
League of Legends Bulk Timestamp Extractor

This script automates the extraction of in-game timestamps from YouTube videos of League of Legends matches. 
It performs the following steps for each YouTube link provided in a CSV file:
- Grab ONE frame near a requested playback time (default 3:40 / 220s)
- Focus ROI on in-game clock and apply OCR to read the timestamp shown in the frame
- Compute the difference between the requested timestamp and the OCR-extracted in-game time
- Save results to timestamps.csv for all matches in data/match_list_with_subs_offset.csv

Possible limitations:
- Not guarantee the frame is at 220s exactly; YouTube streaming + ffmpeg seeking can be imprecise.
- OCR may fail on some frames due to frame not containing the timestamp (e.g. black screen) or OCR errors.
"""

import cv2
import pytesseract
import yt_dlp
import csv
import re
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import subprocess
import shutil
import argparse


pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'
#DEFAULT： 3:40

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

class BulkLoLExtractor:
    def __init__(self, tournament_folder: str, output_csv: str = "timestamps.csv", target_timestamp: int = 390, debug_visuals: bool = False):
        self.output_csv = output_csv
        self.target_timestamp = int(target_timestamp)
        self.debug_visuals = bool(debug_visuals)

        self.project_root = Path(__file__).resolve().parents[1]  # scripts/ -> project root
        self.tournament_dir = self.project_root / get_tournament_dir(tournament_folder)
        self.temp_dir = self.project_root / "temp_processing3"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        print(f"[debug] temp_processing dir: {self.temp_dir}")

        self.default_matches_csv = self.tournament_dir / "match_list_with_youtube.csv"
        self.output_csv = self.tournament_dir / output_csv



    def _probe_pts_time(self, ffmpeg_path: str, media_url: str, header_str: str, seek_s: float) -> Tuple[Optional[float], str]:
        """
        Decode exactly one frame and parse showinfo pts_time from stderr.
        Returns (pts_time_seconds, stderr_text).
        """
        cmd = [
            ffmpeg_path, "-y",
            "-headers", header_str,
            "-copyts",                 
            "-ss", str(seek_s),
            "-i", media_url,
            "-t", "2",
            "-frames:v", "1",
            "-vf", "showinfo",
            "-f", "null", "-",         
            "-loglevel", "info",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)

        if r.returncode != 0:
            return None, r.stderr

        matches = re.findall(r"pts_time:([0-9]+(?:\.[0-9]+)?)", r.stderr)
        return (float(matches[-1]) if matches else None), r.stderr



    def download_frame(self, url: str, timestamp: int = 390) -> Tuple[Optional[str], Optional[float]]:
        """
        Returns (frame_path, derived_actual_video_time_seconds)
        derived_actual_video_time_seconds is computed by:   pts_target - pts_start
        where pts_start is pts near 0.5s, and pts_target is pts near requested timestamp.
        """
        safe_id = re.sub(r"[^A-Za-z0-9_-]+", "_", url)
        out_path = str(self.temp_dir / f"{safe_id}_{timestamp}.jpg")

        ffmpeg_path = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
        if not ffmpeg_path:
            print("<<ERROR>>: ffmpeg not found on PATH.")
            return None, None

        try:
            ydl_opts = {
                "quiet": True, 
                "format": "bestvideo[ext=mp4]/bestvideo/best",
                "nocheckcertificate": True,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                media_url = info.get("url")
                headers = info.get("http_headers") or {}

            if not media_url:
                print("<<ERROR>>: yt-dlp did not return a media URL.")
                return None, None

            header_str = "".join(f"{k}: {v}\r\n" for k, v in headers.items() if v)

            # Probe near start to anchor PTS to "video time"
            pts0, err0 = self._probe_pts_time(ffmpeg_path, media_url, header_str, seek_s=0.5)
            if pts0 is None:
                print("<<ERROR>>: couldn't read start pts_time (probe).")
                # print(err0)  
                return None, None

            # Probe near target seek to learn actual PTS where ffmpeg lands
            ptsT, errT = self._probe_pts_time(ffmpeg_path, media_url, header_str, seek_s=float(timestamp))
            if ptsT is None:
                print("<<ERROR>>: couldn't read target pts_time (probe).")
                # print(errT)  
                return None, None

            derived_actual_video_time = ptsT

            # write the actual JPEG (fast seek to minimise download)
            # ** may not land exactly at `timestamp` on YouTube streams
            cmd = [
                ffmpeg_path, "-y",
                "-headers", header_str,
                "-ss", str(timestamp),
                "-i", media_url,
                "-t", "2",
                "-frames:v", "1",
                "-q:v", "2",
                out_path,
                "-loglevel", "error",
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)

            if r.returncode != 0:
                print("ffmpeg failed writing jpg:")
                print("STDERR:", (r.stderr.strip() or "(no stderr)"))
                return None, derived_actual_video_time

            if not os.path.exists(out_path):
                print("ffmpeg returned 0 but output file was not created.")
                return None, derived_actual_video_time

            return out_path, derived_actual_video_time

        except Exception as e:
            print(f"Download error: {e}")
            return None, None



# OCR 
    def extract_time(self, image_path: str) -> Optional[str]:
        frame = cv2.imread(image_path)
        if frame is None:
            return None

        h, w = frame.shape[:2]

        # Old narrow ROI:
        #y1, y2 = int(h * 0.07), int(h * 0.09)
        #x1, x2 = int(w * 0.49), int(w * 0.52)

        # New robust ROI for "Top-Centre" clocks:
        y1, y2 = int(h * 0.05), int(h * 0.11)  # 5% to 11% height
        x1, x2 = int(w * 0.46), int(w * 0.54)  # 46% to 54% width (8% wide)
        roi = frame[y1:y2, x1:x2]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = 255 - thresh

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        

        if self.debug_visuals:
            vis = frame.copy()
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("full frame - roi window in green", vis)
            cv2.imshow("roi window", roi)
            cv2.imshow("processed", thresh)
            print("press any key to continue")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        text = pytesseract.image_to_string(
            thresh,
            config="--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789:"
        )
        text = text.strip().replace("\n", "").replace(" ", "")
        
        # Save the ROI to see what the script "sees"
        roi_debug_path = image_path.replace(".jpg", "_roi_debug.jpg")
        cv2.imwrite(roi_debug_path, thresh)
        print(f"  [debug] Saved ROI check to: {roi_debug_path}")

        match = re.search(r"(\d{1,2}):(\d{2})", text)
        return match.group(0) if match else None


# csv processing
    def read_youtube_links_from_matches_csv(self, csv_path: Path) -> List[str]:
        if not csv_path.exists():
            print(f"<<Error>>: CSV not found: {csv_path}")
            return []

        links: List[str] = []
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "youtube_url" not in reader.fieldnames:
                print(f"<<Error>>: CSV must contain a 'youtube_url' column. Found: {reader.fieldnames}")
                return []

            for row in reader:
                u = (row.get("youtube_url") or "").strip()
                if u.startswith("http"):
                    links.append(u)

        return links


# main processing loop
    def process_file(self, matches_csv_path: Optional[str] = None):
        csv_path = Path(matches_csv_path).expanduser().resolve() if matches_csv_path else self.default_matches_csv

        links = self.read_youtube_links_from_matches_csv(csv_path)
        if not links:
            print("no valid YouTube links found; exiting.")
            return

        print(f"starting bulk processing of {len(links)} links from: {csv_path}")

        file_exists = os.path.isfile(self.output_csv)
        with open(self.output_csv, "a" if file_exists else "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "youtube_url",
                    "requested_playback_time",
                    "derived_actual_playback_seconds",
                    "derived_drift_seconds",
                    "ingame_time",
                    "difference_seconds"
                ])

            for i, url in enumerate(links, 1):
                print(f"[{i}/{len(links)}] Processing: {url}")

                frame_path, actual_s = self.download_frame(url, timestamp=self.target_timestamp)

                if actual_s is not None:
                    drift = actual_s - self.target_timestamp
                    print(f"  [debug] requested={self.target_timestamp}s derived_actual={actual_s:.3f}s drift={drift:+.3f}s")   # TO REMOVE
                else:
                    drift = None
                    print("  [debug] derived actual time unavailable")

                ingame_time = self.extract_time(frame_path) if frame_path else None

                #requested_label = "3:40" if self.target_timestamp == 220 else str(self.target_timestamp)
                #requested_label = "6:30" if self.target_timestamp == 390 else str(self.target_timestamp)
                requested_label = "7:30" if self.target_timestamp == 450 else str(self.target_timestamp)
                #requested_label = "8:30" if self.target_timestamp == 510 else str(self.target_timestamp)
                #requested_label = "15:00" if self.target_timestamp == 900 else str(self.target_timestamp)

                if ingame_time:
                    m, s = map(int, ingame_time.split(":"))
                    ingame_s = m * 60 + s

                    base = actual_s if actual_s is not None else float(self.target_timestamp)
                    diff = int(round(base - ingame_s))

                    writer.writerow([
                        url,
                        requested_label,
                        f"{actual_s:.3f}" if actual_s is not None else "",
                        f"{drift:+.3f}" if drift is not None else "",
                        ingame_time,
                        diff
                    ])
                    print(f"  Success: ingame={ingame_time} diff={diff}s")
                else:
                    writer.writerow([
                        url,
                        requested_label,
                        f"{actual_s:.3f}" if actual_s is not None else "",
                        f"{drift:+.3f}" if drift is not None else "",
                        "",
                        ""
                    ])
                    print("  Failed to OCR in-game timestamp.")

                if frame_path:
                    print(f"  Saved frame to: {frame_path}")

        print(f"\nBulk processing complete. Results saved to {self.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str, help='Tournament folder (e.g., "WC 2020")')
    args = parser.parse_args()

    extractor = BulkLoLExtractor(
        tournament_folder=args.folder,
        output_csv="timestamps.csv",
        target_timestamp=450, #220/390/450/510/900
        debug_visuals=False
    )
    extractor.process_file()

