#!/usr/bin/env python
# 03_merge_ocr_with_tweets.py
#
# Combine tweet text with any OCR-extracted text from attached images.
# Writes a single Parquet file ready for Hugging Face / PyTorch.

import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm  # progress bar

# ---------------------------------------------------------------------
# --- paths -----------------------------------------------------------
SRC     = Path("data/user_timelinesA.json")      # raw tweets you collected
OCR_DIR = Path("data/ocr")                     # OCR snippets live here
OUTFILE = Path("data/final_datasetA.parquet")

assert SRC.exists(),     f"Tweets file not found: {SRC}"
assert OCR_DIR.exists(), f"OCR folder not found: {OCR_DIR}"

# ---------------------------------------------------------------------
# --- helpers ---------------------------------------------------------
def gather_ocr_snippets(tid, max_imgs: int = 6) -> str:
    """Concatenate OCR text files that belong to the same tweet ID."""
    blobs = []
    for idx in range(max_imgs):
        f = OCR_DIR / f"{tid}_{chr(97 + idx)}.txt"
        if f.exists():
            raw = f.read_bytes()
            txt = raw.decode("utf-8", errors="ignore").strip().replace("\n", " ")
            if txt:
                blobs.append(txt)
    return " ".join(blobs)

def get_created_at(tw: dict) -> str:
    """
    Try common keys for the tweet’s timestamp and return
    an ISO-8601 string; fall back to empty string if missing.
    """
    for key in ("created_at", "timestamp_ms", "date", "time"):
        if key in tw:
            return str(tw[key])
    return ""

# ---------------------------------------------------------------------
# --- main merge loop -------------------------------------------------
rows = []
with SRC.open(encoding="utf-8") as f:
    master_dict = json.load(f)  # top-level {handle: [tweets]}

for handle, tweet_list in tqdm(master_dict.items(), desc="Users"):
    for tw in tweet_list:
        tid = str(tw["id"])

        # ---------- safer user-ID pull ----------
        user_block = tw.get("user", {})
        uid = (
            user_block.get("profile_id") or
            user_block.get("id") or
            user_block.get("user_id") or
            handle                       # fallback to @handle
        )
        uid = str(uid)
        # ----------------------------------------

        text        = tw["text"]
        label       = tw.get("label", 0)
        created_at  = get_created_at(tw)

        ocr_text = gather_ocr_snippets(tid)
        full_text = f"{text} [IMG_TEXT] {ocr_text}" if ocr_text else text

        rows.append(
            {
                "tweet_id"  : tid,
                "user_id"   : uid,
                "created_at": created_at,   # <-- new column
                "text"      : full_text,
                "label"     : label,
            }
        )

df = pd.DataFrame(rows)
OUTFILE.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUTFILE, index=False)
print(f"Saved {len(df):,} rows to {OUTFILE}")
