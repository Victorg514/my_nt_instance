# debug_text_length.py

import json
import pandas as pd
import re
import pathlib

# --- Functions from merge_blip.py ---
CAP_DIR = pathlib.Path("data/captions")

def clean(txt: str) -> str:
    return re.sub(r"http\S+|@\w+", "", str(txt)).strip()

def caption_for(tid: str) -> str:
    caps = []
    for idx in "abcde":
        fp = CAP_DIR / f"{tid}_{idx}.txt"
        if fp.exists():
            try:
                # Read the caption file content
                caption_content = fp.read_text(encoding="utf-8", errors="ignore").strip()
                caps.append(caption_content)
            except Exception as e:
                print(f"Error reading caption for {tid}: {e}")
                caps.append("")
    return " ".join(caps)

def parse_date(s: str) -> pd.Timestamp:
    s = s.replace(" UTC", "").replace(" · ", " ")
    return pd.to_datetime(s, errors="coerce")

# --- Main Debug Logic ---
print("="*70)
print("RUNNING MINIMAL TEXT LENGTH DIAGNOSTIC")
print("="*70)

# We only need to load one file to find the error
file_to_check = "data/user_timelines.json" # Or any of the source files

with open(file_to_check, "r", encoding="utf-8") as f:
    data = json.load(f)

found_error = False
for username, tweets in data.items():
    if found_error:
        break
    for tw in tweets:
        try:
            # 1. Get the original tweet text
            original_text = clean(tw.get("text", ""))

            # 2. Get the caption text
            caption_text = caption_for(tw["id"])

            # 3. Combine them, just like in the main script
            combined_text = original_text + " [IMG_CAP] " + caption_text

            # 4. Check the length
            length = len(combined_text)

            if length > 2000: # Set a high threshold to find the problem
                print(f"\n--- !!! Found abnormally long text !!! ---")
                print(f"Username: {username}")
                print(f"Tweet ID: {tw['id']}")
                print(f"Original Text Length: {len(original_text)}")
                print(f"Caption Text Length: {len(caption_text)}")
                print(f"--> Combined Text Length: {length}")
                print("\nThis indicates a problem with the caption file for this tweet ID.")
                found_error = True
                break

        except Exception:
            continue

if not found_error:
    print("\n--- Diagnostic Complete ---")
    print("No single tweet/caption combination is abnormally long.")
    print("This suggests the error is in the timeline aggregation logic, not the captions.")

print("="*70)