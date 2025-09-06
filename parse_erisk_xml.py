# parse_erisk_xml.py

import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm
import pathlib

def parse_erisk_data(xml_path: str, labels_path: str) -> pd.DataFrame:
    print("--- Parsing eRisk Data ---")
    label_dict = {}
    with open(labels_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                label_dict[parts[0]] = int(parts[1])
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    all_posts = []
    for individual in tqdm(root.findall('INDIVIDUAL'), desc="Processing Users from XML"):
        user_id = individual.get('ID')
        if user_id in label_dict:
            for writing in individual.findall('WRITING'):
                try:
                    title = (writing.find('TITLE').text or "").strip()
                    text = (writing.find('TEXT').text or "").strip()
                    full_text = (title + " " + text).strip()
                    if not full_text: continue
                    all_posts.append({
                        'user_id': user_id,
                        'created': pd.to_datetime(writing.find('DATE').text, errors='coerce'),
                        'text': full_text,
                        'label': label_dict[user_id]
                    })
                except (AttributeError, TypeError):
                    continue

    if not all_posts: raise ValueError("No posts were parsed. Check file paths.")

    df = pd.DataFrame(all_posts).dropna(subset=['created']).sort_values(['user_id', 'created']).reset_index(drop=True)
    df['label_name'] = df['label'].apply(lambda x: 'control' if x == 0 else 'depression')
    return df

if __name__ == '__main__':
    XML_FILE_PATH = 'path/to/your/erisk-2018-t1-test-data.xml' # IMPORTANT: UPDATE THIS
    LABELS_FILE_PATH = 'path/to/your/risk-golden-truth-test.txt' # IMPORTANT: UPDATE THIS
    OUTPUT_PARQUET_PATH = pathlib.Path("data/erisk_parsed_raw.parquet")
    
    OUTPUT_PARQUET_PATH.parent.mkdir(exist_ok=True)
    erisk_df = parse_erisk_data(XML_FILE_PATH, LABELS_FILE_PATH)
    erisk_df.to_parquet(OUTPUT_PARQUET_PATH, index=False)
    print(f"\nSaved clean, raw eRisk data to: {OUTPUT_PARQUET_PATH}")