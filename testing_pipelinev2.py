
import os
import sys
import json
import re
import requests
import pathlib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
from tqdm import tqdm
from PIL import Image

# Core ML/NLP Libraries
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    AutoTokenizer, RobertaModel, AutoModelForSequenceClassification, AutoConfig, Trainer
)
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from datasets import Dataset
from torch.utils.data import DataLoader

# Scraper (assuming ntscraper is in the environment)
from ntscraper import Nitter
import ntscraper.nitter as ntr


class TwitterMentalHealthPipeline:
    """
    End-to-end pipeline for scraping a new Twitter user, processing their data
    with advanced behavioral features, and running predictions with trained models.
    """
    def __init__(self,
                 base_dir: str = "pipeline_data",
                 nitter_instance: str = "http://localhost:8080",
                 min_tweets: int = 50,
                 baseline_frac: float = 0.40):
        # (Initialization is largely the same, just adding model paths)
        self.base_dir = pathlib.Path(base_dir)
        self.nitter_instance = nitter_instance
        self.min_tweets = min_tweets
        self.baseline_frac = baseline_frac
        
        self.data_dir = self.base_dir / "data"
        self.images_dir = self.base_dir / "images"
        self.captions_dir = self.base_dir / "captions"
        self.parquet_dir = self.base_dir / "parquets"
        for dir_path in [self.data_dir, self.images_dir, self.captions_dir, self.parquet_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self._blip_processor = None
        self._blip_model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sia = SentimentIntensityAnalyzer()
        self._apply_nitter_patch()

    # (Helper properties and patch are the same)
    def _apply_nitter_patch(self):
        _orig_get_user = ntr.Nitter._get_user
        def _safe_get_user(self, tweet, is_encrypted):
            try: return _orig_get_user(self, tweet, is_encrypted)
            except IndexError:
                uname = tweet.find("a", class_="username"); fname = tweet.find("a", class_="fullname")
                return {"id": None, "username": uname.text.lstrip("@") if uname else "unknown", "fullname": fname.text if fname else "unknown", "avatar_url": None}
        ntr.Nitter._get_user = _safe_get_user

    @property
    def blip_processor(self):
        if self._blip_processor is None: self._blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        return self._blip_processor

    @property
    def blip_model(self):
        if self._blip_model is None: self._blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self._device)
        return self._blip_model

    # (scrape_user and caption_images are the same)
    def scrape_user(self, username: str, max_tweets: int = -1) -> Dict:
        # ... (function body is identical to your original script)
        username = username.lstrip("@"); user_dir = self.images_dir / username; user_dir.mkdir(parents=True, exist_ok=True)
        print(f"▸ Scraping timeline for @{username}")
        scraper = Nitter(log_level=1, skip_instance_check=True)
        try:
            result = scraper.get_tweets(username, mode="user", number=max_tweets, instance=self.nitter_instance)
            timeline = result.get("tweets", [])
        except Exception as e: print(f"  • Error fetching tweets: {e}"); raise
        print(f"  • {len(timeline)} tweets fetched")
        json_path = self.data_dir / f"{username}_raw.json"
        with open(json_path, 'w', encoding='utf-8') as f: json.dump(timeline, f, ensure_ascii=False, indent=2)
        print(f"  • Timeline written to {json_path}")
        img_urls, tweet_images = [], {}
        for tw in timeline:
            tweet_id = tw.get("id"); tweet_images[tweet_id] = []
            for url in tw.get("photos", []):
                if url not in img_urls: img_urls.append(url); tweet_images[tweet_id].append(len(img_urls) - 1)
        print(f"  • {len(img_urls)} unique images to download")
        sess = requests.Session(); downloaded_images = {}
        for idx, url in enumerate(img_urls):
            ext = pathlib.Path(url).suffix.split("?")[0] or ".jpg"; fname = user_dir / f"{idx:04d}{ext}"
            try:
                with sess.get(url, timeout=30, stream=True) as r:
                    r.raise_for_status()
                    with open(fname, "wb") as f:
                        for chunk in r.iter_content(chunk_size=65536): f.write(chunk)
                downloaded_images[idx] = str(fname)
            except Exception as e: print(f"    ✗ Failed {url} → {e}"); downloaded_images[idx] = None
        return {"username": username, "tweets": timeline, "tweet_images": tweet_images, "downloaded_images": downloaded_images, "json_path": str(json_path)}

    def caption_images(self, scrape_data: Dict) -> Dict:
        # ... (function body is identical to your original script)
        username = scrape_data["username"]; tweet_images = scrape_data["tweet_images"]; downloaded_images = scrape_data["downloaded_images"]
        print(f"\n▸ Generating image captions for @{username}")
        image_captions = {}
        for img_idx, img_path in tqdm(downloaded_images.items(), desc="Captioning"):
            if img_path is None: continue
            caption_file = self.captions_dir / f"{username}_{img_idx:04d}.txt"
            if caption_file.exists():
                try: image_captions[img_idx] = caption_file.read_text(encoding='utf-8').strip()
                except UnicodeDecodeError: image_captions[img_idx] = caption_file.read_text(encoding='utf-8', errors='ignore').strip()
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                inputs = self.blip_processor(images=img, return_tensors="pt").to(self._device)
                out = self.blip_model.generate(**inputs, max_new_tokens=20)
                caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
                caption_file.write_text(caption, encoding='utf-8'); image_captions[img_idx] = caption
            except Exception as e: print(f"Failed to caption {img_path}: {e}"); image_captions[img_idx] = ""
        tweet_captions = {}
        for tweet_id, img_indices in tweet_images.items():
            captions = [image_captions[idx] for idx in img_indices if idx in image_captions and image_captions[idx]]
            tweet_captions[tweet_id] = " ".join(captions)
        return tweet_captions
    
    def clean_text(self, text: str) -> str:
        return re.sub(r"http\S+|@\w+", "", text).strip()
    
    def parse_date(self, s: str) -> pd.Timestamp:
        s = s.replace(" UTC", "").replace(" · ", " ")
        return pd.to_datetime(s, errors='coerce')

    def compute_features(self, username: str, tweets: List[Dict],
                         tweet_captions: Dict, label: int = 0) -> pd.DataFrame:
        """
        --- UPDATED: This now performs the full feature engineering from merge_blip_v4.py ---
        """
        print(f"\n▸ Computing advanced features for @{username}")
        
        # Create initial dataframe
        rows = []
        for tw in tweets:
            try:
                created = self.parse_date(tw["date"])
                if pd.isna(created): continue
                text = self.clean_text(tw.get("text", ""))
                rows.append({"tweet_id": tw["id"], "user_id": tw["user"]["profile_id"], "created": created, "text": text, "label": label})
            except (KeyError, TypeError): continue
        
        df = pd.DataFrame(rows).sort_values("created").reset_index(drop=True)

        if len(df) < self.min_tweets:
            print(f"Warning: User has only {len(df)} tweets (minimum: {self.min_tweets}). Cannot process.")
            return pd.DataFrame()

        # This is the full feature engineering function adapted from merge_blip_v4
        return self._process_user_timeline_with_all_features(df, tweet_captions)

    def _process_user_timeline_with_all_features(self, group, tweet_captions):
        # (This function is a direct copy of the one in merge_blip.py)
        n_base = max(10, int(len(group) * self.baseline_frac))
        baseline_df = group.head(n_base)
        later_df = group.iloc[n_base:].copy()
        if later_df.empty: return pd.DataFrame()
        base_text_lens = baseline_df['text'].str.len().fillna(0)
        base_sentiments = [self.sia.polarity_scores(t) for t in baseline_df['text']]
        base_sent_df = pd.DataFrame(base_sentiments)
        days_in_baseline = (baseline_df['created'].max() - baseline_df['created'].min()).days or 1
        base_hours = baseline_df['created'].dt.hour
        baseline_stats = {
            'len_mu': base_text_lens.mean(), 'len_std': base_text_lens.std() + 1e-8,
            'sent_compound_mu': base_sent_df['compound'].mean(), 'sent_compound_std': base_sent_df['compound'].std() + 1e-8,
            'tpd_mu': len(baseline_df) / days_in_baseline,
            'hour_median': base_hours.median(), 'hour_std': base_hours.std() + 1e-8,
        }
        all_tweet_features = []
        for i in range(len(later_df)):
            current_tweet = later_df.iloc[i]
            history_df = group.head(n_base + i + 1)
            features = {}
            original_text = current_tweet['text']
            caption_text = tweet_captions.get(current_tweet['tweet_id'], "")
            combined_text = (original_text + " [IMG_CAP] " + caption_text)[:4096]
            features['text'] = combined_text
            features['len_val'] = len(combined_text)
            current_sentiment = self.sia.polarity_scores(combined_text)
            features.update({f'sent_{k}': v for k, v in current_sentiment.items()})
            features['len_dev'] = features['len_val'] - baseline_stats['len_mu']
            features['len_z'] = features['len_dev'] / baseline_stats['len_std']
            features['sent_compound_dev'] = features['sent_compound'] - baseline_stats['sent_compound_mu']
            features['sent_compound_z'] = features['sent_compound_dev'] / baseline_stats['sent_compound_std']
            features['hour'] = current_tweet['created'].hour
            features['dow'] = current_tweet['created'].dayofweek
            features['circadian_deviation'] = (features['hour'] - baseline_stats['hour_median']) / baseline_stats['hour_std']
            features['late_night'] = float((features['hour'] >= 23) | (features['hour'] <= 3))
            interval = (current_tweet['created'] - history_df.iloc[-2]['created']).total_seconds() / 3600 if len(history_df) > 1 else 24
            features['burst_posting'] = float(interval < 1)
            features['silence_period'] = float(interval > 72)
            for days in [3, 7, 14, 30]:
                cutoff = current_tweet['created'] - pd.Timedelta(days=days)
                features[f'tpd_{days}d'] = len(history_df[history_df['created'] > cutoff]) / float(days)
                features[f'tpd_{days}d_dev'] = features[f'tpd_{days}d'] - baseline_stats['tpd_mu']
            recent_history = history_df.tail(10)
            recent_sents = [self.sia.polarity_scores(t) for t in recent_history['text']]
            recent_compounds = [s['compound'] for s in recent_sents]
            features['emotion_volatility_10tw'] = np.std(recent_compounds) if len(recent_compounds) > 1 else 0
            features['polarity_shift'] = float(features['sent_compound'] * recent_compounds[-2] < 0) if len(recent_compounds) > 1 else 0
            tokens = combined_text.split()
            features['self_focus'] = sum(word in ['i', 'me', 'my', 'myself'] for word in tokens) / (len(tokens) + 1e-8)
            features['exclamation_count'] = combined_text.count('!')
            features['mention_count'] = combined_text.count('@')
            features['isolation_score'] = sum(word in ['alone', 'lonely', 'nobody'] for word in tokens)
            features['anomaly_score'] = abs(features['len_z']) + abs(features['sent_compound_z']) + abs(features['circadian_deviation'])
            features['manic_indicator'] = float(features['burst_posting'] and features['sent_compound'] > 0.5 and features['late_night'])
            features['depressive_indicator'] = float(features['silence_period'] and features['sent_compound'] < -0.3)
            features['tweet_id'] = current_tweet['tweet_id']
            features['user_id'] = current_tweet['user_id']
            features['created'] = current_tweet['created']
            features['label'] = current_tweet['label']
            all_tweet_features.append(features)
        return pd.DataFrame(all_tweet_features)

    def process_user(self, username: str, label: int = 0, max_tweets: int = -1) -> str:
        # (This function is the same as before)
        username = username.lstrip("@"); print(f"\n{'='*60}\nProcessing user: @{username}\n{'='*60}")
        scrape_data = self.scrape_user(username, max_tweets)
        tweet_captions = self.caption_images(scrape_data)
        df = self.compute_features(username, scrape_data["tweets"], tweet_captions, label)
        if df.empty:
            print(f"✗ Pipeline halted for @{username} due to insufficient data.")
            return ""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parquet_path = self.parquet_dir / f"{username}_{timestamp}.parquet"
        df.to_parquet(parquet_path, index=False)
        print(f"\n✓ Pipeline complete! Processed {len(df)} tweets. Saved to: {parquet_path}")
        return str(parquet_path)

    def predict_user(self, parquet_path: str, baseline_dir: str, delta_dir: str) -> pd.DataFrame:
        """
        --- NEW: Loads trained models and predicts on a processed user file. ---
        """
        if not parquet_path:
            print("Parquet path is empty, skipping prediction.")
            return pd.DataFrame()
            
        print("\n--- Running Prediction on New User ---")
        
        # Load the models
        baseline_model = AutoModelForSequenceClassification.from_pretrained(baseline_dir).to(self._device)
        baseline_model.eval()
        
        delta_config = AutoConfig.from_pretrained(delta_dir)
        num_features_trained = delta_config.custom_num_features
        delta_model = AdvancedDeltaModel(config=delta_config).to(self._device)
        weights_path = os.path.join(delta_dir, "model.safetensors")
        if not os.path.exists(weights_path): weights_path = os.path.join(delta_dir, "pytorch_model.bin")
        from safetensors.torch import load_file
        state_dict = load_file(weights_path, device=str(self._device)) if weights_path.endswith(".safetensors") else torch.load(weights_path, map_location=self._device)
        delta_model.load_state_dict(state_dict)
        delta_model.eval()

        # Prepare data
        new_user_df = pd.read_parquet(parquet_path)
        tokenizer = AutoTokenizer.from_pretrained(delta_config._name_or_path)

        # We need the same feature list that the delta model was trained on
        # To get this, we need to load the original training data to run selection
        # Note: In a production system, this feature list would be saved with the model
        training_df = pd.read_parquet("data/deviation_advanced_final.parquet")
        selected_features = self._get_feature_list(training_df, n_features=num_features_trained)
        
        # Make sure the new user df has all the required feature columns
        for feat in selected_features:
            if feat not in new_user_df.columns:
                new_user_df[feat] = 0.0 # Add missing feature with a neutral value
        
        # Baseline prediction
        def tok_only(batch): return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
        plain_ds = Dataset.from_pandas(new_user_df).map(tok_only, batched=True, remove_columns=new_user_df.columns.tolist())
        plain_ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        plain_loader = DataLoader(plain_ds, batch_size=32)
        
        baseline_logits = []
        with torch.no_grad():
            for batch in tqdm(plain_loader, desc="Predicting with Baseline"):
                batch = {k: v.to(self._device) for k, v in batch.items()}
                logits = baseline_model(**batch).logits
                baseline_logits.append(logits.cpu().numpy())
        
        new_user_df['baseline_pred'] = np.concatenate(baseline_logits).argmax(axis=1)

        # Delta prediction
        def tok_plus(batch):
            encoding = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
            encoding["delta"] = [[float(batch[c][i]) for c in selected_features] for i in range(len(batch["text"]))]
            return encoding
        delta_ds = Dataset.from_pandas(new_user_df).map(tok_plus, batched=True, remove_columns=new_user_df.columns.tolist())
        delta_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'delta'])
        delta_loader = DataLoader(delta_ds, batch_size=32)

        delta_logits = []
        with torch.no_grad():
            for batch in tqdm(delta_loader, desc="Predicting with Delta"):
                batch = {k: v.to(self._device) for k, v in batch.items()}
                logits = delta_model(**batch)['logits']
                delta_logits.append(logits.cpu().numpy())
        
        new_user_df['delta_pred'] = np.concatenate(delta_logits).argmax(axis=1)

        # Add readable labels
        id2label = delta_config.id2label
        new_user_df['baseline_label'] = new_user_df['baseline_pred'].map(id2label)
        new_user_df['delta_label'] = new_user_df['delta_pred'].map(id2label)

        return new_user_df[['text', 'label', 'baseline_label', 'delta_label']]
    
    def _get_feature_list(self, df, n_features):
        exclude_cols = ['tweet_id', 'user_id', 'text', 'label', 'label_name', 'split', 'created']
        feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64']]
        if not feature_cols: return []
        X = df[feature_cols].fillna(0).values; y = df['label'].values
        mi_scores = mutual_info_classif(X, y, random_state=42)
        feature_scores = pd.DataFrame({'feature': feature_cols, 'score': mi_scores}).sort_values('score', ascending=False)
        return feature_scores.head(n_features)['feature'].tolist()


# --- This requires a separate file for the model class ---
# In a new file, e.g., `model_def.py`
# from transformers import RobertaModel
# class AdvancedDeltaModel(nn.Module):
#     ... (full class definition)

if __name__ == '__main__':
    # This is an example of how to run the full pipeline from the command line
    
    # --- CONFIGURATION ---
    TARGET_USERNAME = "anxietytxtmsgs"
    EXPECTED_LABEL = 2 # 0=control, 1=depression, 2=anxiety, 3=bipolar
    MAX_TWEETS_TO_SCRAPE = 500
    BASELINE_MODEL_DIR = "out/baseline_final"
    DELTA_MODEL_DIR = "out/delta_final_v3"
    
    # --- EXECUTION ---
    pipeline = TwitterMentalHealthPipeline(base_dir="live_demo_data")
    
    # 1. Process the user and create the feature parquet
    user_parquet_path = pipeline.process_user(
        TARGET_USERNAME, 
        label=EXPECTED_LABEL, 
        max_tweets=MAX_TWEETS_TO_SCRAPE
    )
    
    # 2. Run predictions on the generated parquet
    if user_parquet_path:
        # We need the model class definition to be available
        from train_delta import AdvancedDeltaModel # Assumes you've created this file
        
        predictions_df = pipeline.predict_user(
            user_parquet_path,
            baseline_dir=BASELINE_MODEL_DIR,
            delta_dir=DELTA_MODEL_DIR
        )
        
        print("\n--- Prediction Results ---")
        print(predictions_df.head())
        
        print("\nBaseline Model Predictions:")
        print(predictions_df['baseline_label'].value_counts())

        print("\nDelta Model Predictions:")
        print(predictions_df['delta_label'].value_counts())