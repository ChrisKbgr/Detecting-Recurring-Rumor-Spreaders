import os
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from dateutil.parser import parse as parse_date  # More robust datetime parsing
import logging

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Config ---
RUMOR_RATIO_THRESHOLD = 0.6
PKL_PATH = "pheme_threads.pkl"
DATASET_PATH = "./pheme_dataset/all-rnr-annotated-threads"
MULTI_EVENT_CSV = "pheme_user_features_multi_event.csv"
ALL_USERS_CSV = "pheme_user_features.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# --- BERT Sentiment Setup ---
class BertSentimentAnalyzer:
    """Sentiment analyzer using BERT. Caches results for speed."""
    def __init__(self, model_name="nlptown/bert-base-multilingual-uncased-sentiment"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.cache = {}

    @torch.no_grad()
    def predict_sentiment(self, text):
        if text in self.cache:
            return self.cache[text]
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        scores = torch.arange(1, 6, device=self.device)
        sentiment = (probs * scores).sum().item()
        norm_sentiment = 2 * (sentiment - 1) / 4 - 1
        self.cache[text] = norm_sentiment
        return norm_sentiment

# --- Data Loading ---
def load_pheme_threads(base_path):
    """Load threads from PHEME dataset."""
    threads = []
    for label in ['rumours', 'non-rumours']:
        label_dir = os.path.join(base_path, label)
        for thread_dir in tqdm(glob(os.path.join(label_dir, '*')), desc=f"{label}"):
            try:
                src_dir = os.path.join(thread_dir, 'source-tweets')
                src_files = [f for f in os.listdir(src_dir) if f.endswith('.json')]
                if not src_files:
                    continue
                with open(os.path.join(src_dir, src_files[0]), encoding='utf-8') as f:
                    source = json.load(f)
                reactions = []
                react_dir = os.path.join(thread_dir, 'reactions')
                if os.path.exists(react_dir):
                    for fname in os.listdir(react_dir):
                        if fname.endswith('.json'):
                            with open(os.path.join(react_dir, fname), encoding='utf-8') as f:
                                try:
                                    reactions.append(json.load(f))
                                except json.JSONDecodeError:
                                    logging.warning(f"JSON decode error in {fname}")
                                    continue
                threads.append({
                    'label': label,
                    'source': source,
                    'reactions': reactions,
                    'event': os.path.basename(base_path)
                })
            except Exception as e:
                logging.error(f"Error in {thread_dir}: {e}")
    return threads

# --- Feature Extraction ---
def extract_user_features(threads, sentiment_analyzer):
    """Extract user-level features from threads."""
    user_stats = defaultdict(lambda: {
        'rumor_tweets': 0, 'non_rumor_tweets': 0, 'total_tweets': 0,
        'events': set(), 'avg_tweet_length': 0, 'avg_sentiment': 0,
        'response_times': [], 'engagement_rate': 0,
        'first_activity': None, 'last_activity': None,
        'neighbors': set()
    })
    for thread in tqdm(threads, desc="Processing threads"):
        event, label = thread['event'], thread['label']
        src = thread['source']
        try:
            src_time = parse_date(src['created_at'])
        except Exception:
            logging.warning(f"Failed to parse date: {src['created_at']}")
            continue
        _update_user(src, event, label, src_time, user_stats, sentiment_analyzer, is_source=True)
        src_uid = src['user']['id_str']
        for react in thread['reactions']:
            try:
                react_time = parse_date(react['created_at'])
            except Exception:
                logging.warning(f"Failed to parse date: {react['created_at']}")
                continue
            resp_time = (react_time - src_time).total_seconds()
            _update_user(react, event, label, src_time, user_stats, sentiment_analyzer, is_source=False, response_time=resp_time)
            react_uid = react['user']['id_str']
            user_stats[src_uid]['neighbors'].add(react_uid)
            user_stats[react_uid]['neighbors'].add(src_uid)
    _finalize_user_stats(user_stats)
    for uid, stats in user_stats.items():
        stats['events'] = list(stats['events'])
        stats['neighbors'] = list(stats['neighbors'])
    return user_stats

def _update_user(tweet, event, label, src_time, user_stats, sentiment_analyzer, is_source=False, response_time=None):
    """Update user statistics for a tweet."""
    uid = tweet['user']['id_str']
    stats = user_stats[uid]
    stats['total_tweets'] += 1
    stats['events'].add(event)
    if label == 'rumours':
        stats['rumor_tweets'] += 1
    else:
        stats['non_rumor_tweets'] += 1
    text = tweet['text']
    sentiment = sentiment_analyzer.predict_sentiment(text)
    n = stats['total_tweets']
    stats['avg_sentiment'] = ((stats['avg_sentiment'] * (n-1)) + sentiment) / n
    stats['avg_tweet_length'] = ((stats['avg_tweet_length'] * (n-1)) + len(text)) / n
    engagement = (tweet.get('retweet_count', 0) + tweet.get('favorite_count', 0)) / 2
    stats['engagement_rate'] = ((stats['engagement_rate'] * (n-1)) + engagement) / n
    try:
        t_time = parse_date(tweet['created_at'])
    except Exception:
        t_time = None
    if t_time:
        stats['first_activity'] = min(stats['first_activity'], t_time) if stats['first_activity'] else t_time
        stats['last_activity'] = max(stats['last_activity'], t_time) if stats['last_activity'] else t_time
    if not is_source and response_time is not None:
        stats['response_times'].append(response_time)

def _finalize_user_stats(user_stats):
    """Finalize user statistics after all updates."""
    for uid, stats in user_stats.items():
        try:
            if stats['first_activity'] and stats['last_activity']:
                dur = (stats['last_activity'] - stats['first_activity']).total_seconds()
                stats['activity_duration'] = dur
                stats['tweets_per_day'] = stats['total_tweets'] / (dur / 86400) if dur > 0 else stats['total_tweets']
            else:
                stats['activity_duration'] = 0
                stats['tweets_per_day'] = 0
            stats['degree'] = len(stats['neighbors'])
            nbs = stats['neighbors']
            if len(nbs) < 2:
                stats['clustering'] = 0.0
            else:
                links = sum(1 for u, v in itertools.combinations(nbs, 2) if u in user_stats and v in user_stats[u]['neighbors'])
                possible = len(nbs) * (len(nbs) - 1) / 2
                stats['clustering'] = links / possible if possible > 0 else 0.0
            if stats['response_times']:
                stats['avg_response_time'] = np.mean(stats['response_times'])
                stats['std_response_time'] = np.std(stats['response_times'])
            else:
                stats['avg_response_time'] = 0
                stats['std_response_time'] = 0
            stats['rumor_ratio'] = stats['rumor_tweets'] / stats['total_tweets'] if stats['total_tweets'] > 0 else 0
        except Exception as e:
            logging.error(f"Stats error for user {uid}: {e}")

# --- Analysis & Visualization ---
def describe_dataset(df, label):
    """Return a summary DataFrame for the user dataset."""
    summary = {
        "Group": label,
        "Users": len(df),
        "Tweets": df['total_tweets'].sum(),
        "Events": len(set(itertools.chain(*df['events']))),
        "Multi-event users": len(df[df['events'].apply(len) > 1]),
        "Multi-event %": len(df[df['events'].apply(len) > 1]) / len(df) if len(df) > 0 else 0,
        "Rumor-only": len(df[df['rumor_ratio'] == 1.0]),
        "Non-rumor-only": len(df[df['rumor_ratio'] == 0.0]),
        "Mixed": len(df[(df['rumor_ratio'] > 0) & (df['rumor_ratio'] < 1)]),
        "Avg. response time (s)": df['avg_response_time'].mean(),
        "Median response time (s)": df['avg_response_time'].median(),
        "Avg. activity duration (days)": df['activity_duration'].mean() / (24 * 3600),
        "Avg. engagement": df['engagement_rate'].mean(),
        "Avg. tweets/user": df['total_tweets'].mean(),
        "Rumor tweets": df['rumor_tweets'].sum(),
        "Non-rumor tweets": df['non_rumor_tweets'].sum(),
        "Rumor tweet ratio": df['rumor_tweets'].sum() / (df['rumor_tweets'].sum() + df['non_rumor_tweets'].sum()) if (df['rumor_tweets'].sum() + df['non_rumor_tweets'].sum()) > 0 else 0,
    }
    return pd.DataFrame([summary])

def plot_rumor_ratio_distribution(df, title):
    """Plot the distribution of rumor ratios."""
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(7,5))
    sns.histplot(df['rumor_ratio'], bins=20)
    plt.title('Rumor Ratio Distribution - ' + title)
    plt.xlabel('Rumor Ratio')
    plt.ylabel('Number of Users')
    plt.tight_layout()
    plt.show()

def main():
    # --- Load or Extract Threads ---
    if os.path.exists(PKL_PATH):
        with open(PKL_PATH, 'rb') as f:
            threads = pickle.load(f)
        logging.info(f"Loaded threads from {PKL_PATH}")
    else:
        threads = []
        for event_dir in os.listdir(DATASET_PATH):
            full_path = os.path.join(DATASET_PATH, event_dir)
            if os.path.isdir(full_path) and not event_dir.startswith('.'):
                threads.extend(load_pheme_threads(full_path))
        with open(PKL_PATH, "wb") as f:
            pickle.dump(threads, f)
        logging.info(f"Extracted and saved threads to {PKL_PATH}")

    # --- Feature Extraction ---
    sentiment_analyzer = BertSentimentAnalyzer()
    user_stats = extract_user_features(threads, sentiment_analyzer)
    df = pd.DataFrame.from_dict(user_stats, orient='index')

    # --- Feature Selection ---
    features = [
        'total_tweets', 'avg_tweet_length', 'avg_sentiment',
        'avg_response_time', 'activity_duration', 'engagement_rate',
        'degree', 'clustering', 'tweets_per_day'
    ]
    X = df[features]
    y = ((df['rumor_ratio'] > RUMOR_RATIO_THRESHOLD) & (df['events'].apply(len) > 1)).astype(int)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    selector = SelectKBest(f_classif, k=5)
    X_selected = selector.fit_transform(X_scaled, y)
    selected_feature_indices = selector.get_support(indices=True)
    selected_features = [features[i] for i in selected_feature_indices]
    logging.info(f"Selected features: {selected_features}")

    df_selected = df[selected_features + [
        'rumor_ratio', 'events', 'neighbors', 'rumor_tweets', 'non_rumor_tweets', 'total_tweets'
    ]].copy()

    for feat, score in zip(features, selector.scores_):
        logging.info(f"{feat}: {score:.3f}")

    logging.info("Correlation between avg_sentiment and rumor_ratio:")
    logging.info(df[['avg_sentiment', 'rumor_ratio']].corr())

    # --- Save CSVs ---
    multi_event_df = df_selected[df_selected['events'].apply(lambda x: len(x) > 1)].copy()
    multi_event_df['neighbors'] = multi_event_df['neighbors'].apply(json.dumps)
    multi_event_df['events'] = multi_event_df['events'].apply(json.dumps)
    multi_event_df.to_csv(MULTI_EVENT_CSV)
    logging.info(f"Saved multi-event user features to {MULTI_EVENT_CSV}")

    df_selected['neighbors'] = df_selected['neighbors'].apply(json.dumps)
    df_selected['events'] = df_selected['events'].apply(json.dumps)
    df_selected.to_csv(ALL_USERS_CSV)
    logging.info(f"Saved all user features to {ALL_USERS_CSV}")

    # --- Visualization ---
    plot_rumor_ratio_distribution(df_selected, "All Users")
    plot_rumor_ratio_distribution(multi_event_df, "Multi-Event Users")

    # --- Analysis Table ---
    analysis_all = describe_dataset(df, "All Users")
    analysis_multi = describe_dataset(df[df['events'].apply(len) > 1], "Multi-Event Users")
    analysis_table = pd.concat([analysis_all, analysis_multi], ignore_index=True)
    print("\n=== Analysis Table ===")
    print(analysis_table)

if __name__ == "__main__":
    main()