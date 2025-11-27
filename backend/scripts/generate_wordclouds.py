"""
Generate word clouds for different Reddit eras.
Creates word clouds for: 2006-2010, 2011-2017, 2018-2024
"""

import os
import pandas as pd
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

# Ensure output directory exists
os.makedirs("frontend/public", exist_ok=True)

# Common stop words and Reddit-specific terms to filter
STOP_WORDS = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with', 
    'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 
    'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 
    'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 
    'no', 'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 
    'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 
    'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even', 
    'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us', 'is', 'are', 'was', 
    'were', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 
    'could', 'may', 'might', 'must', 'can', 'cannot', 'don', 'doesn', 'isn', 'aren', 'wasn', 'weren',
    'haven', 'hasn', 'hadn', 'won', 'wouldn', 'shouldn', 'couldn', 'mayn', 'mightn', 'mustn', 'cant',
    'dont', 'doesnt', 'isnt', 'arent', 'wasnt', 'werent', 'havent', 'hasnt', 'hadnt', 'wont', 
    'wouldnt', 'shouldnt', 'couldnt', 'maynt', 'mightnt', 'mustnt', 'cant'
}

# Reddit bot/moderation terms to filter
REDDIT_BOT_TERMS = {
    'bot', 'removed', 'moderator', 'subreddit', 'message', 'performed', 'automatically', 'action', 
    'contact', 'compose', 'rule', 'concern', 'question', 'please', 'mod', 'admin', 'automod',
    'reddit', 'post', 'comment', 'thread', 'reply', 'edit', 'deleted', 'removed', 'banned',
    'violation', 'violates', 'rules', 'guidelines', 'policy', 'report', 'reported'
}

def is_bot_message(text):
    """Check if text is likely a bot/moderation message."""
    text_lower = text.lower()
    bot_indicators = [
        'performed automatically', 'i am a bot', 'beep boop', 'this action was performed',
        'contact the moderators', 'message the moderators', 'your submission was',
        'your comment was', 'has been removed', 'violates our rules'
    ]
    return any(indicator in text_lower for indicator in bot_indicators)

def clean_text(text):
    """Clean and preprocess text for word cloud."""
    if pd.isna(text) or text == "[deleted]":
        return ""
    
    # Skip bot messages
    if is_bot_message(str(text)):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove Reddit-specific patterns
    text = re.sub(r'/r/\w+', '', text)  # Remove subreddit links
    text = re.sub(r'/u/\w+', '', text)  # Remove user links
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # Remove markdown links
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def filter_words(text):
    """Filter out stop words and short words."""
    words = text.split()
    filtered = []
    for word in words:
        # Keep words that are:
        # - At least 3 characters
        # - Not a stop word
        # - Not a Reddit bot term
        if (len(word) >= 3 and 
            word not in STOP_WORDS and 
            word not in REDDIT_BOT_TERMS):
            filtered.append(word)
    return ' '.join(filtered)

def load_era_data(data_dir, start_year, end_year, max_samples=5000):
    """Load Reddit comments for a specific era."""
    all_texts = []
    
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            path = os.path.join(data_dir, f"RC_{year}-{month:02d}.csv")
            try:
                df = pd.read_csv(path, header=None, nrows=max_samples)
                df.columns = ["subreddit", "subreddit_id", "body", "date_created_utc"]
                
                # Filter out deleted comments and bot messages
                df = df[df["body"] != "[deleted]"]
                df = df[~df["body"].astype(str).str.lower().str.contains('performed automatically|i am a bot|beep boop', na=False)]
                
                # Clean and collect text
                for text in df["body"]:
                    cleaned = clean_text(text)
                    if cleaned:
                        filtered = filter_words(cleaned)
                        if filtered:
                            all_texts.append(filtered)
                        
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue
        
        print(f"  Loaded {year}")
    
    return ' '.join(all_texts)

def generate_wordcloud(text, era_name, output_path):
    """Generate and save a word cloud."""
    if not text or len(text.strip()) < 100:
        print(f"⚠️  Not enough text for {era_name}, creating placeholder")
        # Create a simple placeholder
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.text(0.5, 0.5, f'Word Cloud\n{era_name}\n(Not enough data)', 
                ha='center', va='center', fontsize=20, color='gray')
        ax.axis('off')
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return
    
    # Create word cloud with better settings for distinctive vocabulary
    wordcloud = WordCloud(
        width=800,
        height=800,
        background_color='white',
        max_words=80,
        colormap='Set2',
        relative_scaling=0.4,
        min_font_size=12,
        collocations=False,  # Don't show word pairs
        prefer_horizontal=0.7
    ).generate(text)
    
    # Plot and save (no title - it's in the UI)
    plt.figure(figsize=(10, 10), facecolor='white')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', pad_inches=0)
    plt.close()
    
    print(f"✅ Generated word cloud for {era_name}")

print("Generating word clouds for different Reddit eras...\n")

data_dir = "data/sampled_comments"

# Era 1: 2006-2010
print("1. Loading data for 2006-2010...")
era1_text = load_era_data(data_dir, 2006, 2010, max_samples=2000)
generate_wordcloud(era1_text, "2006-2010", "frontend/public/word-cloud-2006-2010.png")

# Era 2: 2011-2017
print("\n2. Loading data for 2011-2017...")
era2_text = load_era_data(data_dir, 2011, 2017, max_samples=2000)
generate_wordcloud(era2_text, "2011-2017", "frontend/public/word-cloud-2011-2017.png")

# Era 3: 2018-2024
print("\n3. Loading data for 2018-2024...")
era3_text = load_era_data(data_dir, 2018, 2024, max_samples=2000)
generate_wordcloud(era3_text, "2018-2024", "frontend/public/word-cloud-2018-2024.png")

print("\n✅ All word clouds generated successfully!")
print("📁 Files saved to: frontend/public/")
print("   - word-cloud-2006-2010.png")
print("   - word-cloud-2011-2017.png")
print("   - word-cloud-2018-2024.png")

