import pandas as pd
import os
import re

# Configuration
INPUT_FILE = os.path.join("data", "raw", "reviews.csv")
OUTPUT_DIR = os.path.join("data", "processed")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "reviews_cleaned.csv")

def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove emojis (basic regex)
    # text = text.encode('ascii', 'ignore').decode('ascii') # simplistic way, removes accents too. Better to keep accents for French.
    
    # 3. Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # 4. Remove special characters but keep French accents
    # This regex keeps alphanumeric, spaces, and common punctuation.
    # Adjust based on needs.
    text = re.sub(r'[^\w\s\.,!?\'àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ]', ' ', text)
    
    # 5. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def process_reviews(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        return

    print("Loading raw data...")
    df = pd.read_csv(input_path)
    
    print("Cleaning content...")
    df['cleaned_content'] = df['content'].apply(clean_text)
    
    # Filter out empty reviews after cleaning
    df = df[df['cleaned_content'].str.len() > 2]
    
    print(f"Saving processed data to {output_path}...")
    df.to_csv(output_path, index=False, encoding='utf-8')
    print("Done.")

if __name__ == "__main__":
    ensure_directory_exists(OUTPUT_DIR)
    process_reviews(INPUT_FILE, OUTPUT_FILE)
