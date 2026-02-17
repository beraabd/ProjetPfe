import pandas as pd
from google_play_scraper import Sort, reviews_all, app
import os
from datetime import datetime

# Configuration
APP_ID = 'com.orange.orangemoney'  # ID for 'Max it - Maroc' (formerly Orange et moi)
LANG = 'fr'
COUNTRY = 'ma'
OUTPUT_DIR = os.path.join("data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "reviews.csv")

def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def scrape_reviews(app_id, lang, country):
    print(f"Fetching reviews for {app_id} ({lang}-{country})...")
    
    # Fetch all reviews (careful with large apps, might want to limit count)
    # Using reviews_all for simplicity, but for very large apps use 'reviews' with pagination
    try:
        result = reviews_all(
            app_id,
            sleep_milliseconds=0, # Defaults to 0
            lang=lang, 
            country=country, 
            sort=Sort.NEWEST, # Sort by newest
        )
        print(f"Successfully fetched {len(result)} reviews.")
        return result
    except Exception as e:
        print(f"Error fetching reviews: {e}")
        return []

def save_to_csv(data, filepath):
    if not data:
        print("No data to save.")
        return

    df = pd.DataFrame(data)
    
    # Keep relevant columns
    columns_to_keep = ['reviewId', 'userName', 'content', 'score', 'thumbsUpCount', 'reviewCreatedVersion', 'at', 'replyContent', 'repliedAt']
    # Filter columns that exist in the dataframe
    columns_to_keep = [col for col in columns_to_keep if col in df.columns]
    df = df[columns_to_keep]

    df.to_csv(filepath, index=False, encoding='utf-8')
    print(f"Reviews saved to {filepath}")

if __name__ == "__main__":
    ensure_directory_exists(OUTPUT_DIR)
    reviews_data = scrape_reviews(APP_ID, LANG, COUNTRY)
    save_to_csv(reviews_data, OUTPUT_FILE)
