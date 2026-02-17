import pandas as pd
from textblob import TextBlob
from textblob_fr import PatternTagger, PatternAnalyzer
import os

# Configuration
INPUT_FILE = os.path.join("data", "processed", "reviews_cleaned.csv")
OUTPUT_FILE = os.path.join("data", "processed", "reviews_with_sentiment.csv")

def analyze_sentiment(text):
    if not isinstance(text, str):
        return 0, "Neutre"
    
    # Analyse spécifique pour le français
    blob = TextBlob(text, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
    polarity = blob.sentiment[0] # La polarité est entre -1 (très négatif) et +1 (très positif)
    
    if polarity > 0.1:
        sentiment = "Positif"
    elif polarity < -0.1:
        sentiment = "Négatif"
    else:
        sentiment = "Neutre"
        
    return polarity, sentiment

def run_analysis():
    if not os.path.exists(INPUT_FILE):
        print(f"Erreur : Le fichier {INPUT_FILE} n'existe pas.")
        return

    print("Chargement des données...")
    df = pd.read_csv(INPUT_FILE)
    
    print("Analyse des sentiments en cours... (ça peut prendre un peu de temps)")
    # Appliquer la fonction sur chaque ligne
    df[['polarity', 'sentiment']] = df['cleaned_content'].apply(
        lambda x: pd.Series(analyze_sentiment(x))
    )
    
    print(f"Sauvegarde des résultats dans {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    
    # Petit résumé pour toi
    print("\n--- Résumé ---")
    print(df['sentiment'].value_counts())
    print("Terminé.")

if __name__ == "__main__":
    run_analysis()