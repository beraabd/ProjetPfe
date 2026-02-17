import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import os

# Configuration
INPUT_FILE = os.path.join("data", "processed", "reviews_with_sentiment.csv")
OUTPUT_FILE = os.path.join("data", "processed", "reviews_topics.csv")
NUM_TOPICS = 5

# Liste noire des mots vides (Stopwords)
STOP_WORDS = [
    # Français
    'le', 'la', 'les', 'de', 'du', 'des', 'un', 'une', 'et', 'est', 'il', 'elle', 
    'je', 'tu', 'nous', 'vous', 'ils', 'elles', 'ce', 'cet', 'cette', 'pour', 
    'dans', 'sur', 'au', 'aux', 'mon', 'ma', 'mes', 'ne', 'pas', 'plus', 'très', 
    'mais', 'ou', 'où', 'donc', 'or', 'ni', 'car', 'que', 'qui', 'quoi', 'sa', 
    'son', 'ses', 'par', 'avec', 'tout', 'tous', 'sont', 'ont', 'a', 'à', 'y', 
    'en', 'ca', 'ça', 'cest', 'c\'est', 'j\'ai', 'jai', 'qu\'on', 'quon',
    'application', 'app', 'orange', 'maroc', 'merci', 'bonjour', # Mots trop génériques
    
    # Arabe / Dialecte (transcrit ou brut)
    'من', 'على', 'في', 'لا', 'و', 'يا', 'ما', 'مع', 'هذا', 'انا', 'ان', 'كان',
    'تطبيق', 'برنامج', 'orange', 'li', 'fi', '3la', 'dyal', 'ana', 'hada'
]

def run_topic_modeling():
    if not os.path.exists(INPUT_FILE):
        print(f"Erreur : Le fichier {INPUT_FILE} n'existe pas.")
        return

    print("Chargement des données...")
    df = pd.read_csv(INPUT_FILE)
    
    # On garde les reviews > 3 mots
    reviews = df[df['cleaned_content'].str.split().str.len() > 3]['cleaned_content'].dropna().tolist()

    print("Vectorisation (avec exclusion des mots vides)...")
    # On ajoute notre liste 'stop_words' pour ignorer les mots inutiles
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=STOP_WORDS)
    tfidf = tfidf_vectorizer.fit_transform(reviews)

    print(f"Recherche de {NUM_TOPICS} sujets principaux...")
    nmf_model = NMF(n_components=NUM_TOPICS, random_state=1, l1_ratio=.5, init='nndsvd')
    nmf_model.fit(tfidf)
    
    print("\n--- Les VRAIS Sujets ---")
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    for topic_idx, topic in enumerate(nmf_model.components_):
        print(f"\nSujet #{topic_idx + 1}:")
        # Top 10 des mots
        print(", ".join([feature_names[i] for i in topic.argsort()[:-11:-1]]))

if __name__ == "__main__":
    run_topic_modeling()