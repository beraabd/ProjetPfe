"""
Script de Valorisation Croisée (Axe 4)
Connecte l'analyse des feedbacks avec la base de connaissances du chatbot.

Ce script :
1. Lit les avis négatifs analysés
2. Identifie les problèmes récurrents (topics)
3. Compare avec la FAQ existante
4. Suggère de nouvelles questions/réponses à ajouter
"""

import os
import pandas as pd
from collections import Counter
import re

# Configuration
SENTIMENT_FILE = os.path.join("data", "processed", "reviews_with_sentiment.csv")
FAQ_FILE = os.path.join("data", "faq_orange.txt")
OUTPUT_SUGGESTIONS = os.path.join("data", "processed", "faq_suggestions.txt")

def load_negative_reviews():
    """Charge les avis négatifs"""
    if not os.path.exists(SENTIMENT_FILE):
        print(f"Erreur : Fichier {SENTIMENT_FILE} introuvable.")
        return None
    
    df = pd.read_csv(SENTIMENT_FILE)
    # Filtre les avis négatifs uniquement
    negative = df[df['sentiment'] == 'Négatif']
    print(f"✓ {len(negative)} avis négatifs chargés.")
    return negative

def extract_keywords(reviews, top_n=15):
    """Extrait les mots-clés les plus fréquents des avis négatifs"""
    # Mots vides à exclure (génériques + adjectifs négatifs)
    stopwords = ['le', 'la', 'les', 'de', 'du', 'des', 'un', 'une', 'et', 'est', 
                 'il', 'elle', 'avec', 'pour', 'dans', 'sur', 'pas', 'que', 'qui',
                 'se', 'ne', 'ce', 'cette', 'mon', 'ma', 'mes', 'ca', 'ça',
                 # Adjectifs négatifs à exclure
                 'nul', 'nulle', 'mauvais', 'mauvaise', 'très', 'plus', 'tout',
                 'mais', 'être', 'avoir', 'faire', 'rien', 'jamais', 'toujours',
                 'vraiment', 'trop', 'encore', 'depuis', 'après', 'aucun']
    
    # Combine tous les avis en un seul texte
    all_text = " ".join(reviews['content'].dropna().astype(str))
    
    # 1. Extraction de bi-grammes (2 mots consécutifs)
    bigrams = re.findall(r'\b([a-zàâéèêëîïôùûüç]{3,})\s+([a-zàâéèêëîïôùûüç]{3,})\b', all_text.lower())
    bigram_phrases = [f"{w1} {w2}" for w1, w2 in bigrams if w1 not in stopwords and w2 not in stopwords]
    bigram_counts = Counter(bigram_phrases).most_common(10)
    
    # 2. Extraction des mots simples (3 caractères minimum)
    words = re.findall(r'\b[a-zàâéèêëîïôùûüç]{3,}\b', all_text.lower())
    
    # Filtrage et comptage
    filtered_words = [w for w in words if w not in stopwords]
    word_counts = Counter(filtered_words).most_common(top_n)
    
    print("\n📌 Expressions fréquentes (2 mots) :")
    for phrase, count in bigram_counts:
        print(f"  • {phrase} : {count} fois")
    
    return word_counts
def load_faq():
    """Charge la FAQ existante"""
    if not os.path.exists(FAQ_FILE):
        print(f"Erreur : FAQ {FAQ_FILE} introuvable.")
        return ""
    
    with open(FAQ_FILE, 'r', encoding='utf-8') as f:
        faq_content = f.read()
    
    print(f"✓ FAQ chargée ({len(faq_content)} caractères).")
    return faq_content.lower()

def generate_suggestions(keywords, faq_content):
    """Génère des suggestions de nouvelles questions FAQ"""
    suggestions = []
    
    print("\n🔍 Analyse des problèmes récurrents vs FAQ existante...")
    
    for word, count in keywords:
        # Si le mot n'est PAS dans la FAQ (ou très peu présent)
        if faq_content.count(word) < 2:
            suggestions.append({
                "keyword": word,
                "occurrences": count,
                "suggestion": f"Ajouter une entrée FAQ pour : '{word}' (mentionné {count} fois dans les avis négatifs)"
            })
    
    return suggestions

def save_suggestions(suggestions):
    """Sauvegarde les suggestions dans un fichier"""
    if not suggestions:
        print("\n✓ Aucune nouvelle suggestion. La FAQ couvre déjà les principaux problèmes !")
        return
    
    with open(OUTPUT_SUGGESTIONS, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("SUGGESTIONS D'AMÉLIORATION DE LA FAQ (Valorisation Croisée)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Généré automatiquement par l'analyse des feedbacks négatifs.\n")
        f.write(f"Nombre total de suggestions : {len(suggestions)}\n\n")
        
        for i, sugg in enumerate(suggestions, 1):
            f.write(f"{i}. {sugg['suggestion']}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("ACTIONS RECOMMANDÉES :\n")
        f.write("- Créer des réponses pour ces sujets dans faq_orange.txt\n")
        f.write("- Relancer ingest_knowledge.py pour mettre à jour le chatbot\n")
        f.write("=" * 70 + "\n")
    
    print(f"\n✓ {len(suggestions)} suggestions sauvegardées dans : {OUTPUT_SUGGESTIONS}")

def main():
    print("\n" + "="*70)
    print("SCRIPT DE VALORISATION CROISÉE (AXE 4)")
    print("="*70 + "\n")
    
    # 1. Charger les avis négatifs
    negative_reviews = load_negative_reviews()
    if negative_reviews is None or len(negative_reviews) == 0:
        print("Aucun avis négatif trouvé. Arrêt.")
        return
    
    # 2. Extraire les mots-clés récurrents
    print("\n📊 Extraction des problèmes récurrents...")
    keywords = extract_keywords(negative_reviews, top_n=15)
    
    print("\nTop 15 des mots les plus fréquents dans les avis négatifs :")
    for word, count in keywords:
        print(f"  • {word} : {count} fois")
    
    # 3. Charger la FAQ existante
    faq_content = load_faq()
    if not faq_content:
        return
    
    # 4. Générer les suggestions
    suggestions = generate_suggestions(keywords, faq_content)
    
    # 5. Sauvegarder
    save_suggestions(suggestions)
    
    print("\n" + "="*70)
    print("ANALYSE TERMINÉE !")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
