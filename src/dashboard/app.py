import streamlit as st
import pandas as pd
import os
import sys

# Ajout du dossier racine au path pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.chatbot.rag_chatbot import load_db, get_response

# Configuration de la page
st.set_page_config(page_title="Orange PFE - Assistant & Analyse", page_icon="🍊", layout="wide")

# --- FONCTIONS ---
@st.cache_data
def load_data():
    path = os.path.join("data", "processed", "reviews_with_sentiment.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_resource
def get_chatbot_db():
    return load_db()

# --- INTERFACE ---
st.title("🍊 Orange Maroc - Assistant & Analyse PFE")

# Sidebar
page = st.sidebar.selectbox("Navigation", ["📊 Analyse des Feedbacks", "🤖 Assistant Intelligent"])

# --- PAGE 1 : ANALYSE ---
if page == "📊 Analyse des Feedbacks":
    st.header("Analyse des Avis Clients (Play Store)")
    
    df = load_data()
    if df is None:
        st.error("Aucune donnée trouvée. Veuillez lancer le scraping d'abord.")
    else:
        # KPIs
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Avis", len(df))
        col2.metric("Avis Positifs", len(df[df['sentiment'] == 'Positif']))
        col3.metric("Avis Négatifs", len(df[df['sentiment'] == 'Négatif']))
        
        # Graphiques
        st.subheader("Répartition des Sentiments")
        sentiment_counts = df['sentiment'].value_counts()
        st.bar_chart(sentiment_counts)
        
        # Derniers avis négatifs
        st.subheader("⚠️ Derniers avis négatifs (à traiter)")
        neg_reviews = df[df['sentiment'] == 'Négatif'][['at', 'content', 'score']].head(10)
        st.dataframe(neg_reviews, use_container_width=True)

# --- PAGE 2 : CHATBOT ---
elif page == "🤖 Assistant Intelligent":
    st.header("Chatbot Assistant Client")
    st.markdown("Posez une question sur les offres, les factures ou l'assistance technique.")

    # Initialisation de l'historique
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Bonjour ! Je suis l'assistant virtuel d'Orange. Comment puis-je vous aider ?"}]

    # Affichage des messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Zone de saisie
    if prompt := st.chat_input("Votre question..."):
        # Affiche la question utilisateur
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Réponse du bot
        db = get_chatbot_db()
        with st.chat_message("assistant"):
            with st.spinner("Je cherche dans la base de connaissances..."):
                if db:
                    response = get_response(db, prompt)
                else:
                    response = "Erreur : Base de données non chargée."
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
