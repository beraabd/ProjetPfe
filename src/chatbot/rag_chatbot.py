import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq

# Charge le fichier .env automatiquement
load_dotenv()

# Configuration
DB_DIR = os.path.join("data", "chroma_db")

# --- CLÉ API GROQ (lue depuis .env ou variable d'environnement) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "gsk_...")

def load_db():
    if not os.path.exists(DB_DIR):
        print(f"Erreur : La base de données {DB_DIR} n'existe pas. Lancez d'abord ingest_knowledge.py")
        return None

    # On utilise toujours le même embedding pour la recherche
    embedding_function = SentenceTransformerEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    db = Chroma(persist_directory=DB_DIR, embedding_function=embedding_function)
    return db

def get_llm_response(db, query, api_key):
    if not api_key or api_key == "gsk_...":
        return "Erreur : Clé API Groq manquante. Vérifiez votre fichier .env ou définissez la variable GROQ_API_KEY."

    # 1. Recherche des documents pertinents
    docs = db.similarity_search(query, k=3)
    
    if not docs:
        return "Désolé, je n'ai pas trouvé d'information dans ma base de connaissances."
    
    # 2. Construction du contexte
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 3. Configuration du LLM (Llama 3.3 via Groq)
    llm = ChatGroq(
        temperature=0, 
        groq_api_key=api_key, 
        model_name="llama-3.3-70b-versatile"
    )

    # 4. Construction du prompt (bilingue Français / Darija)
    prompt = f"""Tu es un assistant virtuel expert pour Orange Maroc. Tu peux parler en français et en darija (arabe marocain).

RÈGLE IMPORTANTE : Détecte la langue de la question de l'utilisateur et réponds DANS LA MÊME LANGUE.
- Si la question est en français → réponds en français, de manière polie et professionnelle.
- Si la question est en darija (arabe marocain, parfois mélangé avec du français) → réponds en darija, de manière sympa et naturelle.

Si tu ne connais pas la réponse, dis-le honnêtement sans inventer.

Utilise le contexte suivant pour répondre :
{context}

Question : {query}

Réponse :"""

    # 5. Appel au LLM
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Erreur avec Groq : {str(e)}"

# Fonction simplifiée pour Streamlit (rétro-compatibilité)
def get_response(db, query):
    global GROQ_API_KEY
    return get_llm_response(db, query, GROQ_API_KEY)

def chat():
    global GROQ_API_KEY
    db = load_db()
    if db is None:
        return

    print("\n--- CHATBOT ORANGE (IA Llama 3.3) ---")
    print("Posez votre question (ou tapez 'quit' pour quitter)\n")
    
    while True:
        query = input("\nVous : ")
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        print("🤖 Réflexion en cours...", end="\r")
        response = get_llm_response(db, query, GROQ_API_KEY)
        print(f"🤖 Bot : {response}\n")

if __name__ == "__main__":
    chat()
