import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Gestion des versions de LangChain pour l'import
try:
    from langchain.text_splitter import CharacterTextSplitter
except ImportError:
    from langchain_text_splitters import CharacterTextSplitter

# Configuration
FAQ_FILE = os.path.join("data", "faq_orange.txt")
DB_DIR = os.path.join("data", "chroma_db")

def ingest_data():
    if not os.path.exists(FAQ_FILE):
        print(f"Erreur : Le fichier {FAQ_FILE} n'existe pas.")
        return

    print("Lecture du fichier FAQ...")
    loader = TextLoader(FAQ_FILE, encoding='utf-8')
    documents = loader.load()

    print("Découpage du texte en morceaux...")
    # On utilise RecursiveCharacterTextSplitter pour mieux couper
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    # Chunk size réduit pour avoir une question par morceau (environ)
    # On coupe strictement sur "Q: " pour avoir une question par morceau
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\nQ: ", "Q: "], # Coupe quand il voit une nouvelle question
        chunk_size=200, 
        chunk_overlap=0,
        keep_separator=True
    )
    texts = text_splitter.split_documents(documents)
    print(f"{len(texts)} morceaux créés.")

    print("Création de la base de données vectorielle (ChromaDB)...")
    # On utilise un modèle multilingue gratuit
    embedding_function = SentenceTransformerEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # Création et sauvegarde de la DB
    db = Chroma.from_documents(texts, embedding_function, persist_directory=DB_DIR)
    db.persist()
    
    print(f"Succès ! Base de données créée dans {DB_DIR}")

if __name__ == "__main__":
    ingest_data()