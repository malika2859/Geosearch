import streamlit as st
from PIL import Image
import datetime
import json
import os
import numpy as np
from dotenv import load_dotenv
import plotly.express as px
import pandas as pd
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings

st.write("Langchain imported successfully!" if "langchain" in sys.modules else "Langchain NOT imported.")
st.write("HuggingFaceEmbeddings imported successfully!" if "langchain.embeddings" in sys.modules else "HuggingFaceEmbeddings NOT imported.")

# Vérification des modules critiques
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ModuleNotFoundError:
    st.error("Le module `langchain` ou `HuggingFaceEmbeddings` n'est pas installé.")
    st.stop()

try:
    from transformers import pipeline
except ModuleNotFoundError:
    st.error("Le module `transformers` n'est pas installé.")
    st.stop()

try:
    import faiss
except ModuleNotFoundError:
    st.error("Le module `faiss-cpu` n'est pas installé.")
    st.stop()

# Chargement des variables d'environnement
load_dotenv()

# Définition du répertoire de base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration de la page
st.set_page_config(page_title="GeoSearch", page_icon="🌍", layout="wide")

# Chemins des fichiers
IMG_PATH = os.path.join(BASE_DIR, "IM3.png")
IMG_ACCUEIL_PATH = os.path.join(BASE_DIR, "Img.png")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index.bin")
METADATA_FILE = os.path.join(BASE_DIR, "metadata_faiss.json")

MAP_PATHS = {
    "Carte interactive du monde": os.path.join(BASE_DIR, "fixed_interactive_world_map.html"),
    "Cartographie de l'occupation des sols": os.path.join(BASE_DIR, "modis-viewer.html"),
    "Exploration de fond de carte": os.path.join(BASE_DIR, "map_layer_fix.html")
}

# Vérification de l'existence de FAISS
def verify_faiss_index():
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            index = faiss.read_index(FAISS_INDEX_PATH)
            return index
        except Exception as e:
            st.sidebar.error(f"Erreur lors du chargement de l'index FAISS : {e}")
            return None
    else:
        st.sidebar.error(f"Index FAISS introuvable : {FAISS_INDEX_PATH}")
        return None

faiss_index = verify_faiss_index()

# Pipeline de résumé amélioré
def summarize_content(content, threshold=500, max_input_length=1024):
    """Résumé d'un contenu textuel avec gestion des erreurs et des limites de longueur."""
    if not content or len(content) < threshold:
        return content  # Retourne le contenu s'il est trop court

    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        summary = summarizer(content[:max_input_length], max_length=130, min_length=30, do_sample=False)
        return summary[0]["summary_text"]
    except Exception as e:
        return f"Résumé non disponible en raison d'une erreur : {e}"

# Fonction de recherche FAISS
def faiss_search(query, top_k=5):
    faiss_docs = []
    if faiss_index:
        try:
            hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            query_vector = hf_embeddings.embed_query(query)
            distances, indices = faiss_index.search(np.array([query_vector], dtype=np.float32), top_k)
            
            if os.path.exists(METADATA_FILE):
                with open(METADATA_FILE, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                
                for idx in indices[0]:
                    if 0 <= idx < len(metadata):
                        faiss_docs.append({
                            "content": metadata[idx].get("segment_content", "Contenu non disponible"),
                            "source": metadata[idx].get("file_name", "Source inconnue")
                        })
                    else:
                        st.warning(f"Index FAISS invalide : {idx}")
            else:
                st.error(f"Fichier de métadonnées introuvable : {METADATA_FILE}")
        except Exception as e:
            st.error(f"Erreur lors de la recherche FAISS : {e}")
    return faiss_docs

# Barre latérale
if os.path.exists(IMG_PATH):
    st.sidebar.image(IMG_PATH, width=200)
else:
    st.sidebar.warning("L'image de la barre latérale est introuvable.")

st.sidebar.title("GeoSearch 🌍")
st.sidebar.markdown("Un moteur de recherche innovant pour les sciences géospatiales.")

menu = ["Accueil", "Visualisation géospatiale", "Recherche avancée", "Statistiques", "À propos"]
choice = st.sidebar.selectbox("Navigation", menu)

if choice == "Accueil":
    st.title("Bienvenue sur GeoSearch")
    st.markdown(
        """
        **GeoSearch** est une plateforme collaborative de pointe pour les sciences géospatiales.
        
        ### Fonctionnalités principales :
        - 🔎 Recherche avancée dans des sources scientifiques
        - 🗺️ Visualisation de données géospatiales
        - 📊 Analyse statistique des résultats
        - 🤝 Collaboration entre chercheurs et professionnels
        
        Commencez votre exploration dès maintenant !
        """
    )
    if os.path.exists(IMG_ACCUEIL_PATH):
        st.image(IMG_ACCUEIL_PATH, caption="Analyse géospatiale avancée")
    else:
        st.warning("L'image de l'accueil est introuvable.")

elif choice == "Visualisation géospatiale":
    st.title("Visualisation géospatiale avancée 🌍")
    
    map_type = st.selectbox("Choisissez un type de carte :", list(MAP_PATHS.keys()))
    
    if st.button("Afficher la carte", key="show_map"):
        map_path = MAP_PATHS.get(map_type)
        if map_path and os.path.exists(map_path):
            try:
                with open(map_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=600, width=None, scrolling=True)
            except Exception as e:
                st.error(f"Erreur lors du chargement de la carte : {e}")
        else:
            st.warning(f"Aucune carte trouvée pour : {map_type}.")

elif choice == "Recherche avancée":
    st.title("Recherche avancée 🔎")
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Entrez votre requête :", "Changement climatique")
    with col2:
        top_k = st.slider("Nombre de résultats :", 1, 10, 5)

    if st.button("Rechercher"):
        with st.spinner("Recherche en cours..."):
            results = faiss_search(query, top_k=top_k)

        if not results:
            st.warning("Aucun résultat trouvé pour la requête.")
        else:
            for doc in results:
                with st.expander(f"Source : {doc.get('source', 'Inconnue')}"):
                    content = doc.get('content', '')
                    if content:
                        summary = summarize_content(content)
                        st.write(summary)
                    else:
                        st.write("Contenu non disponible")

elif choice == "Statistiques":
    st.title("Statistiques de recherche")

    data = {
        'Catégorie': ['Climat', 'Urbanisme', 'Biodiversité', 'Océanographie', 'Géologie'],
        'Nombre de recherches': [1200, 800, 600, 400, 300]
    }
    df = pd.DataFrame(data)

    fig = px.bar(df, x='Catégorie', y='Nombre de recherches', title="Répartition des recherches par catégorie")
    st.plotly_chart(fig)

elif choice == "À propos":
    st.title("À propos de GeoSearch")
    st.markdown(
        """
        **GeoSearch** est une plateforme de pointe conçue pour les chercheurs et professionnels en géomatique.
        
        ### Notre mission
        Faciliter l'accès et l'analyse des données géospatiales pour accélérer la recherche et l'innovation dans le domaine des sciences de la Terre.
        
        ### Technologie
        - Moteur de recherche basé sur l'IA
        - Intégration de multiples sources de données
        - Visualisation avancée des données géospatiales
        
        ### Équipe
        Une équipe passionnée de data scientists, géographes et développeurs.
        """
    )

# Pied de page
st.sidebar.markdown("---")
now = datetime.datetime.now()
st.sidebar.write(f"📅 Date : {now.strftime('%Y-%m-%d')} | ⏱️ Heure : {now.strftime('%H:%M')}")
st.sidebar.info("© 2025 GeoSearch. Tous droits réservés.")
