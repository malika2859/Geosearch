Application streamlit


import streamlit as st
from PIL import Image
import datetime
from langchain.embeddings import HuggingFaceEmbeddings
import json
import os
import faiss
import numpy as np
from dotenv import load_dotenv
import plotly.express as px
import pandas as pd
from transformers import pipeline

# Chargement des variables d'environnement
load_dotenv()

# Configuration de la page
st.set_page_config(page_title="GeoSearch", page_icon="\U0001F30D", layout="wide")

# Styles CSS personnalisés
st.markdown(
    """
    <style>
    .main {
        background-color: #000000;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stSelectbox {
        background-color: #333333;
        color: white;
    }
    .stExpander {
        background-color: #333333;
        border: 1px solid #555555;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .stExpander .st-cb {
        color: white !important;
    }
    .stMarkdown {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Configuration des chemins pour les fichiers des cartes
MAP_PATHS = {
    "Carte interactive du monde": r"D:\\carte_interactive\\fixed_interactive_world_map.html",
    "Cartographie de l'occupation des sols": r"D:\\Cartographie de l'occupation des sols\\modis-viewer.html",
    "Exploration de fond de carte": r"D:\\Fonds_carte\\map_layer_fix.html"
}

# FAISS Configurations
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "C:\\FAISS\\faiss_index.bin")
METADATA_FILE = os.getenv("METADATA_FILE", "C:\\FAISS\\metadata_faiss.json")

# Vérification de l'existence de FAISS
def verify_faiss_index():
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            index = faiss.read_index(FAISS_INDEX_PATH)
            # Suppression de l'affichage du message de succès
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

    # Tronquer le contenu s'il dépasse la limite du modèle
    truncated_content = content[:max_input_length]

    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        summary = summarizer(truncated_content, max_length=130, min_length=30, do_sample=False)
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
            distances, indices = faiss_index.search(np.array([query_vector], dtype="float32"), top_k)
            st.write(f"Indices FAISS trouvés : {indices}")
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
            st.write(f"Nombre de documents FAISS trouvés : {len(faiss_docs)}")
        except Exception as e:
            st.error(f"Erreur lors de la recherche FAISS : {e}")
    return faiss_docs

# Barre latérale
st.sidebar.image("C:\\Users\\ec\\OneDrive\\Bureau\\Generative_IA\\Projet_Generative_IA\\IM3.png", width=200)
st.sidebar.title("GeoSearch \U0001F30D")
st.sidebar.markdown("Un moteur de recherche innovant pour les sciences géospatiales.")

menu = ["Accueil", "Visualisation géospatiale", "Recherche avancée", "Statistiques", "À propos"]
choice = st.sidebar.selectbox("Navigation", menu)

if choice == "Accueil":
    st.title("Bienvenue sur GeoSearch")
    st.markdown(
        """
        **GeoSearch** est une plateforme collaborative de pointe pour les sciences géospatiales.
        
        ### Fonctionnalités principales:
        - \U0001F50D Recherche avancée dans des sources scientifiques
        - \U0001F5FA\ufe0f Visualisation de données géospatiales
        - \U0001F4CA Analyse statistique des résultats
        - \U0001F91D Collaboration entre chercheurs et professionnels
        
        Commencez votre exploration dès maintenant!
        """
    )
    st.image("D:\\Img.png", caption="Analyse géospatiale avancée")

elif choice == "Visualisation géospatiale":
    st.title("Visualisation géospatiale avancée \U0001F30E")
    
    map_type = st.selectbox("Choisissez un type de carte :", list(MAP_PATHS.keys()))
    
    if st.button("Afficher la carte", key="show_map"):
        map_path = MAP_PATHS.get(map_type)
        if map_path and os.path.exists(map_path):
            try:
                with open(map_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=600, width=None, scrolling=True)
                
                if map_type == "Carte interactive du monde":
                    st.info("Cette carte interactive vous permet d'explorer différentes régions du monde en temps réel.")
                elif map_type == "Cartographie de l'occupation des sols":
                    st.info("Visualisez les différents types d'occupation des sols à l'aide des données MODIS.")
                elif map_type == "Exploration de fond de carte":
                    st.info("Explorez différents styles de fonds de carte pour vos projets de cartographie.")
            except Exception as e:
                st.error(f"Erreur lors du chargement de la carte : {e}")
        else:
            st.warning(f"Aucune carte trouvée pour : {map_type}.")
    
    st.subheader("Outils d'analyse géospatiale")
    tool = st.selectbox("Sélectionnez un outil :", ["Mesure de distance", "Calcul de superficie", "Analyse de densité"])
    st.write(f"L'outil '{tool}' sera bientôt disponible pour améliorer votre analyse géospatiale.")

elif choice == "Recherche avancée":
    st.title("Recherche avancée \U0001F50D")
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
st.sidebar.write(f"\U0001F4C5 Date : {now.strftime('%Y-%m-%d')} | \U0001F551 Heure : {now.strftime('%H:%M')}")
st.sidebar.info("\u00a9 2025 GeoSearch. Tous droits réservés.")
