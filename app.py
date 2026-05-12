import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Chargement du modèle entraîné
# Assurez-vous que le fichier 'best.pt' est dans le même dossier
model = YOLO('best.pt') 

# Configuration de la page
st.title("Détecteur de Monnaie Marocaine 🇲🇦")
st.write("Système d'assistance pour malvoyants : Identification automatique du billet de 200 DH.")

# Interface de la caméra
img_file_buffer = st.camera_input("Veuillez présenter le billet devant l'objectif")

if img_file_buffer is not None:
    # Traitement de l'image pour YOLO
    image = Image.open(img_file_buffer)
    img_array = np.array(image)
    
    # Prédiction (Inférence)
    results = model(img_array)
    
    # Affichage des résultats
    detected = False
    for r in results:
        if len(r.boxes) > 0:
            detected = True
            st.success("✅ Billet détecté : 200 Dirhams Marocains (MAD)")
            # Note : L'intégration gTTS (Audio) peut être ajoutée ici
        else:
            st.warning("⚠️ Aucun billet reconnu. Essayez de rapprocher le billet ou d'améliorer l'éclairage.")
    
    # Visualisation des boîtes de détection
    st.image(results[0].plot(), caption="Analyse en temps réel")