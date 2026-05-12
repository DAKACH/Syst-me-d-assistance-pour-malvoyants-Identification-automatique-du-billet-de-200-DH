import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from gtts import gTTS
import io

# Chargement du modèle
model = YOLO('best.pt') 

def speak(text):
    """Génère et joue un son à partir du texte"""
    tts = gTTS(text=text, lang='fr')
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    st.audio(fp, format='audio/mp3', autoplay=True)

st.title("Assistant Vocal - Monnaie Marocaine 🇲🇦")
st.write("Le système annoncera vocalement la valeur du billet détecté.")

img_file_buffer = st.camera_input("Présentez le billet pour identification")

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    img_array = np.array(image)
    
    # Inférence
    results = model(img_array)
    
    detected_any = False
    for r in results:
        if len(r.boxes) > 0:
            detected_any = True
            msg = "Billet de deux cents dirhams détecté"
            st.success(f"✅ {msg}")
            # Appel de la fonction vocale
            speak(msg)
        else:
            st.warning("⚠️ Aucun billet reconnu. Veuillez ajuster la position.")
            speak("Aucun billet reconnu")
    
    st.image(results[0].plot())