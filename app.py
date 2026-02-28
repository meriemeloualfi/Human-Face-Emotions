import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

# =======================
# CONFIG PAGE
# =======================
st.set_page_config(page_title="Emotion AI", page_icon="😊", layout="wide")

# Masquer le menu Streamlit
st.markdown("<style>header {visibility: hidden;} footer {visibility: hidden;}</style>", unsafe_allow_html=True)

# =======================
# LOAD MODEL
# =======================
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("emotion_model.h5")

model = load_my_model()
emotions = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise']

# =======================
# SIDEBAR
# =======================
st.sidebar.title("😊 Emotion AI")
mode_sombre = st.sidebar.toggle("🌙 Mode sombre", value=True)
source = st.sidebar.radio("Source de l'image", ["📁 Importer une image", "📷 Webcam"])

# =======================
# THÈME ET DESIGN (REDESIGN MODE SOMBRE)
# =======================
if mode_sombre:
    # Mode Sombre Sophistiqué
    bg = "#0f172a"  # Fond Midnight Blue
    card_bg = "rgba(30, 41, 59, 0.7)" # Fond de carte Slate
    card_border = "1px solid rgba(255, 255, 255, 0.1)"
    text_color = "#f8fafc"
    accent_color = "#38bdf8" # Bleu ciel électrique
    sidebar_bg = "#1e293b"
else:
    # Mode Clair Doux
    bg = "linear-gradient(135deg, #e0eafc, #cfdef3)"
    card_bg = "rgba(255, 255, 255, 0.85)"
    card_border = "1px solid rgba(255, 255, 255, 0.3)"
    text_color = "#1e293b"
    accent_color = "#2563eb"
    sidebar_bg = "#f8fafc"

# =======================
# CSS GLOBAL (STRUCTURE SANS ESPACES)
# =======================
st.markdown(f"""
<style>
/* Fond Principal */
[data-testid="stAppViewContainer"] {{ 
    background: {bg}; 
    transition: all 0.3s ease;
}}

/* Sidebar Design */
[data-testid="stSidebar"] {{
    background-color: {sidebar_bg} !important;
}}

.main .block-container {{ padding-top: 2rem !important; }}

/* Style de la Carte (Correction Espaces Blancs) */
div[data-testid="stVerticalBlock"] > div:has(div.custom-card) {{
    background: {card_bg};
    backdrop-filter: blur(12px);
    border-radius: 24px;
    border: {card_border};
    padding: 25px;
    margin-bottom: 30px;
    box-shadow: 0 15px 35px rgba(0,0,0,0.2);
    color: {text_color};
}}

.custom-card {{ display: none; }}

/* Suppression des gaps internes */
div[data-testid="stVerticalBlock"] > div:has(div.custom-card) > div[data-testid="stVerticalBlock"] {{
    gap: 0px !important;
}}

/* Titres et Textes */
h1 {{ color:{text_color}; text-align:center; font-weight:800; font-size: 3rem; margin-bottom: 0px; }}
p {{ color:{text_color}; text-align:center; margin-bottom: 30px; opacity: 0.8; }}
.card-title {{ 
    font-size: 22px; 
    font-weight: 700; 
    margin-bottom: 15px; 
    display: block; 
    color: {accent_color}; 
}}

img {{ border-radius: 18px; margin-top: 10px; border: {card_border}; }}

/* Design des barres de progression en mode sombre */
div[data-testid="stProgress"] > div > div > div > div {{
    background-color: {accent_color} !important;
}}

/* Ajustement espacement colonnes */
[data-testid="stHorizontalBlock"] {{
    gap: 2rem !important;
}}
</style>
""", unsafe_allow_html=True)

# =======================
# HEADER
# =======================
st.markdown("<h1>Emotion AI</h1>", unsafe_allow_html=True)
st.markdown("<p>Analyse intelligente des expressions faciales</p>", unsafe_allow_html=True)

# =======================
# INPUT IMAGE
# =======================
image = None
if source == "📁 Importer une image":
    file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    if file: image = Image.open(file)
else:
    cam = st.camera_input("Prendre une photo")
    if cam: image = Image.open(cam)

# =======================
# ANALYSE ET AFFICHAGE
# =======================
if image is not None:
    # --- PRÉTRAITEMENT ---
    img_rgb = image.convert('RGB')
    img_resized = img_rgb.resize((126, 126)) 
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyse du visage..."):
        preds = model.predict(img_array)[0]
        emotion_final = emotions[np.argmax(preds)]

    # --- AFFICHAGE ---
    col1, col2 = st.columns(2)

    with col1:
        with st.container():
            # Fusion du marqueur et du titre
            st.markdown('<div class="custom-card"></div><span class="card-title">📷 Image analysée</span>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)

    with col2:
        with st.container():
            st.markdown(f'<div class="custom-card"></div><span class="card-title">🎯 Émotion : <b>{emotion_final}</b></span>', unsafe_allow_html=True)
            for emo, score in zip(emotions, preds):
                st.progress(float(score), text=f"{emo} : {score*100:.1f}%")

    # --- GRAPHIQUE ---
    with st.container():
        st.markdown('<div class="custom-card"></div><span class="card-title">📊 Probabilités détaillées</span>', unsafe_allow_html=True)
        df = pd.DataFrame({"Émotion": emotions, "Score": preds})
        st.bar_chart(df.set_index("Émotion"))