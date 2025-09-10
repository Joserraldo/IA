# app.py
import streamlit as st
import joblib
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# --- Cargar modelos ---
gender_encoder = joblib.load("gender_encoder.joblib")
rf_model = joblib.load("random_forest_model.joblib")

# --- Configuración de la página ---
st.set_page_config(
    page_title="Predictor de Speech de Ventas de Bicicletas",
    layout="centered"
)

# --- Título ---
st.title("Predictor de Speech de Ventas de Bicicletas")
st.markdown("**Realizado por: José Alejandro**")

# --- Introducción ---
st.write("""
Esta aplicación permite predecir si una persona comprará o no una bicicleta según su edad y género.
La predicción se realiza utilizando un modelo de Random Forest previamente entrenado.
""")
st.write("""
Seleccione la edad y el género de la persona en los controles de abajo y haga clic en **Enviar** para obtener la predicción.
""")

# --- Input del usuario ---
edad = st.slider("Edad:", min_value=20, max_value=50, step=1)
genero = st.selectbox("Género:", options=["F", "M"])

# --- Botón de predicción ---
if st.button("Enviar"):
    # Codificar género
    genero_codificado = gender_encoder.transform([genero])[0]

    # Preparar features
    X = np.array([[edad, genero_codificado]])

    # Predicción
    pred = rf_model.predict(X)[0]

    # Mostrar resultado con colores
    if pred == 1:
        st.success("La persona probablemente **comprará** la bicicleta.")
        # Imagen de compra
        url_img = "https://media.tacdn.com/media/attractions-splice-spp-674x446/15/87/19/b7.jpg"
        response = requests.get(url_img)
        img = Image.open(BytesIO(response.content))
        st.image(img, use_container_width=True)
    else:
        st.error("La persona probablemente **NO comprará** la bicicleta.")
        # Imagen de no compra
        url_img = "https://as2.ftcdn.net/jpg/00/29/89/87/1000_F_29898774_n0Txc82krUH4IYpjXGuiCGKl7DKpRaH5.jpg"
        response = requests.get(url_img)
        img = Image.open(BytesIO(response.content))
        st.image(img, use_container_width=True)

# --- Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("© Derechos reservados Unab 2025", unsafe_allow_html=True)

