import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

@st.cache_resource
def load_model():
    try: 
        model = tf.keras.models.load_model('waste_classifier_model.h5')
        return model
    except: 
        st.error("Error al cargar el modelo. Asegúrate de que el archivo 'waste_classifier_model.h5' existe.")
        return None

def main ():
    st.title("Sistema de Clasificacion de Residuos 🗑️")

    st.sidebar.title("Configuracion")

    uploaded_file = st.file_uploader("Sube una imagen de un residuo", type=["jpg", "jpeg", "png"])