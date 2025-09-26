import streamlit as st

def main ():
    st.title("Sistema de Clasificacion de Residuos 🗑️")

    st.sidebar.title("Configuracion")

    uploaded_file = st.file_uploader("Sube una imagen de un residuo", type=["jpg", "jpeg", "png"])