import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import full_model as fm

# page config

st.set_page_config(
    page_title="Clasificador de Residuos",
    page_icon="🗑️",
    layout="wide"
)

# class names

CLASS_NAMES = ['battery', 'biological', 'brown Glass', 'cardboard', 'clothes', 'green Glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white Glass']

# dictionary for class descriptions

CLASS_DESCRIPTIONS = {
    'battery': 'Residuos peligrosos que contienen sustancias tóxicas y metales pesados. Deben ser reciclados adecuadamente para evitar la contaminación ambiental.',
    'biological': 'Residuos orgánicos que pueden descomponerse naturalmente. Incluyen restos de comida, residuos de jardinería y otros materiales biodegradables.',
    'brown Glass': 'Vidrio marrón utilizado comúnmente para envases de alimentos y bebidas. Debe ser reciclado para reducir la necesidad de producir nuevo vidrio.',
    'cardboard': 'Material de embalaje hecho de pulpa de madera. Es reciclable y puede ser utilizado para fabricar nuevos paquetes.',
    'clothes': 'Vestimenta y accesorios usados que pueden ser reciclados para fabricar nuevos productos.',
    'green Glass': 'Vidrio verde utilizado para envases de alimentos y bebidas. Debe ser reciclado para reducir la necesidad de producir nuevo vidrio.',
    'metal': 'Material metales utilizado para fabricar objetos y herramientas. Debe ser reciclado para reducir la necesidad de producir nuevos materiales metales.',
    'paper': 'Material de papel utilizado para envases de alimentos y bebidas. Es reciclable y puede ser utilizado para fabricar nuevos paquetes.',
    'plastic': 'Material de plastico utilizado para envases de alimentos y bebidas. Es reciclable y puede ser utilizado para fabricar nuevos paquetes.',
    'shoes': 'Zapatos y accesorios usados que pueden ser reciclados para fabricar nuevos productos.',
    'trash': 'Residuos no clasificados. Deben ser reciclados adecuadamente para evitar la contaminación ambiental.',
    'white Glass': 'Vidrio blanco utilizado comúnmente para envases de alimentos y bebidas. Debe ser reciclado para reducir la necesidad de producir nuevo vidrio.'
}

@st.cache_resource
def load_model():
    try: 
        model = tf.keras.models.load_model(fm.MODEL_SAVE_PATH)
        return model
    except: 
        st.error("Error al cargar el modelo. Asegúrate de que el archivo 'waste_classifier_model.h5' existe.")
        return None
    
def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image)
    img_array = image_array / 255.0
    image_array = np.expand_dims(img_array, axis=0)

    return image_array

def main ():
    st.title("Sistema de Clasificacion de Residuos 🗑️")

    st.markdown("""
    Este sistema utiliza inteligencia artificial para clasificar diferentes tipos de residuos 
    a través de imágenes. Sube una foto de un residuo y la IA te dirá de qué tipo es y cómo reciclarlo.
    """)

    # load model
    model = load_model()

    uploaded_file = st.file_uploader("Sube una imagen de un residuo", type=["jpg", "jpeg", "png"])

    col1, col2 = st.columns(2)

    with col1:
        if uploaded_file is not None:
            image=(Image.open(uploaded_file))
            st.image(image, caption='Imagen subida.', use_column_width=True)

            # show image

            if model is not None: 
                with st.spinner('Clasificando...'):
                    processed_image = preprocess_image(Image.open(uploaded_file))
                    predictions = model.predict(processed_image)
                    predicted_class = np.argmax(predictions[0])
                    confidence = predictions[0][predicted_class]

                    # results

                    st.success(f"**Clasificación:** {CLASS_NAMES[predicted_class]}")
                    st.info(f"**Confianza:** {confidence*100:.2f}%")

                    # show all probabilities

                    st.subheader("Probabilidades de cada clase:")

                    for i, prob in enumerate(predictions[0]):
                        st.write(f"{CLASS_NAMES[i]}: {prob*100:.2f}%")

    with col2:
        if uploaded_file is not None and model is not None:
            predicted_class = np.argmax(predictions[0])
            class_name = CLASS_NAMES[predicted_class]

            st.subheader(f"Descripción del residuo: {class_name}")
            st.info(CLASS_DESCRIPTIONS.get(class_name, "Consulta la guía local de reciclaje para más información."))

            # probability bar chart

            fig, ax = plt.subplots(figsize=(10, 6))
            y_pos = np.arange(len(CLASS_NAMES))
            ax.barh(y_pos, predictions[0]*100)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(CLASS_NAMES)
            ax.set_xlabel('Probabilidad (%)')
            ax.set_title('Probabilidades de Clasificación')
            st.pyplot(fig)
    
    # extra info (optional)

    st.subheader("Información del modelo")
    st.markdown("""
    - **Arquitectura:** Modelo de red neuronal convolucional (CNN) con MobileNetV2 como base.
    - **Precision esperada:** ~85-90% (dependiendo del entrenamiento)
    - **Clases:** Imágenes de residuos etiquetadas en 12 categorías diferentes.
    - **Dataset:** Garbage Classification Dataset de Kaggle.
    - **Nota:** La precisión del modelo puede variar según la calidad y el ángulo de la imagen subida.
    """)

if __name__ == "__main__":
    main()