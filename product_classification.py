import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import model_from_json

@st.cache_data
def load_model():
    with open("models\product_classification\modelo_cnn.pkl", "rb") as file:
        modelo_serializado = pickle.load(file)
    model_json = modelo_serializado["modelo_json"]
    model = model_from_json(model_json)
    model.set_weights(modelo_serializado["pesos"])
    return model

def predecir_imagen_con_umbral(imagen, model, umbral=0.88):

    # Cargar y preprocesar la imagen
    img_width, img_height = 150, 150
    imagen = imagen.resize((img_width, img_height))
    imagen = img_to_array(imagen) / 255.0
    imagen = np.expand_dims(imagen, axis=0)

    prediccion = model.predict(imagen)
    max_prob = np.max(prediccion)

    print("Probabilidad máxima:", max_prob)

    class_indices = {"jeans": 0, "sofa": 1, "tshirt": 2, "tv": 3}
    # Si la probabilidad máxima es menor al umbral, se clasifica como desconocido
    if max_prob < umbral:
        return "Desconocido"
    else:
        indice_clase = np.argmax(prediccion)
        class_labels = {v: k for k, v in class_indices.items()}
        etiqueta = class_labels[indice_clase]
        return etiqueta

def display_product_classification():
    st.subheader("Clasificación automática de productos")
    st.write("Este módulo utiliza un modelo de red neuronal convolucional (CNN) para clasificar imágenes de productos en una de las siguientes categorías: jeans, sofá, camiseta o televisor.")
    with st.expander("➡️Aquí tienes un video guía para utilizar este módulo"):
        st.video("videos\Productos.mp4")

    files = st.file_uploader("Carga aquí las imágenes de productos que deseas clasificar", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if files:
        model = load_model()
        for file in files:
            st.write("Imagen cargada correctamente")
            st.image(file, caption="Imagen a clasificar", use_container_width=True)

            # Realizar la clasificación de la imagen
            image = load_img(file)
            etiqueta = predecir_imagen_con_umbral(image, model)
            st.subheader(f"Categoría del producto: {etiqueta}")