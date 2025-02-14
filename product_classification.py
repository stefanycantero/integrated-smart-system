import streamlit as st

model = None # Cargar modelo (Pendiente)

def display_product_classification():

    st.subheader("Clasificación automática de productos")
    st.write("Este módulo te permite organizar tu inventario con clasificación automática de productos.")
    st.write("AÑADIR Breve descripción del desarrllo del modelo y las métricas de evaluación obtenidas en entrenamiento y validación.")

    # Realizar una nueva clasificación
    with st.expander("➡️Aquí tienes un video guía para utilizar este módulo"):
        st.video('https://www.youtube.com/watch?v=i5cHUTSnkGQ')

    file = st.file_uploader("Carga aquí la imagen que deseas clasificar", type=["jpg", "jpeg", "png"])

    if file is not None:
        # Procesar datos (Pendiente el preprocesamiento)
        datos = file

        # Realizar predicción
        classification = model.predict(datos)

        # Mostrar resultados
        st.image(datos, caption='Imagen cargada')
        st.write("Categoría del producto")
        st.write(classification)