import streamlit as st
import pandas as pd

# Cargar modelo (Pendiente)
model = None

def display_demand_prediction():
    # Widget para cargar archivo
    st.subheader("Predicción de Demanda")
    st.write("Este módulo te permite predecir la demanda de tus productos para los próximos 30 días.")
    st.write("AÑADIR Breve descripción del desarrllo del modelo y las métricas de evaluación obtenidas en entrenamiento y validación.")
    st.image('https://drive.google.com')
    st.divider()

    # Realizar una nueva predicción
    st.subheader("Realizar una nueva predicción")
    with st.expander("➡️Aquí tienes un video guía para utilizar este módulo"):
        st.video('https://www.youtube.com/watch?v=i5cHUTSnkGQ')

    file = st.file_uploader("Carga un archivo CSV con datos históricos", type=["csv"])

    if file is not None:
        # Procesar datos (Pendiente el preprocesamiento)
        datos = pd.read_csv(file)

        # Realizar predicciones
        demand_predictions = model.predict(datos)

        # Mostrar resultados
        st.write("Predicciones para los próximos 30 días")
        st.write(demand_predictions)

        # Mostrar gráfico
        st.write("Gráfico de Predicción")

        # Mostrar métricas de evaluación
        st.write("Métricas de Evaluación")

