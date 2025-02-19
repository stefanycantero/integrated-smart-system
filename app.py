import streamlit as st
import os
import demand_prediction, product_classification, recommendation_system

st.set_page_config(page_title="Sistema Inteligente Integrado", page_icon="🧠", layout="wide")

def display_menu():
    menu = ["Inicio", "Predicción de demanda", "Clasificación de productos", "Recomendación de productos"]
    choice = st.sidebar.selectbox("Menú", menu)
    return choice

def display_home():
    # Contenido de inicio
    st.write("Este sistema te permite mejorar la toma de decisiones en tu empresa y optimizar recursos a través de tres funcionalidades: \n - Predecir la demanda de tus productos a 30 días \n - Organizar tu inventario con clasificación automática de productos \n - Recomendar productos a tus clientes basado en sus preferencias")
    
    with st.expander("Aquí tienes un video guía para saber más del sistema"):
        st.video(os.path.join(os.getcwd(), "videos", "General.mp4"))

    st.write("↖️Selecciona una opción del menú lateral para comenzar. Para conocer más acerca del desarrollo del proyecto puedes visitar:")
    st.write("📄 [Reporte técnico](https://www.notion.so/Trabajo-3-Aplicaciones-de-Redes-Neuronales-19e283d7bb8180e7ac64ef806c5c4a14)")
    st.write("📦 [Repositorio](https://github.com/stefanycantero/integrated-smart-system)")

# Encabezado
st.title("Sistema Inteligente Integrado")
st.divider()
# Menú de navegación
choice = display_menu()

if choice == "Predicción de demanda":
    demand_prediction.display_demand_prediction()
elif choice == "Clasificación de productos":
    product_classification.display_product_classification()
elif choice == "Recomendación de productos":
    recommendation_system.display_product_recommendation()
else:
    display_home()

# Enlace al reporte técnico y al repositorio
st.divider()
st.write("Desarrollado con fines académicos para la asignatura Redes Neuronales y Algoritmos Bioinspirados, Universidad Nacional de Colombia sede Medellín.")
st.divider()