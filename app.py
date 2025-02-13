import streamlit as st
import demand_prediction, product_classification

def display_menu():
    menu = ["Inicio", "Predicción de demanda", "Clasificación de productos", "Recomendación de productos"]
    choice = st.sidebar.selectbox("Menú", menu)
    return choice

def display_home():
    # Contenido de inicio
    st.write("Este sistema te permite mejorar la toma de decisiones en tu empresa y optimizar recursos a través de tres funcionalidades: \n - Predecir la demanda de tus productos a 30 días \n - Organizar tu inventario con clasificación automática de productos \n - Recomendar productos a tus clientes basado en sus preferencias")
    
    with st.expander("Aquí tienes un video guía para saber más del sistema"):
        st.video("https://www.youtube.com/watch?v=i5cHUTSnkGQ")

    st.write("↖️Selecciona una opción del menú lateral para comenzar. Para conocer más acerca del desarrollo del proyecto puedes visitar:")
    st.write("📄 [Reporte técnico](https://drive.google.com)")
    st.write("📦 [Repositorio](https://github.com/stefanycantero/integrated-smart-system)")

# Cambiar el título de la página
st.set_page_config(page_title="Sistema Inteligente Integrado", page_icon="🧠", layout="wide")

# Encabezado
st.title("Sistema Inteligente Integrado")
st.divider()
# Menú de navegación
choice = display_menu()

if choice == "Inicio":
    display_home()
elif choice == "Predicción de demanda":
    demand_prediction.display_demand_prediction()
elif choice == "Clasificación de productos":
    product_classification.display_product_classification()
elif choice == "Recomendación de productos":
    st.write("🚧 En construcción...")

# Enlace al reporte técnico y al repositorio
st.divider()
st.write("Desarrollado con fines académicos para la asignatura Redes Neuronales y Algoritmos Bioinspirados, Universidad Nacional de Colombia sede Medellín.")
st.divider()