import streamlit as st

def display_menu():
    menu = ['Predicción de demanda', 'Clasificación de productos', 'Recomendación de productos']
    choice = st.sidebar.selectbox('Menú', menu)
    return choice

# Cambiar el título de la página
st.set_page_config(page_title='Sistema Inteligente Integrado', page_icon='🧠', layout='wide')

# Encabezado
st.title('Sistema Inteligente Integrado')
st.subheader('Predicción de demanda | Clasificación automática de productos | Recomendación personalizada')
st.divider()

# Menú de navegación
choice = display_menu()

if choice == 'Predicción de demanda':
    pass
elif choice == 'Clasificación de productos':
    pass
else:
    pass

# Contenido de la app
st.write('Este sistema te permite mejorar la toma de decisiones en tu empresa y optimizar recursos a través de tres funcionalidades: \n - Predicción de demanda a 30 días \n - Organizar tu inventario con clasificación automática de productos \n - Recomendar productos a tus clientes basado en sus preferencias')
st.write('↖️Selecciona una opción del menú lateral para comenzar.')
st.write('⬇️Si tienes dudas, mira el video para una guía rápida:')
st.video('https://www.youtube.com/watch?v=i5cHUTSnkGQ')
st.divider()

# Enlace al reporte técnico y al repositorio

st.write('📄 [Reporte técnico](https://drive.google.com)')
st.write('📦 [Repositorio](https://github.com/stefanycantero/integrated-smart-system)')
st.write('Desarrollado con fines académicos para la asignatura Redes Neuronales y Algoritmos Bioinspirados, Universidad Nacional de Colombia sede Medellín.')