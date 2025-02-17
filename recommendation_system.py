import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data():
    df = pd.read_csv('models/amazon_products.csv')  
    df = df[['name', 'main_category', 'sub_category', 'ratings']].fillna('')
    df['text_features'] = df['name'] + ' ' + df['main_category'] + ' ' + df['sub_category']
    return df

def train_recommender(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['text_features'])
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return vectorizer, similarity_matrix

def get_recommendations(product_name, df, vectorizer, similarity_matrix):
    idx = df[df['name'].str.lower() == product_name.lower()].index
    if len(idx) == 0:
        return []
    idx = idx[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in scores[1:6]]  # 5 mejores recomendaciones
    return df.iloc[top_indices]

def display_product_recommendation():
    st.write("Este módulo te permite obtener recomendaciones personalizadas de productos.")
    st.write("Añadir una breve descripción del desarrollo del modelo y las métricas de evaluación obtenidas en entrenamiento y validación.")
    
    with st.expander("➡️Aquí tienes un video guía para utilizar este módulo"):
        st.video('https://www.youtube.com/watch?v=i5cHUTSnkGQ')
    
    st.divider()

    df = load_data()
    if df.empty:
        st.write("Error al cargar los datos.")
        return

    vectorizer, similarity_matrix = train_recommender(df)
    all_products = df['name'].tolist()
    product_name = st.selectbox("Selecciona el producto", all_products)
    if st.button('Obtener recomendaciones'):
        recommendations = get_recommendations(product_name, df, vectorizer, similarity_matrix)
        if not recommendations.empty:
            st.write(recommendations)
        else:
            st.write('No se encontraron recomendaciones.')