import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

# Cargar el modelo TF-IDF y la matriz de similitud
with open('modelo_recomendacion.pkl', 'rb') as file:
    tfidf, cosine_sim, df = pickle.load(file)  # Asegúrate de que df también se incluya en el pickle

# Función para obtener recomendaciones para un nuevo usuario
def get_recommendations(new_user_profile, tfidf_model, tfidf_matrix, cosine_sim):
    # Transformar el perfil del nuevo usuario a un vector TF-IDF
    new_user_vector = tfidf_model.transform([new_user_profile])

    # Calcular la similitud con los usuarios existentes
    sim_scores = linear_kernel(new_user_vector, tfidf_matrix).flatten()

    # Obtener los índices de los usuarios más similares
    top_user_indices = sim_scores.argsort()[-10:][::-1]  # Top 10 usuarios similares

    # Devolver las recomendaciones
    return df['iid'].iloc[top_user_indices]

# Interfaz de usuario en Streamlit
st.title('Sistema de Recomendación de Citas')

with st.form("my_form"):
    age = st.number_input('Edad', min_value=18, max_value=100, step=1)
    gender = st.selectbox('Género', ['Hombre', 'Mujer', 'Otro'])
    race = st.selectbox('Raza', ['Raza 1', 'Raza 2', 'Raza 3'])  # Ajusta las opciones según tus datos

    submitted = st.form_submit_button("Obtener Recomendaciones")
    if submitted:
        user_profile = f"{age} {race} {gender}"
        recommendations = get_recommendations(user_profile, tfidf, tfidf, cosine_sim)
        st.write("Recomendaciones:", recommendations)
