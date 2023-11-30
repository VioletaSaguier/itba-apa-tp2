import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Cargar los datos
# Asegúrate de que la ruta al archivo CSV sea accesible por tu aplicación Streamlit

# URL del archivo CSV en formato raw desde GitHub
url = 'https://raw.githubusercontent.com/VioletaSaguier/itba-apa-tp2/main/basemodif.csv'

# Cargar los datos desde GitHub
df = pd.read_csv(url)
# Preprocesamiento de datos (aquí deberías agregar tu lógica de preprocesamiento)
# Por ejemplo, combinar características relevantes en una columna 'combined_features'
df['combined_features'] = df.apply(lambda x: f"{x['age']} {x['race']} {x['gender']}", axis=1)

# Crear el modelo TF-IDF y calcular la matriz de similitud
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Función para obtener recomendaciones para un nuevo usuario
def get_recommendations(new_user_profile, tfidf_matrix, cosine_sim):
    # Transformar el perfil del nuevo usuario a un vector TF-IDF
    new_user_vector = tfidf.transform([new_user_profile])

    # Calcular la similitud con los usuarios existentes
    sim_scores = linear_kernel(new_user_vector, tfidf_matrix).flatten()

    # Obtener los índices de los usuarios más similares
    top_user_indices = sim_scores.argsort()[-10:][::-1]  # Top 10 usuarios similares

    # Devolver las recomendaciones
    return df['iid'].iloc[top_user_indices]

# Interfaz de usuario en Streamlit
st.markdown(
    """
    <link rel="stylesheet" href="styles.css">
    """,
    unsafe_allow_html=True
)

# Title and page color
st.markdown("<h1 class='title'>🩷El amor de tu vida🩷</h1>", unsafe_allow_html=True)
st.markdown("### y otros por si no funciona 🤭")

with st.form("my_form"):
    age = st.number_input('Edad', min_value=18, max_value=100, step=1)
    gender = st.selectbox('Género', ['Hombre', 'Mujer', 'Otro'])
    race = st.selectbox('Raza', ['Black/African American', 'European/Caucasian-American', 'Latino/Hispanic American','Asian/Pacific Islander/Asian-American','Native American','Other'])  # Ajusta las opciones según tus datos
    
    submitted = st.form_submit_button("Obtener Recomendaciones")
    if submitted:
        user_profile = f"{age} {race} {gender}"
        recommendations = get_recommendations(user_profile, tfidf_matrix, cosine_sim)
        st.write("Recomendaciones:", recommendations)
