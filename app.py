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
import streamlit as st

# Custom CSS
import streamlit as st

st.markdown(
    """
    <style>
    .title {
        color: pink; /* Updated color to pink */
        font-size: 30px;
        text-align: center;
        padding-top: 50px;
        padding-bottom: 30px;
    }

    body {
        background-image: url('https://emojicombos.com/wp-content/uploads/2020/04/hearts100.png'); /* Added hearts background image */
        background-repeat: repeat;
        background-size: contain;
        background-color: #f5f5f5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Rest of the code remains the same
st.markdown("<h1 class='title'>Sistema de Recomendación de Citas</h1>", unsafe_allow_html=True)
st.markdown("<div style='background-color: white; padding: 20px;'>", unsafe_allow_html=True)

with st.form("my_form"):
    age = st.number_input('Edad', min_value=18, max_value=100, step=1)
    gender = st.selectbox('Género', ['Hombre', 'Mujer', 'Otro'])
    race = st.selectbox('Raza', ['Raza 1', 'Raza 2', 'Raza 3'])  # Ajusta las opciones según tus datos

    submitted = st.form_submit_button("Obtener Recomendaciones")
    if submitted:
        user_profile = f"{age} {race} {gender}"
        recommendations = get_recommendations(user_profile, tfidf_matrix, cosine_sim)
        st.write("Recomendaciones:", recommendations)
        
st.markdown("</div>", unsafe_allow_html=True)

