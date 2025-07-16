import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
from funciones_simulacion import entrenar_modelo_gam, simular_eventos
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")

st.title("Simulador de eventos sísmicos")

# Cargar datos
@st.cache_data
def cargar_datos():
    df = pd.read_csv("data/violencia.csv", parse_dates=["Fecha"])
    gdf = gpd.read_file("data/zonas.geojson")
    return df, gdf

df, gdf_zona = cargar_datos()

# Mostrar mapa de eventos originales
st.subheader("Eventos originales")
df_map = df.rename(columns={'Lat': 'lat', 'Long': 'lon'})
st.map(df_map[['lat', 'lon']])

# Parámetros de entrenamiento y simulación
st.sidebar.header("Parámetros de entrenamiento")
fecha_min = df['Fecha'].min()
fecha_max = df['Fecha'].max()

fecha_inicio_train = st.sidebar.date_input("Fecha inicio entrenamiento", value=fecha_min, min_value=fecha_min, max_value=fecha_max)
fecha_fin_train = st.sidebar.date_input("Fecha fin entrenamiento", value=fecha_max, min_value=fecha_min, max_value=fecha_max)

st.sidebar.header("Parámetros de simulación")
fecha_inicio_sim = st.sidebar.date_input("Fecha inicio simulación", value=fecha_max, min_value=fecha_min, max_value=fecha_max)
fecha_fin_sim = st.sidebar.date_input("Fecha fin simulación", value=fecha_max, min_value=fecha_min, max_value=fecha_max)

mu_boost = st.sidebar.slider("Intensidad base (mu_boost)", 0.1, 5.0, 1.0, step=0.1)
alpha = st.sidebar.slider("Alpha (excitación)", 0.0, 1.0, 0.5, step=0.05)
beta = st.sidebar.slider("Beta (decaimiento temporal)", 0.01, 1.0, 0.1, step=0.01)
gamma = st.sidebar.slider("Gamma (decaimiento espacial)", 0.01, 1.0, 0.05, step=0.01)
max_eventos = st.sidebar.number_input("Máximo número de eventos", min_value=100, max_value=10000, value=1000, step=100)
usar_hora = st.sidebar.checkbox("Usar precisión por hora", value=False)
simular = st.sidebar.button("Simular")

if simular:
    st.subheader("Simulación en curso...")
    with st.spinner("Entrenando modelo y generando eventos..."):
        modelo_gam, min_fecha, factor_ajuste = entrenar_modelo_gam(df, fecha_inicio_train, fecha_fin_train, usar_hora)
        gdf_simulados = simular_eventos(
            df, fecha_inicio_train, fecha_fin_train,
            fecha_inicio_sim, fecha_fin_sim,
            gdf_zona, modelo_gam, min_fecha,
            factor_ajuste, mu_boost,
            alpha, beta, gamma,
            max_eventos, seed=42,
            usar_hora=usar_hora
        )

    st.success(f"{len(gdf_simulados)} eventos simulados.")
    if not gdf_simulados.empty:
        df_sim_map = gdf_simulados.rename(columns={'Lat': 'lat', 'Long': 'lon'})
        st.map(df_sim_map[['lat', 'lon']])

        # Mostrar histograma
        fig, ax = plt.subplots(figsize=(10, 4))
        gdf_simulados['Fecha'].dt.to_period("D").value_counts().sort_index().plot(kind='bar', ax=ax)
        ax.set_title("Eventos simulados por día")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Nº de eventos")
        st.pyplot(fig)
    else:
        st.warning("No se generaron eventos en la simulación.")
