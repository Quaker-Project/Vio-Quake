import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import timedelta
from pygam import GAM, s
from shapely.geometry import Point
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
import io

st.set_page_config(page_title="Simulador de Eventos", layout="wide")

st.title("🔢 Simulador Espacio-Temporal de Eventos")

st.sidebar.header("1. Cargar Datos")
uploaded_file = st.sidebar.file_uploader("Sube un archivo .csv o .xlsx con columnas: Fecha, Hora (opcional), Long, Lat", type=['csv', 'xlsx'])
usar_hora = st.sidebar.checkbox("🕒 ¿Incluir hora en la simulación?", value=False)

@st.cache_data

def cargar_datos(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    # Eliminar filas sin coordenadas
    df = df.dropna(subset=['Long', 'Lat'])

    # Si se usa la hora, combinar Fecha + Hora
    if usar_hora and 'Hora' in df.columns:
        df['Fecha'] = pd.to_datetime(df['Fecha'].astype(str) + ' ' + df['Hora'].astype(str), errors='coerce')
    else:
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')

    df = df.dropna(subset=['Fecha'])
    return df

if uploaded_file:
    df = cargar_datos(uploaded_file)
    st.success(f"{len(df)} eventos cargados correctamente.")
    st.map(df[['Lat', 'Long']])

    st.sidebar.header("2. Entrenamiento del Modelo")
    fecha_min, fecha_max = df['Fecha'].min(), df['Fecha'].max()
    fecha_inicio = st.sidebar.date_input("Fecha inicio entrenamiento", value=fecha_min, min_value=fecha_min, max_value=fecha_max)
    fecha_fin = st.sidebar.date_input("Fecha fin entrenamiento", value=fecha_max, min_value=fecha_min, max_value=fecha_max)

    from datetime import datetime
    if isinstance(fecha_inicio, datetime): fecha_inicio = fecha_inicio.date()
    if isinstance(fecha_fin, datetime): fecha_fin = fecha_fin.date()

    from funciones_simulacion import entrenar_modelo_gam

    gam_model, min_fecha_train, factor_ajuste = entrenar_modelo_gam(df, fecha_inicio, fecha_fin, usar_hora)
    st.sidebar.markdown(f"**Factor ajuste:** {factor_ajuste:.2f}")

    st.sidebar.header("3. Simulación")
    fecha_inicio_sim = st.sidebar.date_input("Fecha inicio simulación", value=fecha_max + timedelta(days=1))
    fecha_fin_sim = st.sidebar.date_input("Fecha fin simulación", value=fecha_max + timedelta(days=10))

    mu_boost = st.sidebar.slider("μ Boost (Multiplicador de intensidad)", 0.1, 5.0, 1.0, 0.1)
    alpha = st.sidebar.slider("α (Autoexcitación temporal)", 0.0, 2.0, 0.5, 0.1)
    beta = st.sidebar.slider("β (Decaimiento temporal)", 0.01, 1.0, 0.1, 0.01)
    gamma = st.sidebar.slider("γ (Decaimiento espacial)", 0.01, 1.0, 0.05, 0.01)

    simular = st.sidebar.button("▶️ Ejecutar Simulación")

    if simular:
        # Crear polígonos artificiales (buffer de 0.01 grados)
        from shapely.geometry import Polygon
        buffers = [Point(xy).buffer(0.01) for xy in zip(df['Long'], df['Lat'])]
        union = gpd.GeoSeries(buffers).unary_union
        gdf_zona = gpd.GeoDataFrame(geometry=[union], crs="EPSG:4326").explode(index_parts=False).reset_index(drop=True)

        from funciones_simulacion import simular_eventos
        gdf_sim = simular_eventos(
            df, fecha_inicio, fecha_fin, fecha_inicio_sim, fecha_fin_sim,
            gdf_zona, gam_model, min_fecha_train,
            factor_ajuste=factor_ajuste,
            mu_boost=mu_boost,
            alpha=alpha, beta=beta, gamma=gamma,
            usar_hora=usar_hora
        )

        st.subheader("🌍 Mapa de eventos simulados")
        st.map(gdf_sim[['Lat', 'Long']])

        st.subheader("🔄 Comparativa de frecuencias")
        df['tipo'] = 'real'
        gdf_sim['tipo'] = 'simulado'
        df_plot = pd.concat([df[['Fecha', 'tipo']], gdf_sim[['Fecha', 'tipo']]])

        if usar_hora:
            df_plot['Fecha'] = df_plot['Fecha'].dt.floor('H')
        else:
            df_plot['Fecha'] = df_plot['Fecha'].dt.date

        resumen = df_plot.groupby(['Fecha', 'tipo']).size().reset_index(name='conteo')
        resumen = resumen.pivot(index='Fecha', columns='tipo', values='conteo').fillna(0)

        fig, ax = plt.subplots(figsize=(12, 4))
        resumen.plot(ax=ax)
        plt.ylabel("Eventos diarios")
        st.pyplot(fig)

        st.download_button(
            label="💾 Descargar resultados",
            data=gdf_sim.to_csv(index=False).encode(),
            file_name="eventos_simulados.csv",
            mime="text/csv"
        )

else:
    st.info("Por favor, sube un archivo para comenzar.")
