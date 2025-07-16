import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from datetime import datetime
from simulador import entrenar_modelo_gam, simular_eventos

st.set_page_config(layout="wide", page_title="Simulador Espacio-Temporal de Eventos")

st.title("🌀 Simulador Espacio-Temporal de Eventos")

# --- Carga de datos
st.sidebar.header("1. Carga de datos")
uploaded_file = st.sidebar.file_uploader("Sube un CSV con columnas: Fecha, Long, Lat, [Hora]", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Validación de columnas
    required_cols = {'Fecha', 'Long', 'Lat'}
    if not required_cols.issubset(df.columns):
        st.error("❌ El archivo debe contener al menos las columnas: Fecha, Long, Lat")
        st.stop()

    df['Fecha'] = pd.to_datetime(df['Fecha'])
    
    usar_hora = False
    if 'Hora' in df.columns:
        usar_hora = st.sidebar.checkbox("¿Incluir columna Hora?", value=False)
        if usar_hora:
            df['Hora'] = pd.to_timedelta(df['Hora'])
            df['Fecha'] = df['Fecha'] + df['Hora']

    st.map(df, latitude="Lat", longitude="Long")

    # --- Zona de simulación
    st.sidebar.header("2. Zona de simulación")
    zona_file = st.sidebar.file_uploader("Sube un archivo GeoJSON con polígonos", type=["geojson", "json"])

    if zona_file:
        gdf_zona = gpd.read_file(zona_file)
        gdf_zona = gdf_zona.to_crs(epsg=4326)

        # --- Parámetros de entrenamiento y simulación
        st.sidebar.header("3. Fechas y parámetros")
        col1, col2 = st.sidebar.columns(2)
        fecha_inicio_train = col1.date_input("Inicio entrenamiento", df['Fecha'].min().date())
        fecha_fin_train = col2.date_input("Fin entrenamiento", df['Fecha'].max().date())

        col3, col4 = st.sidebar.columns(2)
        fecha_inicio_sim = col3.date_input("Inicio simulación", df['Fecha'].max().date())
        fecha_fin_sim = col4.date_input("Fin simulación", df['Fecha'].max().date())

        mu_boost = st.sidebar.slider("Escala de intensidad (μ boost)", 0.1, 3.0, 1.0, 0.1)
        alpha = st.sidebar.slider("α (autoexcitación)", 0.0, 1.0, 0.3, 0.05)
        beta = st.sidebar.slider("β (decaimiento temporal)", 0.01, 1.0, 0.1, 0.01)
        gamma = st.sidebar.slider("γ (decaimiento espacial)", 0.01, 1.0, 0.05, 0.01)
        seed = st.sidebar.number_input("Seed aleatoria (opcional)", value=42)

        st.sidebar.header("4. Ejecutar simulación")
        if st.sidebar.button("Entrenar y simular"):
            with st.spinner("Entrenando modelo GAM..."):
                modelo_gam, min_fecha_train, factor_ajuste = entrenar_modelo_gam(
                    df,
                    fecha_inicio=str(fecha_inicio_train),
                    fecha_fin=str(fecha_fin_train),
                    usar_hora=usar_hora
                )

            with st.spinner("Simulando eventos..."):
                gdf_sim = simular_eventos(
                    df,
                    fecha_inicio_train=str(fecha_inicio_train),
                    fecha_fin_train=str(fecha_fin_train),
                    fecha_inicio_sim=str(fecha_inicio_sim),
                    fecha_fin_sim=str(fecha_fin_sim),
                    gdf_zona=gdf_zona,
                    modelo_gam=modelo_gam,
                    min_fecha_train=min_fecha_train,
                    factor_ajuste=factor_ajuste,
                    mu_boost=mu_boost,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    max_eventos=10000,
                    seed=int(seed),
                    usar_hora=usar_hora
                )

            # Resultados
            st.success(f"✅ Simulación completada. Eventos generados: {len(gdf_sim)}")

            colA, colB, colC = st.columns(3)
            with colA:
                dias_real = (pd.to_datetime(fecha_fin_train) - pd.to_datetime(fecha_inicio_train)).days + 1
                media_real = len(df[(df['Fecha'] >= pd.to_datetime(fecha_inicio_train)) &
                                    (df['Fecha'] <= pd.to_datetime(fecha_fin_train))]) / dias_real
                st.metric("Media diaria real", f"{media_real:.2f}")

            with colB:
                dias_sim = (pd.to_datetime(fecha_fin_sim) - pd.to_datetime(fecha_inicio_sim)).days + 1
                media_sim = len(gdf_sim) / dias_sim if dias_sim > 0 else 0
                st.metric("Media diaria simulada", f"{media_sim:.2f}")

            with colC:
                st.metric("μ boost", f"{mu_boost} | α: {alpha}")

            st.subheader("Eventos simulados")
            st.map(gdf_sim, latitude="Lat", longitude="Long")

            st.subheader("📥 Descargar eventos simulados")
            csv = gdf_sim.drop(columns="geometry").to_csv(index=False)
            st.download_button("Descargar CSV", csv, file_name="eventos_simulados.csv", mime="text/csv")
