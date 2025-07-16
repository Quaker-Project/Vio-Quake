import streamlit as st
import pandas as pd
import geopandas as gpd
from vioquake.modelo import entrenar_modelo_gam
from vioquake.simulacion import simular_eventos
from vioquake.utils import cargar_archivo_datos, cargar_shapefile_zip
import tempfile
import os
import io

# Estilos visuales
st.set_page_config(
    page_title="VIO-QUAKE Simulador",
    layout="wide",
    initial_sidebar_state="expanded"
)

def css_estilo():
    st.markdown("""
    <style>
        body { background-color: #111111; color: #EEEEEE; }
        .stApp { background-color: #111111; color: #EEEEEE; font-family: 'Segoe UI', sans-serif; }
        .stSidebar { background-color: #1c1c1c; }
        .stButton>button, .stDownloadButton>button {
            background-color: #ff4b4b; color: white; border-radius: 8px; border: none; padding: 0.5em 1em; font-weight: bold; transition: 0.3s;
        }
        .stButton>button:hover, .stDownloadButton>button:hover { background-color: #ff1c1c; transform: scale(1.05); }
    </style>
    """, unsafe_allow_html=True)

css_estilo()

def main():
    st.title("🧨 VIO-QUAKE | Simulador de Eventos Delictivos Multirréplica")

    st.markdown("""
    **Simulación de eventos espacio-temporales con autoexcitación y replicación**
    
    Este sistema permite simular múltiples escenarios de evolución delictiva con control sobre parámetros clave. El sistema incluye un proceso Hawkes con GAM y diferenciación de eventos.
    """)

    archivo_datos = st.file_uploader("📂 Suba datos de eventos (CSV/Excel)", type=["csv", "xls", "xlsx"])
    df = cargar_archivo_datos(archivo_datos)

    archivo_zip = st.file_uploader("📍 Suba shapefile ZIP de zona (área de simulación)", type=["zip"])
    gdf_zona = cargar_shapefile_zip(archivo_zip)

    if df is not None:
        cols_requeridas = ['Long', 'Lat', 'Fecha']
        if not all(c in df.columns for c in cols_requeridas):
            st.error(f"Faltan columnas. Debe tener {cols_requeridas}")
            return
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
        if df['Fecha'].isnull().any():
            st.error("Hay fechas no convertibles en 'Fecha'. Corríjalo.")
            return

    if df is not None and gdf_zona is not None:
        st.sidebar.header("⚙️ Configuración de simulación")

        fecha_inicio_train = st.sidebar.date_input("Fecha inicio entrenamiento", value=df['Fecha'].min())
        fecha_fin_train = st.sidebar.date_input("Fecha fin entrenamiento", value=df['Fecha'].max())

        fecha_inicio_sim = st.sidebar.date_input("Fecha inicio simulación", value=df['Fecha'].max() + pd.Timedelta(days=1))
        fecha_fin_sim = st.sidebar.date_input("Fecha fin simulación", value=df['Fecha'].max() + pd.Timedelta(days=30))

        mu_boost = st.sidebar.slider("Multiplicador de intensidad base (mu_boost)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

        st.sidebar.subheader("🌐 Autoexcitación espacio-temporal")
        alpha = st.sidebar.slider("Alpha (nivel de autoexcitación)", 0.0, 2.0, 0.5, 0.1)
        beta = st.sidebar.slider("Beta (decaimiento temporal)", 0.01, 1.0, 0.1, 0.01)
        gamma = st.sidebar.slider("Gamma (decaimiento espacial)", 0.01, 1.0, 0.05, 0.01)

        max_eventos = st.sidebar.number_input("Máximo de eventos por réplica", min_value=100, max_value=100000, value=5000, step=100)
        n_replicas = st.sidebar.number_input("Número de réplicas (escenarios)", min_value=1, max_value=100, value=10, step=1)

        usar_semilla = st.sidebar.checkbox("Fijar semilla aleatoria", value=False)

        if st.button("🚀 Entrenar modelo y simular eventos"):
            with st.spinner("🔧 Entrenando modelo GAM espaciotemporal..."):
                modelo_gam, min_fecha_train, factor_ajuste = entrenar_modelo_gam(df, fecha_inicio_train, fecha_fin_train)
                st.info(f"Factor de ajuste automático (histórico): {factor_ajuste:.2f}")
                st.info(f"Boost aplicado por el usuario (mu_boost): {mu_boost:.2f}")

            with st.spinner("🎲 Simulando eventos Hawkes multirréplica..."):
                gdf_sim = simular_eventos(df, fecha_inicio_train, fecha_fin_train,
                                          fecha_inicio_sim, fecha_fin_sim,
                                          gdf_zona, modelo_gam, min_fecha_train,
                                          factor_ajuste=factor_ajuste,
                                          mu_boost=mu_boost,
                                          alpha=alpha, beta=beta, gamma=gamma,
                                          max_eventos=max_eventos,
                                          n_replicas=n_replicas,
                                          seed=42 if usar_semilla else None)

            st.success(f"✅ Simuladas {n_replicas} réplicas con un total de {len(gdf_sim)} eventos")

            # Estadísticas básicas
            resumen = gdf_sim.groupby(['Replica', 'Tipo']).size().unstack(fill_value=0)
            st.write("### 📊 Eventos por réplica y tipo de evento")
            st.dataframe(resumen)

            # Preparar Excel para descarga
            gdf_sim_wgs84 = gdf_sim.to_crs(epsg=4326)
            excel_buffer = gdf_sim_wgs84.copy()
            excel_buffer['Long'] = excel_buffer.geometry.x
            excel_buffer['Lat'] = excel_buffer.geometry.y
            excel_buffer = excel_buffer[['Replica', 'IdEvento', 'Tipo', 'Long', 'Lat', 'Fecha']]

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                excel_buffer.to_excel(writer, index=False)
            output.seek(0)

            st.download_button(
                label="📥 Descargar Excel con eventos simulados",
                data=output,
                file_name="eventos_simulados_multireplica.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == "__main__":
    main()
