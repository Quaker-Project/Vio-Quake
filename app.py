import streamlit as st
import pandas as pd
import geopandas as gpd
from simulador import entrenar_modelo_gam, simular_eventos
import tempfile
import os
import io

# Configuración visual
st.set_page_config(page_title="VIO-QUAKE Simulador", layout="wide", initial_sidebar_state="expanded")

def css_estilo():
    st.markdown("""
    <style>
        body {
            background-color: #111111;
            color: #EEEEEE;
        }
        .stApp {
            background-color: #111111;
            color: #EEEEEE;
            font-family: 'Segoe UI', sans-serif;
        }
        .stSidebar {
            background-color: #1c1c1c;
        }
        .stButton>button, .stDownloadButton>button {
            background-color: #ff4b4b;
            color: white;
            border-radius: 8px;
            padding: 0.5em 1em;
            font-weight: bold;
        }
        .stButton>button:hover, .stDownloadButton>button:hover {
            background-color: #ff1c1c;
            transform: scale(1.05);
        }
    </style>
    """, unsafe_allow_html=True)

css_estilo()

def cargar_archivo_datos(archivo):
    if archivo is None:
        return None
    try:
        if archivo.name.endswith('.csv'):
            df = pd.read_csv(archivo)
        elif archivo.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(archivo)
        else:
            st.error("Formato no soportado. Use CSV o Excel.")
            return None
    except Exception as e:
        st.error(f"Error cargando archivo: {e}")
        return None
    return df

def cargar_shapefile_zip(archivo_zip):
    if archivo_zip is None:
        return None
    import zipfile
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with zipfile.ZipFile(archivo_zip) as z:
                z.extractall(tmpdir)
            shp_files = [f for f in os.listdir(tmpdir) if f.endswith('.shp')]
            if len(shp_files) != 1:
                st.error("El ZIP debe contener un único archivo .shp")
                return None
            gdf = gpd.read_file(os.path.join(tmpdir, shp_files[0]))
            return gdf
        except Exception as e:
            st.error(f"Error leyendo shapefile ZIP: {e}")
            return None

def main():
    st.title("🧨 VIO-QUAKE | Simulador de Eventos Delictivos Basado en Procesos Hawkes")

    st.markdown("""
    **Simulación de eventos espacio-temporales con autoexcitación**

    Este sistema permite simular patrones de delitos replicando comportamientos observados en los datos históricos. Ajusta parámetros en la barra lateral y lanza simulaciones.
    """)

    archivo_datos = st.file_uploader("📂 Suba datos de eventos (CSV/Excel)", type=["csv", "xls", "xlsx"])
    df = cargar_archivo_datos(archivo_datos)

    incluye_hora = False
    if df is not None and 'Hora' in df.columns:
        incluye_hora = st.checkbox("¿El dataset incluye columna de hora?", value=True)

    archivo_zip = st.file_uploader("📍 Suba shapefile ZIP de zona (área de simulación)", type=["zip"])
    gdf_zona = cargar_shapefile_zip(archivo_zip)

    if df is not None:
        columnas_requeridas = ['Fecha', 'Lat', 'Long']
        if not all(col in df.columns for col in columnas_requeridas):
            st.error(f"Faltan columnas obligatorias: {columnas_requeridas}")
            return

        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
        if incluye_hora:
            if 'Hora' not in df.columns:
                st.error("Se indicó que hay hora, pero falta la columna 'Hora'")
                return
            df['Hora'] = pd.to_timedelta(df['Hora'].astype(str))
            df['FechaHora'] = df['Fecha'] + df['Hora']
        else:
            df['FechaHora'] = df['Fecha']

        if df['FechaHora'].isnull().any():
            st.error("Fechas/Horas mal formateadas. Corríjalas.")
            return

    if df is not None and gdf_zona is not None:
        st.sidebar.header("⚙️ Configuración de simulación")

        fecha_min = df['FechaHora'].min().date()
        fecha_max = df['FechaHora'].max().date()

        fecha_inicio_train = st.sidebar.date_input("Fecha inicio entrenamiento", value=fecha_min)
        fecha_fin_train = st.sidebar.date_input("Fecha fin entrenamiento", value=fecha_max)

        fecha_inicio_sim = st.sidebar.date_input("Fecha inicio simulación", value=fecha_max + pd.Timedelta(days=1))
        fecha_fin_sim = st.sidebar.date_input("Fecha fin simulación", value=fecha_max + pd.Timedelta(days=30))

        mu_boost = st.sidebar.slider("Multiplicador de intensidad base (mu_boost)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

        st.sidebar.subheader("🌐 Autoexcitación espacio-temporal")
        alpha = st.sidebar.slider("Alpha (autoexcitación)", 0.0, 2.0, 0.5, 0.1)
        beta = st.sidebar.slider("Beta (decaimiento temporal)", 0.01, 1.0, 0.1, 0.01)
        gamma = st.sidebar.slider("Gamma (decaimiento espacial)", 0.01, 1.0, 0.05, 0.01)

        max_eventos = st.sidebar.number_input("Máximo de eventos simulados", min_value=100, max_value=100000, value=5000, step=100)
        usar_semilla = st.sidebar.checkbox("Fijar semilla aleatoria", value=False)

        if st.button("🚀 Entrenar modelo y simular eventos"):
            with st.spinner("🔧 Entrenando modelo GAM espaciotemporal..."):
                modelo_gam, min_fecha_train, factor_ajuste = entrenar_modelo_gam(
                    df.rename(columns={'FechaHora': 'Fecha'}), fecha_inicio_train, fecha_fin_train
                )
                st.info(f"Factor de ajuste: {factor_ajuste:.2f} | mu_boost: {mu_boost:.2f}")

            with st.spinner("🎲 Simulando eventos..."):
                gdf_sim = simular_eventos(
                    df.rename(columns={'FechaHora': 'Fecha'}),
                    fecha_inicio_train, fecha_fin_train,
                    fecha_inicio_sim, fecha_fin_sim,
                    gdf_zona, modelo_gam, min_fecha_train,
                    factor_ajuste=factor_ajuste,
                    mu_boost=mu_boost,
                    alpha=alpha, beta=beta, gamma=gamma,
                    max_eventos=max_eventos,
                    seed=42 if usar_semilla else None
                )

            st.success(f"✅ Simulados {len(gdf_sim)} eventos")

            # Exportar
            gdf_sim_wgs84 = gdf_sim.to_crs(epsg=4326)
            df_export = gdf_sim_wgs84[['geometry', 'Fecha']].copy()
            df_export['Long'] = df_export.geometry.x
            df_export['Lat'] = df_export.geometry.y
            df_export = df_export.drop(columns='geometry')
            df_export = df_export[['Long', 'Lat', 'Fecha']]

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_export.to_excel(writer, index=False)
            output.seek(0)

            st.download_button(
                label="📥 Descargar Excel con eventos simulados",
                data=output,
                file_name="eventos_simulados.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == "__main__":
    main()
