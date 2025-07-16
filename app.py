import streamlit as st
import pandas as pd
import geopandas as gpd
import io
import os
import tempfile
from simulador import entrenar_modelo_gam, simular_eventos

# Configuración inicial de la app
st.set_page_config(
    page_title="VIO-QUAKE Simulador",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo visual personalizado
def css_estilo():
    st.markdown("""
    <style>
        .stApp { background-color: #111; color: #eee; font-family: 'Segoe UI', sans-serif; }
        .stSidebar { background-color: #1c1c1c; }
        .stButton>button, .stDownloadButton>button {
            border-radius: 8px;
            border: none;
            padding: 0.5em 1em;
            font-weight: bold;
            transition: 0.3s;
        }
        .stButton>button { background-color: #ff4b4b; color: white; }
        .stButton>button:hover { background-color: #ff1c1c; transform: scale(1.05); }
        .stDownloadButton>button { background-color: #4b6fff; color: white; }
        .stDownloadButton>button:hover { background-color: #1c44ff; transform: scale(1.05); }
    </style>
    """, unsafe_allow_html=True)

css_estilo()

# Función para cargar datos de eventos
def cargar_archivo_datos(archivo):
    if archivo is None:
        return None
    try:
        if archivo.name.endswith('.csv'):
            df = pd.read_csv(archivo, parse_dates=['Fecha'])
        elif archivo.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(archivo, parse_dates=['Fecha'])
        else:
            st.error("Formato no soportado. Use CSV o Excel.")
            return None
    except Exception as e:
        st.error(f"Error cargando archivo: {e}")
        return None
    return df

# Función para cargar shapefile desde ZIP
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

# Función principal de la app
def main():
    st.title("🧨 VIO-QUAKE | Simulador de Eventos Delictivos Basado en Procesos Hawkes")

    st.markdown("**Simulación espacio-temporal con autoexcitación. Ajusta parámetros y lanza simulaciones.**")
    
    archivo_datos = st.file_uploader("📂 Suba datos de eventos (CSV/Excel)", type=["csv", "xls", "xlsx"])
    df = cargar_archivo_datos(archivo_datos)

    archivo_zip = st.file_uploader("📍 Suba shapefile ZIP de zona (área de simulación)", type=["zip"])
    gdf_zona = cargar_shapefile_zip(archivo_zip)

    if df is not None:
        columnas_necesarias = ['Long', 'Lat', 'Fecha']
        if not all(col in df.columns for col in columnas_necesarias):
            st.error(f"El archivo debe contener las columnas: {columnas_necesarias}")
            return
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
        if df['Fecha'].isnull().any():
            st.error("Hay valores inválidos en la columna 'Fecha'. Verifique que contienen fecha y hora correctamente.")
            return

    if df is not None and gdf_zona is not None:
        st.sidebar.header("⚙️ Configuración de simulación")

        fecha_inicio_train = st.sidebar.date_input("Fecha inicio entrenamiento", value=df['Fecha'].min().date())
        fecha_fin_train = st.sidebar.date_input("Fecha fin entrenamiento", value=df['Fecha'].max().date())

        fecha_inicio_sim = st.sidebar.date_input("Fecha inicio simulación", value=(df['Fecha'].max() + pd.Timedelta(days=1)).date())
        fecha_fin_sim = st.sidebar.date_input("Fecha fin simulación", value=(df['Fecha'].max() + pd.Timedelta(days=30)).date())

        mu_boost = st.sidebar.slider("Multiplicador de intensidad base (mu_boost)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

        st.sidebar.subheader("🌐 Autoexcitación espacio-temporal")
        alpha = st.sidebar.slider("Alpha (autoexcitación)", 0.0, 2.0, 0.5, 0.1)
        beta = st.sidebar.slider("Beta (decaimiento temporal)", 0.01, 1.0, 0.1, 0.01)
        gamma = st.sidebar.slider("Gamma (decaimiento espacial)", 0.01, 1.0, 0.05, 0.01)

        max_eventos = st.sidebar.number_input("Máximo eventos simulados", min_value=100, max_value=100000, value=5000, step=100)
        usar_semilla = st.sidebar.checkbox("Fijar semilla aleatoria", value=False)

        if st.button("🚀 Entrenar modelo y simular eventos"):
            with st.spinner("Entrenando modelo GAM..."):
                modelo_gam, min_fecha_train, factor_ajuste = entrenar_modelo_gam(df, fecha_inicio_train, fecha_fin_train)
                st.info(f"Factor de ajuste: {factor_ajuste:.2f} | Boost: {mu_boost:.2f}")

            with st.spinner("Simulando eventos..."):
                gdf_sim = simular_eventos(
                    df, fecha_inicio_train, fecha_fin_train,
                    fecha_inicio_sim, fecha_fin_sim,
                    gdf_zona, modelo_gam, min_fecha_train,
                    factor_ajuste=factor_ajuste,
                    mu_boost=mu_boost,
                    alpha=alpha, beta=beta, gamma=gamma,
                    max_eventos=max_eventos,
                    seed=42 if usar_semilla else None
                )

            st.success(f"✅ Simulados {len(gdf_sim)} eventos")

            # Calcular medias reales y simuladas con fechas + horas
            fecha_inicio_sim_dt = pd.to_datetime(fecha_inicio_sim)
            fecha_fin_sim_dt = pd.to_datetime(fecha_fin_sim + pd.Timedelta(days=1)) - pd.Timedelta(seconds=1)
            duracion_sim = (fecha_fin_sim_dt - fecha_inicio_sim_dt).total_seconds() / 86400
            duracion_sim = max(duracion_sim, 1)

            media_real = df[(df['Fecha'] >= fecha_inicio_sim_dt) & (df['Fecha'] <= fecha_fin_sim_dt)].shape[0] / duracion_sim
            media_simulada = len(gdf_sim) / duracion_sim
            st.write(f"📊 Media diaria real: **{media_real:.2f}**")
            st.write(f"📊 Media diaria simulada: **{media_simulada:.2f}**")

            # Generar Excel descargable
            gdf_sim_wgs84 = gdf_sim.to_crs(epsg=4326)
            df_out = gdf_sim_wgs84[['Fecha', 'geometry']].copy()
            df_out['Long'] = df_out.geometry.x
            df_out['Lat'] = df_out.geometry.y
            df_out = df_out.drop(columns='geometry')
            df_out = df_out[['Long', 'Lat', 'Fecha']]  # Orden columnas

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_out.to_excel(writer, index=False)
            output.seek(0)

            st.download_button(
                label="📥 Descargar Excel con eventos simulados (con hora)",
                data=output,
                file_name="eventos_simulados.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == "__main__":
    main()
