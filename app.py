import streamlit as st
import pandas as pd
import geopandas as gpd
from simulador import entrenar_modelo_gam, simular_eventos
import tempfile
import os
import io

st.set_page_config(
    page_title="VIO-QUAKE Simulador",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        .stButton>button {
            background-color: #ff4b4b;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.5em 1em;
            font-weight: bold;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #ff1c1c;
            transform: scale(1.05);
        }
        .stDownloadButton>button {
            background-color: #4b6fff;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.5em 1em;
            font-weight: bold;
            transition: 0.3s;
        }
        .stDownloadButton>button:hover {
            background-color: #1c44ff;
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
        elif archivo.name.endswith('.xls') or archivo.name.endswith('.xlsx'):
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

    archivo_zip = st.file_uploader("📍 Suba shapefile ZIP de zona (área de simulación)", type=["zip"])
    gdf_zona = cargar_shapefile_zip(archivo_zip)

    if df is not None:
        tiene_hora = st.checkbox("¿Tiene columna de hora separada?", value=False)

        cols_requeridas = ['Long', 'Lat', 'Fecha'] + (['Hora'] if tiene_hora else [])
        if not all(c in df.columns for c in cols_requeridas):
            st.error(f"Faltan columnas. Se requieren: {cols_requeridas}")
            return

        try:
            if tiene_hora:
                df['Fecha'] = pd.to_datetime(df['Fecha'].astype(str) + " " + df['Hora'].astype(str), errors='coerce')
            else:
                df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
        except Exception as e:
            st.error(f"Error interpretando fechas/horas: {e}")
            return

        if df['Fecha'].isnull().any():
            st.error("Algunas fechas no se pudieron convertir correctamente.")
            return

    if df is not None and gdf_zona is not None:
        st.sidebar.header("⚙️ Configuración de simulación")

        fecha_inicio_train = st.sidebar.date_input("Fecha inicio entrenamiento", value=df['Fecha'].min().date())
        fecha_fin_train = st.sidebar.date_input("Fecha fin entrenamiento", value=df['Fecha'].max().date())

        fecha_inicio_sim = st.sidebar.date_input("Fecha inicio simulación", value=df['Fecha'].max().date() + pd.Timedelta(days=1))
        fecha_fin_sim = st.sidebar.date_input("Fecha fin simulación", value=df['Fecha'].max().date() + pd.Timedelta(days=30))

        mu_boost = st.sidebar.slider("Multiplicador de intensidad base (mu_boost)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

        st.sidebar.subheader("🌐 Autoexcitación espacio-temporal")
        alpha = st.sidebar.slider("Alpha (nivel de autoexcitación)", 0.0, 2.0, 0.5, 0.1)
        beta = st.sidebar.slider("Beta (decaimiento temporal)", 0.01, 1.0, 0.1, 0.01)
        gamma = st.sidebar.slider("Gamma (decaimiento espacial)", 0.01, 1.0, 0.05, 0.01)

        max_eventos = st.sidebar.number_input("Máximo de eventos simulados", min_value=100, max_value=100000, value=5000, step=100)
        usar_semilla = st.sidebar.checkbox("Fijar semilla aleatoria", value=False)

        if st.button("🚀 Entrenar modelo y simular eventos"):
            with st.spinner("🔧 Entrenando modelo GAM espaciotemporal..."):
                modelo_gam, min_fecha_train, factor_ajuste = entrenar_modelo_gam(df, fecha_inicio_train, fecha_fin_train)
                st.info(f"📈 Factor de ajuste automático (histórico): **{factor_ajuste:.2f}**")
                st.info(f"📌 mu_boost seleccionado por el usuario: **{mu_boost:.2f}**")
                st.info(f"📌 alpha seleccionado por el usuario: **{alpha:.2f}**")

            with st.spinner("🎲 Simulando eventos Hawkes espacio-temporal..."):
                gdf_sim = simular_eventos(df, fecha_inicio_train, fecha_fin_train,
                                          fecha_inicio_sim, fecha_fin_sim,
                                          gdf_zona, modelo_gam, min_fecha_train,
                                          factor_ajuste=factor_ajuste,
                                          mu_boost=mu_boost,
                                          alpha=alpha, beta=beta, gamma=gamma,
                                          max_eventos=max_eventos,
                                          seed=42 if usar_semilla else None)

            st.success(f"✅ Simulados {len(gdf_sim)} eventos")

            fecha_inicio_sim_dt = pd.to_datetime(fecha_inicio_sim)
            fecha_fin_sim_dt = pd.to_datetime(fecha_fin_sim)
            dias_sim = max(1, (fecha_fin_sim_dt - fecha_inicio_sim_dt).days + 1)

            media_real = df[(df['Fecha'] >= fecha_inicio_sim_dt) & (df['Fecha'] <= fecha_fin_sim_dt)].shape[0] / dias_sim
            media_simulada = len(gdf_sim) / dias_sim
            st.write(f"📊 Media diaria real: **{media_real:.2f}**")
            st.write(f"📊 Media diaria simulada: **{media_simulada:.2f}**")

            gdf_sim_wgs84 = gdf_sim.to_crs(epsg=4326)
            excel_buffer = gdf_sim_wgs84[['geometry', 'Fecha']].copy()
            excel_buffer['Long'] = excel_buffer.geometry.x
            excel_buffer['Lat'] = excel_buffer.geometry.y
            excel_buffer = excel_buffer.drop(columns='geometry')
            excel_buffer = excel_buffer[['Long', 'Lat', 'Fecha']]

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                excel_buffer.to_excel(writer, index=False)
            output.seek(0)

            st.download_button(
                label="📥 Descargar Excel con eventos simulados",
                data=output,
                file_name="eventos_simulados.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == "__main__":
    main()
