import streamlit as st
import pandas as pd
import geopandas as gpd
from simulador import entrenar_modelo_gam, simular_eventos
import tempfile
import os
import io
from dateutil import parser

st.set_page_config(
    page_title="VIO-QUAKE Simulador",
    layout="wide",
    initial_sidebar_state="expanded"
)

def css_estilo():
    st.markdown("""
    <style>
        .stApp { background-color: #111111; color: #EEEEEE; font-family: 'Segoe UI', sans-serif; }
        .stSidebar, .css-1d391kg { background-color: #1c1c1c; }
        .stButton>button, .stDownloadButton>button {
            background-color: #ff4b4b; color: white; border-radius: 8px; padding: 0.5em 1em; font-weight: bold;
        }
        .stButton>button:hover { background-color: #ff1c1c; transform: scale(1.05); }
        .stDownloadButton>button { background-color: #4b6fff; }
        .stDownloadButton>button:hover { background-color: #1c44ff; transform: scale(1.05); }
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

def homogeneizar_fechas(df, columna='Fecha'):
    try:
        df[columna] = df[columna].apply(lambda x: parser.parse(str(x)))
        return df
    except Exception as e:
        st.error(f"No se pudieron convertir las fechas: {e}")
        return None

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

    archivo_datos = st.file_uploader("📂 Suba datos de eventos (CSV/Excel)", type=["csv", "xls", "xlsx"])
    df = cargar_archivo_datos(archivo_datos)

    archivo_zip = st.file_uploader("📍 Suba shapefile ZIP de zona (área de simulación)", type=["zip"])
    gdf_zona = cargar_shapefile_zip(archivo_zip)

    usar_hora = st.checkbox("¿Tus eventos incluyen hora? (por ejemplo 13:45:00)", value=True)

    if df is not None:
        cols_requeridas = ['Long', 'Lat', 'Fecha']
        if not all(c in df.columns for c in cols_requeridas):
            st.error(f"Faltan columnas. Debe tener {cols_requeridas}")
            return

        df = homogeneizar_fechas(df, 'Fecha')
        if df is None or df['Fecha'].isnull().any():
            st.error("Algunas fechas no pudieron convertirse. Verifica el archivo.")
            return

    if df is not None and gdf_zona is not None:
        st.sidebar.header("⚙️ Configuración de simulación")

        fecha_inicio_train = st.sidebar.date_input("Fecha inicio entrenamiento", value=df['Fecha'].min().date())
        fecha_fin_train = st.sidebar.date_input("Fecha fin entrenamiento", value=df['Fecha'].max().date())

        fecha_inicio_sim = st.sidebar.date_input("Fecha inicio simulación", value=df['Fecha'].max().date() + pd.Timedelta(days=1))
        fecha_fin_sim = st.sidebar.date_input("Fecha fin simulación", value=df['Fecha'].max().date() + pd.Timedelta(days=30))

        mu_boost = st.sidebar.slider("Multiplicador de intensidad base (mu_boost)", 0.1, 5.0, 1.0, 0.1)

        st.sidebar.subheader("🌐 Autoexcitación espacio-temporal")
        alpha = st.sidebar.slider("Alpha", 0.0, 2.0, 0.5, 0.1)
        beta = st.sidebar.slider("Beta (decaimiento temporal)", 0.01, 1.0, 0.1, 0.01)
        gamma = st.sidebar.slider("Gamma (decaimiento espacial)", 0.01, 1.0, 0.05, 0.01)

        max_eventos = st.sidebar.number_input("Máximo eventos", 100, 100000, 5000, 100)
        usar_semilla = st.sidebar.checkbox("Fijar semilla", value=False)

        if st.button("🚀 Entrenar modelo y simular eventos"):
            with st.spinner("🔧 Entrenando modelo GAM..."):
                modelo_gam, min_fecha_train, factor_ajuste = entrenar_modelo_gam(df, fecha_inicio_train, fecha_fin_train, usar_hora)
                st.info(f"Factor ajuste: {factor_ajuste:.2f}")

            with st.spinner("🎲 Simulando eventos..."):
                gdf_sim = simular_eventos(df, fecha_inicio_train, fecha_fin_train,
                                          fecha_inicio_sim, fecha_fin_sim,
                                          gdf_zona, modelo_gam, min_fecha_train,
                                          factor_ajuste=factor_ajuste,
                                          mu_boost=mu_boost,
                                          alpha=alpha, beta=beta, gamma=gamma,
                                          max_eventos=max_eventos,
                                          usar_hora=usar_hora,
                                          seed=42 if usar_semilla else None)

            st.success(f"✅ Simulados {len(gdf_sim)} eventos")

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
