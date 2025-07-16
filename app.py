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
        .stApp { background-color: #111111; color: #EEEEEE; }
        .stSidebar { background-color: #1c1c1c; }
        .stButton>button {
            background-color: #ff4b4b; color: white;
            border-radius: 8px; font-weight: bold;
        }
        .stButton>button:hover { background-color: #ff1c1c; }
        .stDownloadButton>button {
            background-color: #4b6fff; color: white;
            border-radius: 8px; font-weight: bold;
        }
        .stDownloadButton>button:hover { background-color: #1c44ff; }
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
            st.error("Formato no soportado. Usa CSV o Excel.")
            return None
    except Exception as e:
        st.error(f"Error al cargar archivo: {e}")
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

    archivo_datos = st.file_uploader("📂 Suba datos de eventos (CSV/Excel)", type=["csv", "xls", "xlsx"])
    archivo_zip = st.file_uploader("📍 Suba shapefile ZIP (zona de simulación)", type=["zip"])

    tiene_hora = st.checkbox("¿Tus datos incluyen hora? (HH:MM:SS)", value=False)

    df = cargar_archivo_datos(archivo_datos)
    gdf_zona = cargar_shapefile_zip(archivo_zip)

    if df is not None:
        columnas = df.columns
        if not all(col in columnas for col in ['Fecha', 'Long', 'Lat']):
            st.error("Las columnas obligatorias son: Fecha, Long, Lat")
            return

        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
        if df['Fecha'].isnull().any():
            st.error("Hay fechas no válidas. Corrige los datos.")
            return

        if not tiene_hora:
            df['Fecha'] = df['Fecha'].dt.normalize()

    if df is not None and gdf_zona is not None:
        st.sidebar.header("⚙️ Parámetros de simulación")

        fecha_inicio_train = st.sidebar.date_input("Inicio entrenamiento", value=df['Fecha'].min())
        fecha_fin_train = st.sidebar.date_input("Fin entrenamiento", value=df['Fecha'].max())
        fecha_inicio_sim = st.sidebar.date_input("Inicio simulación", value=df['Fecha'].max() + pd.Timedelta(days=1))
        fecha_fin_sim = st.sidebar.date_input("Fin simulación", value=df['Fecha'].max() + pd.Timedelta(days=30))

        mu_boost = st.sidebar.slider("Multiplicador intensidad base", 0.1, 5.0, 1.0, 0.1)

        st.sidebar.subheader("🌐 Autoexcitación espacio-temporal")
        alpha = st.sidebar.slider("Alpha", 0.0, 2.0, 0.5, 0.1)
        beta = st.sidebar.slider("Beta (decaimiento temporal)", 0.01, 1.0, 0.1, 0.01)
        gamma = st.sidebar.slider("Gamma (decaimiento espacial)", 0.01, 1.0, 0.05, 0.01)

        max_eventos = st.sidebar.number_input("Máximo de eventos", 100, 100000, 5000, 100)
        usar_semilla = st.sidebar.checkbox("Fijar semilla aleatoria", value=False)

        if st.button("🚀 Ejecutar simulación"):
            with st.spinner("Entrenando modelo..."):
                modelo_gam, min_fecha_train, factor_ajuste = entrenar_modelo_gam(df, fecha_inicio_train, fecha_fin_train)
                st.success(f"Modelo entrenado. Ajuste: {factor_ajuste:.2f}")

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

            st.success(f"✅ {len(gdf_sim)} eventos simulados")

            dias_sim = max(1, (pd.to_datetime(fecha_fin_sim) - pd.to_datetime(fecha_inicio_sim)).days + 1)
            media_real = df[(df['Fecha'] >= fecha_inicio_sim) & (df['Fecha'] <= fecha_fin_sim)].shape[0] / dias_sim
            media_sim = len(gdf_sim) / dias_sim

            st.write(f"📊 Media diaria real: **{media_real:.2f}**")
            st.write(f"📊 Media diaria simulada: **{media_sim:.2f}**")

            # Exportar a Excel
            gdf_sim_wgs84 = gdf_sim.to_crs(epsg=4326)
            df_out = gdf_sim_wgs84[['Fecha']].copy()
            df_out['Long'] = gdf_sim_wgs84.geometry.x
            df_out['Lat'] = gdf_sim_wgs84.geometry.y
            df_out = df_out[['Long', 'Lat', 'Fecha']]

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_out.to_excel(writer, index=False)
            output.seek(0)

            st.download_button(
                label="📥 Descargar eventos simulados (.xlsx)",
                data=output,
                file_name="eventos_simulados.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == "__main__":
    main()
