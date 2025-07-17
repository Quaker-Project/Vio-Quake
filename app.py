import streamlit as st
import pandas as pd
import geopandas as gpd
from simulador import entrenar_modelo_gam, simular_eventos
import tempfile
import os
import io
import zipfile

# Estilo visual
st.set_page_config(
    page_title="VIO-QUAKE Simulador",
    layout="wide",
    initial_sidebar_state="expanded"
)

def css_estilo():
    st.markdown("""
    <style>
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
        elif archivo.name.endswith('.xls') or archivo.name.endswith('.xlsx'):
            df = pd.read_excel(archivo)
        else:
            st.error("Formato no soportado. Use CSV o Excel.")
            return None
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error cargando archivo: {e}")
        return None

def cargar_shapefile_zip(archivo_zip):
    if archivo_zip is None:
        return None
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
        cols_requeridas = ['Long', 'Lat', 'Fecha']
        if not all(col in df.columns for col in cols_requeridas):
            st.error(f"Faltan columnas requeridas: {cols_requeridas}")
            return
        if df['Fecha'].isnull().any():
            st.error("Existen fechas no convertibles. Corrija el archivo.")
            return

    if df is not None and gdf_zona is not None:
        st.sidebar.header("⚙️ Configuración de simulación")

        usar_hora = st.sidebar.checkbox("¿Usar hora en los eventos?", value=True)

        fecha_inicio_train = st.sidebar.date_input("Fecha inicio entrenamiento", value=df['Fecha'].min().date())
        fecha_fin_train = st.sidebar.date_input("Fecha fin entrenamiento", value=df['Fecha'].max().date())

        fecha_inicio_sim = st.sidebar.date_input("Fecha inicio simulación", value=df['Fecha'].max().date() + pd.Timedelta(days=1))
        fecha_fin_sim = st.sidebar.date_input("Fecha fin simulación", value=df['Fecha'].max().date() + pd.Timedelta(days=30))

        # Convertir fechas a datetime completos si se usa hora
        if usar_hora:
            fecha_inicio_train = pd.to_datetime(fecha_inicio_train)
            fecha_fin_train = pd.to_datetime(fecha_fin_train) + pd.Timedelta(hours=23, minutes=59)
            fecha_inicio_sim = pd.to_datetime(fecha_inicio_sim)
            fecha_fin_sim = pd.to_datetime(fecha_fin_sim) + pd.Timedelta(hours=23, minutes=59)
        else:
            df['Fecha'] = df['Fecha'].dt.normalize()
            fecha_inicio_train = pd.to_datetime(fecha_inicio_train)
            fecha_fin_train = pd.to_datetime(fecha_fin_train)
            fecha_inicio_sim = pd.to_datetime(fecha_inicio_sim)
            fecha_fin_sim = pd.to_datetime(fecha_fin_sim)

        mu_boost = st.sidebar.slider("Multiplicador de intensidad base (mu_boost)", 0.1, 5.0, 1.0, 0.1)

        st.sidebar.subheader("🌐 Autoexcitación espacio-temporal")
        alpha = st.sidebar.slider("Alpha (nivel de autoexcitación)", 0.0, 2.0, 0.5, 0.1)
        beta = st.sidebar.slider("Beta (decaimiento temporal)", 0.01, 1.0, 0.1, 0.01)
        gamma = st.sidebar.slider("Gamma (decaimiento espacial)", 0.01, 1.0, 0.05, 0.01)

        max_eventos = st.sidebar.number_input("Máximo de eventos simulados", 100, 100000, 5000, 100)
        usar_semilla = st.sidebar.checkbox("Fijar semilla aleatoria", value=False)

        if st.button("🚀 Entrenar modelo y simular eventos"):
            with st.spinner("🔧 Entrenando modelo GAM espaciotemporal..."):
                modelo_gam, min_fecha_train, factor_ajuste = entrenar_modelo_gam(df, fecha_inicio_train, fecha_fin_train, usar_hora=usar_hora)
                st.info(f"Factor de ajuste automático (histórico): {factor_ajuste:.2f}")
                st.info(f"Boost aplicado por el usuario (mu_boost): {mu_boost:.2f}")

                # Calcular intensidad esperada diaria del GAM en el periodo de simulación
                fechas_pred = pd.date_range(fecha_inicio_sim, fecha_fin_sim, freq='D')
                intensidades = []
                for fecha in fechas_pred:
                    if usar_hora:
                        t_norm = (fecha - min_fecha_train).total_seconds() / 86400.0
                    else:
                        t_norm = (fecha.normalize() - min_fecha_train.normalize()).days

                    centroide = gdf_zona.unary_union.centroid
                    lon, lat = centroide.x, centroide.y

                    mu_dia = modelo_gam.predict([[t_norm, lon, lat]])[0]
                    mu_dia = max(mu_dia * factor_ajuste * mu_boost, 0)
                    intensidades.append(mu_dia)

                df_mu = pd.DataFrame({'Fecha': fechas_pred, 'Mu esperada (intensidad GAM)': intensidades})
                media_mu = np.mean(intensidades)

                st.write(f"📈 Media diaria GAM esperada en periodo de simulación: **{media_mu:.2f}**")
                with st.expander("🔍 Ver detalle de predicción diaria GAM"):
                    st.dataframe(df_mu)

            with st.spinner("🎲 Simulando eventos Hawkes espacio-temporal..."):
                gdf_sim = simular_eventos(df, fecha_inicio_train, fecha_fin_train,
                                          fecha_inicio_sim, fecha_fin_sim,
                                          gdf_zona, modelo_gam, min_fecha_train,
                                          factor_ajuste=factor_ajuste,
                                          mu_boost=mu_boost,
                                          alpha=alpha, beta=beta, gamma=gamma,
                                          max_eventos=max_eventos,
                                          seed=42 if usar_semilla else None,
                                          usar_hora=usar_hora)

            st.success(f"✅ Simulados {len(gdf_sim)} eventos")

            dias_sim = max(1, (fecha_fin_sim - fecha_inicio_sim).days + 1)
            media_real = df[(df['Fecha'] >= fecha_inicio_sim) & (df['Fecha'] <= fecha_fin_sim)].shape[0] / dias_sim
            media_simulada = len(gdf_sim) / dias_sim

            st.write(f"📊 Media diaria real: **{media_real:.2f}**")
            st.write(f"📊 Media diaria simulada: **{media_simulada:.2f}**")

            gdf_sim_wgs84 = gdf_sim.to_crs(epsg=4326)
            export_df = gdf_sim_wgs84[['geometry', 'Fecha']].copy()
            export_df['Long'] = export_df.geometry.x
            export_df['Lat'] = export_df.geometry.y
            export_df = export_df.drop(columns='geometry')
            export_df = export_df[['Long', 'Lat', 'Fecha']]

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                export_df.to_excel(writer, index=False)
            output.seek(0)

            st.download_button(
                label="📥 Descargar Excel con eventos simulados",
                data=output,
                file_name="eventos_simulados.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == "__main__":
    main()
