import streamlit as st
import pandas as pd
import geopandas as gpd
import io
import os
import tempfile
from simulador import entrenar_modelo_gam, simular_eventos

st.set_page_config(page_title="VIO-QUAKE Simulador", layout="wide")

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
    st.title("🧨 VIO-QUAKE | Simulador de Eventos")

    archivo_datos = st.file_uploader("📂 Suba datos de eventos (CSV/Excel)", type=["csv", "xls", "xlsx"])
    df = cargar_archivo_datos(archivo_datos)

    archivo_zip = st.file_uploader("📍 Suba shapefile ZIP de la zona", type=["zip"])
    gdf_zona = cargar_shapefile_zip(archivo_zip)

    if df is not None:
        columnas_necesarias = ['Long', 'Lat', 'Fecha']
        if not all(col in df.columns for col in columnas_necesarias):
            st.error(f"El archivo debe contener las columnas: {columnas_necesarias}")
            return

        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
        if df['Fecha'].isnull().any():
            st.error("Hay valores inválidos en la columna 'Fecha'.")
            return

    if df is not None and gdf_zona is not None:
        st.sidebar.header("⚙️ Parámetros de simulación")

        usar_hora = st.sidebar.checkbox("¿Usar hora?", value=False)

        fecha_inicio_train = st.sidebar.date_input("Fecha inicio entrenamiento", value=df['Fecha'].min().date())
        fecha_fin_train = st.sidebar.date_input("Fecha fin entrenamiento", value=df['Fecha'].max().date())
        fecha_inicio_sim = st.sidebar.date_input("Fecha inicio simulación", value=(df['Fecha'].max() + pd.Timedelta(days=1)).date())
        fecha_fin_sim = st.sidebar.date_input("Fecha fin simulación", value=(df['Fecha'].max() + pd.Timedelta(days=30)).date())

        mu_boost = st.sidebar.slider("Multiplicador de intensidad base (mu_boost)", 0.1, 5.0, 1.0, 0.1)
        alpha = st.sidebar.slider("Alpha (autoexcitación)", 0.0, 2.0, 0.5, 0.1)
        beta = st.sidebar.slider("Beta (decaimiento temporal)", 0.01, 1.0, 0.1, 0.01)
        gamma = st.sidebar.slider("Gamma (decaimiento espacial)", 0.01, 1.0, 0.05, 0.01)
        max_eventos = st.sidebar.number_input("Máximo eventos simulados", 100, 100000, 5000, 100)
        usar_semilla = st.sidebar.checkbox("Fijar semilla aleatoria", value=False)

        if st.button("🚀 Ejecutar simulación"):
            with st.spinner("Entrenando modelo GAM..."):
                modelo_gam, min_fecha_train, factor_ajuste = entrenar_modelo_gam(
                    df, fecha_inicio_train, fecha_fin_train, usar_hora=usar_hora
                )
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
                    seed=42 if usar_semilla else None,
                    usar_hora=usar_hora
                )

            st.success(f"✅ Se generaron {len(gdf_sim)} eventos simulados")

            # Cálculo de medias reales/simuladas según usar_hora
            if usar_hora:
                inicio_dt = pd.to_datetime(fecha_inicio_sim)
                fin_dt = pd.to_datetime(fecha_fin_sim + pd.Timedelta(days=1)) - pd.Timedelta(seconds=1)
                dias_sim = max((fin_dt - inicio_dt).total_seconds() / 86400, 1)
            else:
                dias_sim = (fecha_fin_sim - fecha_inicio_sim).days + 1

            media_real = df[(df['Fecha'] >= fecha_inicio_sim) & (df['Fecha'] <= fecha_fin_sim)].shape[0] / dias_sim
            media_simulada = len(gdf_sim) / dias_sim

            st.write(f"📊 Media diaria real: **{media_real:.2f}**")
            st.write(f"📊 Media diaria simulada: **{media_simulada:.2f}**")

            # Preparar archivo descargable
            gdf_sim = gdf_sim.to_crs(epsg=4326)
            df_out = gdf_sim[['Fecha', 'geometry']].copy()
            df_out['Long'] = df_out.geometry.x
            df_out['Lat'] = df_out.geometry.y
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
