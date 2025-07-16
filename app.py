import streamlit as st
import pandas as pd
import geopandas as gpd
import zipfile
import io
from simulador import entrenar_modelo_gam, simular_eventos

def cargar_shapefile_zip(file_zip):
    with open("temp_shapefile.zip", "wb") as f:
        f.write(file_zip.getbuffer())
    try:
        gdf = gpd.read_file("zip://temp_shapefile.zip")
    except Exception as e:
        st.error(f"Error leyendo shapefile desde ZIP: {e}")
        return None
    return gdf

def cargar_datos_eventos(file):
    if file.type == "text/csv":
        df = pd.read_csv(file)
    elif file.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
        df = pd.read_excel(file)
    else:
        st.error("Formato no soportado. Usa CSV o XLSX.")
        return None
    return df

def main():
    st.title("Simulador Vio-Quake")

    archivo_datos = st.file_uploader("Sube archivo CSV o XLSX con eventos", type=["csv", "xlsx"])
    archivo_limites = st.file_uploader("Sube ZIP con shapefile de límites administrativos", type=["zip"])
    usar_hora = st.checkbox("Usar horas para entrenamiento y simulación", value=False)

    if archivo_datos and archivo_limites:
        df = cargar_datos_eventos(archivo_datos)
        if df is None:
            return
        if not {'Fecha', 'Long', 'Lat'}.issubset(df.columns):
            st.error("El archivo debe contener columnas 'Fecha', 'Long' y 'Lat'")
            return
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
        df = df.dropna(subset=['Fecha', 'Long', 'Lat'])

        gdf_limites = cargar_shapefile_zip(archivo_limites)
        if gdf_limites is None:
            return

        st.write("Límites administrativos cargados:")
        st.write(gdf_limites.head())

        fecha_min = df['Fecha'].min().date()
        fecha_max = df['Fecha'].max().date()

        fecha_inicio_train = st.date_input("Fecha inicio entrenamiento", value=fecha_min, min_value=fecha_min, max_value=fecha_max)
        fecha_fin_train = st.date_input("Fecha fin entrenamiento", value=fecha_max, min_value=fecha_min, max_value=fecha_max)

        fecha_inicio_sim = st.date_input("Fecha inicio simulación", value=fecha_min, min_value=fecha_min, max_value=fecha_max)
        fecha_fin_sim = st.date_input("Fecha fin simulación", value=fecha_max, min_value=fecha_min, max_value=fecha_max)

        if fecha_inicio_train > fecha_fin_train:
            st.error("La fecha inicio de entrenamiento debe ser anterior a la fecha fin de entrenamiento")
            return
        if fecha_inicio_sim > fecha_fin_sim:
            st.error("La fecha inicio de simulación debe ser anterior a la fecha fin de simulación")
            return

        st.write("Entrenando modelo GAM...")
        modelo_gam, min_fecha_train, factor_ajuste = entrenar_modelo_gam(df, fecha_inicio_train, fecha_fin_train, usar_hora=usar_hora)
        st.success("Modelo entrenado")

        st.write("Simulando eventos...")
        gdf_simulados = simular_eventos(
            df,
            fecha_inicio_train,
            fecha_fin_train,
            fecha_inicio_sim,
            fecha_fin_sim,
            gdf_limites,
            modelo_gam,
            min_fecha_train,
            factor_ajuste,
            mu_boost=1.0,
            alpha=0.5,
            beta=0.1,
            gamma=0.05,
            max_eventos=10000,
            seed=42,
            usar_hora=usar_hora
        )

        st.write(f"Eventos simulados: {len(gdf_simulados)}")
        if not gdf_simulados.empty:
            st.map(gdf_simulados[['Lat', 'Long']])
        else:
            st.warning("No se generaron eventos simulados")

if __name__ == "__main__":
    main()
