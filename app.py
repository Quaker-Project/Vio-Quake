import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
from simulador import entrenar_modelo_gam, simular_eventos
from shapely.geometry import Polygon
import io

st.set_page_config(page_title="Simulador de Eventos", layout="wide")

def main():
    st.title("🔁 Simulador Espaciotemporal de Eventos")

    st.sidebar.header("1. Subir archivos")
    archivo_datos = st.sidebar.file_uploader("📄 Datos históricos (.csv)", type=["csv"])
    archivo_zona = st.sidebar.file_uploader("📐 Zona de simulación (.geojson)", type=["geojson", "json"])

    tiene_hora = st.sidebar.checkbox("¿Los datos incluyen hora?", value=False)

    if archivo_datos and archivo_zona:
        df = pd.read_csv(archivo_datos)

        if 'Long' not in df.columns or 'Lat' not in df.columns or 'Fecha' not in df.columns:
            st.error("❌ El archivo CSV debe tener columnas llamadas 'Long', 'Lat' y 'Fecha'.")
            return

        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')

        if not tiene_hora:
            df['Fecha'] = df['Fecha'].dt.normalize()

        if tiene_hora and (df['Fecha'].dt.hour == 0).all():
            st.warning("⚠️ Los datos parecen no tener información horaria, aunque se activó esa opción.")

        gdf_zona = gpd.read_file(archivo_zona)

        st.sidebar.header("2. Configurar periodo de entrenamiento")
        fecha_min = df['Fecha'].min().date()
        fecha_max = df['Fecha'].max().date()
        fecha_inicio_train = st.sidebar.date_input("📅 Fecha inicio entrenamiento", fecha_min, min_value=fecha_min, max_value=fecha_max)
        fecha_fin_train = st.sidebar.date_input("📅 Fecha fin entrenamiento", fecha_max, min_value=fecha_min, max_value=fecha_max)

        st.sidebar.header("3. Configurar simulación")
        fecha_inicio_sim = st.sidebar.date_input("🟢 Fecha inicio simulación", fecha_max + pd.Timedelta(days=1))
        fecha_fin_sim = st.sidebar.date_input("🔴 Fecha fin simulación", fecha_max + pd.Timedelta(days=10))

        mu_boost = st.sidebar.slider("📈 Intensidad base (mu_boost)", 0.1, 5.0, 1.0, 0.1)
        alpha = st.sidebar.slider("📊 Alpha (auto-excitación)", 0.0, 1.0, 0.5, 0.01)
        beta = st.sidebar.slider("⏱️ Beta (decay temporal)", 0.01, 1.0, 0.1, 0.01)
        gamma = st.sidebar.slider("📍 Gamma (decay espacial)", 0.01, 1.0, 0.05, 0.01)

        max_eventos = st.sidebar.number_input("🔢 Límite máximo de eventos", min_value=1000, max_value=50000, value=10000, step=1000)
        usar_semilla = st.sidebar.checkbox("🎲 Usar semilla fija", value=True)

        if st.sidebar.button("▶️ Iniciar simulación"):
            st.subheader("🔄 Entrenando modelo...")
            modelo_gam, min_fecha_train, factor_ajuste = entrenar_modelo_gam(
                df, fecha_inicio_train, fecha_fin_train, usar_hora=tiene_hora)

            st.subheader("🛠️ Ejecutando simulación...")
            gdf_sim = simular_eventos(
                df, fecha_inicio_train, fecha_fin_train,
                fecha_inicio_sim, fecha_fin_sim,
                gdf_zona, modelo_gam, min_fecha_train,
                factor_ajuste=factor_ajuste,
                mu_boost=mu_boost,
                alpha=alpha, beta=beta, gamma=gamma,
                max_eventos=max_eventos,
                seed=42 if usar_semilla else None,
                usar_hora=tiene_hora
            )

            if gdf_sim.empty:
                st.error("⚠️ No se generaron eventos simulados. Revisa los parámetros.")
                return

            st.success(f"✅ Se generaron {len(gdf_sim)} eventos simulados.")

            st.subheader("📍 Mapa de eventos simulados")

            centroide = gdf_zona.geometry.unary_union.centroid
            m = folium.Map(location=[centroide.y, centroide.x], zoom_start=12)

            folium.GeoJson(gdf_zona).add_to(m)

            for _, row in gdf_sim.iterrows():
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=3,
                    color='red',
                    fill=True,
                    fill_opacity=0.6
                ).add_to(m)

            st_data = st_folium(m, width=700, height=500)

            st.subheader("📥 Descargar resultados")
            buffer = io.StringIO()
            gdf_sim.drop(columns='geometry').to_csv(buffer, index=False)
            st.download_button("⬇️ Descargar CSV", buffer.getvalue(), file_name="eventos_simulados.csv", mime="text/csv")

if __name__ == "__main__":
    main()
