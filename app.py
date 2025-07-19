import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import zipfile
import os
import tempfile
import matplotlib.pyplot as plt
from shapely.geometry import Point
from datetime import datetime

import numpyro.distributions as dist
from bstpp.main import Hawkes_Model

st.set_page_config(page_title="Modelo de Hurtos con Hawkes", layout="wide")
st.title("📍 Modelo Espaciotemporal de Hurtos con Hawkes + Cox")

st.markdown("""
Esta herramienta permite entrenar un modelo Hawkes sobre eventos (hurtos) en un espacio definido.
Puedes subir los datos, definir el rango de entrenamiento, ajustar parámetros y visualizar los resultados.
""")

# ------------------------
# Función para descomprimir shapefiles
# ------------------------
def unzip_shapefile(zip_file):
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(temp_dir)
    shp_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith(".shp")]
    if not shp_files:
        st.error("No se encontró archivo .shp válido en el ZIP")
        return None
    return gpd.read_file(shp_files[0])

# ------------------------
# Cargar archivos del usuario
# ------------------------
col1, col2 = st.columns(2)

with col1:
    zip_eventos = st.file_uploader("Sube el shapefile de hurtos (.zip)", type="zip")
with col2:
    zip_limites = st.file_uploader("Sube el shapefile de límites (.zip)", type="zip")

# Parámetros
fecha_ini = st.date_input("📅 Fecha inicio entrenamiento", value=pd.to_datetime("2017-01-01"))
fecha_fin = st.date_input("📅 Fecha fin entrenamiento", value=pd.to_datetime("2018-05-01"))
steps = st.slider("🎯 Número de pasos de entrenamiento", min_value=100, max_value=3000, step=100, value=1500)

# ------------------------
# Botón para entrenar
# ------------------------
if st.button("🚀 Entrenar modelo"):
    if zip_eventos is None or zip_limites is None:
        st.warning("Debes subir ambos shapefiles.")
    else:
        with st.spinner("Cargando datos y entrenando modelo..."):
            # Leer shapefiles
            gdf_events = unzip_shapefile(zip_eventos)
            gdf_boundaries = unzip_shapefile(zip_limites)
            gdf_events = gdf_events.to_crs("EPSG:4326")
            gdf_boundaries = gdf_boundaries.to_crs("EPSG:4326")

            gdf_events["Fecha"] = pd.to_datetime(gdf_events["Fecha"])
            t0 = gdf_events["Fecha"].min()
            gdf_events["t"] = (gdf_events["Fecha"] - t0).dt.total_seconds() / 86400
            gdf_events = gdf_events.sort_values("t")

            # Train-test split
            gdf_train = gdf_events[gdf_events["Fecha"] < pd.to_datetime(fecha_fin)]
            gdf_test = gdf_events[gdf_events["Fecha"] >= pd.to_datetime(fecha_fin)]
            data_model = gdf_train[["t", "Long", "Lat"]].rename(columns={"t": "T", "Long": "X", "Lat": "Y"})

            # Crear modelo
            model = Hawkes_Model(
                data=data_model,
                A=gdf_boundaries,
                T=gdf_train["t"].max(),
                cox_background=True,
                a_0=dist.Normal(1, 10),
                alpha=dist.Beta(20, 60),
                beta=dist.HalfNormal(2.0),
                sigmax_2=dist.HalfNormal(0.25)
            )

            model.run_svi(lr=0.02, num_steps=steps)

            # Evaluación
            data_test = gdf_test[["t", "Long", "Lat"]].rename(columns={"t": "T", "Long": "X", "Lat": "Y"})
            log_lik = model.log_expected_likelihood(data_test)
            aic = model.expected_AIC()

        st.success("✅ Modelo entrenado exitosamente!")
        st.write(f"**Log Expected Likelihood:** {log_lik:.2f}")
        st.write(f"**Expected AIC:** {aic:.2f}")

        # Visualizaciones
        st.subheader("🔍 Resultados del Modelo")

        st.markdown("**1. Mapa de intensidad espacial (background):**")
        fig1 = model.plot_spatial(include_cov=False)
        st.pyplot(fig1)

        st.markdown("**2. Curva de intensidad temporal:**")
        fig2 = model.plot_temporal()
        st.pyplot(fig2)

        st.markdown("**3. Dispersión espacial del trigger (α, β, σ²):**")
        fig3 = model.plot_trigger_posterior(trace=True)
        st.pyplot(fig3)

        st.markdown("**4. Decaimiento temporal de la autoexcitación:**")
        fig4 = model.plot_trigger_time_decay()
        st.pyplot(fig4)

        st.markdown("**5. Proporción de eventos autoexcitados:**")
        fig5 = model.plot_prop_excitation()
        st.pyplot(fig5)

        # Guardar como imágenes
        output_dir = tempfile.mkdtemp()
        fig1.savefig(f"{output_dir}/spatial.png")
        fig2.savefig(f"{output_dir}/temporal.png")
        fig3.savefig(f"{output_dir}/posterior.png")
        fig4.savefig(f"{output_dir}/decay.png")
        fig5.savefig(f"{output_dir}/excitation.png")

        st.success("Descarga todas las gráficas como ZIP")
        with zipfile.ZipFile(f"{output_dir}/resultados.zip", "w") as zipf:
            zipf.write(f"{output_dir}/spatial.png", arcname="spatial.png")
            zipf.write(f"{output_dir}/temporal.png", arcname="temporal.png")
            zipf.write(f"{output_dir}/posterior.png", arcname="posterior.png")
            zipf.write(f"{output_dir}/decay.png", arcname="decay.png")
            zipf.write(f"{output_dir}/excitation.png", arcname="excitation.png")

        with open(f"{output_dir}/resultados.zip", "rb") as f:
            st.download_button("📥 Descargar Resultados (.zip)", f, file_name="resultados_modelo.zip")
