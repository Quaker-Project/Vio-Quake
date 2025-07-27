import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import box
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import zipfile
import os
import tempfile

st.set_page_config(layout="wide")

st.title("🧠 Predicción Espacial de Hurtos")

# ------------------------
# Subir archivos
# ------------------------

robos_zip = st.file_uploader("📂 Sube el shapefile de hurtos (.zip con .shp, .shx, .dbf, .prj)", type=["zip"])
contorno_zip = st.file_uploader("🗺️ (Opcional) Sube shapefile de fondo o contorno (.zip)", type=["zip"])

# ------------------------
# Función para leer shapefile desde zip
# ------------------------

def cargar_shapefile_zip(uploaded_zip):
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
            zip_ref.extractall(tmpdir)

        shp_files = [f for f in os.listdir(tmpdir) if f.endswith(".shp")]
        if not shp_files:
            st.error("❌ No se encontró ningún archivo .shp en el .zip.")
            st.stop()

        return gpd.read_file(os.path.join(tmpdir, shp_files[0]))

# ------------------------
# Parámetros
# ------------------------

cell_size = st.slider("📏 Tamaño de celda (m)", 100, 1000, 500, step=100)
umbral = st.slider("🎯 Umbral de riesgo", 0.1, 0.9, 0.7, step=0.05)

# ------------------------
# Procesamiento solo si se subió archivo
# ------------------------

if robos_zip:
    gdf = cargar_shapefile_zip(robos_zip)
    gdf = gdf.to_crs(epsg=32630)  # Ajusta a tu zona UTM si es diferente
    gdf["Fecha"] = pd.to_datetime(gdf["Fecha"], errors="coerce", dayfirst=True)
    gdf["month"] = gdf["Fecha"].dt.to_period("M")

    meses_disponibles = sorted(gdf["month"].dropna().unique())

    st.markdown("## ⚙️ Configuración de simulación")

    col1, col2 = st.columns(2)
    with col1:
        mes_test = st.selectbox("📅 Mes de simulación (test)", meses_disponibles[::-1])
    with col2:
        meses_entrenamiento = st.multiselect("📘 Meses de entrenamiento", meses_disponibles, default=meses_disponibles[:-1])

    simular = st.button("🚀 Ejecutar simulación")

    if simular:
        with st.spinner("Simulando..."):

            # Crear rejilla
            xmin, ymin, xmax, ymax = gdf.total_bounds
            cols = list(np.arange(xmin, xmax, cell_size))
            rows = list(np.arange(ymin, ymax, cell_size))

            polygons, cell_ids = [], []
            for i, x in enumerate(cols):
                for j, y in enumerate(rows):
                    poly = box(x, y, x + cell_size, y + cell_size)
                    polygons.append(poly)
                    cell_ids.append(f"{i}_{j}")

            gdf_grid = gpd.GeoDataFrame({'cell_id': cell_ids}, geometry=polygons, crs=gdf.crs)
            gdf_grid["X"] = gdf_grid.geometry.centroid.x
            gdf_grid["Y"] = gdf_grid.geometry.centroid.y

            # Entrenamiento
            data = []
            for m in meses_entrenamiento:
                df_month = gdf[gdf["month"] == m]
                joined = gpd.sjoin(gdf_grid, df_month, predicate="contains", how="left")
                joined["label"] = joined["index_right"].notnull().astype(int)
                grouped = joined.groupby("cell_id").agg(label=("label", "max")).reset_index()
                merged = pd.merge(grouped, gdf_grid, on="cell_id")
                merged["month"] = str(m)
                data.append(merged)

            df_model = pd.concat(data, ignore_index=True)
            X = df_model[["X", "Y"]]
            y = df_model["label"]

            model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
            model.fit(X, y)

            # Predicción
            df_next = gdf_grid.copy()
            probs = model.predict_proba(df_next[["X", "Y"]])[:, 1]
            df_next["predicted_prob"] = probs
            df_next["predicted_risk"] = (probs >= umbral).astype(int)

            df_test = gdf[gdf["month"] == mes_test]

            # Fondo contorno
            gdf_contorno = None
            if contorno_zip:
                try:
                    gdf_contorno = cargar_shapefile_zip(contorno_zip).to_crs(gdf.crs)
                except:
                    st.warning("⚠️ No se pudo cargar el shapefile de contorno")

            # Mapa
            st.markdown("## 🗺️ Riesgo predicho vs hurtos reales")

            fig, ax = plt.subplots(figsize=(10, 8))
            if gdf_contorno is not None:
                gdf_contorno.plot(ax=ax, facecolor="none", edgecolor="gray", alpha=0.6)

            df_next[df_next["predicted_risk"] == 1].plot(ax=ax, column="predicted_risk", cmap="Reds", alpha=0.5, legend=True)
            df_test.plot(ax=ax, color="black", markersize=8, alpha=0.7)

            plt.title(f"Riesgo predicho vs hurtos reales - {mes_test}")
            plt.axis("off")
            st.pyplot(fig)

            # Evaluación
            joined_test = gpd.sjoin(df_next, df_test, predicate='contains', how='left')
            joined_test["actual_label"] = joined_test["index_right"].notnull().astype(int)

            evaluated = joined_test.groupby("cell_id").agg(
                predicted=("predicted_risk", "max"),
                actual=("actual_label", "max")
            ).reset_index()

            y_true = evaluated["actual"]
            y_pred = evaluated["predicted"]

            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)

            st.markdown("## 📊 Métricas de evaluación")
            st.write(f"**Precision:** {precision:.2f}")
            st.write(f"**Recall:** {recall:.2f}")
            st.write(f"**F1-score:** {f1:.2f}")
