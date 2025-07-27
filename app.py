import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import box
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os
from io import BytesIO

st.set_page_config(layout="wide")
st.title("🧠 Predicción Espacial de Riesgo de Hurtos")

# --- Sidebar ---
st.sidebar.header("🔧 Configuración")

# Subida de archivos
robos_file = st.sidebar.file_uploader("Sube el shapefile de hurtos (.zip)", type=["zip"])
contorno_file = st.sidebar.file_uploader("Sube shapefile de contorno (.zip, opcional)", type=["zip"])

cell_size = st.sidebar.slider("Tamaño de celda (m)", 200, 1000, 500, 100)
umbral = st.sidebar.slider("Umbral de riesgo (probabilidad)", 0.0, 1.0, 0.7, 0.05)

# --- Procesamiento ---
def cargar_shapefile_zip(uploaded_zip):
    from zipfile import ZipFile
    import tempfile

    if uploaded_zip is None:
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        with ZipFile(uploaded_zip, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        shp_files = [f for f in os.listdir(tmpdir) if f.endswith(".shp")]
        if not shp_files:
            st.error("No se encontró .shp en el zip")
            return None
        return gpd.read_file(os.path.join(tmpdir, shp_files[0]))

if robos_file:
    gdf = cargar_shapefile_zip(robos_file)
    gdf = gdf.to_crs(epsg=32616)
    gdf["Fecha"] = pd.to_datetime(gdf["Fecha"], errors="coerce", dayfirst=True)
    gdf = gdf.dropna(subset=["Fecha"])
    gdf["month"] = gdf["Fecha"].dt.to_period("M")

    meses_disponibles = sorted(gdf["month"].unique())
    mes_seleccionado = st.sidebar.selectbox("Mes a simular", meses_disponibles, index=len(meses_disponibles)-1)

    # Crear rejilla
    xmin, ymin, xmax, ymax = gdf.total_bounds
    cols = list(np.arange(xmin, xmax, cell_size))
    rows = list(np.arange(ymin, ymax, cell_size))
    polygons = []
    cell_ids = []
    for i, x in enumerate(cols):
        for j, y in enumerate(rows):
            polygons.append(box(x, y, x + cell_size, y + cell_size))
            cell_ids.append(f"{i}_{j}")

    gdf_grid = gpd.GeoDataFrame({'cell_id': cell_ids}, geometry=polygons, crs=gdf.crs)
    gdf_grid["X"] = gdf_grid.geometry.centroid.x
    gdf_grid["Y"] = gdf_grid.geometry.centroid.y

    # Crear dataset
    train_months = [m for m in meses_disponibles if m < mes_seleccionado]
    data = []
    for m in train_months:
        df_month = gdf[gdf["month"] == m]
        joined = gpd.sjoin(gdf_grid, df_month, predicate='contains', how='left')
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

    # Evento real ese mes
    df_test_month = gdf[gdf["month"] == mes_seleccionado]

    # Contorno
    if contorno_file:
        gdf_contorno = cargar_shapefile_zip(contorno_file)
        gdf_contorno = gdf_contorno.to_crs(gdf.crs)
    else:
        gdf_contorno = None

    # Mapa
    st.subheader(f"🗺️ Riesgo predicho vs eventos reales en {mes_seleccionado}")
    fig, ax = plt.subplots(figsize=(10, 8))
    if gdf_contorno is not None:
        gdf_contorno.plot(ax=ax, facecolor="none", edgecolor="gray", alpha=0.5)

    df_next[df_next["predicted_risk"] == 1].plot(ax=ax, color="red", alpha=0.4, label="Riesgo Predicho")
    df_test_month.plot(ax=ax, color="black", markersize=10, label="Eventos Reales")
    plt.legend()
    plt.axis("off")
    st.pyplot(fig)

    # Evaluación
    joined_test = gpd.sjoin(df_next, df_test_month, predicate='contains', how='left')
    joined_test["actual_label"] = joined_test["index_right"].notnull().astype(int)

    evaluated = joined_test.groupby("cell_id").agg(
        predicted=("predicted_risk", "max"),
        actual=("actual_label", "max")
    ).reset_index()

    precision = precision_score(evaluated["actual"], evaluated["predicted"])
    recall = recall_score(evaluated["actual"], evaluated["predicted"])
    f1 = f1_score(evaluated["actual"], evaluated["predicted"])

    st.markdown("### 📊 Evaluación:")
    st.write(f"**Precision:** {precision:.2f}  |  **Recall:** {recall:.2f}  |  **F1-score:** {f1:.2f}")

    # Exportar
    gdf_export = gpd.GeoDataFrame(df_next, geometry="geometry", crs=gdf.crs)
    export_format = st.selectbox("Formato de exportación", ["GeoJSON", "Shapefile"])

    if st.button("📥 Exportar resultados"):
        with st.spinner("Exportando..."):
            if export_format == "GeoJSON":
                geojson = gdf_export.to_json()
                st.download_button("Descargar GeoJSON", geojson, file_name="prediccion.geojson", mime="application/json")
            else:
                import tempfile
                from zipfile import ZipFile
                with tempfile.TemporaryDirectory() as tmpdir:
                    gdf_export.to_file(os.path.join(tmpdir, "pred.shp"))
                    zip_path = os.path.join(tmpdir, "shapefile.zip")
                    with ZipFile(zip_path, 'w') as zipf:
                        for f in os.listdir(tmpdir):
                            if f.startswith("pred."):
                                zipf.write(os.path.join(tmpdir, f), arcname=f)
                    with open(zip_path, "rb") as f:
                        st.download_button("Descargar Shapefile", f, file_name="prediccion_shapefile.zip")
