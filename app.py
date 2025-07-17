import streamlit as st
import pandas as pd
from datetime import datetime
from simulador import entrenar_modelo_gam, simular_eventos

st.set_page_config(page_title="VIO-QUAKE Simulador Hawkes", layout="centered")
st.title("🔮 Simulador de eventos VIO-QUAKE")

# --- 1. Cargar datos ---
st.sidebar.header("📁 Subida de datos")
data_file = st.sidebar.file_uploader("Sube un archivo Excel o CSV", type=["xlsx", "csv"])
has_hour = st.sidebar.checkbox("¿Tus datos tienen hora (además de fecha)?", value=False)

if data_file is not None:
    if data_file.name.endswith(".xlsx"):
        df = pd.read_excel(data_file)
    else:
        df = pd.read_csv(data_file)

    st.success("Datos cargados correctamente")

    fecha_cols = [col for col in df.columns if "fecha" in col.lower()]
    if not fecha_cols:
        st.error("No se encontró una columna de fecha en los datos.")
    else:
        fecha_col = st.sidebar.selectbox("Selecciona la columna de fecha:", fecha_cols)
        df["Fecha"] = pd.to_datetime(df[fecha_col])

        st.write("Ejemplo de datos:")
        st.dataframe(df.head())

        # --- 2. Configuración ---
        st.sidebar.header("⚙️ Parámetros")
        fecha_min, fecha_max = df["Fecha"].min(), df["Fecha"].max()

        fecha_ini_train = st.sidebar.date_input("Fecha inicio entrenamiento", value=fecha_min, min_value=fecha_min, max_value=fecha_max)
        fecha_fin_train = st.sidebar.date_input("Fecha fin entrenamiento", value=fecha_max, min_value=fecha_min, max_value=fecha_max)

        fecha_ini_sim = st.sidebar.date_input("Fecha inicio simulación", value=fecha_max, min_value=fecha_min)
        fecha_fin_sim = st.sidebar.date_input("Fecha fin simulación", value=fecha_max + pd.Timedelta(days=30))

        mu_boost = st.sidebar.slider("μ boost (ajuste de intensidad basal)", min_value=0.0, max_value=3.0, value=1.0, step=0.1)

        df_train = df[(df["Fecha"] >= pd.to_datetime(fecha_ini_train)) & (df["Fecha"] <= pd.to_datetime(fecha_fin_train))]

        # --- 3. Botón de ejecución ---
        if st.button("🚀 Entrenar y Simular"):
            with st.spinner("Entrenando modelo GAM Hawkes..."):
                modelo, info = entrenar_modelo_gam(df_train)

            with st.spinner("Simulando eventos..."):
                eventos_simulados = simular_eventos(
                    modelo,
                    fecha_ini=pd.to_datetime(fecha_ini_sim),
                    fecha_fin=pd.to_datetime(fecha_fin_sim),
                    mu_boost=mu_boost
                )

            real_count = df[(df["Fecha"] >= pd.to_datetime(fecha_ini_sim)) & (df["Fecha"] <= pd.to_datetime(fecha_fin_sim))].shape[0]
            sim_count = len(eventos_simulados)

            dias = (pd.to_datetime(fecha_fin_sim) - pd.to_datetime(fecha_ini_sim)).days + 1

            st.markdown(f"📊 Media diaria real: **{real_count / dias:.2f}**")
            st.markdown(f"📊 Media diaria simulada: **{sim_count / dias:.2f}**")
            st.markdown(f"📈 Total real: {real_count}, Total simulado: {sim_count}")

            # Mostrar parámetros estimados
            st.markdown("### 📌 Parámetros estimados del modelo")
            st.markdown(f"- **Mu promedio diario**: {info['mu_diario']:.4f}")
            st.markdown(f"- **Alfa promedio diario**: {info['alpha_diario']:.4f}")
            st.markdown(f"- **Decay estimado (β)**: {info['decay']:.4f}")

else:
    st.info("Sube un archivo para comenzar.")
