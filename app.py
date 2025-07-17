import streamlit as st
import pandas as pd
from datetime import datetime
from simulador import simular_eventos

st.set_page_config(page_title="VIO-QUAKE Simulador Hawkes", layout="centered")
st.title("🧠 VIO-QUAKE - Simulador de Eventos tipo Hawkes")

st.sidebar.header("📁 Cargar datos")
uploaded_file = st.sidebar.file_uploader("Carga tu archivo CSV", type=["csv", "xlsx"])
usar_hora = st.sidebar.checkbox("¿Tu dataset incluye horas?", value=True)

df = None
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, parse_dates=["Fecha"])
        else:
            df = pd.read_excel(uploaded_file, parse_dates=["Fecha"])
        st.success("✅ Datos cargados correctamente")
    except Exception as e:
        st.error(f"❌ Error al leer el archivo: {e}")

if df is not None:
    min_fecha = df["Fecha"].min()
    max_fecha = df["Fecha"].max()

    st.sidebar.markdown("### 📅 Fechas para entrenamiento")
    fecha_inicio_train = st.sidebar.date_input("Inicio entrenamiento", min_fecha)
    fecha_fin_train = st.sidebar.date_input("Fin entrenamiento", max_fecha)

    st.sidebar.markdown("### 📅 Fechas para simulación")
    fecha_inicio_sim = st.sidebar.date_input("Inicio simulación", min_fecha)
    fecha_fin_sim = st.sidebar.date_input("Fin simulación", max_fecha)

    st.sidebar.markdown("### ⚙️ Parámetros del modelo")
    mu_boost = st.sidebar.slider("Boost (μ)", 0.1, 5.0, 1.0, 0.1)
    max_eventos = st.sidebar.slider("Máx eventos simulados", 100, 5000, 1000, 100)
    seed = st.sidebar.number_input("Seed aleatoria (opcional)", value=42)

    if st.sidebar.button("🚀 Entrenar y Simular"):
        with st.spinner("Entrenando y simulando eventos..."):
            df_sim = simular_eventos(
                df,
                pd.to_datetime(fecha_inicio_train),
                pd.to_datetime(fecha_fin_train),
                pd.to_datetime(fecha_inicio_sim),
                pd.to_datetime(fecha_fin_sim),
                mu_boost=mu_boost,
                max_eventos=max_eventos,
                seed=int(seed),
                usar_hora=usar_hora
            )

            st.success("✅ Simulación completada")
            st.write("### Eventos simulados")
            st.dataframe(df_sim)

            dias_sim = (pd.to_datetime(fecha_fin_sim) - pd.to_datetime(fecha_inicio_sim)).days + 1
            media_real = df[
                (df["Fecha"] >= pd.to_datetime(fecha_inicio_sim)) & 
                (df["Fecha"] <= pd.to_datetime(fecha_fin_sim))
            ].shape[0] / dias_sim
            media_sim = df_sim.shape[0] / dias_sim

            st.markdown(f"📊 **Media diaria real:** {media_real:.2f}")
            st.markdown(f"📊 **Media diaria simulada:** {media_sim:.2f}")
