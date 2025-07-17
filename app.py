import streamlit as st
import pandas as pd
from datetime import datetime
from simulador import simular_eventos

st.set_page_config(page_title="VIO-QUAKE Simulador Hawkes", layout="centered")
st.title("📈 VIO-QUAKE - Simulador de Eventos tipo Hawkes")

# ---- Carga de datos ----
st.sidebar.header("📁 Cargar datos")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV o Excel", type=["csv", "xlsx"])
usar_hora = st.sidebar.checkbox("¿Tu dataset incluye hora?", value=True)

df = None
if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, parse_dates=["Fecha"])
        else:
            df = pd.read_excel(uploaded_file, parse_dates=["Fecha"])
        if "Fecha" not in df.columns:
            st.error("El archivo debe contener una columna llamada 'Fecha'")
            df = None
        else:
            df = df.sort_values("Fecha")
            st.success("✅ Datos cargados correctamente")
    except Exception as e:
        st.error(f"❌ Error al leer el archivo: {e}")
        df = None

if df is not None:
    # Mostrar datos brevemente
    st.write("Vista previa de los datos cargados:")
    st.dataframe(df.head())

    # Rango completo del dataset
    min_fecha = df["Fecha"].min()
    max_fecha = df["Fecha"].max()

    st.sidebar.markdown("### 🏁 Fechas de entrenamiento")
    fecha_inicio_train = st.sidebar.date_input("Inicio entrenamiento", min_value=min_fecha.date(), max_value=max_fecha.date(), value=min_fecha.date())
    fecha_fin_train = st.sidebar.date_input("Fin entrenamiento", min_value=min_fecha.date(), max_value=max_fecha.date(), value=max_fecha.date())

    st.sidebar.markdown("### 🎯 Fechas de simulación")
    fecha_inicio_sim = st.sidebar.date_input("Inicio simulación", min_value=min_fecha.date(), max_value=max_fecha.date(), value=min_fecha.date())
    fecha_fin_sim = st.sidebar.date_input("Fin simulación", min_value=min_fecha.date(), max_value=max_fecha.date(), value=max_fecha.date())

    st.sidebar.markdown("### ⚙️ Parámetros de simulación")
    mu_boost = st.sidebar.slider("Boost (μ)", 0.1, 5.0, 1.0, 0.1)
    max_eventos = st.sidebar.slider("Máx. eventos simulados", 100, 5000, 1000, 100)
    seed = st.sidebar.number_input("Semilla aleatoria (opcional)", value=42, step=1)

    ejecutar = st.sidebar.button("🚀 Entrenar y Simular")

    if ejecutar:
        try:
            with st.spinner("⏳ Entrenando modelo y generando eventos..."):
                df_sim = simular_eventos(
                    df=df,
                    fecha_inicio_train=pd.to_datetime(fecha_inicio_train),
                    fecha_fin_train=pd.to_datetime(fecha_fin_train),
                    fecha_inicio_sim=pd.to_datetime(fecha_inicio_sim),
                    fecha_fin_sim=pd.to_datetime(fecha_fin_sim),
                    mu_boost=mu_boost,
                    max_eventos=max_eventos,
                    seed=int(seed),
                    usar_hora=usar_hora
                )

            st.success("✅ Simulación completada")
            st.write("### Eventos simulados")
            st.dataframe(df_sim)

            dias_sim = (pd.to_datetime(fecha_fin_sim) - pd.to_datetime(fecha_inicio_sim)).days + 1
            eventos_reales = df[(df["Fecha"] >= pd.to_datetime(fecha_inicio_sim)) & (df["Fecha"] <= pd.to_datetime(fecha_fin_sim))]
            media_real = eventos_reales.shape[0] / dias_sim
            media_sim = df_sim.shape[0] / dias_sim

            st.markdown(f"📊 **Media diaria real:** {media_real:.2f}")
            st.markdown(f"📊 **Media diaria simulada:** {media_sim:.2f}")

        except Exception as e:
            st.error(f"❌ Error durante la simulación: {e}")
else:
    st.warning("🔄 Carga un archivo para comenzar.")
