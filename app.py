import streamlit as st
import pandas as pd
from simulador import HawkesSimulator
import io

st.set_page_config(
    page_title="Simulador de Eventos Hawkes",
    layout="wide",
    initial_sidebar_state="expanded"
)

def css_estilo():
    st.markdown("""
    <style>
        .stApp { background-color: #111111; color: #EEEEEE; font-family: 'Segoe UI', sans-serif; }
        .stSidebar { background-color: #1c1c1c; }
        .stButton>button, .stDownloadButton>button {
            border-radius: 8px;
            padding: 0.5em 1em;
            font-weight: bold;
            transition: 0.3s;
        }
        .stButton>button { background-color: #ff4b4b; color: white; }
        .stButton>button:hover { background-color: #ff1c1c; transform: scale(1.05); }
        .stDownloadButton>button { background-color: #4b6fff; color: white; }
        .stDownloadButton>button:hover { background-color: #1c44ff; transform: scale(1.05); }
    </style>
    """, unsafe_allow_html=True)

css_estilo()

st.title("📈 Simulador de Eventos Hawkes (solo temporal)")

st.markdown("""
Este sistema ajusta un modelo de procesos Hawkes temporal (con autoexcitación) y simula nuevos eventos a partir de datos históricos.
""")

archivo_datos = st.file_uploader("📂 Suba archivo Excel con eventos (debe incluir columna 'Fecha')", type=["xlsx"])

if archivo_datos:
    try:
        df = pd.read_excel(archivo_datos)
        if 'Fecha' not in df.columns:
            st.error("El archivo debe tener una columna llamada 'Fecha'")
        else:
            df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
            df = df.dropna(subset=['Fecha'])
            st.success(f"✅ Datos cargados correctamente. {len(df)} eventos válidos.")

            fecha_min = df['Fecha'].min().date()
            fecha_max = df['Fecha'].max().date()

            st.sidebar.header("⚙️ Parámetros de simulación")
            fecha_ini_sim = st.sidebar.date_input("Fecha inicio simulación", value=fecha_max + pd.Timedelta(days=1))
            fecha_fin_sim = st.sidebar.date_input("Fecha fin simulación", value=fecha_max + pd.Timedelta(days=30))
            mu_boost = st.sidebar.slider("Multiplicador mu_boost", 0.1, 5.0, 1.0, step=0.1)

            if st.button("🚀 Entrenar y simular"):
                with st.spinner("Entrenando modelo GAM Hawkes..."):
                    sim = HawkesSimulator()
                    sim.fit(df)

                with st.spinner("Simulando eventos..."):
                    eventos_sim = sim.simulate_eventos(pd.to_datetime(fecha_ini_sim), pd.to_datetime(fecha_fin_sim), mu_boost=mu_boost)

                dias_sim = max(1, (pd.to_datetime(fecha_fin_sim) - pd.to_datetime(fecha_ini_sim)).days + 1)
                media_real = df[(df['Fecha'] >= pd.to_datetime(fecha_ini_sim)) & (df['Fecha'] <= pd.to_datetime(fecha_fin_sim))].shape[0] / dias_sim
                media_sim = len(eventos_sim) / dias_sim

                st.subheader("📊 Resultados")
                st.write(f"📌 Total real: {df.shape[0]:,}, Total simulado: {len(eventos_sim):,}")
                st.write(f"📊 Media diaria real: {media_real:.2f}")
                st.write(f"📊 Media diaria simulada: {media_sim:.2f}")

                st.markdown("""
                #### 📌 Parámetros estimados
                """)
                st.code(f"Mu promedio diario: {sim.info['mu_diario']:.4f}\n"
                        f"Alfa promedio diario: {sim.info['alpha_diario']:.4f}\n"
                        f"Decay estimado (\u03b2): {sim.info['decay']:.4f}")

                df_sim = pd.DataFrame({"Fecha": [sim.info['t0'] + pd.Timedelta(days=d) for d in eventos_sim]})

                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df_sim.to_excel(writer, index=False)
                output.seek(0)

                st.download_button(
                    label="📥 Descargar eventos simulados (Excel)",
                    data=output,
                    file_name="eventos_simulados.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
