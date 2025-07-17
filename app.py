import streamlit as st
import pandas as pd
from simulador import HawkesSimulator
from datetime import datetime
import io

st.set_page_config(
    page_title="Simulador Hawkes",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 Simulador de Eventos con Modelo Hawkes")

# --- Carga de archivo ---
archivo = st.file_uploader("📂 Carga un archivo Excel con una columna 'Fecha'", type=["xlsx", "xls"])

if archivo:
    try:
        df = pd.read_excel(archivo)
        if "Fecha" not in df.columns:
            st.error("❌ El archivo debe contener una columna llamada 'Fecha'")
        else:
            df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
            df = df.dropna(subset=["Fecha"])

            fecha_min, fecha_max = df["Fecha"].min(), df["Fecha"].max()
            st.success(f"✅ Archivo cargado correctamente. Fechas desde {fecha_min.date()} hasta {fecha_max.date()}")

            # --- Selección de fechas ---
            st.subheader("🗓️ Selecciona el período de entrenamiento y simulación")
            col1, col2 = st.columns(2)
            with col1:
                fecha_ini_train = st.date_input("Inicio entrenamiento", value=fecha_min, min_value=fecha_min, max_value=fecha_max)
                fecha_ini_sim = st.date_input("Inicio simulación", value=fecha_max + pd.Timedelta(days=1))
            with col2:
                fecha_fin_train = st.date_input("Fin entrenamiento", value=fecha_max, min_value=fecha_min, max_value=fecha_max)
                fecha_fin_sim = st.date_input("Fin simulación", value=fecha_max + pd.Timedelta(days=30))

            mu_boost = st.slider("🔧 Multiplicador de intensidad base (mu_boost)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

            if st.button("🚀 Entrenar modelo y simular"):
                with st.spinner("Entrenando modelo Hawkes temporal..."):
                    # Guardar archivo temporal
                    temp_buffer = io.BytesIO()
                    df.to_excel(temp_buffer, index=False)
                    temp_buffer.seek(0)

                    # Inicializar y entrenar modelo
                    modelo = HawkesSimulator(temp_buffer)
                    mask = (df["Fecha"] >= pd.to_datetime(fecha_ini_train)) & (df["Fecha"] <= pd.to_datetime(fecha_fin_train))
                    df_train = df[mask]
                    if df_train.empty:
                        st.error("❌ El periodo de entrenamiento no contiene datos válidos.")
                        st.stop()
                    buffer_train = io.BytesIO()
                    with pd.ExcelWriter(buffer_train, engine='xlsxwriter') as writer:
                        df_train.to_excel(writer, index=False)
                    buffer_train.seek(0)
                    modelo = HawkesSimulator(buffer_train)


                with st.spinner("Simulando eventos..."):
                    eventos = modelo.simulate(pd.to_datetime(fecha_ini_sim), pd.to_datetime(fecha_fin_sim), mu_boost=mu_boost)

                st.success(f"✅ Simulados {len(eventos)} eventos entre {fecha_ini_sim} y {fecha_fin_sim}")

                # Estadísticas
                dias_sim = max((pd.to_datetime(fecha_fin_sim) - pd.to_datetime(fecha_ini_sim)).days + 1, 1)
                real_count = df[(df["Fecha"] >= pd.to_datetime(fecha_ini_sim)) & (df["Fecha"] <= pd.to_datetime(fecha_fin_sim))].shape[0]
                resumen = modelo.summary()

                st.markdown(f"""
                📊 Media diaria real: **{real_count / dias_sim:.2f}**
                
                📊 Media diaria simulada: **{len(eventos) / dias_sim:.2f}**

                📈 Total real: **{real_count}**, Total simulado: **{len(eventos)}**

                📌 Parámetros estimados del modelo
                - Mu promedio diario: **{resumen['mu_avg']:.4f}**
                - Alfa promedio diario: **{resumen['alpha_avg']:.4f}**
                - Decay estimado (β): **{resumen['decay']:.4f}**
                """)

                # Descargar resultados
                out_df = pd.DataFrame({"Fecha": [modelo.t0 + pd.Timedelta(days=float(t)) for t in eventos]})
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    out_df.to_excel(writer, index=False)
                output.seek(0)

                st.download_button(
                    label="📥 Descargar eventos simulados en Excel",
                    data=output,
                    file_name="eventos_simulados.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {e}")
