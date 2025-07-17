import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from simulador import entrenar_modelo, simulate_hawkes

st.set_page_config(page_title="Simulador Hawkes", layout="wide")
st.title("📊 Simulador de Eventos con Proceso de Hawkes")

# --- Subir archivo Excel ---
archivo = st.file_uploader("📁 Sube un archivo Excel con una columna 'Fecha'", type=["xlsx"])

if archivo:
    df = pd.read_excel(archivo)
    df['Fecha'] = pd.to_datetime(df['Fecha'])

    st.success(f"✅ {len(df)} eventos cargados.")
    fecha_min, fecha_max = df['Fecha'].min(), df['Fecha'].max()

    # --- Fechas de entrenamiento ---
    fecha_inicio = st.date_input("📅 Fecha inicio entrenamiento", value=fecha_min.date(), min_value=fecha_min.date(), max_value=fecha_max.date())
    fecha_fin = st.date_input("📅 Fecha fin entrenamiento", value=fecha_max.date(), min_value=fecha_min.date(), max_value=fecha_max.date())

    # Convertir fechas a pandas.Timestamp
    fecha_inicio = pd.to_datetime(fecha_inicio)
    fecha_fin = pd.to_datetime(fecha_fin)

    # --- Entrenar modelo ---
    if st.button("🚀 Entrenar modelo Hawkes"):
        modelo = entrenar_modelo(df, fecha_inicio, fecha_fin)
        st.session_state["modelo_hawkes"] = modelo
        st.success("✅ Modelo entrenado con éxito")

    # --- Simulación solo si ya hay modelo en sesión ---
    if "modelo_hawkes" in st.session_state:
        modelo = st.session_state["modelo_hawkes"]

        st.subheader("🔮 Simulación de eventos futuros")

        pred_inicio = st.date_input("Fecha inicio predicción", value=fecha_fin + pd.Timedelta(days=1))
        pred_fin = st.date_input("Fecha fin predicción", value=fecha_fin + pd.Timedelta(days=30))

        # Convertir
        pred_inicio = pd.to_datetime(pred_inicio)
        pred_fin = pd.to_datetime(pred_fin)

        if st.button("🎯 Simular eventos"):
            t_ini = (pred_inicio - modelo['t0']).total_seconds() / (3600 * 24)
            t_fin = (pred_fin - modelo['t0']).total_seconds() / (3600 * 24)

            eventos = simulate_hawkes(
                modelo['mu_interp'],
                modelo['alpha_interp'],
                modelo['decay'],
                t_ini,
                t_fin
            )

            fechas_simuladas = [modelo['t0'] + pd.Timedelta(days=d) for d in eventos]

            st.success(f"✅ Se simularon {len(fechas_simuladas)} eventos.")
            st.dataframe(pd.DataFrame({'Fecha simulada': fechas_simuladas}))

            # --- Gráfico corregido ---
            df_sim = pd.DataFrame({'Fecha simulada': fechas_simuladas})
            df_sim['Fecha_dia'] = df_sim['Fecha simulada'].dt.date
            conteo_por_dia = df_sim.groupby('Fecha_dia').size()

            fig, ax = plt.subplots()
            conteo_por_dia.plot(kind='bar', ax=ax)
            ax.set_title("Eventos simulados por día")
            ax.set_ylabel("Frecuencia")
            ax.set_xlabel("Fecha")
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)
    else:
        st.warning("🔄 Entrena el modelo antes de simular eventos.")
