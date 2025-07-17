# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from simulador import entrenar_modelo, simulate_hawkes

st.set_page_config(page_title="Simulador Hawkes", layout="wide")
st.title("📊 Simulador de Eventos con Proceso de Hawkes")

# --- Subir archivo ---
archivo = st.file_uploader("Sube un archivo Excel con eventos (columna 'Fecha')", type=["xlsx"])

if archivo:
    df = pd.read_excel(archivo)
    df['Fecha'] = pd.to_datetime(df['Fecha'])

    st.success(f"{len(df)} eventos cargados.")
    fecha_min, fecha_max = df['Fecha'].min(), df['Fecha'].max()

    # Entrenamiento
    fecha_inicio = st.date_input("📅 Fecha inicio entrenamiento", value=fecha_min.date(), min_value=fecha_min.date(), max_value=fecha_max.date())
    fecha_fin = st.date_input("📅 Fecha fin entrenamiento", value=fecha_max.date(), min_value=fecha_min.date(), max_value=fecha_max.date())

    # ✅ Convertir a Timestamp (necesario para evitar errores de tipo)
    fecha_inicio = pd.to_datetime(fecha_inicio)
    fecha_fin = pd.to_datetime(fecha_fin)

    if st.button("🚀 Entrenar modelo"):
        modelo = entrenar_modelo(df, fecha_inicio, fecha_fin)
        st.success("✅ Modelo entrenado con éxito.")

        # Fechas de predicción
        pred_inicio = st.date_input("🔮 Fecha inicio predicción", value=fecha_fin + pd.Timedelta(days=1))
        pred_fin = st.date_input("🔮 Fecha fin predicción", value=fecha_fin + pd.Timedelta(days=30))

        # ✅ Convertir a Timestamp también
        pred_inicio = pd.to_datetime(pred_inicio)
        pred_fin = pd.to_datetime(pred_fin)

        if st.button("🎯 Simular eventos"):
            t_ini = (pred_inicio - modelo['t0']).total_seconds() / (3600 * 24)
            t_fin = (pred_fin - modelo['t0']).total_seconds() / (3600 * 24)

            eventos = simulate_hawkes(modelo['mu_interp'], modelo['alpha_interp'], modelo['decay'], t_ini, t_fin)
            fechas_simuladas = [modelo['t0'] + pd.Timedelta(days=d) for d in eventos]

            st.write(f"🔢 Total eventos simulados: {len(fechas_simuladas)}")
            st.dataframe(pd.DataFrame({'Fecha simulada': fechas_simuladas}))

            # Gráfico de barras por día
            fig, ax = plt.subplots()
            pd.Series(fechas_simuladas).groupby(lambda d: d.date()).size().plot(kind='bar', ax=ax)
            ax.set_title("Eventos simulados por día")
            ax.set_ylabel("Frecuencia")
            ax.set_xlabel("Fecha")
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)
