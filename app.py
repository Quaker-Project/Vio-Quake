import streamlit as st
import pandas as pd
import io
from datetime import datetime
from simulador import simular_eventos

st.set_page_config(page_title="VIO-QUAKE Simulador Hawkes", layout="centered")

def cargar_datos(file):
    if file is None:
        return None
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
        df = df.dropna(subset=['Fecha'])
        return df
    except Exception as e:
        st.error(f"Error cargando archivo: {e}")
        return None

def main():
    st.title("🧨 VIO-QUAKE | Simulador Hawkes")

    st.markdown("""
    Sube tus datos con columnas mínimas:  
    - Long (longitud)  
    - Lat (latitud)  
    - Fecha (datetime con o sin hora)
    """)

    if "df" not in st.session_state:
        archivo = st.file_uploader("Carga archivo CSV o Excel", type=['csv', 'xls', 'xlsx'])
        if archivo:
            df = cargar_datos(archivo)
            if df is not None:
                if not all(col in df.columns for col in ['Long', 'Lat', 'Fecha']):
                    st.error("El archivo debe contener las columnas: Long, Lat y Fecha.")
                else:
                    st.session_state.df = df
                    st.success("Archivo cargado correctamente.")
                    st.experimental_rerun()
    else:
        df = st.session_state.df
        min_fecha = df['Fecha'].min()
        max_fecha = df['Fecha'].max()

        st.write(f"Datos cargados: {len(df)} eventos desde {min_fecha.date()} hasta {max_fecha.date()}")

        st.sidebar.header("Configuración de fechas y parámetros")
        usar_hora = st.sidebar.checkbox("¿Los datos incluyen hora?", value=True)

        fecha_inicio_train = st.sidebar.date_input("Fecha inicio entrenamiento", min_value=min_fecha.date(), max_value=max_fecha.date(), value=min_fecha.date())
        fecha_fin_train = st.sidebar.date_input("Fecha fin entrenamiento", min_value=min_fecha.date(), max_value=max_fecha.date(), value=max_fecha.date())

        fecha_inicio_sim = st.sidebar.date_input("Fecha inicio simulación", value=max_fecha.date())
        fecha_fin_sim = st.sidebar.date_input("Fecha fin simulación", value=max_fecha.date())

        mu_boost = st.sidebar.slider("Multiplicador mu_boost", 0.1, 5.0, 1.0, 0.1)
        max_eventos = st.sidebar.number_input("Máximo eventos simulados", min_value=100, max_value=100000, value=5000, step=100)
        usar_semilla = st.sidebar.checkbox("Fijar semilla aleatoria", value=False)

        if fecha_fin_train < fecha_inicio_train:
            st.error("La fecha fin de entrenamiento debe ser posterior o igual a la fecha inicio.")
            return
        if fecha_fin_sim < fecha_inicio_sim:
            st.error("La fecha fin de simulación debe ser posterior o igual a la fecha inicio.")
            return

        if st.button("Entrenar y Simular"):
            with st.spinner("Entrenando modelo y simulando eventos..."):
                try:
                    if not usar_hora:
                        fecha_inicio_train_dt = pd.to_datetime(fecha_inicio_train)
                        fecha_fin_train_dt = pd.to_datetime(fecha_fin_train) + pd.Timedelta(hours=23, minutes=59, seconds=59)
                        fecha_inicio_sim_dt = pd.to_datetime(fecha_inicio_sim)
                        fecha_fin_sim_dt = pd.to_datetime(fecha_fin_sim) + pd.Timedelta(hours=23, minutes=59, seconds=59)
                    else:
                        fecha_inicio_train_dt = pd.to_datetime(fecha_inicio_train)
                        fecha_fin_train_dt = pd.to_datetime(fecha_fin_train) + pd.Timedelta(hours=23, minutes=59, seconds=59)
                        fecha_inicio_sim_dt = pd.to_datetime(fecha_inicio_sim)
                        fecha_fin_sim_dt = pd.to_datetime(fecha_fin_sim) + pd.Timedelta(hours=23, minutes=59, seconds=59)

                    df_sim = simular_eventos(df, fecha_inicio_train_dt, fecha_fin_train_dt,
                                             fecha_inicio_sim_dt, fecha_fin_sim_dt,
                                             mu_boost=mu_boost, max_eventos=max_eventos,
                                             seed=42 if usar_semilla else None,
                                             usar_hora=usar_hora)

                    eventos_reales = df[(df['Fecha'] >= fecha_inicio_sim_dt) & (df['Fecha'] <= fecha_fin_sim_dt)]
                    n_reales = len(eventos_reales)
                    n_simulados = len(df_sim)
                    dias_sim = (fecha_fin_sim_dt - fecha_inicio_sim_dt).days + 1

                    st.markdown(f"**Eventos reales periodo simulación:** {n_reales} (media diaria {n_reales/dias_sim:.2f})")
                    st.markdown(f"**Eventos simulados periodo simulación:** {n_simulados} (media diaria {n_simulados/dias_sim:.2f})")

                    towrite = io.BytesIO()
                    df_sim.to_excel(towrite, index=False)
                    towrite.seek(0)
                    st.download_button("Descargar eventos simulados (Excel)", data=towrite,
                                       file_name="eventos_simulados.xlsx",
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                except Exception as e:
                    st.error(f"Error durante simulación: {e}")

        if st.button("Limpiar datos cargados"):
            st.session_state.pop("df", None)
            st.experimental_rerun()

if __name__ == "__main__":
    main()
