import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial.distance import cdist
from datetime import timedelta

def preparar_probabilidades_poligonos(gdf_zona, df_train):
    conteos = gdf_zona.copy()
    conteos['conteo'] = 0

    for idx, poligono in conteos.iterrows():
        puntos_en_poligono = df_train[df_train.geometry.within(poligono.geometry)]
        conteos.at[idx, 'conteo'] = len(puntos_en_poligono)

    total_eventos = conteos['conteo'].sum()
    if total_eventos == 0:
        # Si no hay eventos, asignar probabilidad uniforme
        return np.ones(len(gdf_zona)) / len(gdf_zona)

    return conteos['conteo'].values / total_eventos

def samplear_punto_en_poligono(poligono):
    minx, miny, maxx, maxy = poligono.bounds
    while True:
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        punto = gpd.points_from_xy([x], [y])[0]
        if poligono.contains(punto):
            return x, y

def simular_eventos(df, fecha_inicio_train, fecha_fin_train,
                    fecha_inicio_sim, fecha_fin_sim,
                    gdf_zona, modelo_gam, min_fecha_train,
                    factor_ajuste=1.0, mu_boost=1.0,
                    alpha=0.5, beta=0.1, gamma=0.05,
                    max_eventos=10000, seed=None,
                    usar_hora=False):

    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed()

    sim_events = []

    # Definir escala temporal según usar_hora
    if usar_hora:
        escala_temporal = 'segundos'
        unidad_tiempo = 1.0  # segundos normalizados a días
    else:
        escala_temporal = 'días'
        unidad_tiempo = 86400.0  # segundos por día

    # Convertir fechas a datetime si no lo están
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    fecha_inicio_train = pd.to_datetime(fecha_inicio_train)
    fecha_fin_train = pd.to_datetime(fecha_fin_train)
    fecha_inicio_sim = pd.to_datetime(fecha_inicio_sim)
    fecha_fin_sim = pd.to_datetime(fecha_fin_sim)

    # Prepara probabilidades por polígono
    df_train = df[(df['Fecha'] >= fecha_inicio_train) & (df['Fecha'] <= fecha_fin_train)].copy()
    probs_poligonos = preparar_probabilidades_poligonos(gdf_zona, df_train)

    t_ini = 0.0
    if usar_hora:
        t_fin = (fecha_fin_sim - fecha_inicio_sim).total_seconds()
    else:
        t_fin = (fecha_fin_sim - fecha_inicio_sim).days + 1

    t = t_ini

    while t < t_fin:
        if len(sim_events) >= max_eventos:
            print("Se alcanzó el límite de eventos por seguridad.")
            break

        u1 = np.random.uniform()
        w = -np.log(u1) / 30.0  # parámetro base Poisson

        t_candidate = t + w
        if t_candidate > t_fin:
            break

        if usar_hora:
            fecha_sim = fecha_inicio_sim + timedelta(seconds=t_candidate)
        else:
            fecha_sim = fecha_inicio_sim + timedelta(days=t_candidate)

        # Selección de polígono según conteo histórico
        idx_poligono = np.random.choice(len(gdf_zona), p=probs_poligonos)
        poligono = gdf_zona.iloc[idx_poligono].geometry
        lon, lat = samplear_punto_en_poligono(poligono)

        # Normalizar tiempo para el GAM
        if usar_hora:
            t_norm = (fecha_sim - min_fecha_train).total_seconds() / unidad_tiempo
        else:
            t_norm = (fecha_sim - min_fecha_train).total_seconds() / 86400.0

        mu = modelo_gam.predict([[t_norm, lon, lat]])[0]
        mu = max(mu * factor_ajuste * mu_boost, 1e-6)

        # Autoexcitación espacio-temporal
        excitation = 0.0
        if sim_events:
            eventos_previos = np.array([[e['t'], e['Long'], e['Lat']] for e in sim_events])
            tiempos_previos = eventos_previos[:, 0]
            coords_previos = eventos_previos[:, 1:3]

            dt = t_candidate - tiempos_previos
            dist = cdist([[lon, lat]], coords_previos)[0]

            excitation = np.sum(alpha * np.exp(-beta * dt) * np.exp(-gamma * dist))

        intensidad_total = max(mu + excitation, 1e-6)
        lambda_max = max(intensidad_total * 1.5, 1e-6)

        u2 = np.random.uniform()
        if u2 <= intensidad_total / lambda_max:
            sim_events.append({'Fecha': fecha_sim, 'Long': lon, 'Lat': lat, 't': t_candidate})

        t = t_candidate

    df_sim = pd.DataFrame(sim_events)
    if df_sim.empty:
        return gpd.GeoDataFrame(columns=['Fecha', 'Long', 'Lat', 'geometry'], crs=gdf_zona.crs)

    gdf_sim = gpd.GeoDataFrame(df_sim,
                               geometry=gpd.points_from_xy(df_sim['Long'], df_sim['Lat']),
                               crs=gdf_zona.crs)
    return gdf_sim


import streamlit as st
import pandas as pd
import numpy as np
from simulador import entrenar_modelo_gam, simular_eventos

st.title("Simulador de Violencia de Género")

# Sidebar - Parámetros del modelo
st.sidebar.header("Configuración del Modelo")
tamano_muestra = st.sidebar.slider("Tamaño de la muestra", 100, 10000, 500)
np.random.seed(123)

# Generar datos sintéticos (puedes sustituir esto por tus datos reales si los tienes)
datos = pd.DataFrame({
    'edad_agresor': np.random.normal(35, 10, tamano_muestra),
    'orden_alejamiento': np.random.choice([0, 1], tamano_muestra),
    'denuncias_previas': np.random.poisson(1, tamano_muestra),
    'tipo_maltrato': np.random.choice(['físico', 'psicológico', 'sexual'], tamano_muestra),
    'reincidencia': np.random.choice([0, 1], tamano_muestra)
})

# Entrenar el modelo GAM
st.write("### Datos de entrenamiento")
st.dataframe(datos.head())

modelo, datos, variables = entrenar_modelo_gam(datos)

st.sidebar.header("Simulación de Nuevos Eventos")
n_sim = st.sidebar.slider("Número de simulaciones", 10, 1000, 100)

if st.sidebar.button("Simular eventos futuros"):
    nuevos_eventos = simular_eventos(modelo, datos, variables, n_sim)
    
    st.write("### Nuevos eventos simulados")
    st.dataframe(nuevos_eventos)

    st.write("### Distribución de reincidencias simuladas")
    st.bar_chart(nuevos_eventos['reincidencia'].value_counts())
