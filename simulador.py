import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import timedelta
from pygam import GAM, s
from shapely.geometry import Point
from shapely.ops import unary_union
from scipy.spatial.distance import cdist
import warnings
import streamlit as st

warnings.filterwarnings("ignore")

def cargar_datos(path):
    df = pd.read_csv(path)
    if 'Fecha' not in df.columns:
        raise ValueError("La columna 'Fecha' no está en el archivo.")

    # Homogeneizar formato de fecha
    df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce', dayfirst=True, infer_datetime_format=True)

    if df['Fecha'].isnull().any():
        st.warning("Algunas fechas no se pudieron parsear correctamente y se eliminarán.")
        df = df.dropna(subset=['Fecha'])

    return df

def entrenar_modelo_gam(df, fecha_inicio, fecha_fin, incluir_hora=False):
    df_train = df[(df['Fecha'] >= pd.to_datetime(fecha_inicio)) & 
                  (df['Fecha'] <= pd.to_datetime(fecha_fin))].copy()
    df_train = df_train.dropna(subset=['Long', 'Lat', 'Fecha'])

    min_fecha = df_train['Fecha'].min()
    
    if incluir_hora:
        df_train['t'] = (df_train['Fecha'] - min_fecha).dt.total_seconds() / 86400.0
    else:
        df_train['Fecha_sin_hora'] = df_train['Fecha'].dt.floor('D')
        df_train['t'] = (df_train['Fecha_sin_hora'] - min_fecha.floor('D')).dt.days.astype(float)

    X = df_train[['t', 'Long', 'Lat']].values
    y = np.ones(len(df_train))

    gam = GAM(s(0) + s(1) + s(2), verbose=False).fit(X, y)

    pred_mu = gam.predict(X)
    total_predicho = np.sum(pred_mu)
    total_real = len(df_train)
    factor_ajuste = total_real / total_predicho if total_predicho > 0 else 1.0

    return gam, min_fecha, factor_ajuste

def preparar_probabilidades_poligonos(gdf_zona, df_train):
    gdf_train = gpd.GeoDataFrame(df_train,
                                  geometry=gpd.points_from_xy(df_train['Long'], df_train['Lat']),
                                  crs=gdf_zona.crs)

    sjoined = gpd.sjoin(gdf_train, gdf_zona, how='inner')
    conteo = sjoined.groupby('index_right').size()

    probs = np.zeros(len(gdf_zona))
    for idx, n in conteo.items():
        probs[idx] = n

    if probs.sum() == 0:
        # Si no hay eventos en ningún polígono, asignar probabilidades uniformes
        probs = np.ones(len(gdf_zona))
    probs = probs / probs.sum()
    return probs

def samplear_punto_en_poligono(poligono):
    minx, miny, maxx, maxy = poligono.bounds
    while True:
        p = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        if poligono.contains(p):
            return p.x, p.y

def simular_eventos(df, fecha_inicio_train, fecha_fin_train,
                    fecha_inicio_sim, fecha_fin_sim,
                    gdf_zona, modelo_gam, min_fecha_train,
                    factor_ajuste=1.0, mu_boost=1.0,
                    alpha=0.5, beta=0.1, gamma=0.05,
                    max_eventos=10000, seed=None,
                    incluir_hora=False):

    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed()

    sim_events = []

    t_ini = 0
    t_fin = max(1, (pd.to_datetime(fecha_fin_sim) - pd.to_datetime(fecha_inicio_sim)).days + 1)

    # Prepara probabilidades por polígono
    df_train = df[(df['Fecha'] >= pd.to_datetime(fecha_inicio_train)) & 
                  (df['Fecha'] <= pd.to_datetime(fecha_fin_train))].copy()
    probs_poligonos = preparar_probabilidades_poligonos(gdf_zona, df_train)

    t = t_ini

    while t < t_fin:
        if len(sim_events) >= max_eventos:
            print("Se alcanzó el límite de eventos por seguridad.")
            break

        u1 = np.random.uniform()
        w = -np.log(u1) / 30.0
        t_candidate = t + w
        if t_candidate > t_fin:
            break

        fecha_sim = pd.to_datetime(fecha_inicio_sim) + timedelta(days=t_candidate)

        # Selección de polígono según conteo histórico
        idx_poligono = np.random.choice(len(gdf_zona), p=probs_poligonos)
        poligono = gdf_zona.iloc[idx_poligono].geometry
        lon, lat = samplear_punto_en_poligono(poligono)

        # Calcular t_norm teniendo en cuenta si se usa hora o no
        if incluir_hora:
            t_norm = (fecha_sim - min_fecha_train).total_seconds() / 86400.0
        else:
            fecha_sim_sin_hora = fecha_sim.floor('D')
            t_norm = (fecha_sim_sin_hora - min_fecha_train.floor('D')).days

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
