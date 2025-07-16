import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import timedelta
from pygam import GAM, s
from shapely.geometry import Point
from shapely.ops import unary_union
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings("ignore")

def entrenar_modelo_gam(df, fecha_inicio, fecha_fin, usar_hora=False):
    df_train = df[(df['Fecha'] >= fecha_inicio) & (df['Fecha'] <= fecha_fin)].copy()
    df_train = df_train.dropna(subset=['Long', 'Lat', 'Fecha'])

    min_fecha = df_train['Fecha'].min()

    if usar_hora:
        df_train['t'] = (df_train['Fecha'] - min_fecha).dt.total_seconds() / 86400.0
    else:
        df_train['t'] = (df_train['Fecha'].dt.normalize() - min_fecha.normalize()).dt.days.astype(float)

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
        probs = np.ones(len(gdf_zona))
    probs = probs / probs.sum()
    return probs

def samplear_punto_en_poligono(poligono):
    minx, miny, maxx, maxy = poligono.bounds
    while True:
        p = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        if poligono.contains(p):
            return p.x, p.y

def simular_eventos_aprendiendo(df, fecha_inicio_train, fecha_fin_train,
                                fecha_inicio_sim, fecha_fin_sim,
                                gdf_zona, modelo_gam, min_fecha_train,
                                factor_ajuste=1.0, mu_boost=1.0,
                                seed=None, usar_hora=False):

    if seed is not None:
        np.random.seed(seed)

    # Extraer datos de entrenamiento (para densidad espacial)
    df_train = df[(df['Fecha'] >= fecha_inicio_train) & (df['Fecha'] <= fecha_fin_train)].copy()
    df_train = df_train.dropna(subset=['Long', 'Lat', 'Fecha'])

    # Calcular duración de simulación
    if usar_hora:
        duracion_sim = (fecha_fin_sim - fecha_inicio_sim).total_seconds() / 86400.0
    else:
        duracion_sim = (fecha_fin_sim.normalize() - fecha_inicio_sim.normalize()).days + 1

    # Calcular duración del entrenamiento
    if usar_hora:
        duracion_train = (fecha_fin_train - fecha_inicio_train).total_seconds() / 86400.0
    else:
        duracion_train = (fecha_fin_train.normalize() - fecha_inicio_train.normalize()).days + 1

    # Media diaria observada en entrenamiento
    media_diaria_real = len(df_train) / duracion_train

    # Total de eventos a simular
    n_eventos = int(media_diaria_real * duracion_sim * mu_boost)

    # Estimar probabilidades espaciales
    probs_poligonos = preparar_probabilidades_poligonos(gdf_zona, df_train)

    eventos_sim = []

    for _ in range(n_eventos):
        # Fecha aleatoria dentro del rango
        if usar_hora:
            offset_dias = np.random.uniform(0, duracion_sim)
            fecha_sim = fecha_inicio_sim + timedelta(days=offset_dias)
            t_norm = (fecha_sim - min_fecha_train).total_seconds() / 86400.0
        else:
            offset_dias = np.random.randint(0, duracion_sim)
            fecha_sim = fecha_inicio_sim + timedelta(days=offset_dias)
            t_norm = (fecha_sim.normalize() - min_fecha_train.normalize()).days

        # Elegir zona espacial
        idx_poligono = np.random.choice(len(gdf_zona), p=probs_poligonos)
        poligono = gdf_zona.iloc[idx_poligono].geometry
        lon, lat = samplear_punto_en_poligono(poligono)

        # Validar con GAM (intensidad)
        mu = modelo_gam.predict([[t_norm, lon, lat]])[0]
        mu *= factor_ajuste * mu_boost

        # Umbral de aceptación
        if mu > 0:
            eventos_sim.append({'Fecha': fecha_sim, 'Long': lon, 'Lat': lat})

    # Convertir a GeoDataFrame
    if not eventos_sim:
        return gpd.GeoDataFrame(columns=['Fecha', 'Long', 'Lat', 'geometry'], crs=gdf_zona.crs)

    df_sim = pd.DataFrame(eventos_sim)
    gdf_sim = gpd.GeoDataFrame(df_sim,
                               geometry=gpd.points_from_xy(df_sim['Long'], df_sim['Lat']),
                               crs=gdf_zona.crs)
    return gdf_sim
