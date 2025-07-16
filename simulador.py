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


def entrenar_modelo_gam(df, fecha_inicio, fecha_fin):
    df_train = df[(df['Fecha'] >= pd.to_datetime(fecha_inicio)) &
                  (df['Fecha'] <= pd.to_datetime(fecha_fin))].copy()
    df_train = df_train.dropna(subset=['Long', 'Lat', 'Fecha'])

    min_fecha = df_train['Fecha'].min()
    df_train['t'] = (df_train['Fecha'] - min_fecha).dt.total_seconds() / 86400.0

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


import geopandas as gpd
import pandas as pd
import numpy as np

def simular_eventos(df, fecha_inicio_train, fecha_fin_train,
                   fecha_inicio_sim, fecha_fin_sim,
                   gdf_zona, modelo_gam, min_fecha_train,
                   factor_ajuste=1.0,
                   mu_boost=1.0,
                   alpha=0.5, beta=0.1, gamma=0.05,
                   max_eventos=5000,
                   seed=None,
                   st_out=None):
    """
    Simula eventos espacio-temporales tipo Hawkes.
    Si se pasa st_out, se usan st_out.write() para mostrar mensajes, 
    si no, print().

    Parámetros similares a tu implementación actual.
    """

    def msg(mensaje):
        if st_out is not None:
            st_out.write(mensaje)
        else:
            print(mensaje)

    # Ejemplo de uso de la función de mensajes:
    msg(f"Factor de ajuste automático (histórico): {factor_ajuste:.2f}")
    msg(f"Boost aplicado por el usuario (mu_boost): {mu_boost:.2f}")

    # Para el ejemplo, supongamos que simulamos eventos con numpy:
    np.random.seed(seed)

    dias_sim = (fecha_fin_sim - fecha_inicio_sim).days + 1
    media_diaria_real = df[(df['Fecha'] >= fecha_inicio_sim) & (df['Fecha'] <= fecha_fin_sim)].shape[0] / dias_sim

    # Aquí deberías incluir tu lógica real de simulación, esto es un dummy:
    num_eventos = min(max_eventos, int(media_diaria_real * dias_sim * mu_boost))

    # Creamos un GeoDataFrame ficticio con los eventos simulados:
    lons = np.random.uniform(gdf_zona.total_bounds[0], gdf_zona.total_bounds[2], num_eventos)
    lats = np.random.uniform(gdf_zona.total_bounds[1], gdf_zona.total_bounds[3], num_eventos)
    fechas = pd.date_range(start=fecha_inicio_sim, periods=dias_sim).to_pydatetime().tolist()
    fechas_eventos = np.random.choice(fechas, num_eventos)

    gdf_sim = gpd.GeoDataFrame({
        'Fecha': fechas_eventos,
        'geometry': gpd.points_from_xy(lons, lats)
    }, crs=gdf_zona.crs)

    media_diaria_sim = num_eventos / dias_sim

    msg(f"✅ Simulados {num_eventos} eventos")
    msg(f"📊 Media diaria real: {media_diaria_real:.2f}")
    msg(f"📊 Media diaria simulada: {media_diaria_sim:.2f}")

    return gdf_sim
