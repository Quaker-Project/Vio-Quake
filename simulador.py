# simulador.py
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from datetime import timedelta
import numpyro.distributions as dist
from bstpp.main import Hawkes_Model


def entrenar_modelo(gdf_events, gdf_boundaries, fecha_inicio, fecha_split, num_steps=1500):
    t0 = gdf_events["Fecha"].min()
    gdf_events["t"] = (gdf_events["Fecha"] - t0).dt.total_seconds() / 86400
    gdf_events = gdf_events.sort_values(by="t").reset_index(drop=True)

    gdf_train = gdf_events[(gdf_events["Fecha"] >= fecha_inicio) & (gdf_events["Fecha"] < fecha_split)]
    gdf_test = gdf_events[gdf_events["Fecha"] >= fecha_split]

    data_model = gdf_train[["t", "Long", "Lat"]].rename(columns={"t": "T", "Long": "X", "Lat": "Y"})

    model = Hawkes_Model(
        data=data_model,
        A=gdf_boundaries,
        T=gdf_train["t"].max(),
        cox_background=True,
        a_0=dist.Normal(1, 10),
        alpha=dist.Beta(20, 60),
        beta=dist.HalfNormal(2.0),
        sigmax_2=dist.HalfNormal(0.25)
    )

    model.run_svi(lr=0.02, num_steps=num_steps)
    return model, gdf_train, gdf_test, t0


def simular_eventos(model):
    # Ajustar tamaño de la rejilla si da error de longitud
    n_cells = model.args["xy_events"].shape[1]
    model.args["n_xy"] = int(np.sqrt(n_cells))
    model.run_svi(lr=0.001, num_steps=10)

    eventos_simulados = model.simulate(model.samples)
    if len(eventos_simulados) == 0:
        return gpd.GeoDataFrame(columns=["t", "X", "Y", "Fecha", "geometry"], crs="EPSG:4326")

    df_sim = pd.DataFrame(eventos_simulados, columns=["t", "X", "Y"])
    df_sim["Fecha"] = model.data["T"].min() + pd.to_timedelta(df_sim["t"], unit="D")
    geometry = [Point(xy) for xy in zip(df_sim["X"], df_sim["Y"])]
    gdf_simulados = gpd.GeoDataFrame(df_sim, geometry=geometry, crs="EPSG:4326")
    return gdf_simulados
