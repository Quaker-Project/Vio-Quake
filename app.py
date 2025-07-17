import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import tempfile
import os
import io
import zipfile
from scipy.optimize import minimize
from scipy.interpolate import BSpline, interp1d
from numba import njit

# --- Funciones para cargar archivos ---

def cargar_archivo_datos(archivo):
    if archivo is None:
        return None
    try:
        if archivo.name.endswith('.csv'):
            df = pd.read_csv(archivo)
        elif archivo.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(archivo)
        else:
            st.error("Formato no soportado. Use CSV o Excel.")
            return None
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error cargando archivo: {e}")
        return None

def cargar_shapefile_zip(archivo_zip):
    if archivo_zip is None:
        return None
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with zipfile.ZipFile(archivo_zip) as z:
                z.extractall(tmpdir)
            shp_files = [f for f in os.listdir(tmpdir) if f.endswith('.shp')]
            if len(shp_files) != 1:
                st.error("El ZIP debe contener un único archivo .shp")
                return None
            gdf = gpd.read_file(os.path.join(tmpdir, shp_files[0]))
            return gdf
        except Exception as e:
            st.error(f"Error leyendo shapefile ZIP: {e}")
            return None

# --- Funciones spline y modelo Hawkes ---

def crear_spline_basis(t, knots, degree):
    knots_extended = np.concatenate((np.repeat(knots[0], degree), knots, np.repeat(knots[-1], degree)))
    n_coef = len(knots) + degree - 1
    t = np.atleast_1d(t)
    basis_matrix = np.zeros((len(t), n_coef))
    for i in range(n_coef):
        c = np.zeros(n_coef)
        c[i] = 1
        spline = BSpline(knots_extended, c, degree)
        basis_matrix[:, i] = spline(t)
    return basis_matrix

@njit
def log_likelihood_numba(times, mu_vals, alpha_vals, decay):
    n = len(times)
    ll = 0.0
    for i in range(n):
        excitation = 0.0
        for j in range(i):
            dt = times[i] - times[j]
            excitation += alpha_vals[j] * decay * np.exp(-decay * dt)
        intensity = mu_vals[i] + excitation
        if intensity <= 0:
            return -1e10
        ll += np.log(intensity)
    return ll

def entrenar_hawkes(times_train_np, T_train, lambda_reg=0.01, K=8, degree=3):
    knots = np.linspace(0, T_train, K)
    spline_mat = crear_spline_basis(times_train_np, knots, degree)
    n_coef = spline_mat.shape[1]

    def mu_vals_from_coef(c_mu):
        return spline_mat @ c_mu

    def alpha_vals_from_coef(c_alpha):
        return spline_mat @ c_alpha

    def objetivo(params):
        c_mu = params[:n_coef]
        c_alpha = params[n_coef:2*n_coef]
        decay = params[-1]
        if np.any(c_mu < 0) or np.any(c_alpha < 0) or decay <= 0:
            return 1e10
        mu_vals = mu_vals_from_coef(c_mu)
        alpha_vals = alpha_vals_from_coef(c_alpha)
        ll = log_likelihood_numba(times_train_np, mu_vals, alpha_vals, decay)
        mu_integral = np.trapz(spline_mat @ c_mu, times_train_np)
        hawkes_integral = (np.mean(alpha_vals) / decay) * len(times_train_np)
        penalty = lambda_reg * np.sum(c_alpha**2)
        return -(ll - (mu_integral + hawkes_integral)) + penalty

    init_c_mu = np.full(n_coef, len(times_train_np)/T_train / n_coef)
    init_c_alpha = np.full(n_coef, 0.1)
    init_decay = 1.0
    initial_params = np.concatenate([init_c_mu, init_c_alpha, [init_decay]])
    bounds = [(0, None)]*(2*n_coef) + [(1e-3, 100)]
    res = minimize(objetivo, initial_params, bounds=bounds, method='L-BFGS-B', options={'maxiter':2000})
    c_mu_fit, c_alpha_fit, decay_fit = res.x[:n_coef], res.x[n_coef:2*n_coef], res.x[-1]

    knots_extended = np.concatenate((np.repeat(knots[0], degree), knots, np.repeat(knots[-1], degree)))
    spline_grid = crear_spline_basis(np.linspace(0, T_train, 1000), knots, degree)

    mu_interp = interp1d(np.linspace(0, T_train, 1000), spline_grid @ c_mu_fit, kind='cubic', fill_value="extrapolate")
    alpha_interp = interp1d(np.linspace(0, T_train, 1000), spline_grid @ c_alpha_fit, kind='cubic', fill_value="extrapolate")

    return mu_interp, alpha_interp, decay_fit, knots, degree, T_train

def simular_ogata(mu_fn, alpha_fn, decay, t_start, t_end, max_events=10000, seed=None):
    if seed is not None:
        np.random.seed(seed)
    t = t_start
    eventos = []
    while t < t_end and len(eventos) < max_events:
        excitation = 0.0
        for s in eventos:
            if s < t:
                excitation += alpha_fn(s) * decay * np.exp(-decay * (t - s))
        lambda_t = mu_fn(t) + excitation
        if lambda_t <= 0:
            break
        u = np.random.uniform()
        w = -np.log(u) / lambda_t
        t_candidate = t + w
        if t_candidate > t_end:
            break
        excitation_candidate = 0.0
        for s in eventos:
            if s < t_candidate:
                excitation_candidate += alpha_fn(s) * decay * np.exp(-decay * (t_candidate - s))
        lambda_candidate = mu_fn(t_candidate) + excitation_candidate
        if np.random.uniform() <= lambda_candidate / lambda_t:
            eventos.append(t_candidate)
        t = t_candidate
    return np.array(eventos)

def main():
    st.title("🧨 VIO-QUAKE | Simulador Hawkes espacio-temporal")

    st.markdown("""
    **Simulación de eventos delictivos basada en procesos Hawkes con ajuste spline.**
    """)

    archivo = st.file_uploader("Sube datos de eventos (CSV/Excel) con columnas 'Long', 'Lat', 'Fecha'", type=['csv', 'xls', 'xlsx'])
    gdf_zona_zip = st.file_uploader("Sube shapefile ZIP para zona de simulación", type=['zip'])

    df = cargar_archivo_datos(archivo)
    gdf_zona = cargar_shapefile_zip(gdf_zona_zip)

    if df is not None:
        cols_req = ['Long', 'Lat', 'Fecha']
        if not all(col in df.columns for col in cols_req):
            st.error(f"Faltan columnas requeridas: {cols_req}")
            return
        if df['Fecha'].isnull().any():
            st.error("Fechas inválidas en datos. Corrige el archivo.")
            return

    if df is not None and gdf_zona is not None:
        usar_hora = st.sidebar.checkbox("¿Usar hora en las fechas?", value=True)

        fechas_validas = df['Fecha'].dropna()
        min_fecha = fechas_validas.min()
        max_fecha = fechas_validas.max()

        fecha_inicio_train = st.sidebar.date_input("Fecha inicio entrenamiento", value=min_fecha.date(), min_value=min_fecha.date(), max_value=max_fecha.date())
        fecha_fin_train = st.sidebar.date_input("Fecha fin entrenamiento", value=max_fecha.date(), min_value=min_fecha.date(), max_value=max_fecha.date())
        fecha_inicio_sim = st.sidebar.date_input("Fecha inicio simulación", value=(max_fecha + pd.Timedelta(days=1)).date())
        fecha_fin_sim = st.sidebar.date_input("Fecha fin simulación", value=(max_fecha + pd.Timedelta(days=30)).date())

        if usar_hora:
            fecha_inicio_train = pd.to_datetime(fecha_inicio_train)
            fecha_fin_train = pd.to_datetime(fecha_fin_train) + pd.Timedelta(hours=23, minutes=59)
            fecha_inicio_sim = pd.to_datetime(fecha_inicio_sim)
            fecha_fin_sim = pd.to_datetime(fecha_fin_sim) + pd.Timedelta(hours=23, minutes=59)
        else:
            df['Fecha'] = df['Fecha'].dt.normalize()
            fecha_inicio_train = pd.to_datetime(fecha_inicio_train)
            fecha_fin_train = pd.to_datetime(fecha_fin_train)
            fecha_inicio_sim = pd.to_datetime(fecha_inicio_sim)
            fecha_fin_sim = pd.to_datetime(fecha_fin_sim)

        mu_boost = st.sidebar.slider("Multiplicador mu_boost", 0.1, 5.0, 1.0, 0.1)
        max_eventos = st.sidebar.number_input("Máximo eventos simulados", min_value=100, max_value=100000, value=5000, step=100)
        usar_semilla = st.sidebar.checkbox("Fijar semilla aleatoria", value=False)

        if st.button("Entrenar y simular"):
            with st.spinner("Entrenando modelo..."):
                # Preparamos datos para entrenamiento
                df_train = df[(df['Fecha'] >= fecha_inicio_train) & (df['Fecha'] <= fecha_fin_train)].copy()
                df_train = df_train.sort_values('Fecha').reset_index(drop=True)
                t0_train = df_train['Fecha'].min()
                times_train = (df_train['Fecha'] - t0_train).dt.total_seconds() / (3600*24)
                times_train_np = times_train.values
                T_train = times_train.max()

                mu_interp, alpha_interp, decay_fit, knots, degree, T_train = entrenar_hawkes(times_train_np, T_train)

            with st.spinner("Simulando eventos..."):
                T_ini = (fecha_inicio_sim - t0_train).total_seconds() / (3600*24)
                T_fin = (fecha_fin_sim - t0_train).total_seconds() / (3600*24)

                eventos_sim = simular_ogata(mu_interp, alpha_interp, decay_fit, T_ini, T_fin,
                                            max_events=max_eventos, seed=42 if usar_semilla else None)

                df_sim = pd.DataFrame({
                    'Fecha_simulada': pd.to_datetime(eventos_sim * 3600*24, unit='s', origin=t0_train)
                })
                count_sim = len(df_sim)
                count_real = len(df[(df['Fecha'] >= fecha_inicio_sim) & (df['Fecha'] <= fecha_fin_sim)])

                media_diaria_real = count_real / ((fecha_fin_sim - fecha_inicio_sim).days + 1)
                media_diaria_sim = count_sim / ((fecha_fin_sim - fecha_inicio_sim).days + 1)

                st.write(f"Eventos reales en periodo de simulación: {count_real} (media diaria {media_diaria_real:.2f})")
                st.write(f"Eventos simulados en periodo de simulación: {count_sim} (media diaria {media_diaria_sim:.2f})")

                # Descargar simulados
                towrite = io.BytesIO()
                df_sim.to_excel(towrite, index=False)
                towrite.seek(0)
                st.download_button(label="Descargar eventos simulados (Excel)", data=towrite,
                                   file_name="eventos_simulados.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == "__main__":
    main()
