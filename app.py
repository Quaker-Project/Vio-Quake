import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import BSpline, interp1d
from numba import njit
import io
import matplotlib.pyplot as plt

st.set_page_config(page_title="Hawkes Spline Simulador", layout="wide")

# 1. Funciones spline para base y alpha ----------------------------------
def spline_basis(t, knots, degree, n_coef):
    t = np.atleast_1d(t)
    knots_extended = np.concatenate((
        np.repeat(knots[0], degree), 
        knots, 
        np.repeat(knots[-1], degree)
    ))
    basis_matrix = np.zeros((len(t), n_coef))
    for i in range(n_coef):
        c = np.zeros(n_coef)
        c[i] = 1
        spline = BSpline(knots_extended, c, degree)
        basis_matrix[:, i] = spline(t)
    return basis_matrix

# 2. Log-verosimilitud y optimización --------------------------------------
lambda_reg = 0.01

@njit
def compute_log_likelihood_numba(times, mu_vals, alpha_vals, decay):
    n = len(times)
    ll = 0.0
    for i in range(n):
        excitation = 0.0
        for j in range(i):
            dt = times[i] - times[j]
            excitation += alpha_vals[j] * decay * np.exp(-decay * dt)
        intensity = mu_vals[i] + excitation
        if intensity <= 0:
            return -np.inf
        ll += np.log(intensity)
    return ll

def log_likelihood(params, times, spline_mat, mu_integral_coeffs):
    n_coef = spline_mat.shape[1]
    c_mu = params[:n_coef]
    c_alpha = params[n_coef:2*n_coef]
    decay = params[-1]
    
    if np.any(c_mu < 0) or np.any(c_alpha < 0) or decay <= 0:
        return np.inf
    
    mu_vals = spline_mat @ c_mu
    alpha_vals = spline_mat @ c_alpha
    
    ll = compute_log_likelihood_numba(times, mu_vals, alpha_vals, decay)
    mu_integral = mu_integral_coeffs @ c_mu
    hawkes_integral = (np.mean(alpha_vals) / decay) * len(times)
    penalty = lambda_reg * np.sum(c_alpha**2)
    
    return -(ll - (mu_integral + hawkes_integral)) + penalty

# 3. Simulación Ogata ------------------------------------------------------
def simulate_hawkes_ogata(mu_fn, alpha_fn, decay, t_start, t_end, max_jumps=10000, seed=None):
    rng = np.random.default_rng(seed)
    t = t_start
    events = []
    while t < t_end and len(events) < max_jumps:
        excitation = np.sum([alpha_fn(s) * decay * np.exp(-decay * (t - s)) for s in events if s < t])
        lambda_t = mu_fn(t) + excitation
        if lambda_t <= 0:
            break
        u = rng.uniform()
        w = -np.log(u) / lambda_t
        t_candidate = t + w
        if t_candidate > t_end:
            break
        excitation_candidate = np.sum([alpha_fn(s) * decay * np.exp(-decay * (t_candidate - s)) for s in events if s < t_candidate])
        lambda_candidate = mu_fn(t_candidate) + excitation_candidate
        if rng.uniform() <= lambda_candidate / lambda_t:
            events.append(t_candidate)
        t = t_candidate
    return np.array(events)

# 4. Streamlit app ---------------------------------------------------------
def main():
    st.title("🧨 Hawkes Spline Simulador Espacio-Temporal")

    archivo = st.file_uploader("Sube archivo CSV o Excel con columnas: Fecha (datetime), Long, Lat", type=['csv', 'xls', 'xlsx'])
    if archivo is None:
        st.info("Carga los datos para continuar.")
        return
    
    try:
        if archivo.name.endswith('.csv'):
            df = pd.read_csv(archivo)
        else:
            df = pd.read_excel(archivo)
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
        if df['Fecha'].isnull().any():
            st.error("Algunas fechas no son válidas.")
            return
        if not all(c in df.columns for c in ['Long', 'Lat']):
            st.error("Faltan columnas 'Long' y 'Lat'.")
            return
    except Exception as e:
        st.error(f"Error al cargar archivo: {e}")
        return
    
    # Filtrar periodo entrenamiento
    min_fecha = df['Fecha'].min()
    max_fecha = df['Fecha'].max()
    
    st.sidebar.header("Configuración entrenamiento y simulación")
    fecha_inicio_train = st.sidebar.date_input("Fecha inicio entrenamiento", value=min_fecha.date(), min_value=min_fecha.date(), max_value=max_fecha.date())
    fecha_fin_train = st.sidebar.date_input("Fecha fin entrenamiento", value=max_fecha.date(), min_value=min_fecha.date(), max_value=max_fecha.date())
    fecha_inicio_sim = st.sidebar.date_input("Fecha inicio simulación", value=max_fecha.date())
    fecha_fin_sim = st.sidebar.date_input("Fecha fin simulación", value=max_fecha.date() + pd.Timedelta(days=30))
    
    if fecha_inicio_train > fecha_fin_train:
        st.error("Fecha inicio entrenamiento debe ser anterior a fin entrenamiento.")
        return
    if fecha_inicio_sim > fecha_fin_sim:
        st.error("Fecha inicio simulación debe ser anterior a fin simulación.")
        return
    
    df_train = df[(df['Fecha'] >= pd.to_datetime(fecha_inicio_train)) & (df['Fecha'] <= pd.to_datetime(fecha_fin_train))].copy()
    if df_train.empty:
        st.error("No hay datos en el periodo de entrenamiento.")
        return
    
    # Preparar tiempos para spline
    t0_train = df_train['Fecha'].min()
    times_train = (df_train['Fecha'] - t0_train).dt.total_seconds() / (3600*24)
    times_train_np = times_train.values
    T_train = times_train.max()
    
    K = st.sidebar.slider("Número de knots splines", 4, 20, 8)
    degree = 3
    knots = np.linspace(0, T_train, K)
    knots_extended = np.concatenate((
        np.repeat(knots[0], degree),
        knots,
        np.repeat(knots[-1], degree)
    ))
    n_coef = len(knots) + degree - 1
    spline_mat = spline_basis(times_train_np, knots, degree, n_coef)
    
    time_grid = np.linspace(0, T_train, 1000)
    spline_grid = spline_basis(time_grid, knots, degree, n_coef)
    mu_integral_coeffs = np.trapz(spline_grid, time_grid, axis=0)
    
    st.write(f"Entrenando modelo con {len(df_train)} eventos... esto puede tardar un momento.")
    
    initial_params = np.concatenate([
        np.full(n_coef, len(times_train_np) / T_train / n_coef),
        np.full(n_coef, 0.1),
        [1.0]
    ])
    bounds = [(0, None)]*(2*n_coef) + [(1e-3, 100)]
    
    @st.cache_data(show_spinner=False)
    def entrenar(params):
        res = minimize(log_likelihood, params,
                       args=(times_train_np, spline_mat, mu_integral_coeffs),
                       bounds=bounds,
                       method='L-BFGS-B',
                       options={'maxiter': 2000})
        return res
    
    resultado = entrenar(initial_params)
    
    if not resultado.success:
        st.error("No convergió la optimización.")
        return
    
    c_mu_fit = resultado.x[:n_coef]
    c_alpha_fit = resultado.x[n_coef:2*n_coef]
    decay_fit = resultado.x[-1]
    
    mu_vals_fit = spline_mat @ c_mu_fit
    alpha_vals_fit = spline_mat @ c_alpha_fit
    
    st.success("Modelo entrenado correctamente.")
    st.write(f"Decay estimado (beta): {decay_fit:.3f}")
    
    # Construir funciones interpoladas
    mu_interp = interp1d(time_grid, spline_grid @ c_mu_fit, kind='cubic', fill_value='extrapolate')
    alpha_interp = interp1d(time_grid, spline_grid @ c_alpha_fit, kind='cubic', fill_value='extrapolate')
    
    mu_boost = st.sidebar.slider("Multiplicador mu (boost)", 0.1, 3.0, 1.0, 0.1)
    max_events_sim = st.sidebar.number_input("Máximo eventos simulados", 1000, 100000, 10000, step=1000)
    seed = st.sidebar.number_input("Semilla aleatoria (0=sin semilla)", 0, 99999, 42, step=1)
    seed = None if seed == 0 else seed
    
    # Convertir fechas simulación a tiempos relativos
    t_sim_start = (pd.to_datetime(fecha_inicio_sim) - t0_train).total_seconds() / (3600*24)
    t_sim_end = (pd.to_datetime(fecha_fin_sim) - t0_train).total_seconds() / (3600*24)
    
    # Funciones boosteadas
    def mu_fn_boosted(t): return mu_boost * mu_interp(t)
    def alpha_fn(t): return alpha_interp(t)
    
    if st.button("Simular eventos en periodo seleccionado"):
        with st.spinner("Simulando eventos..."):
            eventos_sim = simulate_hawkes_ogata(
                mu_fn=mu_fn_boosted,
                alpha_fn=alpha_fn,
                decay=decay_fit,
                t_start=t_sim_start,
                t_end=t_sim_end,
                max_jumps=max_events_sim,
                seed=seed
            )
            st.success(f"Simulación completada: {len(eventos_sim)} eventos generados.")
            
            # Mostrar resumen
            df_sim = pd.DataFrame({
                'Tiempo_días': eventos_sim,
                'Fecha': pd.to_datetime(eventos_sim, unit='D', origin=t0_train)
            })
            st.dataframe(df_sim)
            
            # Gráfico simulación temporal
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(eventos_sim, bins=50, alpha=0.7)
            ax.set_xlabel("Tiempo (días desde inicio simulación)")
            ax.set_ylabel("Eventos")
            ax.set_title("Histograma de eventos simulados")
            st.pyplot(fig)
            
            # Botón para descargar resultados
            csv_buffer = io.StringIO()
            df_sim.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Descargar eventos simulados CSV",
                data=csv_buffer.getvalue(),
                file_name="eventos_simulados.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
