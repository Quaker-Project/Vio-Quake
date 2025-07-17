import numpy as np
import pandas as pd
from scipy.interpolate import BSpline, interp1d
from scipy.optimize import minimize
from numba import njit

# --- 1. Preparación de base spline ---
def preparar_spline(times_train_np, K=8, degree=3):
    T_train = times_train_np.max()
    knots = np.linspace(0, T_train, K)
    knots_extended = np.concatenate((np.repeat(knots[0], degree), knots, np.repeat(knots[-1], degree)))
    n_coef = len(knots) + degree - 1

    def spline_basis(t):
        t = np.atleast_1d(t)
        basis_matrix = np.zeros((len(t), n_coef))
        for i in range(n_coef):
            c = np.zeros(n_coef)
            c[i] = 1
            spline = BSpline(knots_extended, c, degree)
            basis_matrix[:, i] = spline(t)
        return basis_matrix

    spline_mat = spline_basis(times_train_np)
    return spline_mat, knots_extended, n_coef, T_train

# --- 2. Funciones para mu y alpha desde coef ---
def mu_vals_from_coef(c_mu, spline_mat):
    return spline_mat @ c_mu

def alpha_vals_from_coef(c_alpha, spline_mat):
    return spline_mat @ c_alpha

# --- 3. Log-likelihood con numba ---
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

# --- 4. Función de optimización ---
def entrenar_modelo_gam(df, fecha_inicio_train, fecha_fin_train, lambda_reg=0.01, usar_hora=True):
    df_train = df[(df['Fecha'] >= fecha_inicio_train) & (df['Fecha'] <= fecha_fin_train)].copy()
    df_train = df_train.sort_values('Fecha').reset_index(drop=True)
    t0_train = df_train['Fecha'].min()

    if usar_hora:
        times_train = (df_train['Fecha'] - t0_train).dt.total_seconds() / (3600 * 24)
    else:
        times_train = (df_train['Fecha'].dt.floor('D') - t0_train).dt.total_seconds() / (3600 * 24)

    times_train_np = times_train.values

    spline_mat, knots_extended, n_coef, T_train = preparar_spline(times_train_np)

    def log_likelihood(params):
        c_mu = params[:n_coef]
        c_alpha = params[n_coef:2*n_coef]
        decay = params[-1]

        if np.any(c_mu < 0) or np.any(c_alpha < 0) or decay <= 0:
            return np.inf

        mu_vals = mu_vals_from_coef(c_mu, spline_mat)
        alpha_vals = alpha_vals_from_coef(c_alpha, spline_mat)

        ll = compute_log_likelihood_numba(times_train_np, mu_vals, alpha_vals, decay)

        # Integral aproximado (trapecio) para penalización regularización
        time_grid = np.linspace(0, T_train, 1000)
        spline_grid = np.zeros((len(time_grid), n_coef))
        for i in range(n_coef):
            c = np.zeros(n_coef)
            c[i] = 1
            spline_grid[:, i] = BSpline(knots_extended, c, 3)(time_grid)

        mu_integral = np.trapz(spline_grid @ c_mu, time_grid)
        hawkes_integral = (np.mean(alpha_vals) / decay) * len(times_train_np)
        penalty = lambda_reg * np.sum(c_alpha ** 2)

        return -(ll - (mu_integral + hawkes_integral)) + penalty

    initial_params = np.concatenate([
        np.full(n_coef, len(times_train_np) / T_train / n_coef),
        np.full(n_coef, 0.1),
        [1.0]
    ])

    bounds = [(0, None)] * (2 * n_coef) + [(1e-3, 100)]

    res = minimize(log_likelihood, initial_params, bounds=bounds, method='L-BFGS-B', options={'maxiter': 2000})

    c_mu_fit, c_alpha_fit, decay_fit = res.x[:n_coef], res.x[n_coef:2*n_coef], res.x[-1]

    # Funciones interpoladas para usar luego
    time_grid = np.linspace(0, T_train, 1000)
    spline_grid = np.zeros((len(time_grid), n_coef))
    for i in range(n_coef):
        c = np.zeros(n_coef)
        c[i] = 1
        spline_grid[:, i] = BSpline(knots_extended, c, 3)(time_grid)

    mu_interp_fn = interp1d(time_grid, spline_grid @ c_mu_fit, kind='cubic', fill_value="extrapolate")
    alpha_interp_fn = interp1d(time_grid, spline_grid @ c_alpha_fit, kind='cubic', fill_value="extrapolate")

    # Factor ajuste base histórico
    factor_ajuste = len(times_train_np) / (np.trapz(mu_interp_fn(time_grid), time_grid) + (np.mean(alpha_interp_fn(time_grid)) / decay_fit) * len(times_train_np))

    return {
        'mu_interp_fn': mu_interp_fn,
        'alpha_interp_fn': alpha_interp_fn,
        'decay': decay_fit,
        't0_train': t0_train,
        'T_train': T_train,
        'factor_ajuste': factor_ajuste
    }

# --- 5. Simulación Ogata ---
def simulate_hawkes_ogata(mu_fn, alpha_fn, decay, t_start, t_end, max_jumps=10000, seed=None):
    if seed is not None:
        np.random.seed(seed)
    t = t_start
    events = []
    while t < t_end and len(events) < max_jumps:
        excitation = 0.0
        for s in events:
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
        for s in events:
            if s < t_candidate:
                excitation_candidate += alpha_fn(s) * decay * np.exp(-decay * (t_candidate - s))
        lambda_candidate = mu_fn(t_candidate) + excitation_candidate
        if np.random.uniform() <= lambda_candidate / lambda_t:
            events.append(t_candidate)
        t = t_candidate
    return np.array(events)

# --- 6. Función principal de simulación para la app ---
def simular_eventos(df, fecha_inicio_train, fecha_fin_train, fecha_inicio_sim, fecha_fin_sim,
                    mu_boost=1.0, max_eventos=5000, seed=None, usar_hora=True):

    modelo = entrenar_modelo_gam(df, fecha_inicio_train, fecha_fin_train, usar_hora=usar_hora)
    mu_interp_fn = modelo['mu_interp_fn']
    alpha_interp_fn = modelo['alpha_interp_fn']
    decay = modelo['decay']
    t0_train = modelo['t0_train']
    factor_ajuste = modelo['factor_ajuste']

    # Ajuste boost para mu base
    def mu_fn(t):
        return mu_boost * factor_ajuste * mu_interp_fn(t)

    alpha_fn = alpha_interp_fn

    T_ini = (fecha_inicio_sim - t0_train).total_seconds() / (3600 * 24)
    T_fin = (fecha_fin_sim - t0_train).total_seconds() / (3600 * 24)

    eventos_sim = simulate_hawkes_ogata(mu_fn, alpha_fn, decay, T_ini, T_fin, max_jumps=max_eventos, seed=seed)

    # Convertir a fechas reales
    fechas_sim = pd.to_datetime(eventos_sim * 24 * 3600, unit='s', origin=t0_train)

    # Generar GeoDataFrame si quieres añadir coords (ejemplo random en zona, aquí solo devuelve DataFrame)
    df_sim = pd.DataFrame({'Fecha': fechas_sim})

    return df_sim

