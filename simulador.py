import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.interpolate import BSpline, interp1d
from numba import njit

# --- Función de entrenamiento ---
def entrenar_modelo_gam(df_train):
    df_train = df_train.sort_values("Fecha").reset_index(drop=True)
    t0 = df_train["Fecha"].min()
    times = (df_train["Fecha"] - t0).dt.total_seconds() / (3600 * 24)
    T = times.max()
    times_np = times.values

    K, degree = 8, 3
    knots = np.linspace(0, T, K)
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

    spline_mat = spline_basis(times_np)
    time_grid = np.linspace(0, T, 1000)
    spline_grid = spline_basis(time_grid)
    mu_integral_coeffs = np.trapz(spline_grid, time_grid, axis=0)

    lambda_reg = 0.01

    @njit
    def compute_log_likelihood(times, mu_vals, alpha_vals, decay):
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

    def log_likelihood(params):
        c_mu = params[:n_coef]
        c_alpha = params[n_coef:2 * n_coef]
        decay = params[-1]

        if np.any(c_mu < 0) or np.any(c_alpha < 0) or decay <= 0:
            return np.inf

        mu_vals = spline_mat @ c_mu
        alpha_vals = spline_mat @ c_alpha
        ll = compute_log_likelihood(times_np, mu_vals, alpha_vals, decay)

        mu_integral = mu_integral_coeffs @ c_mu
        hawkes_integral = (np.mean(alpha_vals) / decay) * len(times_np)
        penalty = lambda_reg * np.sum(c_alpha ** 2)

        return -(ll - (mu_integral + hawkes_integral)) + penalty

    initial_params = np.concatenate([
        np.full(n_coef, len(times_np) / T / n_coef),
        np.full(n_coef, 0.1),
        [1.0]
    ])
    bounds = [(0, None)] * (2 * n_coef) + [(1e-3, 100)]
    res = minimize(log_likelihood, initial_params, bounds=bounds, method='L-BFGS-B', options={'maxiter': 2000})

    c_mu, c_alpha, decay = res.x[:n_coef], res.x[n_coef:2 * n_coef], res.x[-1]
    mu_fn = interp1d(time_grid, spline_grid @ c_mu, fill_value="extrapolate")
    alpha_fn = interp1d(time_grid, spline_grid @ c_alpha, fill_value="extrapolate")

    info = {
        "t0": t0,
        "T_train": T,
        "mu_fn": mu_fn,
        "alpha_fn": alpha_fn,
        "decay": decay,
        "mu_diario": np.mean(spline_mat @ c_mu),
        "alpha_diario": np.mean(spline_mat @ c_alpha)
    }
    return info, info  # modelo e info iguales por ahora


# --- Simulación Ogata ---
def simulate_hawkes(mu_fn, alpha_fn, decay, t_start, t_end, max_jumps=10000):
    np.random.seed(42)
    t = t_start
    events = []
    while t < t_end and len(events) < max_jumps:
        excitation = np.sum([alpha_fn(s) * decay * np.exp(-decay * (t - s)) for s in events if s < t])
        lambda_t = mu_fn(t) + excitation
        if lambda_t <= 0:
            break
        u = np.random.uniform()
        w = -np.log(u) / lambda_t
        t_candidate = t + w
        if t_candidate > t_end:
            break
        excitation_candidate = np.sum([alpha_fn(s) * decay * np.exp(-decay * (t_candidate - s)) for s in events if s < t_candidate])
        lambda_candidate = mu_fn(t_candidate) + excitation_candidate
        if np.random.uniform() <= lambda_candidate / lambda_t:
            events.append(t_candidate)
        t = t_candidate
    return np.array(events)


# --- Función de simulación pública ---
def simular_eventos(modelo, fecha_ini, fecha_fin, mu_boost=1.0):
    t0 = modelo["t0"]
    T_train = modelo["T_train"]
    mu_fn_base = modelo["mu_fn"]
    alpha_fn_base = modelo["alpha_fn"]
    decay = modelo["decay"]

    T_ini = (fecha_ini - t0).total_seconds() / (3600 * 24)
    T_fin = (fecha_fin - t0).total_seconds() / (3600 * 24)

    def make_seasonal_fns(mu_fn, alpha_fn, t_train_end, t_start_sim, t_end_sim):
        days_in_year = 365
        n_years = 3
        t_sim_grid = np.linspace(t_start_sim, t_end_sim, 1000)
        seasonal_mu = np.zeros_like(t_sim_grid)
        seasonal_alpha = np.zeros_like(t_sim_grid)

        for y in range(1, n_years + 1):
            offset = y * days_in_year
            t_shifted = t_sim_grid - offset
            valid = (t_shifted >= 0) & (t_shifted <= t_train_end)
            seasonal_mu[valid] += mu_fn(t_shifted[valid])
            seasonal_alpha[valid] += alpha_fn(t_shifted[valid])

        seasonal_mu /= n_years
        seasonal_alpha /= n_years

        mu_fn = interp1d(t_sim_grid, mu_boost * seasonal_mu, fill_value="extrapolate")
        alpha_fn = interp1d(t_sim_grid, seasonal_alpha, fill_value="extrapolate")
        return mu_fn, alpha_fn

    mu_fn_adj, alpha_fn_adj = make_seasonal_fns(mu_fn_base, alpha_fn_base, T_train, T_ini, T_fin)
    eventos_simulados = simulate_hawkes(mu_fn_adj, alpha_fn_adj, decay, T_ini, T_fin)

    return eventos_simulados
