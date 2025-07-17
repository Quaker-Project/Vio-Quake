import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.interpolate import BSpline, interp1d
from datetime import datetime
from numba import njit

# --- SPLINE DEFINITION ---
def spline_basis_matrix(t, knots, degree):
    n_coef = len(knots) + degree - 1
    knots_ext = np.concatenate(([knots[0]] * degree, knots, [knots[-1]] * degree))
    spline_mat = np.zeros((len(t), n_coef))
    for i in range(n_coef):
        coeffs = np.zeros(n_coef)
        coeffs[i] = 1
        spline = BSpline(knots_ext, coeffs, degree)
        spline_mat[:, i] = spline(t)
    return spline_mat

# --- LOG-LIKELIHOOD FUNCTION ---
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
            return -1e10  # Penaliza intensidades no positivas
        ll += np.log(intensity)
    return ll

def entrenar_modelo_gam(df_train, lambda_reg=0.01, K=8, degree=3):
    df_train = df_train.sort_values('Fecha').reset_index(drop=True)
    t0 = df_train['Fecha'].min()
    times = (df_train['Fecha'] - t0).dt.total_seconds() / (3600 * 24)
    times_np = times.values
    T_train = times.max()

    knots = np.linspace(0, T_train, K)
    spline_mat = spline_basis_matrix(times_np, knots, degree)
    n_coef = spline_mat.shape[1]
    spline_grid = spline_basis_matrix(np.linspace(0, T_train, 1000), knots, degree)
    mu_integral_coeffs = np.trapz(spline_grid, np.linspace(0, T_train, 1000), axis=0)

    def mu_from_c(c_mu): return spline_mat @ c_mu
    def alpha_from_c(c_alpha): return spline_mat @ c_alpha

    def neg_log_likelihood(params):
        c_mu = params[:n_coef]
        c_alpha = params[n_coef:2 * n_coef]
        decay = params[-1]

        if np.any(c_mu < 0) or np.any(c_alpha < 0) or decay <= 0:
            return 1e10

        mu_vals = mu_from_c(c_mu)
        alpha_vals = alpha_from_c(c_alpha)
        ll = compute_log_likelihood_numba(times_np, mu_vals, alpha_vals, decay)

        mu_integral = mu_integral_coeffs @ c_mu
        hawkes_integral = (np.mean(alpha_vals) / decay) * len(times_np)
        penalty = lambda_reg * np.sum(c_alpha ** 2)
        return -(ll - (mu_integral + hawkes_integral)) + penalty

    initial_params = np.concatenate([
        np.full(n_coef, len(times_np) / T_train / n_coef),  # mu
        np.full(n_coef, 0.1),  # alpha
        [1.0]  # decay
    ])
    bounds = [(0, None)] * (2 * n_coef) + [(1e-3, 50)]
    res = minimize(neg_log_likelihood, initial_params, bounds=bounds, method='L-BFGS-B')

    c_mu_fit, c_alpha_fit, decay_fit = res.x[:n_coef], res.x[n_coef:2 * n_coef], res.x[-1]
    mu_interp_fn = interp1d(
        np.linspace(0, T_train, 1000),
        spline_basis_matrix(np.linspace(0, T_train, 1000), knots, degree) @ c_mu_fit,
        kind='cubic', fill_value="extrapolate"
    )
    alpha_interp_fn = interp1d(
        np.linspace(0, T_train, 1000),
        spline_basis_matrix(np.linspace(0, T_train, 1000), knots, degree) @ c_alpha_fit,
        kind='cubic', fill_value="extrapolate"
    )

    return {
        "t0": t0,
        "T_train": T_train,
        "mu_fn": mu_interp_fn,
        "alpha_fn": alpha_interp_fn,
        "decay": decay_fit
    }

# --- SIMULATION FUNCTION ---
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
        excitation_cand = np.sum([alpha_fn(s) * decay * np.exp(-decay * (t_candidate - s)) for s in events if s < t_candidate])
        lambda_cand = mu_fn(t_candidate) + excitation_cand
        if np.random.uniform() <= lambda_cand / lambda_t:
            events.append(t_candidate)
        t = t_candidate
    return np.array(events)

# --- FORECAST ---
def simular_eventos(modelo, fecha_ini, fecha_fin, mu_boost=1.0):
    t0 = modelo["t0"]
    T_train = modelo["T_train"]
    mu_fn_base = modelo["mu_fn"]
    alpha_fn_base = modelo["alpha_fn"]
    decay = modelo["decay"]

    T_ini = (pd.to_datetime(fecha_ini) - t0).total_seconds() / (3600 * 24)
    T_fin = (pd.to_datetime(fecha_fin) - t0).total_seconds() / (3600 * 24)

    t_grid = np.linspace(T_ini, T_fin, 1000)
    seasonal_mu = np.zeros_like(t_grid)
    seasonal_alpha = np.zeros_like(t_grid)
    for y in range(1, 4):
        offset = y * 365
        shifted = t_grid - offset
        valid = (shifted >= 0) & (shifted <= T_train)
        seasonal_mu[valid] += mu_fn_base(shifted[valid])
        seasonal_alpha[valid] += alpha_fn_base(shifted[valid])
    seasonal_mu /= 3
    seasonal_alpha /= 3

    mu_fn = interp1d(t_grid, mu_boost * seasonal_mu, fill_value="extrapolate")
    alpha_fn = interp1d(t_grid, seasonal_alpha, fill_value="extrapolate")

    simulated_events = simulate_hawkes(mu_fn, alpha_fn, decay, T_ini, T_fin)
    return simulated_events
