# simulador.py

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import BSpline, interp1d
from numba import njit

# ---------- SPLINE BASIS ----------
def create_spline_basis(t, T_train, K=8, degree=3):
    knots = np.linspace(0, T_train, K)
    knots_extended = np.concatenate((np.repeat(knots[0], degree), knots, np.repeat(knots[-1], degree)))
    n_coef = len(knots) + degree - 1

    t = np.atleast_1d(t)
    basis_matrix = np.zeros((len(t), n_coef))
    for i in range(n_coef):
        c = np.zeros(n_coef)
        c[i] = 1
        spline = BSpline(knots_extended, c, degree)
        basis_matrix[:, i] = spline(t)
    return basis_matrix, knots_extended, n_coef

# ---------- LIKELIHOOD ----------
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

# ---------- ENTRENAMIENTO ----------
def entrenar_modelo(df, fecha_inicio, fecha_fin, lambda_reg=0.01):
    df = df[(df['Fecha'] >= fecha_inicio) & (df['Fecha'] <= fecha_fin)].copy()
    df = df.sort_values('Fecha').reset_index(drop=True)

    t0_train = df['Fecha'].min()
    times_train = (df['Fecha'] - t0_train).dt.total_seconds() / (3600 * 24)
    times_np = times_train.values
    T_train = times_train.max()

    spline_mat, knots_ext, n_coef = create_spline_basis(times_np, T_train)
    time_grid = np.linspace(0, T_train, 1000)
    spline_grid, _, _ = create_spline_basis(time_grid, T_train)
    mu_integral_coeffs = np.trapz(spline_grid, time_grid, axis=0)

    def mu_vals(c_mu): return spline_mat @ c_mu
    def alpha_vals(c_alpha): return spline_mat @ c_alpha

    def log_likelihood(params):
        c_mu = params[:n_coef]
        c_alpha = params[n_coef:2*n_coef]
        decay = params[-1]

        if np.any(c_mu < 0) or np.any(c_alpha < 0) or decay <= 0:
            return np.inf

        mu = mu_vals(c_mu)
        alpha = alpha_vals(c_alpha)

        ll = compute_log_likelihood_numba(times_np, mu, alpha, decay)

        mu_integral = mu_integral_coeffs @ c_mu
        hawkes_integral = (np.mean(alpha) / decay) * len(times_np)
        penalty = lambda_reg * np.sum(c_alpha**2)

        return -(ll - (mu_integral + hawkes_integral)) + penalty

    initial_params = np.concatenate([
        np.full(n_coef, len(times_np)/T_train / n_coef),
        np.full(n_coef, 0.1),
        [1.0]
    ])
    bounds = [(0, None)]*(2*n_coef) + [(1e-3, 100)]

    res = minimize(log_likelihood, initial_params, bounds=bounds, method='L-BFGS-B', options={'maxiter':2000})

    c_mu_fit, c_alpha_fit, decay_fit = res.x[:n_coef], res.x[n_coef:2*n_coef], res.x[-1]
    mu_interp = interp1d(time_grid, spline_grid @ c_mu_fit, kind='cubic', fill_value="extrapolate")
    alpha_interp = interp1d(time_grid, spline_grid @ c_alpha_fit, kind='cubic', fill_value="extrapolate")

    return {
        "mu_interp": mu_interp,
        "alpha_interp": alpha_interp,
        "decay": decay_fit,
        "t0": t0_train,
        "T_train": T_train,
        "params": res.x
    }

# ---------- SIMULACIÓN ----------
def simulate_hawkes(mu_fn, alpha_fn, decay, t_start, t_end, max_jumps=10000):
    np.random.seed(42)
    t = t_start
    events = []
    while t < t_end and len(events) < max_jumps:
        excitation = sum(alpha_fn(s) * decay * np.exp(-decay * (t - s)) for s in events if s < t)
        lambda_t = mu_fn(t) + excitation
        if lambda_t <= 0:
            break
        u = np.random.uniform()
        w = -np.log(u) / lambda_t
        t_candidate = t + w
        if t_candidate > t_end:
            break
        excitation_cand = sum(alpha_fn(s) * decay * np.exp(-decay * (t_candidate - s)) for s in events if s < t_candidate)
        lambda_cand = mu_fn(t_candidate) + excitation_cand
        if np.random.uniform() <= lambda_cand / lambda_t:
            events.append(t_candidate)
        t = t_candidate
    return np.array(events)
