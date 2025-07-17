import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.interpolate import BSpline, interp1d
from numba import njit

# --- 1. Spline bases ---
def preparar_spline(times, K=8, degree=3):
    T = times.max()
    knots = np.linspace(0, T, K)
    knots_ext = np.concatenate((
        np.repeat(knots[0], degree),
        knots,
        np.repeat(knots[-1], degree)
    ))
    n_coef = len(knots) + degree - 1

    def spline_basis(t):
        t = np.atleast_1d(t)
        basis = np.zeros((len(t), n_coef))
        for i in range(n_coef):
            c = np.zeros(n_coef)
            c[i] = 1
            spline = BSpline(knots_ext, c, degree)
            basis[:, i] = spline(t)
        return basis

    return spline_basis, knots_ext, n_coef, T

# --- 2. Log-likelihood ---
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

def entrenar_modelo_gam(df_train):
    df_train = df_train.sort_values("Fecha").reset_index(drop=True)
    t0 = df_train["Fecha"].min()
    times = (df_train["Fecha"] - t0).dt.total_seconds() / (3600 * 24)
    times_np = times.values

    spline_basis_fn, knots_ext, n_coef, T_train = preparar_spline(times_np)
    spline_mat = spline_basis_fn(times_np)
    spline_grid = spline_basis_fn(np.linspace(0, T_train, 1000))
    mu_integral_coeffs = np.trapz(spline_grid, np.linspace(0, T_train, 1000), axis=0)

    lambda_reg = 0.01

    def mu_vals(c): return spline_mat @ c
    def alpha_vals(c): return spline_mat @ c

    def log_likelihood(params):
        c_mu = params[:n_coef]
        c_alpha = params[n_coef:2*n_coef]
        decay = params[-1]

        if np.any(c_mu < 0) or np.any(c_alpha < 0) or decay <= 0:
            return np.inf

        mu = mu_vals(c_mu)
        alpha = alpha_vals(c_alpha)

        ll = compute_log_likelihood(times_np, mu, alpha, decay)

        mu_integral = mu_integral_coeffs @ c_mu
        hawkes_integral = (np.mean(alpha) / decay) * len(times_np)
        penalty = lambda_reg * np.sum(c_alpha**2)

        return -(ll - (mu_integral + hawkes_integral)) + penalty

    initial = np.concatenate([
        np.full(n_coef, len(times_np)/T_train / n_coef),
        np.full(n_coef, 0.1),
        [1.0]
    ])
    bounds = [(0, None)]*(2*n_coef) + [(1e-3, 100)]
    res = minimize(log_likelihood, initial, bounds=bounds, method='L-BFGS-B')

    c_mu, c_alpha, decay = res.x[:n_coef], res.x[n_coef:2*n_coef], res.x[-1]
    mu_grid = spline_grid @ c_mu
    alpha_grid = spline_grid @ c_alpha

    mu_interp = interp1d(np.linspace(0, T_train, 1000), mu_grid, fill_value="extrapolate")
    alpha_interp = interp1d(np.linspace(0, T_train, 1000), alpha_grid, fill_value="extrapolate")

    mu_diario = np.mean(mu_grid)
    alpha_diario = np.mean(alpha_grid)

    modelo = {
        "mu": mu_interp,
        "alpha": alpha_interp,
        "decay": decay,
        "T_train": T_train,
        "t0": t0
    }

    info = {
        "mu_diario": mu_diario,
        "alpha_diario": alpha_diario,
        "decay": decay
    }

    return modelo, info

# --- 3. Simulación Ogata ---
def simulate_hawkes(mu_fn, alpha_fn, decay, t_ini, t_fin, max_jumps=10000):
    np.random.seed(42)
    t = t_ini
    events = []

    while t < t_fin and len(events) < max_jumps:
        excitation = np.sum([alpha_fn(s) * decay * np.exp(-decay * (t - s)) for s in events if s < t])
        lambda_t = mu_fn(t) + excitation
        if lambda_t <= 0:
            break
        u = np.random.uniform()
        w = -np.log(u) / lambda_t
        t_candidate = t + w
        if t_candidate > t_fin:
            break
        excitation_cand = np.sum([alpha_fn(s) * decay * np.exp(-decay * (t_candidate - s)) for s in events if s < t_candidate])
        lambda_cand = mu_fn(t_candidate) + excitation_cand
        if np.random.uniform() <= lambda_cand / lambda_t:
            events.append(t_candidate)
        t = t_candidate

    return np.array(events)

# --- 4. Simular eventos ---
def simular_eventos(modelo, fecha_ini, fecha_fin, mu_boost=1.0):
    t0 = modelo["t0"]
    T_train = modelo["T_train"]
    decay = modelo["decay"]

    T_ini = (fecha_ini - t0).total_seconds() / (3600*24)
    T_fin = (fecha_fin - t0).total_seconds() / (3600*24)

    def seasonal_fns(mu_fn, alpha_fn, t_train_end, t_ini_sim, t_fin_sim):
        days = 365
        n_years = 3
        grid = np.linspace(t_ini_sim, t_fin_sim, 1000)
        seasonal_mu = np.zeros_like(grid)
        seasonal_alpha = np.zeros_like(grid)

        for y in range(1, n_years+1):
            offset = y * days
            t_shifted = grid - offset
            valid = (t_shifted >= 0) & (t_shifted <= t_train_end)
            seasonal_mu[valid] += mu_fn(t_shifted[valid])
            seasonal_alpha[valid] += alpha_fn(t_shifted[valid])

        seasonal_mu /= n_years
        seasonal_alpha /= n_years

        mu_fn_new = interp1d(grid, mu_boost * seasonal_mu, fill_value="extrapolate")
        alpha_fn_new = interp1d(grid, seasonal_alpha, fill_value="extrapolate")
        return mu_fn_new, alpha_fn_new

    mu_fn_sim, alpha_fn_sim = seasonal_fns(modelo["mu"], modelo["alpha"], T_train, T_ini, T_fin)
    sim_days = simulate_hawkes(mu_fn_sim, alpha_fn_sim, decay, T_ini, T_fin)
    sim_dates = [t0 + pd.Timedelta(days=t) for t in sim_days]
    return sim_dates
