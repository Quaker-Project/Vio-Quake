# simulador.py

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import BSpline, interp1d
from numba import njit

class HawkesSimulator:
    def __init__(self, filepath, date_col="Fecha"):
        self.df = pd.read_excel(filepath)
        self.df[date_col] = pd.to_datetime(self.df[date_col])
        self.df = self.df.sort_values(by=date_col).reset_index(drop=True)

        self.t0 = self.df[date_col].min()
        self.times = (self.df[date_col] - self.t0).dt.total_seconds() / (3600 * 24)
        self.T = self.times.max()
        self.times_np = self.times.values

    def fit(self, K=8, degree=3, lambda_reg=0.01):
        knots = np.linspace(0, self.T, K)
        knots_ext = np.concatenate((np.repeat(knots[0], degree), knots, np.repeat(knots[-1], degree)))
        self.n_coef = len(knots) + degree - 1

        def spline_basis(t):
            t = np.atleast_1d(t)
            basis = np.zeros((len(t), self.n_coef))
            for i in range(self.n_coef):
                c = np.zeros(self.n_coef)
                c[i] = 1
                spline = BSpline(knots_ext, c, degree)
                basis[:, i] = spline(t)
            return basis

        self.spline_mat = spline_basis(self.times_np)
        self.mu_vals_from_coef = lambda c: self.spline_mat @ c
        self.alpha_vals_from_coef = lambda c: self.spline_mat @ c

        grid = np.linspace(0, self.T, 1000)
        spline_grid = spline_basis(grid)
        self.mu_integral_coeffs = np.trapz(spline_grid, grid, axis=0)

        @njit
        def compute_log_likelihood_numba(times, mu_vals, alpha_vals, decay):
            ll = 0.0
            for i in range(len(times)):
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
            c_mu = params[:self.n_coef]
            c_alpha = params[self.n_coef:2*self.n_coef]
            decay = params[-1]

            if np.any(c_mu < 0) or np.any(c_alpha < 0) or decay <= 0:
                return np.inf

            mu_vals = self.mu_vals_from_coef(c_mu)
            alpha_vals = self.alpha_vals_from_coef(c_alpha)
            ll = compute_log_likelihood_numba(self.times_np, mu_vals, alpha_vals, decay)
            mu_integral = self.mu_integral_coeffs @ c_mu
            hawkes_integral = (np.mean(alpha_vals) / decay) * len(self.times_np)
            penalty = lambda_reg * np.sum(c_alpha**2)

            return -(ll - (mu_integral + hawkes_integral)) + penalty

        initial_params = np.concatenate([
            np.full(self.n_coef, len(self.times_np)/self.T / self.n_coef),
            np.full(self.n_coef, 0.1),
            [2.0]
        ])

        bounds = [(0, None)]*(2*self.n_coef) + [(1e-3, 10)]
        res = minimize(log_likelihood, initial_params, bounds=bounds, method='L-BFGS-B', options={'maxiter':2000})
        self.c_mu, self.c_alpha, self.decay = res.x[:self.n_coef], res.x[self.n_coef:2*self.n_coef], res.x[-1]

        self.grid = grid
        self.mu_fn = interp1d(grid, spline_grid @ self.c_mu, kind='cubic', fill_value="extrapolate")
        self.alpha_fn = interp1d(grid, spline_grid @ self.c_alpha, kind='cubic', fill_value="extrapolate")

    def simulate(self, start_date, end_date, mu_boost=1.0, n_years=1, max_jumps=10000):
        T_ini = (pd.to_datetime(start_date) - self.t0).total_seconds() / (3600 * 24)
        T_fin = (pd.to_datetime(end_date) - self.t0).total_seconds() / (3600 * 24)
        t_sim_grid = np.linspace(T_ini, T_fin, 1000)
        seasonal_mu = np.zeros_like(t_sim_grid)
        seasonal_alpha = np.zeros_like(t_sim_grid)

        for y in range(1, n_years+1):
            offset = y * 365
            shifted = t_sim_grid - offset
            valid = (shifted >= 0) & (shifted <= self.T)
            seasonal_mu[valid] += self.mu_fn(shifted[valid])
            seasonal_alpha[valid] += self.alpha_fn(shifted[valid])

        seasonal_mu /= n_years
        seasonal_alpha /= n_years

        mu_fn_sim = interp1d(t_sim_grid, mu_boost * seasonal_mu, fill_value="extrapolate")
        alpha_fn_sim = interp1d(t_sim_grid, seasonal_alpha, fill_value="extrapolate")

        def ogata(mu_fn, alpha_fn, decay, t_start, t_end, max_jumps):
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

        return ogata(mu_fn_sim, alpha_fn_sim, self.decay, T_ini, T_fin, max_jumps)

    def summary(self):
        mu_avg = np.mean(self.mu_fn(self.grid))
        alpha_avg = np.mean(self.alpha_fn(self.grid))
        return {
            "mu_avg": mu_avg,
            "alpha_avg": alpha_avg,
            "decay": self.decay,
            "total_real": len(self.times_np),
            "daily_avg_real": len(self.times_np) / self.T
        }
