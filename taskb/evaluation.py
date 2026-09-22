"""
Evaluation: fitted constraint error, held-out residuals, exotic pricing.

All exotic payoffs are functions of the price path only. Every weighted
estimate carries a self-normalized importance-sampling standard error, so
"stable enough to measure" is a checkable statement rather than an impression.
"""

from dataclasses import dataclass
import numpy as np

from heston import Paths
from constraints import ConstraintSet
from projection import wmean, wmean_se
from charfn import carr_madan_call, implied_vol
from config import EXOTICS, Heston


# --------------------------------------------------------------------------
# Exotic payoffs (discounted, path-only)
# --------------------------------------------------------------------------

def exotic_payoff(paths: Paths, kind: str, **kw) -> np.ndarray:
    S, r, dt, H = paths.S, paths.r, paths.dt, paths.H
    T = H * dt
    disc = np.exp(-r * T)

    if kind == "asian":                       # arithmetic average of S_1..S_H
        A = S[:, 1:].mean(axis=1)
        return disc * np.maximum(A - kw["K"], 0.0)

    if kind == "up_and_out":                  # daily-monitored knock-out call
        alive = S[:, 1:].max(axis=1) < kw["B"]
        return disc * np.maximum(S[:, -1] - kw["K"], 0.0) * alive

    if kind == "lookback_float":              # floating-strike lookback call
        return disc * (S[:, -1] - S.min(axis=1))

    raise ValueError(f"unknown exotic '{kind}'")


def price_exotics(paths: Paths, w: np.ndarray = None) -> dict:
    out = {}
    for name, spec in EXOTICS.items():
        pay = exotic_payoff(paths, **spec)
        out[name] = (wmean(pay, w), wmean_se(pay, w))
    return out


# --------------------------------------------------------------------------
# Constraint / held-out diagnostics
# --------------------------------------------------------------------------

@dataclass
class LevelReport:
    level: str
    m: int
    m_mart: int
    m_van: int
    screen_ok: bool
    screen_failed: list
    ess: float
    ess_frac: float
    max_weight_ratio: float
    beta_norm_std: float
    beta_norm_raw: float
    grad_resid: float
    kl: float
    converged: bool
    n_iter: int
    fit_mart_rmse: float
    fit_van_rmse: float
    ho_mart_rmse: float
    ho_mart_rmse_norm: float
    ho_van_rmse: float
    ho_van_mae: float
    ho_van_max: float
    mart_profile_max: float
    exotics: dict
    ho_van_detail: list


def heldout_martingale_residuals(ho: ConstraintSet, w: np.ndarray = None):
    """
    E_w[psi_j(H_s) dStilde_{s+1}] for held-out psi. Zero under any EMM.
    Returns (raw residuals, residuals normalized by the P-sample column sd).
    """
    mask = ho.mask("mart")
    G = ho.G[:, mask]
    sd = G.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    res = np.array([wmean(G[:, j], w) for j in range(G.shape[1])])
    return res, res / sd


def heldout_vanilla_errors(ho: ConstraintSet, w: np.ndarray = None):
    """Weighted price minus Carr-Madan target for each held-out (T, K)."""
    mask = ho.mask("vanilla")
    G, c = ho.G[:, mask], ho.c[mask]
    names = [n for n, k in zip(ho.names, ho.kinds) if k == "vanilla"]
    px = np.array([wmean(G[:, j], w) for j in range(G.shape[1])])
    se = np.array([wmean_se(G[:, j], w) for j in range(G.shape[1])])
    return names, px, c, px - c, se


def martingale_profile(paths: Paths, w: np.ndarray = None) -> np.ndarray:
    """E_w[e^{-r s dt} S_s]/S_0 - 1 for s = 0..H."""
    disc = np.exp(-paths.r * paths.dt * np.arange(paths.H + 1))
    Stil = paths.S * disc
    return np.array([wmean(Stil[:, s], w) for s in range(paths.H + 1)]) / paths.S0 - 1.0


def fitted_errors_by_kind(result) -> dict:
    err = result.fitted_err_raw
    kinds = np.array(result.kinds)
    out = {}
    for k in ("mart", "vanilla"):
        sel = kinds == k
        out[k] = float(np.sqrt(np.mean(err[sel] ** 2))) if sel.any() else 0.0
    return out


# --------------------------------------------------------------------------

def smile(paths: Paths, strikes, step: int, w: np.ndarray = None):
    """Weighted call prices and implied vols at a maturity step (diagnostic)."""
    T = step * paths.dt
    disc = np.exp(-paths.r * T)
    px, iv = [], []
    for K in strikes:
        p = wmean(disc * np.maximum(paths.S[:, step] - K, 0.0), w)
        px.append(p)
        iv.append(implied_vol(p, K, paths.S0, paths.r, T))
    return np.array(px), np.array(iv)


def benchmark_smile(q: Heston, strikes, T: float):
    px = np.array([carr_madan_call(float(K), q, T) for K in strikes])
    iv = np.array([implied_vol(px[i], float(strikes[i]), q.S0, q.r, T)
                   for i in range(len(strikes))])
    return px, iv
