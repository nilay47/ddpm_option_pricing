"""
Step 3 of Task C (execution-plan "step 2"): re-solve beta* on P_theta draw A.

Everything finance-side is taskb, unchanged: constraints.build (same g, same
Carr-Madan targets c from the benchmark Q), projection.solve (same convex dual,
same L-BFGS, same column standardization), evaluation.* (same held-out sets,
exotics, self-normalized SEs). What this module adds is the sample discipline
of DECISIONS.md section 5:

    draw A  ->  beta*, psi_hat_A            (solve)
    draw B  ->  L*(x_i) targets for h_psi   (step 3)
    draw C  ->  every reported expectation  (ESS, held-out, exotics, SEs)

L*(x) = exp(beta_raw . (g(x) - m_A) - lse_A), with m_A the column means of draw
A and lse_A = log (1/N_A) sum_i exp(beta_raw . (g(x_i) - m_A)). The location
shift m_A is A's, not B's or C's, so L* is one fixed function; E_B[L*] and
E_C[L*] are diagnostics (should be ~1) and are never renormalized.
"""

from dataclasses import dataclass
import numpy as np

import taskc  # noqa: F401
from config import (q_params, CONSTRAINT_LEVELS, HELDOUT_TESTFUNS, HELDOUT_VANILLAS,  # taskb
                    EXOTICS)
from constraints import build, build_heldout, feasibility_screen     # taskb
from projection import solve, condition_number, normalized_weights, ess_of, wmean, wmean_se  # taskb
import evaluation as ev                                              # taskb
from heston import Paths                                             # taskb


@dataclass
class Tilt:
    """The projected measure as a function of the path: log L*(x)."""
    level: str
    names: list
    kinds: list
    beta_raw: np.ndarray       # (m,) on the kept columns
    m_A: np.ndarray            # (m,) column means of draw A (kept columns)
    keep: np.ndarray           # (m_all,) bool, degenerate columns dropped by the solver
    lse_A: float               # log-mean-exp on draw A
    c: np.ndarray              # (m,) targets

    def logL(self, G: np.ndarray) -> np.ndarray:
        """log L*(x) for constraint matrix G (n, m_all) built with the same spec."""
        return (G[:, self.keep] - self.m_A) @ self.beta_raw - self.lse_A

    def weights(self, G: np.ndarray) -> np.ndarray:
        """Self-normalized weights on another sample (sum to 1)."""
        return normalized_weights(self.logL(G))


def solve_level(level: str, pathsA: Paths, q=None):
    """Build the level's constraint set on draw A and solve. Returns (Tilt, ProjectionResult, ConstraintSet)."""
    q = q if q is not None else q_params()
    spec = CONSTRAINT_LEVELS[level]
    cs = build(pathsA, spec["testfuns"], spec["vanillas"], q)
    r = solve(cs)
    keep = np.array([n in set(r.names) for n in cs.names])
    m_A = cs.G[:, keep].mean(axis=0)
    z = (cs.G[:, keep] - m_A) @ r.beta_raw
    lse_A = float(np.log(np.mean(np.exp(z - z.max()))) + z.max())
    tilt = Tilt(level=level, names=r.names, kinds=r.kinds, beta_raw=r.beta_raw, m_A=m_A,
                keep=keep, lse_A=lse_A, c=cs.c[keep])
    return tilt, r, cs


def evaluate_on(tilt: Tilt, paths: Paths, q=None):
    """Weighted diagnostics of the tilt on an independent sample (draw C)."""
    q = q if q is not None else q_params()
    spec = CONSTRAINT_LEVELS[tilt.level]
    cs = build(paths, spec["testfuns"], spec["vanillas"], q)
    logL = tilt.logL(cs.G)
    L = np.exp(logL)
    w = normalized_weights(logL)
    fitted = np.array([wmean(cs.G[:, j], w) for j in range(cs.m)]) - cs.c
    ho = build_heldout(paths, HELDOUT_TESTFUNS, HELDOUT_VANILLAS, q)
    res, resn = ev.heldout_martingale_residuals(ho, w)
    hv_names, hpx, hc, herr, hse = ev.heldout_vanilla_errors(ho, w)
    mp = ev.martingale_profile(paths, w)
    ex = ev.price_exotics(paths, w)
    return dict(
        n=paths.n, E_L=float(L.mean()), L_max=float(L.max()),
        ess=ess_of(w), ess_frac=ess_of(w) / paths.n, max_weight_ratio=float(w.max() * paths.n),
        fit_mart_rmse=float(np.sqrt(np.mean(fitted[cs.mask("mart")] ** 2))) if cs.mask("mart").any() else 0.0,
        fit_van_rmse=float(np.sqrt(np.mean(fitted[cs.mask("vanilla")] ** 2))) if cs.mask("vanilla").any() else 0.0,
        fitted_err_raw=fitted, fit_names=cs.names,
        ho_mart_rmse=float(np.sqrt(np.mean(res ** 2))), ho_martn_rmse=float(np.sqrt(np.mean(resn ** 2))),
        ho_van_rmse=float(np.sqrt(np.mean(herr ** 2))), ho_van_max=float(np.abs(herr).max()),
        hv_names=hv_names, hv_px=hpx, hv_target=hc, hv_err=herr, hv_se=hse,
        mart_prof_max=float(np.abs(mp).max()), exotics=ex, w=w, logL=logL)


def baseline_on(paths: Paths, q=None):
    """Unweighted diagnostics of a sample (P_theta itself, or Heston-P/Q floors)."""
    q = q if q is not None else q_params()
    ho = build_heldout(paths, HELDOUT_TESTFUNS, HELDOUT_VANILLAS, q)
    res, resn = ev.heldout_martingale_residuals(ho, None)
    hv_names, hpx, hc, herr, hse = ev.heldout_vanilla_errors(ho, None)
    return dict(ho_mart_rmse=float(np.sqrt(np.mean(res ** 2))), ho_martn_rmse=float(np.sqrt(np.mean(resn ** 2))),
                ho_van_rmse=float(np.sqrt(np.mean(herr ** 2))), hv_names=hv_names, hv_px=hpx, hv_target=hc,
                hv_err=herr, mart_prof_max=float(np.abs(ev.martingale_profile(paths, None)).max()),
                exotics=ev.price_exotics(paths, None))


# Task B reference numbers (taskb/full_run.log, N=1e5, seed 20260920) for side-by-side.
TASKB = {
    "C0": dict(ess_frac=0.9941, beta_raw=5.815, screen_margin=4.075, ho_van=0.1515, ho_mart=8.51e-4, maxw=1.6, kl=0.00296),
    "C1": dict(ess_frac=0.9604, beta_raw=8.402, screen_margin=0.811, ho_van=0.0391, ho_mart=1.24e-2, maxw=4.3, kl=0.01845),
    "C2": dict(ess_frac=0.9392, beta_raw=39.228, screen_margin=0.288, ho_van=0.0031, ho_mart=1.32e-2, maxw=11.7, kl=0.02621),
    "C3": dict(ess_frac=0.9356, beta_raw=33.340, screen_margin=0.147, ho_van=0.0028, ho_mart=1.32e-2, maxw=14.6, kl=0.02651),
}
TASKB_EXOTICS_P = {"asian_call_K100": 1.61319, "uo_barrier_call_K100_B110": 2.05449, "lookback_float_call": 4.43604}
TASKB_EXOTICS_Q = {"asian_call_K100": 1.57207, "uo_barrier_call_K100_B110": 1.76547, "lookback_float_call": 4.62501}
TASKB_EXOTICS_C3 = None   # filled from the log by the notebook if wanted
