"""
Minimum-relative-entropy projection by convex duality.

Primal:   min_Q  KL(Q || P_N)   s.t.  E_Q[g_j(X)] = c_j,   Q << P_N
          where P_N is the empirical measure on the N simulated paths.

Dual:     min_beta  psi(beta) = log( (1/N) sum_i e^{beta . g(X_i)} ) - beta . c
          w_i propto exp(beta . g(X_i)),   grad psi = E_w[g] - c.

Column standardization
----------------------
Increment constraints are O(0.01) and option payoffs are O(1-10); solved raw,
L-BFGS crawls. We solve in standardized coordinates

    gtilde_j = (g_j - m_j) / s_j,     ctilde_j = (c_j - m_j) / s_j

with m_j, s_j the empirical mean and sd of column j under P. The location
shift m_j cancels exactly (it enters psi as a constant), so this is an exact
reparameterization of the same problem: beta_raw_j = betatilde_j / s_j. We
report BOTH ||beta|| in standardized units (the optimizer's own scale) and the
implied raw-unit beta.
"""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from constraints import ConstraintSet, feasibility_screen


@dataclass
class ProjectionResult:
    beta_std: np.ndarray        # dual variable in standardized coordinates
    beta_raw: np.ndarray        # implied dual variable in raw units
    logw: np.ndarray            # unnormalized log weights
    w: np.ndarray               # weights normalized to sum to 1
    ess: float                  # (sum w)^2 / sum w^2, in path units
    ess_frac: float             # ess / N
    max_weight: float           # max normalized weight
    grad_resid: float           # ||E_w[gtilde] - ctilde||_inf (standardized)
    fitted_err_raw: np.ndarray  # E_w[g] - c, raw units
    fitted_err_std: np.ndarray  # standardized units
    kl: float                   # KL(Q* || P_N) = beta.c - psi_logmgf
    logw_sd: float              # sd of beta.g(X_i) -- scale-free tilt magnitude
    converged: bool
    n_iter: int
    message: str
    screen: dict
    names: list
    kinds: list
    dropped: list               # degenerate columns removed before solving


def _standardize(cs: ConstraintSet, sd_floor: float = 1e-12):
    m = cs.G.mean(axis=0)
    s = cs.G.std(axis=0)
    keep = s > sd_floor
    dropped = [cs.names[j] for j in np.where(~keep)[0]]
    Gs = (cs.G[:, keep] - m[keep]) / s[keep]
    cs_std = (cs.c[keep] - m[keep]) / s[keep]
    return Gs, cs_std, m[keep], s[keep], keep, dropped


def normalized_weights(logw: np.ndarray):
    lw = logw - logw.max()
    w = np.exp(lw)
    w /= w.sum()
    return w


def ess_of(w: np.ndarray) -> float:
    """(sum w)^2 / sum w^2. For w normalized to sum 1 this is 1/sum w^2."""
    return float(w.sum() ** 2 / np.sum(w ** 2))


def solve(cs: ConstraintSet,
          ridge: float = 0.0,
          maxiter: int = 2000,
          tol: float = 1e-12,
          verbose: bool = False) -> ProjectionResult:
    """
    Solve the convex dual with L-BFGS-B.

    `ridge` adds (ridge/2)||beta||^2 in standardized coordinates. It is 0 by
    default; a small value can be used when the constraint columns are close
    to collinear and beta is only weakly identified (the *weights* are still
    unique even then, so this changes the reported beta, not the projection).
    """
    N = cs.n
    Gs, c_std, mu, sd, keep, dropped = _standardize(cs)
    m = Gs.shape[1]

    screen = feasibility_screen(cs)

    if m == 0:
        w = np.full(N, 1.0 / N)
        return ProjectionResult(
            beta_std=np.zeros(0), beta_raw=np.zeros(0),
            logw=np.zeros(N), w=w, ess=float(N), ess_frac=1.0,
            max_weight=1.0 / N, grad_resid=0.0,
            fitted_err_raw=np.zeros(0), fitted_err_std=np.zeros(0),
            kl=0.0, logw_sd=0.0, converged=True, n_iter=0, message="no constraints",
            screen=screen, names=[], kinds=[], dropped=dropped)

    logN = np.log(N)

    def obj(beta):
        z = Gs @ beta
        lse = logsumexp(z) - logN                 # log (1/N) sum exp
        f = lse - beta @ c_std
        w = np.exp(z - z.max())
        w /= w.sum()
        grad = Gs.T @ w - c_std
        if ridge:
            f += 0.5 * ridge * beta @ beta
            grad = grad + ridge * beta
        return f, grad

    res = minimize(obj, np.zeros(m), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=maxiter, maxfun=maxiter * 2,
                                ftol=tol, gtol=1e-12, maxcor=50))

    beta = res.x
    z = Gs @ beta
    w = normalized_weights(z)
    ess = ess_of(w)

    err_std = Gs.T @ w - c_std
    err_raw = err_std * sd                        # undo scaling (shift cancels)

    lse = logsumexp(z) - logN
    kl = float(beta @ c_std - lse)                # KL(Q* || P_N) at optimum

    kinds = [k for k, kp in zip(cs.kinds, keep) if kp]
    names = [n for n, kp in zip(cs.names, keep) if kp]

    return ProjectionResult(
        beta_std=beta, beta_raw=beta / sd,
        logw=z, w=w, ess=ess, ess_frac=ess / N,
        max_weight=float(w.max()),
        grad_resid=float(np.max(np.abs(err_std))),
        fitted_err_raw=err_raw, fitted_err_std=err_std,
        kl=kl, logw_sd=float(np.std(z)),
        converged=bool(res.success), n_iter=int(res.nit),
        message=str(res.message), screen=screen,
        names=names, kinds=kinds, dropped=dropped)


# --------------------------------------------------------------------------
# weighted expectation helpers
# --------------------------------------------------------------------------

def wmean(x: np.ndarray, w: np.ndarray = None) -> float:
    if w is None:
        return float(np.mean(x))
    return float(np.dot(w, x) / w.sum())


def wmean_se(x: np.ndarray, w: np.ndarray = None) -> float:
    """
    Standard error of the self-normalized importance-sampling estimator:
        se^2 = sum_i wbar_i^2 (x_i - xbar)^2,   wbar normalized to sum 1.
    Reduces to the usual s/sqrt(N) when weights are uniform.
    """
    if w is None:
        n = len(x)
        return float(np.std(x, ddof=1) / np.sqrt(n))
    wb = w / w.sum()
    xb = np.dot(wb, x)
    return float(np.sqrt(np.sum(wb ** 2 * (x - xb) ** 2)))


def condition_number(cs: ConstraintSet) -> float:
    """Condition number of the standardized constraint correlation matrix."""
    Gs, _, _, _, keep, _ = _standardize(cs)
    if Gs.shape[1] == 0:
        return 1.0
    C = (Gs.T @ Gs) / Gs.shape[0]
    ev = np.linalg.eigvalsh(C)
    ev = np.clip(ev, 1e-18, None)
    return float(ev[-1] / ev[0])


if __name__ == "__main__":
    from config import P_PARAMS, SimConfig, CALIB_TESTFUNS, VANILLA_C2
    from heston import simulate
    from constraints import build

    sim = SimConfig(n_paths=50_000, H=21, seed=3)
    pth = simulate(P_PARAMS, sim, world="P").without_variance()
    cs = build(pth, CALIB_TESTFUNS, VANILLA_C2)

    print(f"m={cs.m}  cond(corr)={condition_number(cs):.3e}")
    r = solve(cs)
    print(f"converged={r.converged} iters={r.n_iter}  msg={r.message}")
    print(f"||beta_std||={np.linalg.norm(r.beta_std):.4f}  "
          f"grad_resid={r.grad_resid:.3e}")
    print(f"ESS={r.ess:,.0f} ({r.ess_frac*100:.2f}%)  "
          f"max_w={r.max_weight:.3e} ({r.max_weight*cs.n:.1f}x uniform)  "
          f"KL={r.kl:.4f}")
    print(f"max |fitted err| raw = {np.abs(r.fitted_err_raw).max():.3e}")
