"""
Gaussian-mixture world with exact ground truth, for the dimension-scaling
experiment (DECISIONS.md section 25).

P = sum_k pi_k N(mu_k, I) in d dimensions. Constraints fix every coordinate's
mean, so m = d and g(x) = x.

Everything the pipeline needs is closed form. For one Gaussian,

    N(mu, I)(x) e^{beta.x} = e^{beta.mu + |beta|^2/2} N(mu + beta, I)(x),

so the tilted mixture is another Gaussian mixture with the SAME covariance,
means shifted by beta and weights reweighted by e^{beta.mu}:

    Q*  =  sum_k pi'_k N(mu_k + beta, I),   pi'_k  propto  pi_k e^{beta.mu_k}
    psi(beta) = log E_P[e^{beta.x}] = |beta|^2/2 + log sum_k pi_k e^{beta.mu_k}
    L*(x)     = exp(beta.x - psi(beta))
    E_Q*[x]   = beta + sum_k pi'_k mu_k
    KL(Q*||P) = beta . E_Q*[x] - psi(beta)

so Q*, psi and L* are exact and the recovered beta can be checked against the
beta that generated the targets.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm


# --------------------------------------------------------------------------
# the mixture
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class GM:
    mu: np.ndarray          # (K, d) component means
    logpi: np.ndarray       # (K,) log mixture weights, normalised

    @property
    def K(self) -> int: return self.mu.shape[0]

    @property
    def d(self) -> int: return self.mu.shape[1]

    @property
    def pi(self) -> np.ndarray: return np.exp(self.logpi)


def make_mixture(d: int, K: int = 4, seed: int = 0, spread: float = 1.5) -> GM:
    """Seeded means, identity covariance, equal weights. One mixture per d."""
    rng = np.random.default_rng(1000 + seed * 97 + d)
    mu = rng.normal(0.0, spread, size=(K, d))
    return GM(mu=mu, logpi=np.full(K, -np.log(K)))


def gm_sample(gm: GM, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = rng.choice(gm.K, size=n, p=gm.pi)
    return gm.mu[k] + rng.standard_normal((n, gm.d))


def gm_logpdf(gm: GM, x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    d2 = ((x[:, None, :] - gm.mu[None, :, :]) ** 2).sum(-1)       # (n, K)
    return logsumexp(gm.logpi[None, :] - 0.5 * d2, axis=1) - 0.5 * gm.d * np.log(2 * np.pi)


# --------------------------------------------------------------------------
# exact tilt
# --------------------------------------------------------------------------

def tilt(gm: GM, beta: np.ndarray) -> GM:
    """Q* = the exponentially tilted mixture, itself a Gaussian mixture."""
    beta = np.asarray(beta, float)
    lw = gm.logpi + gm.mu @ beta
    return GM(mu=gm.mu + beta[None, :], logpi=lw - logsumexp(lw))


def psi(gm: GM, beta: np.ndarray) -> float:
    """log E_P[e^{beta.x}], exact."""
    beta = np.asarray(beta, float)
    return float(0.5 * beta @ beta + logsumexp(gm.logpi + gm.mu @ beta))


def mean_qstar(gm: GM, beta: np.ndarray) -> np.ndarray:
    q = tilt(gm, beta)
    return q.pi @ q.mu


def kl_qstar(gm: GM, beta: np.ndarray) -> float:
    """KL(Q*||P) = beta.E_Q*[x] - psi(beta), exact."""
    beta = np.asarray(beta, float)
    return float(beta @ mean_qstar(gm, beta) - psi(gm, beta))


def log_Lstar(gm: GM, beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    return x @ np.asarray(beta, float) - psi(gm, beta)


def beta_for_kl(gm: GM, direction: np.ndarray, target_kl: float,
                lo: float = 1e-6, hi: float = 50.0, tol: float = 1e-10) -> np.ndarray:
    """Scale `direction` so KL(Q*||P) equals target_kl. KL is increasing in the scale."""
    u = np.asarray(direction, float)
    u = u / np.linalg.norm(u)
    f = lambda s: kl_qstar(gm, s * u) - target_kl
    while f(hi) < 0 and hi < 1e6:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0: hi = mid
        else: lo = mid
        if hi - lo < tol: break
    return 0.5 * (lo + hi) * u


# --------------------------------------------------------------------------
# exact held-out functionals on a mixture
# --------------------------------------------------------------------------

def proj_var(gm: GM, u: np.ndarray) -> float:
    """Var(u.x) under the mixture, u a unit vector (covariance is I)."""
    m = gm.mu @ u
    p = gm.pi
    return float(p @ (1.0 + m ** 2) - (p @ m) ** 2)


def tail_prob(gm: GM, u: np.ndarray, t: float) -> float:
    """P(u.x > t) under the mixture, exact: a mixture of univariate normals."""
    return float(gm.pi @ norm.sf(t - gm.mu @ u))


def heldout_spec(gm_q: GM, n_proj: int = 8, seed: int = 0) -> dict:
    """
    Random-projection variances and tail probabilities, with EXACT Q* values.

    Tail thresholds are set from Q*'s own projected distribution (its 95th
    percentile by bisection on the exact mixture CDF), so the functional is
    informative at every d rather than landing in an empty tail.
    """
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n_proj, gm_q.d))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    out = {"U": U, "var": [], "tail": [], "thresh": []}
    for u in U:
        out["var"].append(proj_var(gm_q, u))
        lo, hi = -50.0, 50.0
        for _ in range(200):                       # exact 95th percentile of u.x under Q*
            mid = 0.5 * (lo + hi)
            if tail_prob(gm_q, u, mid) > 0.05: lo = mid
            else: hi = mid
        t = 0.5 * (lo + hi)
        out["thresh"].append(t)
        out["tail"].append(tail_prob(gm_q, u, t))
    for k in ("var", "tail", "thresh"):
        out[k] = np.asarray(out[k])
    return out


def heldout_empirical(X: np.ndarray, spec: dict, w: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """Sample versions of the same functionals, optionally weighted. Returns (var, tail)."""
    P = X @ spec["U"].T                                            # (n, n_proj)
    if w is None:
        return P.var(axis=0), (P > spec["thresh"][None, :]).mean(axis=0)
    w = w / w.sum()
    m = w @ P
    v = w @ (P ** 2) - m ** 2
    return v, w @ (P > spec["thresh"][None, :]).astype(float)


# --------------------------------------------------------------------------
# the dual solve: recover beta from samples
# --------------------------------------------------------------------------

def solve_beta_samples(X: np.ndarray, c: np.ndarray, beta0: np.ndarray = None,
                       maxiter: int = 500):
    """
    min_beta  log mean exp(X beta) - beta.c     (g(x) = x, so the columns are X itself)

    Returns (beta, w, kl, ess). This is the same convex dual the path experiments
    solve, with the mixture playing the role of the learned prior.
    """
    X = np.asarray(X, float)
    n = X.shape[0]
    logn = np.log(n)
    # Standardise the columns before solving, exactly as the path dual does. Without
    # this an ill-scaled sample overflows exp() and L-BFGS returns its starting point,
    # which looks like a converged beta = 0 rather than a failure.
    m, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Xs = (X - m) / sd
    cs = (np.asarray(c, float) - m) / sd
    b0 = np.zeros(X.shape[1]) if beta0 is None else np.asarray(beta0, float) * sd

    def obj(b):
        z = Xs @ b
        lse = logsumexp(z) - logn
        w = np.exp(z - z.max()); w /= w.sum()
        return lse - b @ cs, Xs.T @ w - cs

    r = minimize(obj, b0, jac=True, method="L-BFGS-B",
                 options=dict(maxiter=maxiter, maxfun=2 * maxiter, ftol=1e-14, gtol=1e-10, maxcor=50))
    r.x = r.x / sd                       # back to the original coordinates
    z = X @ r.x
    lse = float(logsumexp(z) - logn)
    w = np.exp(z - z.max()); w /= w.sum()
    kl = float(w @ z - lse)
    ess = float(1.0 / np.sum(w ** 2) / n)
    return r.x, w, kl, ess


def resample(X: np.ndarray, w: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=n, replace=True, p=w / w.sum())
    return X[idx]


def sliced_w1(A: np.ndarray, B: np.ndarray, n_proj: int = 256, seed: int = 0) -> float:
    """Sliced 1-Wasserstein; equal sample sizes not required."""
    rng = np.random.default_rng(seed)
    d = A.shape[1]
    U = rng.standard_normal((n_proj, d))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    pa = np.sort(A @ U.T, axis=0)
    pb = np.sort(B @ U.T, axis=0)
    n = min(pa.shape[0], pb.shape[0])
    qa = pa[np.linspace(0, pa.shape[0] - 1, n).astype(int)]
    qb = pb[np.linspace(0, pb.shape[0] - 1, n).astype(int)]
    return float(np.abs(qa - qb).mean())
