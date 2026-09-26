"""
Bates world for the "what does the learned prior buy" experiment (DECISIONS.md section 23).

Bates = the existing Heston-P dynamics plus Merton log-normal jumps:

    d log S = (drift - lam*kbar - v/2) dt + sqrt(v) dW^S + dJ,
    dv      = kappa(theta - v) dt + xi sqrt(v) dW^v,   Corr(dW^S, dW^v) = rho,
    J       = sum of N_t ~ Poisson(lam t) jumps, each log-jump ~ N(mu_j, sig_j^2),
    kbar    = e^{mu_j + sig_j^2/2} - 1        (compensator, so E[dS/S] = drift dt)

Nothing here modifies the Heston path. `EXOTICS` in taskb/config.py is deliberately
NOT extended -- this module carries its own exotic dict so no existing table moves.
"""

from dataclasses import dataclass, replace, asdict
from typing import Dict, Sequence, Tuple

import numpy as np

import taskc  # noqa: F401  (sys.path setup for taskb)
from config import Heston, P_PARAMS, SimConfig, VANILLA_C3, CALIB_TESTFUNS  # taskb
from heston import Paths, simulate                                          # taskb
from charfn import heston_cf                                                # taskb
from constraints import build                                               # taskb
import evaluation as ev                                                     # taskb
from taskc.config import TaskCConfig
from taskc.data import PathStandardizer, apply_cap, TrainingSet


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Bates:
    S0: float = 100.0
    v0: float = 0.04
    drift: float = 0.10
    r: float = 0.05
    kappa: float = 3.0
    theta: float = 0.04
    xi: float = 0.35
    rho: float = -0.7
    lam: float = 5.0            # jump intensity, per year
    mu_j: float = -0.03         # mean log jump
    sig_j: float = 0.03         # sd of log jump

    @property
    def kbar(self) -> float:
        return float(np.exp(self.mu_j + 0.5 * self.sig_j ** 2) - 1.0)

    def heston_part(self, drift: float = None) -> Heston:
        """The diffusive part as a taskb Heston, with the drift optionally overridden."""
        return Heston(S0=self.S0, v0=self.v0,
                      drift=self.drift if drift is None else drift,
                      r=self.r, kappa=self.kappa, theta=self.theta, xi=self.xi, rho=self.rho)

    def variance_rate(self, T: float) -> float:
        """Total variance of log S per unit time: diffusive mean + jump contribution."""
        from config import mean_variance
        return mean_variance(self.heston_part(), T) + self.lam * (self.mu_j ** 2 + self.sig_j ** 2)


# DECISIONS.md section 23.1
BATES_P = Bates()
Q_A = replace(BATES_P, drift=BATES_P.r, lam=10.0)                                   # jump premium only
Q_B = replace(BATES_P, drift=BATES_P.r, lam=10.0, kappa=5.0, theta=0.10)            # jump + variance


def variance_matched_heston(b: Bates = BATES_P, T: float = 21 / 252.0) -> Heston:
    """Heston-P with no jumps whose mean variance rate equals the Bates total (section 23.3 arm 3)."""
    vr = b.variance_rate(T)
    return replace(b.heston_part(), v0=vr, theta=vr)


# DECISIONS.md section 23.2 -- local, so taskb's EXOTICS is untouched
BATES_EXOTICS: Dict[str, dict] = {
    "asian_call_K100":           dict(kind="asian", K=100.0),
    "uo_barrier_call_K100_B110": dict(kind="up_and_out", K=100.0, B=110.0),
    "lookback_float_call":       dict(kind="lookback_float"),
    "di_put_K100_B90":           dict(kind="down_and_in", K=100.0, B=90.0),
}


# --------------------------------------------------------------------------
# simulation
# --------------------------------------------------------------------------

def simulate_bates(b: Bates, sim: SimConfig, world: str = "P", chunk: int = 25_000,
                   return_jumps: bool = False):
    """
    Full-truncation Euler on (log S, v) with a compound-Poisson jump added per step.
    Mirrors taskb.heston.simulate exactly on the diffusive part, including the
    correlated-normal construction, so the jump term is the only difference.
    """
    n, H, dt = sim.n_paths, sim.H, sim.dt
    rng = np.random.default_rng(sim.seed)
    S = np.empty((n, H + 1), dtype=np.float64)
    V = np.empty((n, H + 1), dtype=np.float64)
    NJ = np.zeros((n, H), dtype=np.int16)          # jumps per step, for the z-cap diagnostic
    kbar = b.kbar

    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        m = hi - lo
        logS = np.full(m, np.log(b.S0))
        v = np.full(m, b.v0)
        S[lo:hi, 0] = b.S0
        V[lo:hi, 0] = b.v0

        for s in range(H):
            z1 = rng.standard_normal(m)
            z2 = b.rho * z1 + np.sqrt(1.0 - b.rho ** 2) * rng.standard_normal(m)
            vp = np.maximum(v, 0.0)
            sqrt_vp_dt = np.sqrt(vp * dt)

            # compound Poisson over the step: N jumps, sum of N normals
            N = rng.poisson(b.lam * dt, size=m)
            jump = np.where(N > 0,
                            N * b.mu_j + np.sqrt(np.maximum(N, 0)) * b.sig_j * rng.standard_normal(m),
                            0.0)

            logS = logS + (b.drift - b.lam * kbar - 0.5 * vp) * dt + sqrt_vp_dt * z1 + jump
            v = v + b.kappa * (b.theta - vp) * dt + b.xi * sqrt_vp_dt * z2

            S[lo:hi, s + 1] = np.exp(logS)
            V[lo:hi, s + 1] = v
            NJ[lo:hi, s] = N

    P = Paths(S=S, v=V, r=b.r, dt=dt, world=world)
    return (P, NJ) if return_jumps else P


# --------------------------------------------------------------------------
# characteristic function and vanilla pricing
# --------------------------------------------------------------------------

def bates_cf(u, b: Bates, T: float):
    """E[exp(i u log S_T)] under Bates: Heston CF at the compensated drift times the Merton jump CF."""
    u = np.asarray(u, dtype=np.complex128)
    h = b.heston_part(drift=b.drift - b.lam * b.kbar)
    jump = np.exp(b.lam * T * (np.exp(1j * u * b.mu_j - 0.5 * (u ** 2) * b.sig_j ** 2) - 1.0))
    return heston_cf(u, h, T) * jump


def carr_madan_call_bates(strike: float, b: Bates, T: float,
                          alpha: float = 1.5, u_max: float = 200.0, n_u: int = 8192) -> float:
    """Carr-Madan damped-integrand quadrature, same Simpson rule as taskb.charfn."""
    k = np.log(strike)
    n_u = n_u + (n_u % 2)
    u = np.linspace(0.0, u_max, n_u + 1)
    du = u[1] - u[0]
    denom = alpha ** 2 + alpha - u ** 2 + 1j * (2.0 * alpha + 1.0) * u
    psi = np.exp(-b.r * T) * bates_cf(u - (alpha + 1.0) * 1j, b, T) / denom
    integ = np.real(np.exp(-1j * u * k) * psi)
    w = np.ones(n_u + 1)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return float(np.exp(-alpha * k) / np.pi * (du / 3.0) * np.dot(w, integ))


def bates_vanilla_targets(b_q: Bates, vanillas: Sequence[Tuple[int, float]] = VANILLA_C3,
                          dt: float = 1 / 252.0) -> np.ndarray:
    return np.array([carr_madan_call_bates(K, b_q, step * dt) for step, K in vanillas])


def build_bates(paths: Paths, testfuns, vanillas, b_q: Bates, dt: float = 1 / 252.0):
    """
    taskb's `build`, with the vanilla targets replaced by Bates CF prices.

    The martingale columns and their zero targets are unchanged; only the vanilla
    targets move, because those are the only entries that depend on the Q model.
    """
    if paths.v is not None:
        paths = paths.without_variance()           # the builder refuses to see the variance path
    cs = build(paths, testfuns, vanillas)          # Heston targets, overwritten below
    if vanillas:
        tgt = bates_vanilla_targets(b_q, vanillas, dt)
        van = np.array([k == "vanilla" for k in cs.kinds])
        assert van.sum() == len(tgt), f"{van.sum()} vanilla columns vs {len(tgt)} targets"
        cs.c[van] = tgt
    return cs


# --------------------------------------------------------------------------
# training set / reference sample for the learned-prior arm
# --------------------------------------------------------------------------

def bates_training_set(cfg: TaskCConfig, b: Bates = BATES_P) -> TrainingSet:
    """Same construction as taskc.data.build_training_set, on Bates-P."""
    paths = simulate_bates(b, cfg.train_sim(), world="P")
    Y = paths.returns()
    std = PathStandardizer.fit(Y, S0=cfg.S0, r=cfg.r, dt=cfg.dt)
    z = std.to_z(Y)
    max_abs = float(np.abs(z).max())
    z_kept, n_rej = apply_cap(z, cfg.z_cap)
    return TrainingSet(z=z_kept.astype(np.float32), std=std, n_rejected=n_rej,
                       max_abs_z=max_abs, Y_raw=Y)


def bates_reference_paths(cfg: TaskCConfig, b: Bates = BATES_P) -> Paths:
    return simulate_bates(b, cfg.ref_sim(), world="P").without_variance()


# --------------------------------------------------------------------------
# jump checks G5a / G5b (DECISIONS.md section 23.4)
# --------------------------------------------------------------------------

def jump_stats(Y: np.ndarray) -> Dict[str, float]:
    """Tail fraction beyond 3.5 sd and excess kurtosis of pooled daily returns."""
    Y = np.asarray(Y, dtype=np.float64).ravel()
    m, sd = Y.mean(), Y.std()
    zz = (Y - m) / sd
    return dict(tail35=float((np.abs(zz) > 3.5).mean()),
                exkurt=float((zz ** 4).mean() - 3.0))


def jump_bands(ref_Y: np.ndarray, n_ref: int) -> Dict[str, Tuple[float, float]]:
    """Bands rebuilt from the reference sample, exactly as pre-registered."""
    s = jump_stats(ref_Y)
    t = s["tail35"]
    hw_t = max(0.25 * t, 3.0 * np.sqrt(max(t * (1.0 - t), 0.0) / max(n_ref, 1)))
    k = s["exkurt"]
    hw_k = max(0.25 * abs(k), 0.10)
    return dict(g5a_tail35=(t - hw_t, t + hw_t), g5b_exkurt=(k - hw_k, k + hw_k))


def check_jumps(zA: np.ndarray, std: PathStandardizer, ref: Paths, n_ref: int) -> dict:
    """G5a/G5b on a generated draw against the Bates reference. Returns a report dict."""
    YA = std.to_paths(zA, world="Ptheta_A").returns()
    bands = jump_bands(ref.returns(), n_ref)
    got = jump_stats(YA)
    out = {}
    for key, stat in (("g5a_tail35", "tail35"), ("g5b_exkurt", "exkurt")):
        lo, hi = bands[key]
        out[key] = dict(value=got[stat], lo=lo, hi=hi, passed=bool(lo <= got[stat] <= hi))
    out["passed"] = all(v["passed"] for v in out.values() if isinstance(v, dict))
    out["reference"] = jump_stats(ref.returns())
    return out


# --------------------------------------------------------------------------
# desk practice: Heston calibrated under Q to the vanilla quotes
# --------------------------------------------------------------------------

def calibrate_heston_q(quotes: np.ndarray, vanillas: Sequence[Tuple[int, float]],
                       r: float, S0: float, dt: float = 1 / 252.0,
                       x0: Sequence[float] = None, seed: int = 0) -> Tuple[Heston, dict]:
    """
    Least squares over (kappa, theta, xi, rho, v0) against the 12 vanilla quotes,
    using the same Carr-Madan pricer the targets were built with. Arm 4.
    """
    from scipy.optimize import least_squares
    from charfn import carr_madan_call

    lo = np.array([0.20, 0.005, 0.05, -0.95, 0.005])
    hi = np.array([30.0, 0.50, 3.00, 0.10, 0.50])
    starts = [np.array([3.0, 0.04, 0.35, -0.7, 0.04]),
              np.array([8.0, 0.06, 0.60, -0.5, 0.05]),
              np.array([1.0, 0.10, 1.00, -0.8, 0.03]),
              np.array([15.0, 0.05, 1.50, -0.3, 0.06]),
              np.array([5.0, 0.03, 0.25, -0.9, 0.05])]
    if x0 is not None:
        starts = [np.asarray(x0, float)] + starts
    starts = [np.clip(x, lo + 1e-9, hi - 1e-9) for x in starts]

    def unpack(x):
        return Heston(S0=S0, v0=x[4], drift=r, r=r, kappa=x[0], theta=x[1], xi=x[2], rho=x[3])

    def resid(x):
        h = unpack(x)
        px = np.array([carr_madan_call(K, h, step * dt) for step, K in vanillas])
        return px - quotes

    best = None
    for x in starts:                                   # multi-start: the Heston fit to a jumpy
        ri = least_squares(resid, x, bounds=(lo, hi),  # smile is not convex and does pin bounds
                           xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=4000)
        if best is None or np.sum(ri.fun ** 2) < np.sum(best.fun ** 2):
            best = ri                                  # NB: not `r` -- that is the rate, closed over
    res = best
    h = unpack(res.x)
    at_bound = {n: ("lo" if abs(res.x[i] - lo[i]) < 1e-6 else "hi" if abs(res.x[i] - hi[i]) < 1e-6 else "")
                for i, n in enumerate(("kappa", "theta", "xi", "rho", "v0"))}
    info = dict(rmse=float(np.sqrt(np.mean(res.fun ** 2))), max_abs=float(np.max(np.abs(res.fun))),
                nfev=int(res.nfev), success=bool(res.success), n_starts=len(starts),
                at_bound={k: v for k, v in at_bound.items() if v},
                params=dict(kappa=h.kappa, theta=h.theta, xi=h.xi, rho=h.rho, v0=h.v0))
    return h, info


# --------------------------------------------------------------------------
# truth
# --------------------------------------------------------------------------

def truth_exotics(b_q: Bates, n: int, seed: int, H: int = 21, dt: float = 1 / 252.0) -> dict:
    """Large-N Monte Carlo truth under a risk-neutral Bates law."""
    paths = simulate_bates(b_q, SimConfig(n_paths=n, H=H, dt=dt, seed=seed), world="Q")
    out = {}
    for k, sp in BATES_EXOTICS.items():
        pay = ev.exotic_payoff(paths, **sp)
        out[k] = dict(price=float(pay.mean()), se=float(pay.std(ddof=1) / np.sqrt(len(pay))))
    return out


def bates_thresholds(ref: Paths, dt: float = 1 / 252.0):
    """
    G4 bands rebuilt from the Bates reference sample.

    `taskc.gate.thresholds_for` keys off `cfg.heston == P_PARAMS`, and the Bates
    diffusive part IS P_PARAMS field-for-field, so that check would wrongly hand
    back the Heston bands. This applies the same construction directly.
    """
    from taskc.gate import GateThresholds, CV_NULL, sv_stats
    sv = sv_stats(ref.returns(), dt)
    ex = sv["cv_rv"] - CV_NULL
    g4a = tuple(sorted((CV_NULL + 0.75 * ex, CV_NULL + 1.25 * ex)))

    def band(v):
        hw = max(0.25 * abs(v), 0.005)
        return (v - hw, v + hw)

    return GateThresholds(g4a_cv=g4a, g4b_acf1=band(sv["acf1_sq"]), g4c_lev5=band(sv["lev5"]))


# --------------------------------------------------------------------------
# fitting Bates to paths (DECISIONS.md section 27, arm 3)
# --------------------------------------------------------------------------

def fit_bates_mle(rets: np.ndarray, dt: float = 1 / 252.0, block: int = 21,
                  fix_dynamics=None) -> "Bates":
    """
    Two-stage estimator on daily log-returns.

    Stage 1, the jump part, by MLE on the one-day mixture. Over a single day
    P(N = 0) = 1 - lam*dt and P(N = 1) ~ lam*dt, so

        r ~ (1 - lam dt) N(m, v dt)  +  (lam dt) N(m + mu_J, v dt + sig_J^2),

    a two-component Gaussian mixture whose weight is tied to lam. Fitting the
    mixture rather than thresholding avoids the truncation bias that makes a
    threshold estimator miss small jumps (understating lam) and keep only large
    ones (overstating sig_J).

    Stage 2, the diffusive part: kappa, xi, rho from `fit_heston_mom` applied to
    returns with detected jumps truncated, using the fitted jump law to set the
    threshold.
    """
    from scipy.optimize import minimize
    from taskc.aapl import fit_heston_mom   # the AAPL fitter, reused (section 27 arm 2/3)
    r = np.asarray(rets, dtype=np.float64).ravel()

    def nll(p):
        log_v, m, mu_j, log_sj, log_lam = p
        v, sj, lam = np.exp(log_v), np.exp(log_sj), np.exp(log_lam)
        w1 = np.clip(lam * dt, 1e-12, 0.5)
        s0, s1 = np.sqrt(v * dt), np.sqrt(v * dt + sj ** 2)
        a = np.log1p(-w1) - 0.5 * ((r - m) / s0) ** 2 - np.log(s0)
        b = np.log(w1) - 0.5 * ((r - m - mu_j) / s1) ** 2 - np.log(s1)
        return -float(np.sum(np.logaddexp(a, b)))

    x0 = np.array([np.log(max(r.var() / dt, 1e-6)), float(r.mean()), -0.03, np.log(0.04), np.log(5.0)])
    res = minimize(nll, x0, method="Nelder-Mead",
                   options=dict(maxiter=20000, maxfev=20000, xatol=1e-8, fatol=1e-8))
    log_v, m, mu_j, log_sj, log_lam = res.x
    v, sig_j, lam = float(np.exp(log_v)), float(np.exp(log_sj)), float(np.exp(log_lam))

    # stage 2: truncate at 4 sd of the diffusive part and fit the variance dynamics
    thr = 4.0 * np.sqrt(v * dt)
    R2 = np.asarray(rets, dtype=np.float64)
    if fix_dynamics is not None:
        # (kappa, xi, rho) are NOT identifiable from 21-day paths started at v0 = theta:
        # across kappa = 3, 15, 30 the 21-day return sd is 5.800 / 5.795 / 5.790 %, and the
        # measurement noise in a sub-block realized variance is ~3.6x the genuine variation
        # in v. Passing them in isolates the jump misspecification, which is what this
        # experiment is about -- and hands the parametric arms information the learned prior
        # has to find for itself. See DECISIONS.md section 27.2.
        kappa, xi, rho = fix_dynamics
        return Bates(S0=100.0, v0=v, drift=float(m / dt), r=0.05,
                     kappa=kappa, theta=v, xi=xi, rho=rho,
                     lam=lam, mu_j=float(mu_j), sig_j=sig_j)
    if R2.ndim == 2:                      # panel of independent paths: identify within-path
        R_diff = np.where(np.abs(R2 - m) > thr, 0.0, R2)
        kappa, xi, rho = fit_variance_dynamics_panel(R_diff, theta=v, dt=dt)
    else:                                 # one long series: the AAPL estimator is correct
        h = fit_heston_mom(np.where(np.abs(R2 - m) > thr, 0.0, R2), dt=dt, block=block)
        kappa, xi, rho = h.kappa, h.xi, h.rho
    return Bates(S0=100.0, v0=v, drift=float(m / dt), r=0.05,
                 kappa=kappa, theta=v, xi=xi, rho=rho,
                 lam=lam, mu_j=float(mu_j), sig_j=sig_j)


def fit_variance_dynamics_panel(R: np.ndarray, theta: float, dt: float = 1 / 252.0, sub: int = 7):
    """
    kappa, xi, rho from a PANEL of independent H-step paths.

    `taskc.aapl.fit_heston_mom` estimates mean reversion from the lag-1 correlation of
    consecutive realized-variance blocks, which is right for one long series and wrong
    here: consecutive paths are independent, so that correlation is zero and kappa runs
    to its clip. Identification has to come from WITHIN each path. Each path is cut into
    H//sub sub-blocks and the lag-1 correlation is pooled over paths, which identifies
    mean reversion at the sub-block scale.

    Returns (kappa, xi, rho).
    """
    R = np.asarray(R, dtype=np.float64)
    n, H = R.shape
    k = H // sub
    if k < 2:
        raise ValueError(f"need at least two sub-blocks per path: H={H}, sub={sub}")
    B = R[:, :k * sub].reshape(n, k, sub)
    RV = (B ** 2).sum(axis=2) / (sub * dt)                 # (n, k) annualised per sub-block
    a = RV[:, :-1].ravel(); b = RV[:, 1:].ravel()
    a1 = float(np.corrcoef(a, b)[0, 1])
    a1 = min(max(a1, 1e-3), 0.999)
    kappa = float(-np.log(a1) / (sub * dt))
    var_rv = max(RV.ravel().var(ddof=1) - 2.0 * theta ** 2 / sub, 1e-12)
    xi = float(np.sqrt(max(2.0 * kappa * var_rv / max(theta, 1e-12), 1e-12)))
    rho = float(np.corrcoef(B.sum(axis=2).ravel(), RV.ravel())[0, 1])
    return kappa, xi, min(max(rho, -0.95), -0.01)
