"""
Task B configuration: Heston parameters (P and Q), simulation settings,
constraint-level definitions, and evaluation contracts.

Parameters follow 01_PAPER_STATE.md / the NeurIPS submission appendix F.1:
    kappa = 3.0, theta = 0.04, xi = 0.35, rho = -0.7, mu = 0.10, r = 0.05,
    S0 = 100, H = 21, dt = 1/252.
v0 is not stated in the submission; we take v0 = theta (stationary start).
"""

from dataclasses import dataclass, replace
import numpy as np


# --------------------------------------------------------------------------
# Heston parameters
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Heston:
    S0: float = 100.0
    v0: float = 0.04
    drift: float = 0.10      # mu under P, r under Q
    r: float = 0.05          # discount rate (always the risk-free rate)
    kappa: float = 3.0
    theta: float = 0.04
    xi: float = 0.35
    rho: float = -0.7

    @property
    def feller(self) -> float:
        """2*kappa*theta - xi^2; > 0 means Feller condition holds."""
        return 2.0 * self.kappa * self.theta - self.xi ** 2


P_PARAMS = Heston()


# Benchmark Q world: fixed, chosen volatility risk premium expressed directly
# as the risk-neutral mean-reversion pair. v0 is a STATE variable and is
# therefore identical under P and Q -- only kappa and theta move.
KAPPA_Q = 5.0
THETA_Q = 0.10


def q_params(p: Heston = P_PARAMS,
             kappa_q: float = KAPPA_Q,
             theta_q: float = THETA_Q) -> Heston:
    """
    Benchmark risk-neutral world with a FIXED, CHOSEN volatility risk premium.

    Girsanov with kernels on W^S and the orthogonal part W^perp of W^v:

        gamma_S    = (mu - r) / sqrt(v)                       (equity premium)
        gamma_perp = ( a / sqrt(v) + b sqrt(v) ) / sqrt(1-rho^2)   (vol premium)

    gives, under Q,

        dS = r S dt + sqrt(v) S dW^S_Q
        dv = [ kappa*theta - xi*rho*(mu-r) - xi*a - (kappa + xi*b) v ] dt
             + xi sqrt(v) dW^v_Q

    which is affine again, with kappa_Q = kappa + xi*b and
    kappa_Q*theta_Q = kappa*theta - xi*rho*(mu-r) - xi*a. The two free kernel
    coefficients (a, b) therefore place (kappa_Q, theta_Q) anywhere, so we
    specify the premium by its economically meaningful consequence -- the
    risk-neutral variance level -- rather than by a kernel constant.

    This is fix *(b)* from 01_PAPER_STATE.md Sec 6.3: the drift under P is
    constant, so gamma_S = (mu-r)/sqrt(v), and kappa/theta genuinely change
    under Q because rho != 0. Appendix A.3's "Girsanov shifts only the drift"
    claim is false here, as diagnosed.

    The single-parameter Heston convention (kappa_Q = kappa + xi*lam_v with
    kappa_Q*theta_Q held fixed) is available via `q_params_lam` but at H = 21
    days it moves realised vol by < 0.3 vol points from any lam_v, because v0
    dominates at short horizons -- too weak to be an informative go/no-go test.
    """
    if kappa_q <= 0 or theta_q <= 0:
        raise ValueError("kappa_Q and theta_Q must be positive")
    return replace(p, drift=p.r, kappa=kappa_q, theta=theta_q)


def q_params_lam(p: Heston = P_PARAMS, lam_v: float = -1.0) -> Heston:
    """
    Single-parameter convention:

        kappa_Q       = kappa + xi*lam_v
        kappa_Q*theta_Q = kappa*theta - xi*rho*(mu - r)

    NOTE: kappa*theta is NOT preserved. With rho != 0 the Girsanov shift applied
    to W^S propagates into W^v and the variance drift picks up -xi*rho*(mu-r);
    that term is exactly why kappa and theta move under Q at all (see `q_params`,
    which carries the general form with the kernel coefficient a). At lam_v = -1
    the numerator is 0.12 - (0.35)(-0.7)(0.05) = 0.13225, not 0.12, so
    theta_Q = 0.049906 and the 21-day mean vol is 20.25% (+0.25 points over P).
    Preserving kappa*theta would instead give theta_Q = 0.045283 and 20.14%.
    """
    kappa_q = p.kappa + p.xi * lam_v
    if kappa_q <= 0:
        raise ValueError(f"lam_v={lam_v} gives non-positive kappa_Q={kappa_q}")
    theta_q = (p.kappa * p.theta - p.xi * p.rho * (p.drift - p.r)) / kappa_q
    return replace(p, drift=p.r, kappa=kappa_q, theta=theta_q)


def q_from_target_vol(p: Heston = P_PARAMS, target_vol: float = 0.2257,
                      kappa_q: float = KAPPA_Q, H: int = 21,
                      dt: float = 1.0 / 252.0) -> Heston:
    """
    Benchmark Q pinned by the quantity that actually matters for pricing at
    this horizon: the risk-neutral mean variance over [0, H*dt], i.e. the level
    of the 21-day implied-vol surface. v0 is a state variable and is held
    equal to its P value; only (kappa, theta) move.

        (1/T) int_0^T E_Q[v_t] dt = theta_Q + (v0 - theta_Q) f,
        f = (1 - e^{-kappa_Q T}) / (kappa_Q T)
      => theta_Q = (target^2 - v0 f) / (1 - f)
    """
    T = H * dt
    kT = kappa_q * T
    f = (1.0 - np.exp(-kT)) / kT
    theta_q = (target_vol ** 2 - p.v0 * f) / (1.0 - f)
    if theta_q <= 0:
        raise ValueError(f"target_vol={target_vol} unreachable with "
                         f"kappa_Q={kappa_q} (theta_Q={theta_q:.5f})")
    return replace(p, drift=p.r, kappa=kappa_q, theta=theta_q)


def premium_kernel(p: Heston = P_PARAMS, q: Heston = None):
    """Girsanov coefficients (a, b) implied by a chosen (kappa_Q, theta_Q)."""
    q = q if q is not None else q_params(p)
    b = (q.kappa - p.kappa) / p.xi
    a = (p.kappa * p.theta - p.xi * p.rho * (p.drift - p.r)
         - q.kappa * q.theta) / p.xi
    return a, b


def mean_variance(h: Heston, T: float) -> float:
    """(1/T) * int_0^T E[v_t] dt  =  theta + (v0 - theta)(1 - e^{-kT})/(kT)."""
    kT = h.kappa * T
    return h.theta + (h.v0 - h.theta) * (1.0 - np.exp(-kT)) / kT


# --------------------------------------------------------------------------
# Simulation settings
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SimConfig:
    n_paths: int = 100_000
    H: int = 21
    dt: float = 1.0 / 252.0
    seed: int = 0


SIM_P = SimConfig(n_paths=100_000, H=21, seed=20260920)
SIM_Q = SimConfig(n_paths=100_000, H=21, seed=20260921)   # independent benchmark


# --------------------------------------------------------------------------
# Constraint design
#
# Everything below is expressed in STEPS (s = 0..H). Maturities for vanillas
# are step indices, so an option "at 21d" matures at S_21.
# --------------------------------------------------------------------------

# --- dynamic martingale test functions used for CALIBRATION -----------------
# (phi_name, tuple of steps s at which  E[ phi(H_s) * dStilde_{s+1} ] = 0 )
CALIB_TESTFUNS = (
    ("one",     tuple(range(0, 21))),                  # 21 constraints
    ("s_over_s0", (2, 5, 8, 11, 14, 17, 20)),           #  7
    ("s_over_s0_sq", (2, 5, 8, 11, 14, 17, 20)),        #  7
    ("rv5",     (5, 8, 11, 14, 17, 20)),                #  6
)                                                       # total 41

# --- dynamic martingale test functions HELD OUT (never calibrated on) -------
HELDOUT_TESTFUNS = (
    ("s_over_s0",   (4, 10, 16)),      # same family, unseen steps
    ("s_over_s0_cu", (4, 10, 16)),     # cubic  -- unseen family
    ("indicator_down", (4, 10, 16)),   # 1{S_s < S_0}
    ("running_max", (7, 14, 20)),      # (max_{u<=s} S_u)/S0
    ("running_mean", (7, 14, 20)),     # (mean_{u<=s} S_u)/S0
)                                      # total 15

# --- vanilla calls: (maturity step, strike) --------------------------------
VANILLA_C1 = ((21, 100.0),)
VANILLA_C2 = tuple((21, k) for k in (90.0, 95.0, 100.0, 105.0, 110.0))
VANILLA_C3 = (
    tuple((21, k) for k in (85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0))
    + tuple((10, k) for k in (92.0, 96.0, 100.0, 104.0, 108.0))
)

# Held-out vanillas: half-strikes at 21d (never calibrated) and a maturity
# (15d) that appears at no constraint level.
HELDOUT_VANILLAS = (
    tuple((21, k) for k in (87.5, 92.5, 97.5, 102.5, 107.5, 112.5))
    + tuple((15, k) for k in (95.0, 100.0, 105.0))
)

CONSTRAINT_LEVELS = {
    "C0": dict(testfuns=CALIB_TESTFUNS, vanillas=()),
    "C1": dict(testfuns=CALIB_TESTFUNS, vanillas=VANILLA_C1),
    "C2": dict(testfuns=CALIB_TESTFUNS, vanillas=VANILLA_C2),
    "C3": dict(testfuns=CALIB_TESTFUNS, vanillas=VANILLA_C3),
}


# --------------------------------------------------------------------------
# Exotics to price
# --------------------------------------------------------------------------

EXOTICS = {
    "asian_call_K100":        dict(kind="asian",    K=100.0),
    "uo_barrier_call_K100_B110": dict(kind="up_and_out", K=100.0, B=110.0),
    "lookback_float_call":    dict(kind="lookback_float"),
}


def sanity_print():
    p = P_PARAMS
    q = q_params(p)
    print(f"P: kappa={p.kappa:.4f} theta={p.theta:.5f} xi={p.xi} rho={p.rho} "
          f"drift={p.drift} v0={p.v0}  Feller={p.feller:+.4f}")
    print(f"Q: kappa={q.kappa:.4f} theta={q.theta:.5f} xi={q.xi} rho={q.rho} "
          f"drift={q.drift} v0={q.v0}  Feller={q.feller:+.4f}")
    a, b = premium_kernel(p, q)
    print(f"implied vol-premium kernel: a={a:+.4f}  b={b:+.4f}")
    T = SIM_P.H * SIM_P.dt
    for nm, h in (("P", p), ("Q", q)):
        avg = mean_variance(h, T)
        print(f"   {nm}: E[avg variance over {T*252:.0f}d] = {avg:.5f} "
              f"-> vol {np.sqrt(avg)*100:.2f}%")


if __name__ == "__main__":
    sanity_print()
