"""
Heston simulator: full-truncation Euler on (log S, v).

Full truncation (Lord, Koekkoek & van Dijk 2010): the variance is kept as a
signed state but every appearance of v in a coefficient is replaced by v^+.
This is the lowest-bias simple Euler scheme for Heston and is preferred over
the reflection scheme used in the NeurIPS version (appendix F.1).

Returns S paths, and the variance path SEPARATELY so it can be used for
diagnostics but is structurally unavailable to the constraint builder.
"""

from dataclasses import dataclass
import numpy as np

from config import Heston, SimConfig, P_PARAMS, q_params, SIM_P, SIM_Q


@dataclass
class Paths:
    """S: (n, H+1) prices. v: (n, H+1) variance. r, dt, H for downstream use."""
    S: np.ndarray
    v: np.ndarray
    r: float
    dt: float
    world: str          # "P", "Q", or a generator label such as "Ptheta"
    S0: float = None    # common starting price; filled from S[:, 0] and checked

    def __post_init__(self):
        # Every constraint, test function and martingale profile divides by a
        # single scalar S0. That is only correct if all paths share the same
        # starting value, so verify it here instead of silently reading path 0.
        col0 = np.asarray(self.S[:, 0], dtype=np.float64)
        s0 = float(col0[0]) if self.S0 is None else float(self.S0)
        if not np.allclose(col0, s0, rtol=0.0, atol=1e-9 * max(1.0, abs(s0))):
            raise ValueError("Paths.S0 is ambiguous: paths do not share a common "
                             f"starting price (min {col0.min():.6g}, max {col0.max():.6g})")
        self.S0 = s0

    @property
    def H(self) -> int:
        return self.S.shape[1] - 1

    @property
    def n(self) -> int:
        return self.S.shape[0]

    def returns(self) -> np.ndarray:
        """(n, H) log-returns -- the object a path DDPM would model."""
        return np.diff(np.log(self.S), axis=1)

    def without_variance(self) -> "Paths":
        """
        Strip the latent variance path. Constraint construction is run on this
        so that no latent v_s can leak into g(X) even by accident.
        """
        return Paths(S=self.S, v=None, r=self.r, dt=self.dt, world=self.world,
                     S0=self.S0)


def simulate(params: Heston, sim: SimConfig, world: str = "P",
             chunk: int = 25_000) -> Paths:
    """
    Full-truncation Euler:
        v_{s+1} = v_s + kappa (theta - v_s^+) dt + xi sqrt(v_s^+ dt) Z2
        log S_{s+1} = log S_s + (drift - v_s^+/2) dt + sqrt(v_s^+ dt) Z1
    with Corr(Z1, Z2) = rho.

    Chunked so peak memory stays modest for n = 1e5.
    """
    n, H, dt = sim.n_paths, sim.H, sim.dt
    rng = np.random.default_rng(sim.seed)
    sq_dt = np.sqrt(dt)

    S = np.empty((n, H + 1), dtype=np.float64)
    V = np.empty((n, H + 1), dtype=np.float64)

    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        m = hi - lo
        logS = np.full(m, np.log(params.S0))
        v = np.full(m, params.v0)
        S[lo:hi, 0] = params.S0
        V[lo:hi, 0] = params.v0

        for s in range(H):
            z1 = rng.standard_normal(m)
            z2 = params.rho * z1 + np.sqrt(1.0 - params.rho ** 2) * rng.standard_normal(m)
            vp = np.maximum(v, 0.0)                       # v^+ (full truncation)
            sqrt_vp_dt = np.sqrt(vp * dt)

            logS = logS + (params.drift - 0.5 * vp) * dt + sqrt_vp_dt * z1
            v = v + params.kappa * (params.theta - vp) * dt + params.xi * sqrt_vp_dt * z2

            S[lo:hi, s + 1] = np.exp(logS)
            V[lo:hi, s + 1] = v

    return Paths(S=S, v=V, r=params.r, dt=dt, world=world)


def simulate_P(sim: SimConfig = SIM_P,
               params: Heston = P_PARAMS) -> Paths:
    return simulate(params, sim, world="P")


def simulate_Q(sim: SimConfig = SIM_Q,
               params: Heston = P_PARAMS) -> Paths:
    return simulate(q_params(params), sim, world="Q")


# --------------------------------------------------------------------------

def martingale_profile(paths: Paths) -> np.ndarray:
    """E[e^{-r s dt} S_s] / S_0 - 1 for s = 0..H. Zero under any Q."""
    disc = np.exp(-paths.r * paths.dt * np.arange(paths.H + 1))
    return (paths.S * disc).mean(axis=0) / paths.S0 - 1.0


if __name__ == "__main__":
    from config import SimConfig
    small = SimConfig(n_paths=20_000, H=21, seed=1)
    for name, pp in (("P", P_PARAMS), ("Q", q_params())):
        pth = simulate(pp, small, world=name)
        rets = pth.returns()
        ann_vol = rets.std(ddof=1) / np.sqrt(pth.dt)
        print(f"{name}: E[S_H]={pth.S[:, -1].mean():8.4f}  "
              f"ann.vol={ann_vol*100:6.3f}%  "
              f"mart.dev(H)={martingale_profile(pth)[-1]:+.5f}  "
              f"min v={pth.v.min():+.2e}  frac v<0={np.mean(pth.v < 0):.2e}")
