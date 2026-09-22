"""
Constraint builder.

g(X) is built from the PRICE/RETURN PATH ONLY. The latent variance path v_s is
never touched: `build` asserts that the Paths object handed to it has had its
variance stripped, so a leak is a hard error rather than a silent result.

Two families:

  (a) dynamic martingale restrictions
          E[ phi_k(H_s) * dStilde_{s+1} ] = 0,      dStilde_{s+1} = Stilde_{s+1} - Stilde_s
      where H_s is the history of the price path up to step s. These hold under
      ANY equivalent martingale measure -- they carry no model information
      beyond no-arbitrage, so their targets are exactly 0 with no calibration.

  (b) vanilla European calls
          E[ e^{-r T_j} (S_{T_j} - K_j)^+ ] = C_market(T_j, K_j)
      with targets from the Heston characteristic function (Carr-Madan).
"""

from dataclasses import dataclass, field
from typing import Sequence, Tuple
import numpy as np

from config import Heston, q_params
from charfn import carr_madan_call
from heston import Paths


# --------------------------------------------------------------------------
# Test functions phi(H_s) -- all functions of the observed price path only
# --------------------------------------------------------------------------

def _phi(name: str, S: np.ndarray, s: int, S0: float, dt: float,
         r: float = 0.0) -> np.ndarray:
    """
    Value of test function `name` on the history up to and including step s.
    S is (n, H+1). Returns (n,).
    """
    x = S[:, s] / S0

    if name == "one":
        return np.ones(S.shape[0])
    if name == "s_over_s0":
        return x
    if name == "s_over_s0_sq":
        return x ** 2
    if name == "s_over_s0_cu":
        return x ** 3
    if name == "indicator_down":
        return (S[:, s] < S0).astype(np.float64)
    if name == "running_max":
        return S[:, :s + 1].max(axis=1) / S0
    if name == "running_mean":
        return S[:, :s + 1].mean(axis=1) / S0
    if name == "dprev":                         # previous discounted increment
        if s < 1:
            raise ValueError("dprev needs s >= 1")
        d = np.exp(-r * dt * np.arange(s - 1, s + 1))
        return (S[:, s] * d[1] - S[:, s - 1] * d[0]) / S0 * 100.0
    if name == "rvall":                         # realised variance, full history
        if s < 2:
            raise ValueError("rvall needs s >= 2")
        lr = np.diff(np.log(S[:, :s + 1]), axis=1)
        return (lr ** 2).mean(axis=1) / dt
    if name.startswith("rv"):
        w = int(name[2:])                       # e.g. "rv5" -> 5-step window
        if s < w:
            raise ValueError(f"{name} needs s >= {w}, got s={s}")
        seg = S[:, s - w:s + 1]
        lr = np.diff(np.log(seg), axis=1)
        return (lr ** 2).mean(axis=1) / dt      # annualised realised variance
    raise ValueError(f"unknown test function '{name}'")


TESTFUN_LABELS = {
    "one": "1",
    "s_over_s0": "S/S0",
    "s_over_s0_sq": "(S/S0)^2",
    "s_over_s0_cu": "(S/S0)^3",
    "indicator_down": "1{S<S0}",
    "running_max": "max/S0",
    "running_mean": "mean/S0",
    "rv5": "RV5",
    "dprev": "dStil_prev",
    "rvall": "RVall",
}


# --------------------------------------------------------------------------

@dataclass
class ConstraintSet:
    G: np.ndarray            # (n, m) constraint functions evaluated on paths
    c: np.ndarray            # (m,) targets
    names: list = field(default_factory=list)
    kinds: list = field(default_factory=list)   # "mart" | "vanilla"

    @property
    def n(self): return self.G.shape[0]

    @property
    def m(self): return self.G.shape[1]

    def mask(self, kind: str) -> np.ndarray:
        return np.array([k == kind for k in self.kinds])


def discounted_increments(paths: Paths) -> np.ndarray:
    """dStilde_{s+1} for s = 0..H-1, shape (n, H)."""
    disc = np.exp(-paths.r * paths.dt * np.arange(paths.H + 1))
    Stil = paths.S * disc
    return np.diff(Stil, axis=1)


def build_martingale_columns(paths: Paths,
                             spec: Sequence[Tuple[str, Sequence[int]]]):
    """
    spec: ((phi_name, (steps...)), ...). Returns (cols, names).
    Column for (phi, s) is phi(H_s) * dStilde_{s+1}; target is 0.
    """
    assert paths.v is None, "constraint builder must not see the variance path"
    S, S0, dt = paths.S, paths.S0, paths.dt
    dS = discounted_increments(paths)

    cols, names = [], []
    for phi_name, steps in spec:
        for s in steps:
            if s >= paths.H:
                raise ValueError(f"step {s} has no increment (H={paths.H})")
            cols.append(_phi(phi_name, S, s, S0, dt, paths.r) * dS[:, s])
            names.append(f"mart[{TESTFUN_LABELS.get(phi_name, phi_name)}@{s}]")
    return cols, names


def build_vanilla_columns(paths: Paths,
                          spec: Sequence[Tuple[int, float]],
                          q: Heston):
    """
    spec: ((maturity_step, strike), ...). Column is the discounted call payoff;
    target is the Carr-Madan price under the benchmark Q parameters.
    """
    assert paths.v is None, "constraint builder must not see the variance path"
    cols, names, targets = [], [], []
    for (step, K) in spec:
        if step > paths.H:
            raise ValueError(f"maturity step {step} exceeds H={paths.H}")
        T = step * paths.dt
        disc = np.exp(-paths.r * T)
        cols.append(disc * np.maximum(paths.S[:, step] - K, 0.0))
        names.append(f"call[T={step}d,K={K:g}]")
        targets.append(carr_madan_call(K, q, T))
    return cols, names, np.array(targets)


def build(paths: Paths,
          testfuns: Sequence[Tuple[str, Sequence[int]]],
          vanillas: Sequence[Tuple[int, float]],
          q: Heston = None) -> ConstraintSet:
    """Assemble the full constraint set for one constraint level."""
    q = q if q is not None else q_params()
    mcols, mnames = build_martingale_columns(paths, testfuns)
    kinds = ["mart"] * len(mcols)
    targets = [0.0] * len(mcols)
    names = list(mnames)
    cols = list(mcols)

    if vanillas:
        vcols, vnames, vtar = build_vanilla_columns(paths, vanillas, q)
        cols += vcols
        names += vnames
        kinds += ["vanilla"] * len(vcols)
        targets += list(vtar)

    G = np.column_stack(cols) if cols else np.zeros((paths.n, 0))
    return ConstraintSet(G=G, c=np.array(targets, dtype=np.float64),
                         names=names, kinds=kinds)


# --------------------------------------------------------------------------
# Held-out evaluation functionals (never enter the projection)
# --------------------------------------------------------------------------

def build_heldout(paths: Paths, testfuns, vanillas, q: Heston = None) -> ConstraintSet:
    """
    Same machinery, but the result is used only for evaluation. Kept as a
    ConstraintSet so the same weighted-expectation code applies.
    """
    return build(paths, testfuns, vanillas, q)


def feasibility_screen(cs: ConstraintSet) -> dict:
    """
    Coordinate-wise screen: is c_j strictly inside [min_i g_j, max_i g_j]?

    NOTE: this is a SCREEN ONLY. Passing it does not establish that c lies in
    the joint convex hull of {g(X_i)}, and with tens of constraints it must not
    be reported as a feasibility proof.
    """
    lo = cs.G.min(axis=0)
    hi = cs.G.max(axis=0)
    ok = (cs.c > lo) & (cs.c < hi)
    # where in [0,1] the target sits inside the coordinate range
    span = np.where(hi > lo, hi - lo, 1.0)
    pos = (cs.c - lo) / span
    # margin in units of the empirical sd of that column
    sd = cs.G.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    margin = np.minimum(cs.c - lo, hi - cs.c) / sd
    return dict(ok=ok, all_ok=bool(ok.all()), lo=lo, hi=hi,
                pos=pos, margin=margin,
                failed=[cs.names[j] for j in np.where(~ok)[0]])


if __name__ == "__main__":
    from config import (P_PARAMS, SimConfig, CALIB_TESTFUNS, VANILLA_C3,
                        HELDOUT_TESTFUNS, HELDOUT_VANILLAS)
    from heston import simulate

    sim = SimConfig(n_paths=20_000, H=21, seed=7)
    pth = simulate(P_PARAMS, sim, world="P").without_variance()

    cs = build(pth, CALIB_TESTFUNS, VANILLA_C3)
    print(f"C3 constraint set: n={cs.n}, m={cs.m} "
          f"({sum(cs.mask('mart'))} martingale + {sum(cs.mask('vanilla'))} vanilla)")

    scr = feasibility_screen(cs)
    print("coordinate screen passes:", scr["all_ok"], "| failures:", scr["failed"])

    emp = cs.G.mean(axis=0)
    print("\nlargest |E_P[g] - c| gaps (raw units):")
    gap = np.abs(emp - cs.c)
    for j in np.argsort(-gap)[:8]:
        print(f"  {cs.names[j]:26s} E_P={emp[j]:+10.5f} c={cs.c[j]:+10.5f} "
              f"gap={gap[j]:.5f} sd={cs.G[:, j].std():.5f}")

    ho = build_heldout(pth, HELDOUT_TESTFUNS, HELDOUT_VANILLAS)
    print(f"\nheld-out set: m={ho.m} "
          f"({sum(ho.mask('mart'))} test functions + {sum(ho.mask('vanilla'))} vanillas)")
