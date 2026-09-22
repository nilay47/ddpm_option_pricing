"""
How many martingale test functions should the constraint set carry?

The identification sweep (C0 -> C3) varies the amount of *vanilla* information
at a fixed martingale family. This script varies the *martingale* family at a
fixed vanilla level, which is the other axis and the one that decides whether
a weak held-out martingale result is a method failure or an under-specified
constraint set.

Per 03_EXECUTION_PLAN.md Task D: "add constraints until ESS degrades, report
the curve, and let that be the empirical answer to 'why not enforce more?'"

CRITICAL: every enriched family deliberately skips the (family, step) pairs
that appear in HELDOUT_TESTFUNS, so the held-out set is byte-identical across
all variants and the comparison is honest.
"""

import argparse
import numpy as np

from config import (P_PARAMS, q_params, SimConfig, SIM_P,
                    CALIB_TESTFUNS, VANILLA_C3, HELDOUT_TESTFUNS,
                    HELDOUT_VANILLAS, EXOTICS)
from heston import simulate
from constraints import build, build_heldout
from projection import solve, condition_number
import evaluation as ev


HELD = {(f, s) for f, steps in HELDOUT_TESTFUNS for s in steps}


def _steps(family, candidate):
    """Drop any step that is held out for this family."""
    return tuple(s for s in candidate if (family, s) not in HELD)


ALL = tuple(range(0, 21))

M1 = CALIB_TESTFUNS                                     # 41 -- the sweep default

M2 = (
    ("one",          ALL),
    ("s_over_s0",    _steps("s_over_s0", range(1, 21))),
    ("s_over_s0_sq", tuple(range(1, 21))),
    ("rv5",          tuple(range(5, 21))),
)

M3 = M2 + (
    ("dprev", tuple(range(1, 21))),
    ("rvall", tuple(range(2, 21))),
)

VARIANTS = {"M1": M1, "M2": M2, "M3": M3}
DESC = {
    "M1": "baseline: 1 at every step, S/S0 and (S/S0)^2 at 7 steps, RV5 at 6",
    "M2": "all four families at every admissible step",
    "M3": "M2 + previous discounted increment + full-history realised variance",
}


def rmse(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--maxiter", type=int, default=2000)
    args = ap.parse_args()

    p, q = P_PARAMS, q_params()
    simP = SimConfig(n_paths=args.paths, H=SIM_P.H, dt=SIM_P.dt, seed=args.seed)
    simQ = SimConfig(n_paths=args.paths, H=SIM_P.H, dt=SIM_P.dt,
                     seed=args.seed + 500_000)

    pathsP = simulate(p, simP, world="P").without_variance()
    pathsQ = simulate(q, simQ, world="Q").without_variance()

    hoP = build_heldout(pathsP, HELDOUT_TESTFUNS, HELDOUT_VANILLAS, q)
    hoQ = build_heldout(pathsQ, HELDOUT_TESTFUNS, HELDOUT_VANILLAS, q)

    resP, resnP = ev.heldout_martingale_residuals(hoP, None)
    resQ, resnQ = ev.heldout_martingale_residuals(hoQ, None)
    _, _, _, errP, _ = ev.heldout_vanilla_errors(hoP, None)
    _, _, _, errQ, _ = ev.heldout_vanilla_errors(hoQ, None)
    exQ = ev.price_exotics(pathsQ, None)
    exP = ev.price_exotics(pathsP, None)

    print(f"N = {args.paths:,}  seed = {args.seed}   vanilla level fixed at C3 "
          f"({len(VANILLA_C3)} options)")
    print(f"held-out set: {int(sum(hoP.mask('mart')))} test functions "
          f"+ {int(sum(hoP.mask('vanilla')))} vanillas, identical across variants\n")

    hdr = (f"{'var':4s} {'mMart':>6s} {'m':>4s} {'ESS %':>7s} {'maxw':>7s} "
           f"{'sdlogw':>7s} {'KL':>7s} {'fitMart':>9s} {'fitVan':>9s} "
           f"{'hoMart':>9s} {'hoMartN':>9s} {'hoVan':>8s} {'iters':>6s}")
    print(hdr); print("-" * len(hdr))
    print(f"{'P':4s} {'-':>6s} {'-':>4s} {'100.00':>7s} {'1.0':>7s} "
          f"{'0.000':>7s} {'0.0000':>7s} {'-':>9s} {'-':>9s} "
          f"{rmse(resP):9.2e} {rmse(resnP):9.2e} {rmse(errP):8.4f} {'-':>6s}")

    rows = {}
    for name, fam in VARIANTS.items():
        cs = build(pathsP, fam, VANILLA_C3, q)
        r = solve(cs, maxiter=args.maxiter)
        w = r.w
        res, resn = ev.heldout_martingale_residuals(hoP, w)
        _, _, _, herr, _ = ev.heldout_vanilla_errors(hoP, w)
        fe = ev.fitted_errors_by_kind(r)
        ex = ev.price_exotics(pathsP, w)
        nm = int(sum(cs.mask("mart")))
        rows[name] = dict(r=r, ex=ex, nm=nm, m=cs.m,
                          ho=rmse(res), hon=rmse(resn), hv=rmse(herr))
        print(f"{name:4s} {nm:6d} {cs.m:4d} {r.ess_frac*100:7.2f} "
              f"{r.max_weight*cs.n:7.1f} {r.logw_sd:7.3f} {r.kl:7.4f} "
              f"{fe['mart']:9.2e} {fe['vanilla']:9.2e} "
              f"{rmse(res):9.2e} {rmse(resn):9.2e} {rmse(herr):8.4f} "
              f"{r.n_iter:6d}")

    print(f"{'Qflr':4s} {'-':>6s} {'-':>4s} {'-':>7s} {'-':>7s} {'-':>7s} "
          f"{'-':>7s} {'-':>9s} {'-':>9s} "
          f"{rmse(resQ):9.2e} {rmse(resnQ):9.2e} {rmse(errQ):8.4f} {'-':>6s}")

    print("\nfor each variant, the martingale family used:")
    for name, fam in VARIANTS.items():
        tot = sum(len(s) for _, s in fam)
        print(f"  {name}: {DESC[name]}")
        print(f"      " + ", ".join(f"{f}x{len(s)}" for f, s in fam) +
              f"  = {tot}")

    print("\nexotic prices by martingale-family richness (vanillas fixed at C3)")
    print(f"{'exotic':28s} {'Heston-P':>11s} " +
          " ".join(f"{v:>11s}" for v in VARIANTS) + f" {'Heston-Q':>11s}")
    print("-" * (28 + 12 * (len(VARIANTS) + 2)))
    for k in EXOTICS:
        row = f"{k:28s} {exP[k][0]:11.5f} "
        row += " ".join(f"{rows[v]['ex'][k][0]:11.5f}" for v in VARIANTS)
        row += f" {exQ[k][0]:11.5f}"
        print(row)
        row = f"{'  (se)':28s} {exP[k][1]:11.5f} "
        row += " ".join(f"{rows[v]['ex'][k][1]:11.5f}" for v in VARIANTS)
        row += f" {exQ[k][1]:11.5f}"
        print(row)


if __name__ == "__main__":
    main()
