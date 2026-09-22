"""
One-run diagnostic: is the elevated held-out martingale residual at C1-C3 an
under-specified test-function family, or a structural trade-off against the
vanilla constraints?

Design note
-----------
Promoting the tier-1 held-out functions into the calibration set drives THEIR
residuals to ~0 by construction -- they become equality constraints. So that
number alone discriminates nothing. The readout has to come from three places:

  1. a TIER-2 held-out set (families and steps used in neither the baseline
     calibration nor tier 1), which stays genuinely out-of-sample in both arms;
  2. whether the joint problem is still solvable -- feasibility screen,
     convergence, fitted error on BOTH blocks;
  3. whether vanilla generalisation degrades -- tier-1 held-out vanillas are
     kept OUT of calibration in both arms precisely so this stays measurable.

Reading:
  * tier-2 martingale residual falls to the Q noise floor, ESS usable, held-out
    vanilla error unchanged  ->  the family was under-specified.
  * tier-2 residual stays elevated, and/or ESS collapses, and/or held-out
    vanilla error degrades                    ->  the trade-off is structural.
"""

import argparse
import numpy as np

from config import (P_PARAMS, q_params, SimConfig, SIM_P,
                    CALIB_TESTFUNS, VANILLA_C3,
                    HELDOUT_TESTFUNS, HELDOUT_VANILLAS, EXOTICS)
from heston import simulate
from constraints import build, build_heldout
from projection import solve
import evaluation as ev


# Tier-2 held-out: (family, step) pairs appearing in NEITHER CALIB_TESTFUNS
# NOR HELDOUT_TESTFUNS, so they are out-of-sample in both arms of the check.
TIER2_TESTFUNS = (
    ("s_over_s0_cu",   (7, 13, 19)),
    ("running_max",    (4, 10, 16)),
    ("running_mean",   (4, 10, 16)),
    ("indicator_down", (7, 14, 20)),
    ("dprev",          (5, 11, 17)),
    ("rvall",          (6, 12, 18)),
)

# Tier-2 vanillas: strikes/maturities in neither C3 nor HELDOUT_VANILLAS.
TIER2_VANILLAS = (
    (21, 88.0), (21, 103.0), (21, 113.0),
    (18, 94.0), (18, 98.0), (18, 104.0),
)


def rmse(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def _assert_disjoint():
    calib = {(f, s) for f, st in CALIB_TESTFUNS for s in st}
    t1 = {(f, s) for f, st in HELDOUT_TESTFUNS for s in st}
    t2 = {(f, s) for f, st in TIER2_TESTFUNS for s in st}
    assert not (calib & t1), f"tier-1 leaks into calibration: {calib & t1}"
    assert not (calib & t2), f"tier-2 leaks into calibration: {calib & t2}"
    assert not (t1 & t2), f"tier-2 leaks into tier-1: {t1 & t2}"
    v3, v1, v2 = set(VANILLA_C3), set(HELDOUT_VANILLAS), set(TIER2_VANILLAS)
    assert not (v3 & v1) and not (v3 & v2) and not (v1 & v2), "vanilla overlap"
    print(f"disjointness verified: calib {len(calib)} | tier-1 {len(t1)} | "
          f"tier-2 {len(t2)} test functions; vanillas "
          f"{len(v3)}/{len(v1)}/{len(v2)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--maxiter", type=int, default=20_000)
    args = ap.parse_args()

    _assert_disjoint()

    p, q = P_PARAMS, q_params()
    simP = SimConfig(n_paths=args.paths, H=SIM_P.H, dt=SIM_P.dt, seed=args.seed)
    simQ = SimConfig(n_paths=args.paths, H=SIM_P.H, dt=SIM_P.dt,
                     seed=args.seed + 500_000)
    pathsP = simulate(p, simP, world="P").without_variance()
    pathsQ = simulate(q, simQ, world="Q").without_variance()

    # evaluation sets, built identically on P (for the weights) and Q (floor)
    t1P = build_heldout(pathsP, HELDOUT_TESTFUNS, HELDOUT_VANILLAS, q)
    t1Q = build_heldout(pathsQ, HELDOUT_TESTFUNS, HELDOUT_VANILLAS, q)
    t2P = build_heldout(pathsP, TIER2_TESTFUNS, TIER2_VANILLAS, q)
    t2Q = build_heldout(pathsQ, TIER2_TESTFUNS, TIER2_VANILLAS, q)

    floors = dict(
        t1m=rmse(ev.heldout_martingale_residuals(t1Q, None)[0]),
        t2m=rmse(ev.heldout_martingale_residuals(t2Q, None)[0]),
        t1v=rmse(ev.heldout_vanilla_errors(t1Q, None)[3]),
        t2v=rmse(ev.heldout_vanilla_errors(t2Q, None)[3]))
    baseP = dict(
        t1m=rmse(ev.heldout_martingale_residuals(t1P, None)[0]),
        t2m=rmse(ev.heldout_martingale_residuals(t2P, None)[0]),
        t1v=rmse(ev.heldout_vanilla_errors(t1P, None)[3]),
        t2v=rmse(ev.heldout_vanilla_errors(t2P, None)[3]))

    exQ = ev.price_exotics(pathsQ, None)
    exP = ev.price_exotics(pathsP, None)

    arms = {
        "C3": CALIB_TESTFUNS,
        "C3+HO": CALIB_TESTFUNS + HELDOUT_TESTFUNS,
    }

    print(f"\nN = {args.paths:,}  seed = {args.seed}   "
          f"vanilla calibration fixed at C3 ({len(VANILLA_C3)} options) in both arms")
    print("tier-1 vanillas are kept OUT of calibration in both arms, so vanilla")
    print("generalisation stays measurable.\n")

    res = {}
    for name, fam in arms.items():
        cs = build(pathsP, fam, VANILLA_C3, q)
        r = solve(cs, maxiter=args.maxiter)
        w = r.w
        fe = ev.fitted_errors_by_kind(r)
        res[name] = dict(
            r=r, cs=cs, fe=fe,
            t1m=rmse(ev.heldout_martingale_residuals(t1P, w)[0]),
            t2m=rmse(ev.heldout_martingale_residuals(t2P, w)[0]),
            t1v=rmse(ev.heldout_vanilla_errors(t1P, w)[3]),
            t2v=rmse(ev.heldout_vanilla_errors(t2P, w)[3]),
            ex=ev.price_exotics(pathsP, w))

    # ---------------- report ---------------------------------------------
    print("=" * 78)
    print("SOLVE HEALTH")
    print("=" * 78)
    h = (f"{'arm':7s} {'mMart':>6s} {'m':>4s} {'screen':>7s} {'conv':>6s} "
         f"{'iters':>6s} {'gradres':>10s} {'fitMart':>10s} {'fitVan':>10s} "
         f"{'ESS %':>7s} {'maxw':>7s} {'KL':>7s}")
    print(h); print("-" * len(h))
    for name in arms:
        R, cs, fe = res[name]["r"], res[name]["cs"], res[name]["fe"]
        print(f"{name:7s} {int(sum(cs.mask('mart'))):6d} {cs.m:4d} "
              f"{'PASS' if R.screen['all_ok'] else 'FAIL':>7s} "
              f"{str(R.converged):>6s} {R.n_iter:6d} {R.grad_resid:10.2e} "
              f"{fe['mart']:10.2e} {fe['vanilla']:10.2e} "
              f"{R.ess_frac*100:7.2f} {R.max_weight*cs.n:7.1f} {R.kl:7.4f}")

    print("\n" + "=" * 78)
    print("RESIDUALS  (raw units; tier-1 martingale is IN-SAMPLE for C3+HO)")
    print("=" * 78)
    rows = [("t1m", "tier-1 martingale RMSE"),
            ("t2m", "tier-2 martingale RMSE  <- the discriminator"),
            ("t1v", "tier-1 held-out vanilla RMSE"),
            ("t2v", "tier-2 held-out vanilla RMSE")]
    hh = (f"{'diagnostic':42s} {'Heston-P':>11s} " +
          " ".join(f"{a:>11s}" for a in arms) + f" {'Q floor':>11s}")
    print(hh); print("-" * len(hh))
    for k, lab in rows:
        line = f"{lab:42s} {baseP[k]:11.6f} "
        line += " ".join(f"{res[a][k]:11.6f}" for a in arms)
        line += f" {floors[k]:11.6f}"
        print(line)

    print("\n" + "=" * 78)
    print("EXOTICS")
    print("=" * 78)
    print(f"{'exotic':28s} {'Heston-P':>11s} " +
          " ".join(f"{a:>11s}" for a in arms) + f" {'Heston-Q':>11s}")
    print("-" * (28 + 12 * (len(arms) + 2)))
    for k in EXOTICS:
        line = f"{k:28s} {exP[k][0]:11.5f} "
        line += " ".join(f"{res[a]['ex'][k][0]:11.5f}" for a in arms)
        line += f" {exQ[k][0]:11.5f}"
        print(line)
        line = f"{'  (se)':28s} {exP[k][1]:11.5f} "
        line += " ".join(f"{res[a]['ex'][k][1]:11.5f}" for a in arms)
        line += f" {exQ[k][1]:11.5f}"
        print(line)

    # ---------------- verdict --------------------------------------------
    A, B = res["C3"], res["C3+HO"]
    print("\n" + "=" * 78)
    print("READOUT")
    print("=" * 78)
    print(f"tier-2 martingale: {A['t2m']:.6f} -> {B['t2m']:.6f} "
          f"(Q floor {floors['t2m']:.6f}, P {baseP['t2m']:.6f})")
    print(f"  ratio to Q floor: {A['t2m']/floors['t2m']:.2f}x -> "
          f"{B['t2m']/floors['t2m']:.2f}x")
    print(f"tier-1 held-out vanilla: {A['t1v']:.6f} -> {B['t1v']:.6f} "
          f"(Q floor {floors['t1v']:.6f})")
    print(f"ESS: {A['r'].ess_frac*100:.2f}% -> {B['r'].ess_frac*100:.2f}%")


if __name__ == "__main__":
    main()
