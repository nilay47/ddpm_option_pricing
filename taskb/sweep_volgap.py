"""
Headline sweep repeated across P->Q volatility gaps.

The go/no-go must not depend on a conveniently large measure change. This runs
the full C0 -> C3 identification sweep at several sizes of the variance risk
premium and reports ESS, held-out vanilla repricing, and exotic movement at
each.

Gaps
----
  lam_v      the single-parameter Heston convention (kappa_Q = kappa + xi*lam_v
             with kappa*theta preserved). At H = 21 days this can only move
             realised vol by a fraction of a point in either direction, because
             v0 is shared and mean reversion has almost no time to act. This is
             a constraint of the restricted kernel family, NOT an estimate of
             what markets show.
  +1.0 pt    a small but economically real short-dated premium.
  +2.6 pt    the headline setting. For 1-month index options an implied-minus-
             realised gap of 2-3 vol points is the ordinary empirical
             magnitude, so this is the realistic case, not an inflated one.
  +4.0 pt    a stressed premium, to find where ESS starts to bite.
"""

import argparse
import numpy as np

from config import (P_PARAMS, q_params_lam, q_from_target_vol, mean_variance,
                    EXOTICS)
from run_taskb import run_once, LEVELS


def rmse(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=20260920)
    args = ap.parse_args()

    T = 21 / 252.0
    p = P_PARAMS
    volP = np.sqrt(mean_variance(p, T))

    gaps = [
        ("lam_v=-1", q_params_lam(p, -1.0)),
        ("21.0%",    q_from_target_vol(target_vol=0.2100)),
        ("22.57%",   q_from_target_vol(target_vol=0.2257)),
        ("24.0%",    q_from_target_vol(target_vol=0.2400)),
    ]

    print(f"P 21-day mean vol = {volP*100:.2f}%   N = {args.paths:,}   "
          f"seed = {args.seed}\n")
    print(f"{'label':10s} {'kappa_Q':>8s} {'theta_Q':>8s} {'Q vol':>8s} "
          f"{'gap (pts)':>10s} {'Feller_Q':>9s}")
    print("-" * 58)
    for lab, q in gaps:
        vq = np.sqrt(mean_variance(q, T))
        print(f"{lab:10s} {q.kappa:8.3f} {q.theta:8.5f} {vq*100:7.2f}% "
              f"{(vq-volP)*100:+10.2f} {q.feller:+9.4f}")

    runs = {}
    for lab, q in gaps:
        runs[lab] = run_once(args.paths, args.seed, verbose=False, q=q)

    # ---- ESS -------------------------------------------------------------
    print("\n" + "=" * 78)
    print("ESS (% of N) BY CONSTRAINT LEVEL AND VOL GAP")
    print("=" * 78)
    print(f"{'gap':10s} " + " ".join(f"{lv:>9s}" for lv in LEVELS))
    print("-" * (10 + 10 * len(LEVELS)))
    for lab, _ in gaps:
        print(f"{lab:10s} " + " ".join(
            f"{runs[lab]['levels'][lv]['ess_frac']*100:9.2f}" for lv in LEVELS))

    print(f"\n{'gap':10s} " + " ".join(f"{lv:>9s}" for lv in LEVELS) +
          "     <- KL(Q*||P_N), nats")
    for lab, _ in gaps:
        print(f"{lab:10s} " + " ".join(
            f"{runs[lab]['levels'][lv]['kl']:9.4f}" for lv in LEVELS))

    print(f"\n{'gap':10s} " + " ".join(f"{lv:>9s}" for lv in LEVELS) +
          "     <- max weight / uniform")
    for lab, _ in gaps:
        print(f"{lab:10s} " + " ".join(
            f"{runs[lab]['levels'][lv]['maxw_ratio']:9.1f}" for lv in LEVELS))

    # ---- constraint fit --------------------------------------------------
    print("\n" + "=" * 78)
    print("FITTED CONSTRAINT ERROR (raw price units, max over the two blocks)")
    print("=" * 78)
    print(f"{'gap':10s} " + " ".join(f"{lv:>9s}" for lv in LEVELS) +
          f" {'screen':>8s}")
    for lab, _ in gaps:
        L = runs[lab]["levels"]
        ok = all(L[lv]["screen_ok"] for lv in LEVELS)
        print(f"{lab:10s} " + " ".join(
            f"{max(L[lv]['fit_mart'], L[lv]['fit_van']):9.2e}" for lv in LEVELS) +
            f" {'all PASS' if ok else 'FAIL':>8s}")

    # ---- held-out vanillas ----------------------------------------------
    print("\n" + "=" * 78)
    print("HELD-OUT VANILLA RMSE  (9 options never calibrated on)")
    print("=" * 78)
    print(f"{'gap':10s} {'Heston-P':>10s} " +
          " ".join(f"{lv:>9s}" for lv in LEVELS) + f" {'Q floor':>9s} {'C3 gain':>9s}")
    print("-" * (21 + 10 * len(LEVELS) + 20))
    for lab, _ in gaps:
        R = runs[lab]
        base = R["base"]["ho_van_P"]
        c3 = R["levels"]["C3"]["ho_van"]
        print(f"{lab:10s} {base:10.5f} " + " ".join(
            f"{R['levels'][lv]['ho_van']:9.5f}" for lv in LEVELS) +
            f" {R['base']['ho_van_Q']:9.5f} {base/c3:8.1f}x")

    # ---- held-out martingale --------------------------------------------
    print("\n" + "=" * 78)
    print("HELD-OUT MARTINGALE RMSE (reported, not a gate)")
    print("=" * 78)
    print(f"{'gap':10s} {'Heston-P':>10s} " +
          " ".join(f"{lv:>9s}" for lv in LEVELS) + f" {'Q floor':>9s}")
    for lab, _ in gaps:
        R = runs[lab]
        print(f"{lab:10s} {R['base']['ho_mart_P']:10.5f} " + " ".join(
            f"{R['levels'][lv]['ho_mart']:9.5f}" for lv in LEVELS) +
            f" {R['base']['ho_mart_Q']:9.5f}")

    # ---- exotics ---------------------------------------------------------
    for k in EXOTICS:
        print("\n" + "=" * 78)
        print(f"EXOTIC: {k}")
        print("=" * 78)
        print(f"{'gap':10s} {'Heston-P':>10s} " +
              " ".join(f"{lv:>10s}" for lv in LEVELS) +
              f" {'Heston-Q':>10s} {'P->Q':>8s} {'P->C3':>8s} {'closed':>7s}")
        print("-" * (21 + 11 * len(LEVELS) + 30))
        for lab, _ in gaps:
            R = runs[lab]
            pP, pQ = R["exP"][k][0], R["exQ"][k][0]
            p3 = R["levels"]["C3"]["exotics"][k][0]
            span = pQ - pP
            moved = p3 - pP
            frac = moved / span if abs(span) > 1e-12 else np.nan
            print(f"{lab:10s} {pP:10.5f} " + " ".join(
                f"{R['levels'][lv]['exotics'][k][0]:10.5f}" for lv in LEVELS) +
                f" {pQ:10.5f} {span:+8.4f} {moved:+8.4f} "
                f"{frac*100:6.0f}%")
        print(f"{'(se, C3)':10s} {'':10s} " + " ".join(
            f"{runs[gaps[-1][0]]['levels'][lv]['exotics'][k][1]:10.5f}"
            for lv in LEVELS))


if __name__ == "__main__":
    main()
