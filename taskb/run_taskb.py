"""
Task B driver: entropy projection on raw Heston paths, C0 -> C3.

Adds two things a bare sweep does not give you:

  * Monte Carlo NOISE FLOORS. Every held-out diagnostic is also evaluated on
    the independent Heston-Q sample, where the true value is 0 (martingale
    restrictions) or the Carr-Madan price (vanillas). A Q* residual sitting at
    the Q-sample floor is as good as any estimator can be at this N; without
    the floor you cannot separate a real bias from sampling noise.

  * STABILITY across seeds. "Exotic prices stable enough to measure" is a
    claim about run-to-run spread, so we re-solve on independent samples and
    compare the spread to the within-run standard error.

Usage:
    python run_taskb.py [--paths 100000] [--seed 20260920] [--seeds 1] [--quick]
"""

import argparse
import time
import numpy as np

from config import (P_PARAMS, q_params, premium_kernel, mean_variance,
                    SimConfig, SIM_P,
                    CONSTRAINT_LEVELS, HELDOUT_TESTFUNS, HELDOUT_VANILLAS,
                    EXOTICS)
from heston import simulate
from constraints import build, build_heldout
from projection import solve, condition_number
import evaluation as ev


LEVEL_DESC = {
    "C0": "martingale test functions only",
    "C1": "+ ATM vanilla (21d, K=100)",
    "C2": "+ 5 strikes at 21d",
    "C3": "+ 7 strikes at 21d and 5 at 10d",
}
LEVELS = list(CONSTRAINT_LEVELS)


def banner(s, ch="="):
    print("\n" + ch * 78)
    print(s)
    print(ch * 78)


def rmse(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


# --------------------------------------------------------------------------

def run_once(n_paths: int, seed: int, ridge: float = 0.0, verbose: bool = True,
             q=None):
    p = P_PARAMS
    q = q if q is not None else q_params()
    simP = SimConfig(n_paths=n_paths, H=SIM_P.H, dt=SIM_P.dt, seed=seed)
    simQ = SimConfig(n_paths=n_paths, H=SIM_P.H, dt=SIM_P.dt, seed=seed + 500_000)

    pathsP_full = simulate(p, simP, world="P")
    pathsQ_full = simulate(q, simQ, world="Q")
    pathsP = pathsP_full.without_variance()
    pathsQ = pathsQ_full.without_variance()

    hoP = build_heldout(pathsP, HELDOUT_TESTFUNS, HELDOUT_VANILLAS, q)
    hoQ = build_heldout(pathsQ, HELDOUT_TESTFUNS, HELDOUT_VANILLAS, q)

    out = dict(seed=seed, n=n_paths)

    resP, resnP = ev.heldout_martingale_residuals(hoP, None)
    resQ, resnQ = ev.heldout_martingale_residuals(hoQ, None)
    _, pxP, cHO, errP, _ = ev.heldout_vanilla_errors(hoP, None)
    hv_names, pxQ, _, errQ, _ = ev.heldout_vanilla_errors(hoQ, None)

    out["base"] = dict(
        ho_mart_P=rmse(resP), ho_mart_Q=rmse(resQ),
        ho_martn_P=rmse(resnP), ho_martn_Q=rmse(resnQ),
        ho_van_P=rmse(errP), ho_van_Q=rmse(errQ),
        mart_prof_P=float(np.abs(ev.martingale_profile(pathsP, None)).max()),
        mart_prof_Q=float(np.abs(ev.martingale_profile(pathsQ, None)).max()),
        hv_names=hv_names, hv_target=cHO, hv_px_P=pxP, hv_px_Q=pxQ)

    out["exP"] = ev.price_exotics(pathsP, None)
    out["exQ"] = ev.price_exotics(pathsQ, None)

    if verbose:
        a, b = premium_kernel(p, q)
        T = SIM_P.H * SIM_P.dt
        banner("SETUP")
        print(f"P : kappa={p.kappa:.3f} theta={p.theta:.4f} xi={p.xi} "
              f"rho={p.rho} mu={p.drift} v0={p.v0}   Feller={p.feller:+.4f}")
        print(f"Q : kappa={q.kappa:.3f} theta={q.theta:.4f} xi={q.xi} "
              f"rho={q.rho} r={q.drift}  v0={q.v0}   Feller={q.feller:+.4f}")
        print(f"    vol-risk-premium kernel gamma_perp = "
              f"(a/sqrt(v)+b sqrt(v))/sqrt(1-rho^2), a={a:+.4f}, b={b:+.4f}")
        print(f"    21d mean variance  P {mean_variance(p,T):.5f} "
              f"(vol {np.sqrt(mean_variance(p,T))*100:.2f}%)   "
              f"Q {mean_variance(q,T):.5f} "
              f"(vol {np.sqrt(mean_variance(q,T))*100:.2f}%)")
        print(f"N = {n_paths:,} paths, H = {SIM_P.H} daily steps, dt = 1/252, "
              f"seed = {seed}")
        print(f"realised ann. vol:  P "
              f"{pathsP.returns().std(ddof=1)/np.sqrt(pathsP.dt)*100:.3f}%   "
              f"Q {pathsQ.returns().std(ddof=1)/np.sqrt(pathsQ.dt)*100:.3f}%")
        print(f"P variance path (diagnostic only, never in g): min v = "
              f"{pathsP_full.v.min():.3e}, frac v<0 = "
              f"{np.mean(pathsP_full.v < 0):.2e}")
        print(f"\nheld-out set: {int(sum(hoP.mask('mart')))} test functions "
              f"+ {int(sum(hoP.mask('vanilla')))} vanillas, never calibrated on")

        banner("BASELINES AND MONTE CARLO NOISE FLOORS")
        print(f"{'diagnostic':42s} {'Heston-P':>12s} {'Heston-Q floor':>15s}")
        print(f"{'held-out martingale RMSE (raw)':42s} "
              f"{rmse(resP):12.6f} {rmse(resQ):15.6f}")
        print(f"{'held-out martingale RMSE (sd-normalised)':42s} "
              f"{rmse(resnP):12.6f} {rmse(resnQ):15.6f}")
        print(f"{'held-out vanilla RMSE':42s} "
              f"{rmse(errP):12.6f} {rmse(errQ):15.6f}")
        print(f"{'max |discounted mart. deviation|':42s} "
              f"{out['base']['mart_prof_P']:12.6f} "
              f"{out['base']['mart_prof_Q']:15.6f}")
        print("\nexotic benchmarks:")
        for k in EXOTICS:
            print(f"  {k:28s} P {out['exP'][k][0]:9.5f} +-{out['exP'][k][1]:.5f}"
                  f"    Q {out['exQ'][k][0]:9.5f} +-{out['exQ'][k][1]:.5f}")

    out["levels"] = {}
    for level in LEVELS:
        spec = CONSTRAINT_LEVELS[level]
        cs = build(pathsP, spec["testfuns"], spec["vanillas"], q)
        r = solve(cs, ridge=ridge)
        w = r.w

        res, resn = ev.heldout_martingale_residuals(hoP, w)
        _, hpx, _, herr, hse = ev.heldout_vanilla_errors(hoP, w)
        mp = ev.martingale_profile(pathsP, w)
        fe = ev.fitted_errors_by_kind(r)
        ex = ev.price_exotics(pathsP, w)

        L = dict(level=level, m=cs.m,
                 m_mart=int(sum(cs.mask("mart"))),
                 m_van=int(sum(cs.mask("vanilla"))),
                 cond=condition_number(cs),
                 screen_ok=r.screen["all_ok"], screen_failed=r.screen["failed"],
                 screen_margin=float(r.screen["margin"].min()),
                 ess=r.ess, ess_frac=r.ess_frac,
                 maxw_ratio=r.max_weight * cs.n,
                 beta_std=float(np.linalg.norm(r.beta_std)),
                 beta_raw=float(np.linalg.norm(r.beta_raw)),
                 logw_sd=r.logw_sd, kl=r.kl,
                 grad_resid=r.grad_resid,
                 converged=r.converged, n_iter=r.n_iter, msg=r.message,
                 fit_mart=fe["mart"], fit_van=fe["vanilla"],
                 ho_mart=rmse(res), ho_martn=rmse(resn),
                 ho_van=rmse(herr), ho_van_mae=float(np.mean(np.abs(herr))),
                 ho_van_max=float(np.abs(herr).max()),
                 hv_px=hpx, hv_err=herr, hv_se=hse,
                 mart_prof=float(np.abs(mp).max()),
                 exotics=ex)
        out["levels"][level] = L

        if verbose:
            banner(f"LEVEL {level}  --  {LEVEL_DESC[level]}", "-")
            print(f"constraints m = {cs.m} ({L['m_mart']} martingale + "
                  f"{L['m_van']} vanilla)   cond(corr) = {L['cond']:.2e}")
            print(f"feasibility screen (coordinate-wise; NOT a hull proof): "
                  f"{'PASS' if L['screen_ok'] else 'FAIL ' + str(L['screen_failed'])}"
                  f"   min margin {L['screen_margin']:.3f} sd")
            print(f"optimizer: converged={L['converged']} iters={L['n_iter']}")
            print(f"  ||beta|| std {L['beta_std']:8.3f}   raw {L['beta_raw']:8.3f}"
                  f"   sd(log w) {L['logw_sd']:.4f}   "
                  f"KL(Q*||P_N) {L['kl']:.5f} nats")
            print(f"  gradient residual (inf-norm, standardized) "
                  f"{L['grad_resid']:.3e}")
            print(f"\n  *** ESS = {L['ess']:,.0f} / {cs.n:,} = "
                  f"{L['ess_frac']*100:6.2f}%   "
                  f"max weight = {L['maxw_ratio']:.1f}x uniform ***")
            print(f"\nfitted constraint error (raw units): martingale RMSE "
                  f"{L['fit_mart']:.2e}   vanilla RMSE {L['fit_van']:.2e}")
            print(f"\nheld-out martingale ({len(res)} test functions):")
            print(f"  RMSE raw  {L['ho_mart']:.6f}   "
                  f"[P {rmse(resP):.6f}  ->  Q floor {rmse(resQ):.6f}]")
            print(f"  RMSE norm {L['ho_martn']:.6f}   "
                  f"[P {rmse(resnP):.6f}  ->  Q floor {rmse(resnQ):.6f}]")
            print(f"\nheld-out vanillas ({len(herr)}): RMSE {L['ho_van']:.6f}   "
                  f"[P {rmse(errP):.6f}  ->  Q floor {rmse(errQ):.6f}]")
            print(f"  {'option':22s} {'target':>9s} {'P':>9s} {'Q*':>9s} "
                  f"{'err':>9s} {'se':>8s}")
            for j, nmj in enumerate(hv_names):
                print(f"  {nmj:22s} {cHO[j]:9.5f} {pxP[j]:9.5f} {hpx[j]:9.5f} "
                      f"{herr[j]:+9.5f} {hse[j]:8.5f}")
            print(f"\nmax |discounted mart. deviation| {L['mart_prof']:.3e}  "
                  f"[P {out['base']['mart_prof_P']:.3e}]")
            print(f"\nexotics under Q* weights:")
            print(f"  {'':28s} {'Q*':>18s} {'Heston-Q MC':>18s} {'diff':>9s} "
                  f"{'diff/se':>8s}")
            for k in EXOTICS:
                pw, sw = ex[k]
                pq, sq = out["exQ"][k]
                d, se = pw - pq, np.hypot(sw, sq)
                print(f"  {k:28s} {pw:10.5f} +-{sw:6.5f} {pq:10.5f} +-{sq:6.5f} "
                      f"{d:+9.5f} {d/se:8.2f}")

    return out


# --------------------------------------------------------------------------

def print_summary(out):
    b = out["base"]
    banner("SUMMARY TABLE")
    hdr = (f"{'level':5s} {'m':>4s} {'scr':>4s} {'ESS %':>7s} {'ESS':>10s} "
           f"{'maxw':>7s} {'|beta|':>8s} {'sdlogw':>7s} {'KL':>7s} "
           f"{'gradres':>9s} {'fitMart':>9s} {'fitVan':>9s} "
           f"{'hoMart':>9s} {'hoVan':>8s}")
    print(hdr); print("-" * len(hdr))
    print(f"{'P':5s} {'-':>4s} {'-':>4s} {'100.00':>7s} {out['n']:>10,d} "
          f"{'1.0':>7s} {'0.00':>8s} {'0.000':>7s} {'0.0000':>7s} "
          f"{'-':>9s} {'-':>9s} {'-':>9s} "
          f"{b['ho_mart_P']:9.2e} {b['ho_van_P']:8.4f}")
    for lv in LEVELS:
        L = out["levels"][lv]
        print(f"{lv:5s} {L['m']:4d} {'ok' if L['screen_ok'] else 'FAIL':>4s} "
              f"{L['ess_frac']*100:7.2f} {L['ess']:10,.0f} "
              f"{L['maxw_ratio']:7.1f} {L['beta_std']:8.2f} {L['logw_sd']:7.3f} "
              f"{L['kl']:7.4f} {L['grad_resid']:9.2e} "
              f"{L['fit_mart']:9.2e} {L['fit_van']:9.2e} "
              f"{L['ho_mart']:9.2e} {L['ho_van']:8.4f}")
    print(f"{'Qflr':5s} {'-':>4s} {'-':>4s} {'-':>7s} {out['n']:>10,d} "
          f"{'-':>7s} {'-':>8s} {'-':>7s} {'-':>7s} {'-':>9s} "
          f"{'-':>9s} {'-':>9s} "
          f"{b['ho_mart_Q']:9.2e} {b['ho_van_Q']:8.4f}")
    print("\n'Qflr' = the same held-out diagnostics evaluated on an independent")
    print("Heston-Q sample of the same size: the Monte Carlo noise floor.")

    banner("EXOTIC PRICES ACROSS CONSTRAINT LEVELS")
    print(f"{'exotic':28s} {'Heston-P':>11s} " +
          " ".join(f"{lv:>11s}" for lv in LEVELS) + f" {'Heston-Q':>11s}")
    print("-" * (28 + 12 * (len(LEVELS) + 2)))
    for k in EXOTICS:
        row = f"{k:28s} {out['exP'][k][0]:11.5f} "
        row += " ".join(f"{out['levels'][lv]['exotics'][k][0]:11.5f}"
                        for lv in LEVELS)
        row += f" {out['exQ'][k][0]:11.5f}"
        print(row)
        row = f"{'  (se)':28s} {out['exP'][k][1]:11.5f} "
        row += " ".join(f"{out['levels'][lv]['exotics'][k][1]:11.5f}"
                        for lv in LEVELS)
        row += f" {out['exQ'][k][1]:11.5f}"
        print(row)


def print_multiseed(runs):
    banner(f"STABILITY ACROSS {len(runs)} INDEPENDENT SAMPLES")
    print("ESS % by level")
    print(f"  {'level':6s} " + " ".join(f"{'s'+str(i):>9s}"
                                        for i in range(len(runs))) +
          f" {'mean':>9s} {'sd':>8s}")
    for lv in LEVELS:
        v = np.array([r["levels"][lv]["ess_frac"] * 100 for r in runs])
        print(f"  {lv:6s} " + " ".join(f"{x:9.2f}" for x in v) +
              f" {v.mean():9.2f} {v.std(ddof=1):8.3f}")

    print("\nexotic prices: mean +- sd across samples, vs within-run se")
    for k in EXOTICS:
        print(f"\n  {k}")
        print(f"    {'source':10s} {'mean':>10s} {'sd across':>11s} "
              f"{'mean se':>10s} {'sd/se':>7s}")
        items = ([("Heston-P", lambda r: r["exP"][k])] +
                 [(lv, (lambda lv: lambda r: r["levels"][lv]["exotics"][k])(lv))
                  for lv in LEVELS] +
                 [("Heston-Q", lambda r: r["exQ"][k])])
        for lab, get in items:
            v = np.array([get(r)[0] for r in runs])
            s = np.array([get(r)[1] for r in runs])
            sd = v.std(ddof=1) if len(v) > 1 else 0.0
            print(f"    {lab:10s} {v.mean():10.5f} {sd:11.5f} "
                  f"{s.mean():10.5f} {sd/s.mean():7.2f}")

    print("\nheld-out diagnostics: mean across samples")
    print(f"  {'level':6s} {'hoMart raw':>12s} {'hoMart norm':>12s} "
          f"{'hoVan RMSE':>12s}")
    print(f"  {'P':6s} {np.mean([r['base']['ho_mart_P'] for r in runs]):12.6f} "
          f"{np.mean([r['base']['ho_martn_P'] for r in runs]):12.6f} "
          f"{np.mean([r['base']['ho_van_P'] for r in runs]):12.6f}")
    for lv in LEVELS:
        print(f"  {lv:6s} "
              f"{np.mean([r['levels'][lv]['ho_mart'] for r in runs]):12.6f} "
              f"{np.mean([r['levels'][lv]['ho_martn'] for r in runs]):12.6f} "
              f"{np.mean([r['levels'][lv]['ho_van'] for r in runs]):12.6f}")
    print(f"  {'Qflr':6s} {np.mean([r['base']['ho_mart_Q'] for r in runs]):12.6f} "
          f"{np.mean([r['base']['ho_martn_Q'] for r in runs]):12.6f} "
          f"{np.mean([r['base']['ho_van_Q'] for r in runs]):12.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--seeds", type=int, default=1, help="independent repeats")
    ap.add_argument("--ridge", type=float, default=0.0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    n = 20_000 if args.quick else args.paths
    t0 = time.time()
    runs = []
    for i in range(args.seeds):
        out = run_once(n, args.seed + 1000 * i, args.ridge, verbose=(i == 0))
        if i == 0:
            print_summary(out)
        else:
            print(f"\n[repeat {i}: seed {args.seed + 1000*i} done, "
                  f"{time.time()-t0:.0f}s]")
        runs.append(out)

    if args.seeds > 1:
        print_multiseed(runs)

    print(f"\ntotal wall time: {time.time()-t0:.1f}s")
    return runs


if __name__ == "__main__":
    main()
