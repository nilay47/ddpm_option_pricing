"""
Sample-restricted MOT-style price bounds on an existing solve draw.

For each exotic payoff p, solve

    min / max   E_w[p]      over   w >= 0,  sum_i w_i = 1,  G^T w = c

where G and c are exactly the constraint columns and targets the dual was solved
with (C3: 41 martingale test functions + 12 vanillas = 53 equalities). Nothing is
retrained and no new paths are drawn: the atoms are the frozen prior's existing
1e6-path solve draw.

**These are INNER approximations of true MOT bounds.** The optimisation ranges
only over measures supported on the sampled atoms, so the feasible set is a
subset of the set of all measures satisfying the constraints; a true MOT bound
is at least as wide in both directions. Widening comes from two sources we do
not capture: paths the sample does not contain, and the fact that finitely many
unconditional moment conditions do not pin the conditional martingale property.

Method: column generation on the LP dual. The primal has 1e6 variables but only
54 equalities, so an optimal basic solution puts mass on at most 54 atoms. We
solve a restricted master on a working set, price out all atoms against the
duals, add the most violated one, and repeat to optimality (reduced cost >= -tol
everywhere). Exact at convergence, and it never forms a 1e6-row LP.

    ~/.venv/bin/python3 -m taskc.mot_bounds [--n 1000000] [--level C3]
"""

import argparse
import json
import os
import time

import numpy as np
from scipy.optimize import linprog

import taskc  # noqa: F401
from taskc.config import CFG, frozen
from taskc.ptheta import load_checkpoint, load_draw
from constraints import build                      # taskb
from config import q_params, CONSTRAINT_LEVELS, EXOTICS   # taskb
import evaluation as ev                            # taskb

HESTON_P = {"asian_call_K100": 1.61319, "uo_barrier_call_K100_B110": 2.05449,
            "lookback_float_call": 4.43604}
HESTON_Q = {"asian_call_K100": 1.57207, "uo_barrier_call_K100_B110": 1.76547,
            "lookback_float_call": 4.62501}


def solve_lp(G, c, p, sense, n_init=400, tol=1e-9, max_iter=2000, seed=0, verbose=True):
    """
    min (sense=+1) or max (sense=-1) of p·w subject to G^T w = c, 1·w = 1, w >= 0,
    by column generation. Returns (value, support, iterations, status).

    The working set starts small -- an optimal basic solution puts mass on at most
    m+1 atoms, so a large initial set only makes every master solve expensive. If
    the restricted master is infeasible we double the set until it is not.
    """
    n, m = G.shape
    obj = p if sense > 0 else -p
    rng = np.random.default_rng(seed)
    b = np.concatenate([c, [1.0]])
    k = min(n_init, n)
    while True:                                     # phase 0: a feasible working set
        S = np.unique(np.concatenate([rng.choice(n, size=k, replace=False),
                                      [int(np.argmin(p)), int(np.argmax(p))]]))
        A = np.vstack([G[S].T, np.ones((1, len(S)))])
        if linprog(obj[S], A_eq=A, b_eq=b, bounds=(0, None), method="highs").success:
            break
        k = min(k * 4, n)
        if k >= n:
            return None, None, 0, "no feasible working set"
    for it in range(max_iter):
        A = np.vstack([G[S].T, np.ones((1, len(S)))])
        r = linprog(obj[S], A_eq=A, b_eq=b, bounds=(0, None), method="highs")
        if not r.success:
            return None, None, it, f"restricted master {r.message}"
        y = r.eqlin.marginals                       # duals: 53 for G, 1 for the simplex
        red = obj - (G @ y[:m] + y[m])              # reduced cost of every atom
        j = int(np.argmin(red))
        if red[j] >= -tol * max(1.0, abs(r.fun)):
            w = np.zeros(n); w[S] = r.x
            val = float(p @ w)
            return val, S[r.x > 1e-12], it, "optimal"
        S = np.append(S, j)
        if verbose and it % 25 == 0:
            print(f"      iter {it:3d}  master {sense*r.fun:+.6f}  most-negative reduced cost {red[j]:.2e}  |S| {len(S)}", flush=True)
    return None, None, max_iter, "iteration limit"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="lrema"); ap.add_argument("--level", default="C3")
    ap.add_argument("--n", type=int, default=None, help="use a prefix of the solve draw")
    ap.add_argument("--out", default="artifacts_taskc/lrema/mot_bounds.json")
    args = ap.parse_args()
    t0 = time.time()

    RUN = frozen(artifact_dir=CFG.run_dir(args.tag))
    _, std, _, _ = load_checkpoint(os.path.join(RUN.artifact_dir, RUN.ckpt_name))
    S6 = load_draw(RUN, "S6")
    z = S6.z if args.n is None else S6.z[:args.n]
    paths = std.to_paths(z, "S6")
    q = q_params(); spec = CONSTRAINT_LEVELS[args.level]
    cs = build(paths, spec["testfuns"], spec["vanillas"], q)
    G = np.ascontiguousarray(cs.G, dtype=np.float64)
    c = cs.c.astype(np.float64)
    print(f"atoms {G.shape[0]:,}  constraints {G.shape[1]} ({args.level})  draw seed {S6.seed}  "
          f"({time.time()-t0:.0f}s to build)", flush=True)

    pay = {k: ev.exotic_payoff(paths, **spec_) for k, spec_ in EXOTICS.items()}
    # Q* on the same atoms, for the location of the projected price inside the interval
    from taskc.dual import solve_level
    tilt, r, _ = solve_level(args.level, paths, q)
    w_star = r.w
    out = {"level": args.level, "n_atoms": int(G.shape[0]), "m_constraints": int(G.shape[1]),
           "draw_seed": int(S6.seed), "tag": args.tag, "exotics": {}}

    print(f"\n{'exotic':28s} {'lower':>9s} {'upper':>9s} {'width':>8s} "
          f"{'Q* (frac)':>16s} {'Heston-Q (frac)':>18s} {'Heston-P (frac)':>18s}")
    print("-" * 112)
    for k in EXOTICS:
        p = np.ascontiguousarray(pay[k], dtype=np.float64)
        t1 = time.time()
        lo, s_lo, it_lo, st_lo = solve_lp(G, c, p, +1, verbose=False)
        hi, s_hi, it_hi, st_hi = solve_lp(G, c, p, -1, verbose=False)
        if lo is None or hi is None:
            print(f"  {k:28s} FAILED  ({st_lo} / {st_hi})"); continue
        qstar = float(w_star @ p)
        width = hi - lo
        frac = lambda v: (v - lo) / width if width > 0 else float("nan")
        out["exotics"][k] = dict(lower=lo, upper=hi, width=width,
                                 q_star=qstar, q_star_frac=frac(qstar),
                                 heston_q=HESTON_Q[k], heston_q_frac=frac(HESTON_Q[k]),
                                 heston_q_inside=bool(lo <= HESTON_Q[k] <= hi),
                                 heston_p=HESTON_P[k], heston_p_frac=frac(HESTON_P[k]),
                                 heston_p_inside=bool(lo <= HESTON_P[k] <= hi),
                                 support_lower=int(len(s_lo)), support_upper=int(len(s_hi)),
                                 iters=(it_lo, it_hi), seconds=time.time() - t1)
        e = out["exotics"][k]
        print(f"  {k:28s} {lo:9.4f} {hi:9.4f} {width:8.4f} "
              f"{qstar:9.4f} ({frac(qstar):5.2f}) {HESTON_Q[k]:9.4f} ({frac(HESTON_Q[k]):5.2f}){'' if e['heston_q_inside'] else ' OUT'} "
              f"{HESTON_P[k]:9.4f} ({frac(HESTON_P[k]):5.2f}){'' if e['heston_p_inside'] else ' OUT'}", flush=True)

    out["seconds"] = time.time() - t0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\nSAMPLE-RESTRICTED: these are INNER approximations of true MOT bounds -- the "
          f"optimisation ranges only over measures supported on the {G.shape[0]:,} sampled atoms.")
    print(f"{time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
