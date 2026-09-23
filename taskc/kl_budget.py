"""
KL-budgeted price intervals on an existing solve draw.

    max / min  E_Q[f]   s.t.  E_Q[g] = c,  KL(Q || P_theta) <= k,  Q << P_N

By duality the optimum is the prior tilted by both the constraints and the
payoff,

    w_i  propto  exp( beta . g_i + gamma f_i ),

with gamma >= 0 for the maximisation branch and gamma <= 0 for the minimisation
branch. gamma is swept; at each step beta is re-solved by the same convex dual
(L-BFGS on  log mean exp(beta.g + gamma f) - beta.c ) so the 53 equalities hold
exactly along the whole curve, and

    KL(Q || P_N) = E_w[z] - log mean exp(z),      z_i = beta.g_i + gamma f_i,

which at gamma = 0 reduces to the entropy projection's own KL. Solves are
warm-started along the gamma path, so each step costs a few L-BFGS iterations
rather than a cold solve.

Note KL(Q*||P_theta) is a FLOOR, not zero: every measure meeting the constraints
has at least that much relative entropy. "Budget k" below means total KL, and
the curve starts at the floor.

    ~/.venv/bin/python3 -m taskc.kl_budget [--n 1000000] [--steps 9]
"""

import argparse
import json
import os
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

import taskc  # noqa: F401
from taskc.config import CFG, frozen
from taskc.ptheta import load_checkpoint, load_draw
from constraints import build                                  # taskb
from config import q_params, CONSTRAINT_LEVELS, EXOTICS        # taskb
import evaluation as ev                                        # taskb

HESTON_Q = {"asian_call_K100": 1.57206694, "uo_barrier_call_K100_B110": 1.76546907,
            "lookback_float_call": 4.62500570}
HESTON_P = {"asian_call_K100": 1.61319, "uo_barrier_call_K100_B110": 2.05449,
            "lookback_float_call": 4.43604}


def standardise(G, c, floor=1e-12):
    m, s = G.mean(axis=0), G.std(axis=0)
    keep = s > floor
    return (G[:, keep] - m[keep]) / s[keep], (c[keep] - m[keep]) / s[keep], keep


def solve_beta(Gs, c_std, offset, beta0, maxiter=500):
    """min_beta  log mean exp(Gs beta + offset) - beta . c_std.  Returns (beta, w, z, lse)."""
    N = Gs.shape[0]; logN = np.log(N)

    def obj(b):
        z = Gs @ b + offset
        lse = logsumexp(z) - logN
        w = np.exp(z - z.max()); w /= w.sum()
        return lse - b @ c_std, Gs.T @ w - c_std

    r = minimize(obj, beta0, jac=True, method="L-BFGS-B",
                 options=dict(maxiter=maxiter, maxfun=maxiter * 2, ftol=1e-14, gtol=1e-10, maxcor=50))
    z = Gs @ r.x + offset
    lse = float(logsumexp(z) - logN)
    w = np.exp(z - z.max()); w /= w.sum()
    return r.x, w, z, lse, r


def kl_of(w, z, lse):
    return float(w @ z - lse)


def branch(Gs, c_std, f, beta0, sign, kl_target, steps, verbose=False):
    """Walk gamma outward from 0 in direction `sign`, aiming to reach kl_target."""
    out, beta, gamma = [], beta0.copy(), 0.0
    # scale: gamma of order 1/sd(f) changes the tilt by ~1 nat
    step = sign / max(f.std(), 1e-12) * 0.25
    for i in range(steps):
        gamma += step
        beta, w, z, lse, r = solve_beta(Gs, c_std, gamma * f, beta)
        kl, price = kl_of(w, z, lse), float(w @ f)
        ess = float(1.0 / np.sum(w ** 2) / len(w))
        out.append(dict(gamma=float(gamma), kl=kl, price=price, ess=ess,
                        grad=float(np.max(np.abs(Gs.T @ w - c_std))), nit=int(r.nit)))
        if verbose:
            print(f"      gamma {gamma:+10.4f}  KL {kl:7.4f}  price {price:9.5f}  ESS {ess*100:5.1f}%  it {r.nit}", flush=True)
        if kl >= kl_target:
            break
        # enlarge the step if KL is growing slowly (KL ~ gamma^2 near the floor)
        if i and out[-1]["kl"] - out[-2]["kl"] < 0.15 * kl_target:
            step *= 1.6
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="lrema"); ap.add_argument("--level", default="C3")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--steps", type=int, default=9)
    ap.add_argument("--kl_target", type=float, default=1.0)
    ap.add_argument("--out", default="artifacts_taskc/lrema/kl_budget.json")
    args = ap.parse_args()
    t0 = time.time()

    RUN = frozen(artifact_dir=CFG.run_dir(args.tag))
    _, std, _, _ = load_checkpoint(os.path.join(RUN.artifact_dir, RUN.ckpt_name))
    S6 = load_draw(RUN, "S6")
    z0 = S6.z if args.n is None else S6.z[:args.n]
    paths = std.to_paths(z0, "S6")
    q = q_params(); spec = CONSTRAINT_LEVELS[args.level]
    cs = build(paths, spec["testfuns"], spec["vanillas"], q)
    Gs, c_std, keep = standardise(np.asarray(cs.G, dtype=np.float64), cs.c.astype(np.float64))
    print(f"atoms {Gs.shape[0]:,}  constraints {Gs.shape[1]}  ({time.time()-t0:.0f}s to build)", flush=True)

    beta0, w0, zz, lse0, r0 = solve_beta(Gs, c_std, np.zeros(Gs.shape[0]), np.zeros(Gs.shape[1]))
    kl0 = kl_of(w0, zz, lse0)
    print(f"Q* (gamma = 0): KL {kl0:.5f} nats   ESS {1/np.sum(w0**2)/len(w0)*100:.2f}%   "
          f"grad {np.max(np.abs(Gs.T @ w0 - c_std)):.1e}", flush=True)

    out = dict(level=args.level, n_atoms=int(Gs.shape[0]), kl_floor=kl0, curves={},
               heston_q=HESTON_Q, heston_p=HESTON_P)
    for k, sp in EXOTICS.items():
        f = np.asarray(ev.exotic_payoff(paths, **sp), dtype=np.float64)
        q_star = float(w0 @ f)
        t1 = time.time()
        up = branch(Gs, c_std, f, beta0, +1.0, args.kl_target, args.steps)
        dn = branch(Gs, c_std, f, beta0, -1.0, args.kl_target, args.steps)
        out["curves"][k] = dict(q_star=q_star, upper=up, lower=dn, seconds=time.time() - t1)
        print(f"  {k:28s} Q* {q_star:9.5f}  up to {up[-1]['price']:9.5f} (KL {up[-1]['kl']:.3f})  "
              f"down to {dn[-1]['price']:9.5f} (KL {dn[-1]['kl']:.3f})  {time.time()-t1:.0f}s", flush=True)

    out["seconds"] = time.time() - t0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\n{time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
