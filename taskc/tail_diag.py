"""
Unplanned diagnostic (DECISIONS.md section 15.3): is the far right wing of the
terminal price systematically under-represented by the DDPM, across seeds and priors?

Per prior with artifacts: draw A (unweighted) vs its OWN simulator reference --
quantiles of S_21, mean ratio E_A/E_ref for EVERY vanilla column (12 calibrated +
9 held-out), and the sign of the C3 exotic deviations vs Heston-Q where a sweep
exists. Characterization only; nothing is adjusted.

    ~/.venv/bin/python3 -m taskc.tail_diag
"""
import json, os
from dataclasses import replace
import numpy as np
import taskc  # noqa
from taskc.config import CFG
from taskc.ptheta import load_checkpoint, load_draw
from taskc.data import reference_paths
from taskc.gate import all_columns
from taskc.dual import TASKB_EXOTICS_Q
from config import Heston, EXOTICS
import evaluation as ev

ORDER = ["lrema", "retrain1", "retrain1b", "retrain2", "mu25", "rho02", "kappa6"]
QS = [0.001, 0.005, 0.01, 0.5, 0.99, 0.995, 0.999]


def tail_stats(tag):
    RUN = replace(CFG, artifact_dir=CFG.run_dir(tag))
    ck = os.path.join(RUN.artifact_dir, RUN.ckpt_name)
    if not os.path.exists(ck):
        return None
    _, std, cfg_dict, _ = load_checkpoint(ck)
    h = cfg_dict.get("heston"); RUN = replace(RUN, heston=Heston(**h) if isinstance(h, dict) else RUN.heston)
    A = load_draw(RUN, "A"); ref = reference_paths(RUN)
    pA = std.to_paths(A.z)
    ST_A, ST_R = pA.S[:, -1], ref.S[:, -1]
    qA, qR = np.quantile(ST_A, QS), np.quantile(ST_R, QS)
    GA, names, kinds, calibrated = all_columns(pA); GR, _, _, _ = all_columns(ref)
    van = [j for j, k in enumerate(kinds) if k == "vanilla"]
    # order vanilla columns by (maturity, strike) for a monotonicity read
    def key(n):
        T = int(n.split("T=")[1].split("d")[0]); K = float(n.split("K=")[1].rstrip("]")); return (T, K)
    van = sorted(van, key=lambda j: key(names[j]))
    rows = [(names[j], key(names[j]), float(GA[:, j].mean() / GR[:, j].mean()), float(GA[:, j].std() / GR[:, j].std()),
             float((GA[:, j].mean() - GR[:, j].mean()) / GR[:, j].std())) for j in van]
    # right-tail mass beyond thresholds
    mass = {K: (float((ST_A > K).mean()), float((ST_R > K).mean())) for K in (105, 110, 115, 120)}
    sw = os.path.join(RUN.artifact_dir, "sweep", "sweep.json")
    ex = None
    if os.path.exists(sw):
        s = json.load(open(sw)); ex = {k: s["levels"]["C3"]["exotics_C"][k] for k in EXOTICS}
    return dict(tag=tag, qA=qA.tolist(), qR=qR.tolist(), vanilla=rows, mass=mass, exotics_C3=ex,
                sT_mean=(float(ST_A.mean()), float(ST_R.mean())), sT_sd=(float(ST_A.std()), float(ST_R.std())))


def main(tags=ORDER, verbose=True):
    res = {t: tail_stats(t) for t in tags}
    res = {t: r for t, r in res.items() if r is not None}
    print("=" * 100); print("UNWEIGHTED TERMINAL TAIL OF EACH PRIOR (draw A) vs ITS OWN SIMULATOR REFERENCE"); print("=" * 100)
    print(f"{'prior':10s} " + " ".join(f"{'q'+str(q):>14s}" for q in QS) + f" {'mean':>14s} {'sd':>14s}")
    for t, r in res.items():
        print(f"{t:10s} " + " ".join(f"{a:6.2f}/{b:6.2f}" for a, b in zip(r['qA'], r['qR'])) + f" {r['sT_mean'][0]:6.2f}/{r['sT_mean'][1]:6.2f} {r['sT_sd'][0]:6.2f}/{r['sT_sd'][1]:6.2f}")
    print("  (A / reference).  Right-tail mass P(S_21 > K), A vs reference:")
    for t, r in res.items():
        print(f"    {t:10s} " + "  ".join(f"K{K}: {a*100:.3f}%/{b*100:.3f}% ({a/b:.3f})" for K, (a, b) in r['mass'].items()))
    print("\nMEAN RATIO E_A[payoff]/E_ref[payoff] for every vanilla column, by maturity then strike (1.000 = no bias); sd ratio in parentheses")
    names = [n for n, _, _, _, _ in next(iter(res.values()))["vanilla"]]
    print(f"{'column':22s} " + " ".join(f"{t:>16s}" for t in res))
    for i, n in enumerate(names):
        print(f"{n:22s} " + " ".join(f"{res[t]['vanilla'][i][2]:7.3f} ({res[t]['vanilla'][i][3]:.3f})" for t in res))
    # monotonicity: is the under-representation growing with strike at 21d, for every prior?
    print("\nmonotone-in-strike check at 21d (mean ratio decreasing with K over the 13 strikes 85..115):")
    for t, r in res.items():
        r21 = [(k[1], m) for n, k, m, _, _ in r["vanilla"] if k[0] == 21]
        ratios = [m for _, m in sorted(r21)]
        dec = all(ratios[i] >= ratios[i+1] - 1e-9 for i in range(len(ratios)-1))
        print(f"    {t:10s} {' '.join(f'{m:.3f}' for m in ratios)}   strictly non-increasing: {dec}   (85 -> 115: {ratios[0]:.3f} -> {ratios[-1]:.3f})")
    print("\nC3 EXOTICS vs HESTON-Q (sign of Q* - Q per prior); expected sign under a thin right tail: Asian -, up-and-out barrier +, lookback -")
    for t, r in res.items():
        if r["exotics_C3"] is None: print(f"    {t:10s} (not swept)"); continue
        print(f"    {t:10s} " + "  ".join(f"{k.split('_')[0]:8s} {r['exotics_C3'][k][0]-TASKB_EXOTICS_Q[k]:+.4f} ({(r['exotics_C3'][k][0]-TASKB_EXOTICS_Q[k])/r['exotics_C3'][k][1]:+.1f}SE)" for k in EXOTICS))
    return res


if __name__ == "__main__":
    main()
