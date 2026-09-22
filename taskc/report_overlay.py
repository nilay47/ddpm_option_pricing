"""
Task D overlay report from the per-prior sweep.json / gate_report.json files.

    ~/.venv/bin/python3 -m taskc.report_overlay

Prints: gate outcome per prior; the C3 spread table against the pre-registered
ordering (mu < kappa < rho) with the retrain band as reference; the three exotics
per prior with SEs and span fractions at every level; and, for the corrupted-drift
prior, ||beta_raw|| at C0 alongside its exotic spread.
"""
import json
import os
import numpy as np

import taskc  # noqa
from taskc.config import CFG
from taskc.dual import TASKB_EXOTICS_Q, TASKB_EXOTICS_P
from config import EXOTICS

BASE = "lrema"
ORDER = [("lrema", "base"), ("retrain1", "retrain1 (seed 1)"), ("retrain1b", "retrain1b (seed 3)"),
         ("retrain2", "retrain2 (seed 2)"), ("mu25", "mu=0.25"), ("rho02", "rho=-0.2"), ("kappa6", "kappa=6")]
LEVELS = ["C0", "C1", "C2", "C3"]
SHORT = {"asian_call_K100": "Asian", "uo_barrier_call_K100_B110": "Barrier", "lookback_float_call": "Lookback"}


def load(tag):
    d = CFG.run_dir(tag)
    g = os.path.join(d, "gate_report.json"); s = os.path.join(d, "sweep", "sweep.json")
    return (json.load(open(g)) if os.path.exists(g) else None, json.load(open(s)) if os.path.exists(s) else None)


def k115_across(data):
    """Recompute, per prior with artifacts: K=115 sd ratio and mean ratio (draw A vs own
    reference), unweighted exotics on draw A (relative % vs own reference and absolute),
    and the C3 feasibility-screen margin/position for K=115 on the 1e6 solve draw."""
    from dataclasses import replace
    from taskc.ptheta import load_checkpoint, load_draw
    from taskc.data import reference_paths
    from taskc.gate import all_columns
    from constraints import build, feasibility_screen
    from config import q_params, CONSTRAINT_LEVELS, Heston
    import evaluation as ev
    out = {}
    for tag, _ in ORDER:
        g, s = data[tag]
        if g is None: continue
        RUN = replace(CFG, artifact_dir=CFG.run_dir(tag))
        ck = os.path.join(RUN.artifact_dir, RUN.ckpt_name)
        if not os.path.exists(ck): continue
        _, std, cfg_dict, _ = load_checkpoint(ck)
        h = cfg_dict.get("heston"); RUN = replace(RUN, heston=Heston(**h) if isinstance(h, dict) else RUN.heston)
        A = load_draw(RUN, "A"); ref = reference_paths(RUN)
        pA = std.to_paths(A.z); GA, names, kinds, _ = all_columns(pA); GR, _, _, _ = all_columns(ref)
        j = names.index("call[T=21d,K=115]")
        exA, exR = ev.price_exotics(pA), ev.price_exotics(ref)
        r = dict(sd_ratio=float(GA[:, j].std() / GR[:, j].std()), mean_ratio=float(GA[:, j].mean() / GR[:, j].mean()),
                 unw=[(exA[k][0] - exR[k][0]) / exR[k][0] * 100 for k in EXOTICS], unw_abs=[exA[k][0] for k in EXOTICS],
                 margin=None, pos=None)
        s6 = os.path.join(RUN.artifact_dir, "ptheta_draw_S6.npz")
        if s is not None and os.path.exists(s6):
            S6 = load_draw(RUN, "S6"); spec = CONSTRAINT_LEVELS["C3"]
            cs = build(std.to_paths(S6.z), spec["testfuns"], spec["vanillas"], q_params())
            scr = feasibility_screen(cs); jj = cs.names.index("call[T=21d,K=115]")
            r["margin"] = float(scr["margin"][jj]); r["pos"] = float(scr["pos"][jj])
        out[tag] = r
    return out


def main():
    data = {tag: load(tag) for tag, _ in ORDER}
    print("=" * 100); print("GATE OUTCOMES (each prior vs its own simulator; DECISIONS.md section 14 rule)"); print("=" * 100)
    for tag, lab in ORDER:
        g, s = data[tag]
        if g is None:
            print(f"{lab:22s} -- not run"); continue
        fails = [k for k, v in g.items() if isinstance(v, dict) and "passed" in v and not v["passed"] and k != "G1-info"]
        thr = g.get("thresholds", {})
        own = thr.get("own_simulator_bands", False)
        det = "; ".join(f"{k}={g[k]['value']:.4f} {g[k]['bound']}" for k in fails)
        extra = ""
        if "G4c" in fails and own:
            lo, hi = thr["g4c_lev5"]; extra = f"  [G4c band half-width {(hi-lo)/2:.4f}, own-simulator construction]"
        print(f"{lab:22s} gate {'PASSED' if g['passed'] else 'FAILED'}  {('fails: ' + det) if fails else ''}{extra}  "
              f"(G4 bands {'own simulator' if own else 'section 7'}; swept: {'yes' if s else 'no'})")

    # margins for every retrain (and the base): how close each check sat to its bound
    print("\n" + "=" * 100); print("GATE MARGINS -- retrains and base (value / bound; fraction of bound used; band checks show position in band)"); print("=" * 100)
    for tag, lab in ORDER:
        g = data[tag][0]
        if g is None or not (tag == BASE or tag.startswith("retrain")):
            continue
        parts = []
        for cid in ("G1-tail", "G1-mean", "G2a", "G2b", "G3:asian_call_K100", "G3:uo_barrier_call_K100_B110", "G3:lookback_float_call"):
            v = g[cid]["value"]; bd = float(g[cid]["bound"].replace("<=", "").strip())
            parts.append(f"{cid.replace('G3:', 'G3 ').replace('_call_K100', '').replace('uo_barrier','barrier').replace('_B110','').replace('_float','')} {v:.3f}/{bd:g} ({v/bd*100:3.0f}%)")
        thr = g.get("thresholds") or {"g4a_cv": [0.385, 0.435], "g4b_acf1": [0.030, 0.050], "g4c_lev5": [-0.129, -0.077]}  # section-7 bands if written before the rule
        for cid, key in (("G4a", "g4a_cv"), ("G4b", "g4b_acf1"), ("G4c", "g4c_lev5")):
            v = g[cid]["value"]
            if key in thr:
                lo, hi = thr[key]; pos = (v - lo) / (hi - lo)
                parts.append(f"{cid} {v:+.4f} in [{lo:+.3f},{hi:+.3f}] (pos {pos:.2f})")
            else:
                parts.append(f"{cid} {v:+.4f}")
        print(f"{lab:22s} {'PASS' if g['passed'] else 'FAIL'}\n    " + "\n    ".join(parts))
    # retrain1 unweighted exotics as a width diagnostic (not swept)
    r1 = data.get("retrain1", (None, None))[0]
    if r1 is not None:
        print("\nretrain1 (seed 1, FAILED G2b, not swept) -- unweighted exotics on its draw A vs its own Heston-P reference, from the gate:")
        for cid in ("G3:asian_call_K100", "G3:uo_barrier_call_K100_B110", "G3:lookback_float_call"):
            print(f"    {cid[3:]:28s} {r1[cid]['detail']}")

    # ---- the K=115 column across all priors: sd ratio (draw A vs own reference) and C3 screen margin (on S6)
    print("\n" + "=" * 100); print("THE K=115 (21d) COLUMN ACROSS PRIORS: thinnest tail, tightest baseline gate margin, first screen bite"); print("=" * 100)
    k115 = k115_across(data)
    print(f"  {'prior':22s} {'sdA/sdref':>10s} {'|r-1| / 0.10':>12s} {'E_A/E_ref':>10s} {'screen margin C3 (sd)':>22s} {'pos in [min,max]':>16s} {'unw. exotics vs own ref (A%, B%, L%)':>36s}")
    for tag, lab in ORDER:
        r = k115.get(tag)
        if r is None: continue
        sm = f"{r['margin']:.3f}" if r.get('margin') is not None else "  (not swept)"
        pos = f"{r['pos']:.4f}" if r.get('pos') is not None else ""
        print(f"  {lab:22s} {r['sd_ratio']:10.4f} {abs(r['sd_ratio']-1)/0.10*100:11.0f}% {r['mean_ratio']:10.4f} {sm:>22s} {pos:>16s} "
              f"{r['unw'][0]:+8.2f} {r['unw'][1]:+7.2f} {r['unw'][2]:+7.2f}")

    swept = [(tag, lab) for tag, lab in ORDER if data[tag][1] is not None]
    if BASE not in [t for t, _ in swept]:
        print("baseline sweep missing"); return
    b = data[BASE][1]
    retrains = [t for t, _ in swept if t.startswith("retrain")]
    print("\n" + "=" * 100); print("EXOTICS PER PRIOR: value +- SE on draw C, and % of the Heston P->Q span closed, at each level"); print("=" * 100)
    for k in EXOTICS:
        span = TASKB_EXOTICS_P[k] - TASKB_EXOTICS_Q[k]
        print(f"\n{SHORT[k]}  (Heston-P {TASKB_EXOTICS_P[k]:.4f}, Heston-Q {TASKB_EXOTICS_Q[k]:.4f}, span {span:+.4f})")
        print(f"  {'prior':22s} {'P_theta unw':>14s} " + " ".join(f"{l:>22s}" for l in LEVELS))
        for tag, lab in swept:
            s = data[tag][1]; p0, e0 = s["unweighted_C"]["exotics"][k]
            row = f"  {lab:22s} {p0:8.4f}+-{e0:.4f}"
            for l in LEVELS:
                v, se = s["levels"][l]["exotics_C"][k]
                row += f" {v:8.4f}+-{se:.4f} ({(TASKB_EXOTICS_P[k]-v)/span*100:+4.0f}%)"
            print(row)

    print("\n" + "=" * 100); print("C3 SPREAD vs BASELINE (price; SE units; % of span) -- pre-registered ordering: mu < kappa < rho"); print("=" * 100)
    band = {}
    if len(retrains) >= 1:
        for k in EXOTICS:
            vals = [data[t][1]["levels"]["C3"]["exotics_C"][k][0] for t in retrains] + [b["levels"]["C3"]["exotics_C"][k][0]]
            band[k] = dict(retrain_spread=(max(vals) - min(vals)),
                           max_abs_dev=max(abs(data[t][1]["levels"]["C3"]["exotics_C"][k][0] - b["levels"]["C3"]["exotics_C"][k][0]) for t in retrains))
        print("retrain band at C3, way 1 -- swept retrains only (conditioned on passing the gate), max - min over {base, " + ", ".join(retrains) + "}:\n     " +
              "  ".join(f"{SHORT[k]} {band[k]['retrain_spread']:.4f} ({band[k]['retrain_spread']/(TASKB_EXOTICS_P[k]-TASKB_EXOTICS_Q[k])*100:.0f}% span)" for k in EXOTICS))
        # way 2: unweighted exotic prices on draw A, all retrains including the failed seed
        unw = {t: k115[t]["unw_abs"] for t in ("lrema", "retrain1", "retrain1b", "retrain2") if t in k115}
        if len(unw) >= 2:
            print("retrain band, way 2 (footnote) -- UNWEIGHTED exotics on each seed's draw A, including the failed seed retrain1, max - min over {" + ", ".join(unw) + "}:\n     " +
                  "  ".join(f"{SHORT[k]} {max(v[i] for v in unw.values())-min(v[i] for v in unw.values()):.4f} ({(max(v[i] for v in unw.values())-min(v[i] for v in unw.values()))/(TASKB_EXOTICS_P[k]-TASKB_EXOTICS_Q[k])*100:.0f}% span)" for i, k in enumerate(EXOTICS)))
            unw_pass = {t: v for t, v in unw.items() if t != "retrain1"}
            print("     same, passing seeds only: " + "  ".join(f"{SHORT[k]} {max(v[i] for v in unw_pass.values())-min(v[i] for v in unw_pass.values()):.4f}" for i, k in enumerate(EXOTICS)))
            print("     -> way 1 is conditioned on passing the gate and is narrower than true seed variance by construction; it is NOT widened to compensate.")
    print(f"\n  {'prior':22s} " + " ".join(f"{SHORT[k]:>30s}" for k in EXOTICS) + f" {'ESS_C3%':>8s} {'|b_raw|C0':>10s} {'|b_raw|C3':>10s} {'hoVan_in':>9s}")
    for tag, lab in swept:
        s = data[tag][1]; row = f"  {lab:22s} "
        for k in EXOTICS:
            d = s["levels"]["C3"]["exotics_C"][k][0] - b["levels"]["C3"]["exotics_C"][k][0]
            se = np.hypot(s["levels"]["C3"]["exotics_C"][k][1], b["levels"]["C3"]["exotics_C"][k][1])
            span = TASKB_EXOTICS_P[k] - TASKB_EXOTICS_Q[k]
            flag = ""
            if band and tag not in retrains and tag != BASE:
                flag = " *" if abs(d) > band[k]["retrain_spread"] else "  "
            row += f" {d:+8.4f} ({d/se:+5.1f}SE {d/span*100:+4.0f}%){flag}"
        row += f" {s['levels']['C3']['ess_C']*100:8.2f} {s['levels']['C0']['beta_raw_norm']:10.2f} {s['levels']['C3']['beta_raw_norm']:10.2f} {s['levels']['C3']['ho_van_in']:9.4f}"
        print(row)
    print("  (* = outside the retrain band)")

    # ordering verdict
    mis = [t for t in ("mu25", "kappa6", "rho02") if data[t][1] is not None]
    if len(mis) >= 2:
        print("\nordering by mean |spread| / span over the three exotics at C3:")
        sc = {}
        for t in mis:
            s = data[t][1]
            sc[t] = np.mean([abs(s["levels"]["C3"]["exotics_C"][k][0] - b["levels"]["C3"]["exotics_C"][k][0]) / (TASKB_EXOTICS_P[k] - TASKB_EXOTICS_Q[k]) for k in EXOTICS])
        for t in sorted(sc, key=sc.get): print(f"   {t:8s} {sc[t]*100:6.1f}% of span")
        pred = [t for t in ("mu25", "kappa6", "rho02") if t in sc]
        print("   pre-registered:", " < ".join(pred), "  observed:", " < ".join(sorted(sc, key=sc.get)),
              "  ->", "HELD" if sorted(sc, key=sc.get) == pred else "FALSIFIED")
    if data["mu25"][1] is not None:
        s = data["mu25"][1]
        print(f"\ncorrupted drift (mu=0.25): ||beta_raw|| at C0 = {s['levels']['C0']['beta_raw_norm']:.2f} (baseline {b['levels']['C0']['beta_raw_norm']:.2f}, Task B 5.82); "
              f"ESS C0 {s['levels']['C0']['ess_C']*100:.2f}% (baseline {b['levels']['C0']['ess_C']*100:.2f}%); "
              f"screen C3 {'ok' if s['levels']['C3']['screen_ok'] else 'FAIL'} margin {s['levels']['C3']['screen_margin']:.3f}")
        print("   exotics at C0 (drift removed, no vanillas) vs baseline C0: " + "  ".join(
            f"{SHORT[k]} {s['levels']['C0']['exotics_C'][k][0]-b['levels']['C0']['exotics_C'][k][0]:+.4f} ({(s['levels']['C0']['exotics_C'][k][0]-b['levels']['C0']['exotics_C'][k][0])/np.hypot(s['levels']['C0']['exotics_C'][k][1], b['levels']['C0']['exotics_C'][k][1]):+.1f}SE)" for k in EXOTICS))
        print("   P_theta unweighted (the drift itself): " + "  ".join(
            f"{SHORT[k]} {s['unweighted_C']['exotics'][k][0]-b['unweighted_C']['exotics'][k][0]:+.4f}" for k in EXOTICS))


if __name__ == "__main__":
    main()
