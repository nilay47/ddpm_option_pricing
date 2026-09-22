"""
Task D sweep for one prior: dual solved on the 1e6 solve draw at C0..C3, every
reported number from draw C (1e5) with self-normalized SEs, in-sample numbers
from the solve draw reported alongside and labelled.

    ~/.venv/bin/python3 -m taskc.sweep --tag lrema            # baseline
    ~/.venv/bin/python3 -m taskc.sweep --tag mu25
"""
import argparse, json, os, pickle, time
from dataclasses import replace
import numpy as np, torch

import taskc  # noqa
from taskc.config import CFG
from taskc.ptheta import load_checkpoint, load_draw, make_schedule, sample_ptheta, save_draw, draw_path
from taskc.dual import solve_level, evaluate_on, baseline_on, TASKB, TASKB_EXOTICS_Q, TASKB_EXOTICS_P
from taskc.gate import sv_stats
from config import q_params, EXOTICS
import evaluation as ev

LEVELS = ("C0", "C1", "C2", "C3")


def ensure_draw(cfg, model, sched, name, draw, device):
    p = draw_path(cfg, name)
    if not os.path.exists(p):
        r = sample_ptheta(model, sched, n=draw.n, seed=draw.seed, cfg=cfg, device=device, verbose=True)
        save_draw(cfg, name, r)
    return load_draw(cfg, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    RUN = replace(CFG, artifact_dir=CFG.run_dir(args.tag))
    model, std, cfg_dict, extra = load_checkpoint(os.path.join(RUN.artifact_dir, RUN.ckpt_name), device=device)
    # the prior's own simulator params come from the checkpoint
    from config import Heston
    RUN = replace(RUN, heston=Heston(**cfg_dict["heston"]) if isinstance(cfg_dict.get("heston"), dict) else RUN.heston)
    sched = make_schedule(RUN, device=device)
    out_dir = os.path.join(RUN.artifact_dir, "sweep"); os.makedirs(out_dir, exist_ok=True)
    q = q_params()

    t0 = time.time()
    S6 = ensure_draw(RUN, model, sched, "S6", RUN.solve_draw, device)
    C = ensure_draw(RUN, model, sched, "C", RUN.draws["C"], device)
    print(f"[{args.tag}] solve draw S6: n={S6.z.shape[0]:,} seed {S6.seed} {S6.device} rejected {S6.n_rejected} ({time.time()-t0:.0f}s); "
          f"C: n={C.z.shape[0]:,} rejected {C.n_rejected}", flush=True)
    pS6, pC = std.to_paths(S6.z, "S6"), std.to_paths(C.z, "C")
    lvlS6, lvlC = float(S6.z.mean(0).sum()), float(C.z.mean(0).sum())
    print(f"[{args.tag}] levels (summed z means): S6 {lvlS6:+.4f} (floor {np.sqrt(21/1e6):.4f})  C {lvlC:+.4f} (floor {np.sqrt(21/1e5):.4f})", flush=True)

    base = baseline_on(pC, q)
    sv = sv_stats(pC.returns(), RUN.dt)
    out = dict(tag=args.tag, heston=cfg_dict.get("heston"), n_solve=int(S6.z.shape[0]), n_eval=int(C.z.shape[0]),
               level_S6=lvlS6, level_C=lvlC, sv_C=sv,
               unweighted_C=dict(ho_van=base["ho_van_rmse"], ho_mart=base["ho_mart_rmse"], mart_prof=base["mart_prof_max"],
                                 exotics={k: [float(a), float(b)] for k, (a, b) in base["exotics"].items()}),
               levels={})
    tilts = {}
    hdr = (f"{'lvl':4s} {'m':>3s} {'scr':>4s} {'margin':>7s} {'|b_raw|':>8s} {'TaskB':>7s} {'ESS_S6%':>8s} {'ESS_C%':>7s} {'TaskB':>6s} "
           f"{'KL':>7s} {'E_C[L]':>7s} {'hoVan_in':>9s} {'hoVan_C':>8s} {'TaskB_in':>8s} {'hoMart_C':>9s}")
    print(hdr); print("-" * len(hdr))
    for lvl in LEVELS:
        t1 = time.time()
        tilt, r, cs = solve_level(lvl, pS6, q)
        e_in = evaluate_on(tilt, pS6, q)
        e = evaluate_on(tilt, pC, q)
        tb = TASKB[lvl]
        j = int(np.argmin(r.screen["margin"]))
        print(f"{lvl:4s} {cs.m:3d} {'ok' if r.screen['all_ok'] else 'FAIL':>4s} {r.screen['margin'].min():7.3f} {np.linalg.norm(r.beta_raw):8.2f} {tb['beta_raw']:7.2f} "
              f"{r.ess_frac*100:8.2f} {e['ess_frac']*100:7.2f} {tb['ess_frac']*100:6.2f} {r.kl:7.4f} {e['E_L']:7.4f} "
              f"{e_in['ho_van_rmse']:9.4f} {e['ho_van_rmse']:8.4f} {tb['ho_van']:8.4f} {e['ho_mart_rmse']:9.2e}   "
              f"bite {cs.names[j]}  conv={r.converged} it={r.n_iter} {time.time()-t1:.0f}s", flush=True)
        tilts[lvl] = tilt
        out["levels"][lvl] = dict(
            m=cs.m, screen_ok=bool(r.screen["all_ok"]), screen_margin=float(r.screen["margin"].min()), screen_bite=cs.names[j],
            screen_failed=r.screen["failed"], beta_raw_norm=float(np.linalg.norm(r.beta_raw)), beta_raw=r.beta_raw.tolist(),
            beta_names=r.names, ess_solve=float(r.ess_frac), ess_C=float(e["ess_frac"]), kl=float(r.kl), E_C_L=float(e["E_L"]),
            maxw_C=float(e["max_weight_ratio"]), converged=bool(r.converged), n_iter=int(r.n_iter),
            fit_van_C=float(e["fit_van_rmse"]), fit_mart_C=float(e["fit_mart_rmse"]),
            ho_van_in=float(e_in["ho_van_rmse"]), ho_van_C=float(e["ho_van_rmse"]), ho_van_C_max=float(e["ho_van_max"]),
            ho_mart_in=float(e_in["ho_mart_rmse"]), ho_mart_C=float(e["ho_mart_rmse"]), ho_martn_C=float(e["ho_martn_rmse"]),
            hv_names=e["hv_names"], hv_err_C=e["hv_err"].tolist(), hv_se_C=e["hv_se"].tolist(), hv_err_in=e_in["hv_err"].tolist(),
            mart_prof_C=float(e["mart_prof_max"]),
            exotics_C={k: [float(a), float(b)] for k, (a, b) in e["exotics"].items()},
            exotics_in={k: [float(a), float(b)] for k, (a, b) in e_in["exotics"].items()})
    print(f"\n[{args.tag}] exotics on C (SE self-normalized), P -> C0..C3 -> Heston-Q; span fraction closed = (P - Q*)/(P - Q):")
    for k in EXOTICS:
        p0, s0 = base["exotics"][k]; qq = TASKB_EXOTICS_Q[k]
        row = f"  {k:28s} P {p0:.4f}+-{s0:.4f}"
        for lvl in LEVELS:
            v, se = out["levels"][lvl]["exotics_C"][k]; row += f" | {lvl} {v:.4f}+-{se:.4f} ({(p0-v)/(p0-qq)*100:+5.0f}%)"
        print(row + f" | Q {qq:.4f}")
    pickle.dump(tilts, open(os.path.join(out_dir, "tilts_S6.pkl"), "wb"))
    json.dump(out, open(os.path.join(out_dir, "sweep.json"), "w"), indent=1)
    print(f"\n[{args.tag}] SWEEP DONE {time.time()-t0:.0f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
