"""
Step 3: train h_psi on draw B (targets L* at one constraint level, from A's tilt),
diagnose on draw C. Nothing here touches draw A except through the saved Tilt.

    ~/.venv/bin/python3 -m taskc.run_hpsi --level C3 --device mps
"""
import argparse, json, os, pickle, time
from dataclasses import replace
import numpy as np, torch
import taskc  # noqa
from taskc.config import CFG
from taskc.ptheta import load_checkpoint, load_draw, make_schedule
from taskc.hnet import HNet, train_hnet, tower_curve, h0_vs_L, grad_ratio_curve, save_hnet
from constraints import build
from config import q_params, CONSTRAINT_LEVELS


def targets_for(tilt, std, z, q):
    spec = CONSTRAINT_LEVELS[tilt.level]
    cs = build(std.to_paths(z), spec["testfuns"], spec["vanillas"], q)
    return np.exp(tilt.logL(cs.G))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="lrema"); ap.add_argument("--level", default="C3")
    ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--device", default=None)
    ap.add_argument("--h_min", type=float, default=0.05); ap.add_argument("--hidden", type=int, default=256)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    RUN = replace(CFG, artifact_dir=CFG.run_dir(args.tag))
    out_dir = os.path.join(RUN.artifact_dir, "hpsi"); os.makedirs(out_dir, exist_ok=True)
    eps_model, std, _, _ = load_checkpoint(os.path.join(RUN.artifact_dir, RUN.ckpt_name), device=device)
    sched = make_schedule(RUN, device=device)
    tilt = pickle.load(open(os.path.join(RUN.artifact_dir, "dual", "tilts.pkl"), "rb"))[args.level]
    q = q_params()
    B, C = load_draw(RUN, "B"), load_draw(RUN, "C")
    LB, LC = targets_for(tilt, std, B.z, q), targets_for(tilt, std, C.z, q)
    print(f"== h_psi {args.level} on {device}: targets on B: E={LB.mean():.4f} min={LB.min():.4f} max={LB.max():.3f} | "
          f"C: E={LC.mean():.4f} min={LC.min():.4f} max={LC.max():.3f} | h_min={args.h_min}", flush=True)

    hnet = HNet(CFG.data_dim, args.hidden, 32, args.h_min)
    log = train_hnet(hnet, B.z, LB, sched, device=device, epochs=args.epochs, verbose=False)
    print(f"trained {args.epochs} epochs in {log.seconds/60:.1f} min; MSE first/last epoch {log.epoch_loss[0]:.4f} / {log.epoch_loss[-1]:.4f}; "
          f"floor active {max(log.floor_frac):.2e} (max over epochs), clamp {max(log.clamp_frac):.2e}", flush=True)
    save_hnet(os.path.join(out_dir, f"hpsi_{args.level}.pt"), hnet,
              dict(level=args.level, epochs=args.epochs, device=device, h_min=args.h_min, hidden=args.hidden,
                   train_seconds=log.seconds, epoch_loss=log.epoch_loss, floor_frac=log.floor_frac, clamp_frac=log.clamp_frac,
                   E_L_B=float(LB.mean()), min_L_B=float(LB.min()), max_L_B=float(LB.max())))

    ts = sorted(set([0, 5, 10, 20, 40, 60, 80, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 980, 998]))
    tower = tower_curve(hnet, C.z, sched, ts, device=device)
    print("\ntower property on draw C: E_C[h_tau(Z_tau)] (target 1)")
    print(f"{'t':>4s} {'E[h]':>8s} {'se':>7s} {'dev/se':>7s} {'min':>7s} {'max':>8s} {'near_floor':>10s}")
    for t in ts:
        d = tower[t]; print(f"{t:4d} {d['mean']:8.4f} {d['se']:7.4f} {((d['mean']-1)/d['se']) if d['se'] > 0 else float('nan'):7.2f} {d['min']:7.4f} {d['max']:8.3f} {d['frac_near_floor']:10.2e}")
    h0 = h0_vs_L(hnet, C.z, LC, sched, device=device)
    print(f"\nh_0 vs L* on C: corr {h0['corr']:.4f}  R2 {h0['r2']:.4f}  RMSE {h0['rmse']:.4f}  mean h {h0['mean_h']:.4f} vs mean L {h0['mean_L']:.4f}")
    gr = grad_ratio_curve(hnet, eps_model, C.z, sched, ts, device=device)
    print("\n||grad log h|| / ||grad log p|| on C (mean over 5000 paths)")
    print(f"{'t':>4s} {'ratio':>8s} {'median':>8s} {'|grad h|':>9s} {'|grad p|':>9s} {'|eps corr|':>10s}")
    for t in ts:
        d = gr[t]; print(f"{t:4d} {d['ratio_mean']:8.4f} {d['ratio_median']:8.4f} {d['grad_h_norm']:9.4f} {d['grad_p_norm']:9.3f} {d['eps_corr_norm']:10.4f}")
    json.dump(dict(level=args.level, tower={str(k): v for k, v in tower.items()}, h0=h0,
                   grad_ratio={str(k): v for k, v in gr.items()}, train=dict(seconds=log.seconds, loss=log.epoch_loss,
                   floor_frac=log.floor_frac, clamp_frac=log.clamp_frac)),
              open(os.path.join(out_dir, f"diagnostics_{args.level}.json"), "w"), indent=1)
    print("\nsaved", out_dir)


if __name__ == "__main__":
    main()
