"""
One rung of the escalation ladder end to end: train -> checkpoint -> draws A/B/C
-> gate -> eps-bias diagnostic, everything under artifacts_taskc/<tag>/.

    ~/.venv/bin/python3 -m taskc.run_rung --tag rung1 --epochs 450
    ~/.venv/bin/python3 -m taskc.run_rung --tag rung2 --epochs 450 --snr
    ~/.venv/bin/python3 -m taskc.run_rung --tag rungX --epochs 450 --draw_n 20000 --device mps

Draws are seed-identified, not bitwise reproducible across devices (DECISIONS.md
section 3); MPS is ~3x CPU per draw on Apple silicon.

Same init seed, same data order and same sampler for every rung; only what the
flags say changes. Nothing here reads the gate result to choose anything.
"""

import argparse
import json
import os
import time
from dataclasses import replace

import numpy as np
import torch

import taskc  # noqa: F401
from taskc.config import CFG
from taskc.data import build_training_set, make_loader, reference_paths
from taskc.ptheta import (make_schedule, build_model, train_ptheta, save_checkpoint,
                          sample_ptheta, save_draw)
from taskc.gate import run_gate, print_report, summary_dict


@torch.no_grad()
def eps_bias_diagnostic(model, sched, device, ts=(998, 990, 950, 900, 800, 500, 200, 50),
                        n=200_000, seed=0):
    """Per-coordinate mean of eps_hat over x_t ~ N(0, I) (true value 0) and the
    excess eps-MSE over the Gaussian optimum abar_t, at a few t."""
    torch.manual_seed(seed)
    x = torch.randn(n, CFG.data_dim, device=device)
    out = {}
    for t in ts:
        t01 = torch.full((n,), (t + 0.5) / sched.T, device=device)
        e = model(x, t01)
        b = e.mean(0).cpu().numpy()
        out[t] = dict(bias_rms=float(np.sqrt((b ** 2).mean())), bias_max=float(np.abs(b).max()))
    return out


@torch.no_grad()
def excess_mse(model, z, sched, device, n_bins=10, n=20_000, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(z.shape[0], generator=g)[:n]
    x0 = torch.from_numpy(z[idx.numpy()]).to(device)
    T = sched.T
    rows = []
    for b in range(n_bins):
        lo, hi = int(b * T / n_bins), int((b + 1) * T / n_bins)
        t = torch.randint(lo, hi, (x0.shape[0],), generator=g).to(device)
        ab = sched.alphas_bar[t].view(-1, 1)
        noise = torch.randn(x0.shape, generator=g).to(device)
        xt = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise
        pred = model(xt, (t.float() + 0.5) / T)
        rows.append((lo, hi, float(((pred - noise) ** 2).mean()), float(ab.mean())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", type=int, default=CFG.epochs)
    ap.add_argument("--snr", action="store_true")
    ap.add_argument("--gamma", type=float, default=CFG.snr_gamma)
    ap.add_argument("--draw_n", type=int, default=None,
                    help="draw size; 2e4 for rung comparisons, None (=1e5) for the chosen P_theta")
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: best available)")
    ap.add_argument("--lr_decay", action="store_true", help="cosine lr decay to cfg.lr_min")
    ap.add_argument("--ema", action="store_true", help="EMA of weights for sampling")
    ap.add_argument("--prior", default="base", help="Task D prior tag from taskc.priors (base|retrain1|retrain2|mu25|rho02|kappa6)")
    ap.add_argument("--draws_only", action="store_true",
                    help="skip training; load <tag>/ckpt and only draw + gate (promotion to 1e5)")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    from taskc.priors import PRIORS
    pr = PRIORS[args.prior]
    cfg = replace(CFG, artifact_dir=CFG.run_dir(args.tag), epochs=args.epochs,
                  snr_weighting=args.snr, snr_gamma=args.gamma,
                  lr_decay=args.lr_decay, ema=args.ema,
                  heston=pr.heston, init_seed=pr.init_seed)
    os.makedirs(cfg.artifact_dir, exist_ok=True)
    print(f"== {args.tag}: prior={args.prior} ({pr.note}) heston={cfg.heston} init_seed={cfg.init_seed}", flush=True)
    print(f"== {args.tag}: epochs={cfg.epochs} snr_weighting={cfg.snr_weighting} "
          f"gamma={cfg.snr_gamma} lr_decay={cfg.lr_decay} (lr_min {cfg.lr_min}) ema={cfg.ema} "
          f"({cfg.ema_decay}) device={device} -> {cfg.artifact_dir}", flush=True)

    ts = build_training_set(cfg)
    sched = make_schedule(cfg, device=device)
    model = build_model(cfg).to(device)
    loader = make_loader(ts.z, batch_size=cfg.batch_size, seed=cfg.init_seed)

    ckpt = os.path.join(cfg.artifact_dir, cfg.ckpt_name)
    if args.draws_only:
        from taskc.ptheta import load_checkpoint
        model, std_ck, _, extra = load_checkpoint(ckpt, device=device)
        assert std_ck == ts.std, "checkpoint standardizer != training-set standardizer"
        train_s = float(extra.get("train_seconds", 0.0))
        print(f"loaded {ckpt} (no retraining): {extra}", flush=True)
    else:
        t0 = time.time()
        model = train_ptheta(model, loader, sched, cfg, device=device)
        train_s = time.time() - t0
        model.eval()
        save_checkpoint(ckpt, model, ts.std, cfg,
                        extra=dict(tag=args.tag, epochs=cfg.epochs, snr_weighting=cfg.snr_weighting,
                                   snr_gamma=cfg.snr_gamma, lr_decay=cfg.lr_decay, lr_min=cfg.lr_min,
                                   ema=cfg.ema, ema_decay=cfg.ema_decay,
                                   train_seconds=train_s, device=device))
        print(f"trained in {train_s/60:.1f} min -> {ckpt}", flush=True)

    diag = dict(eps_bias=eps_bias_diagnostic(model, sched, device),
                excess_mse=excess_mse(model, ts.z, sched, device))
    print("eps_hat per-coordinate mean bias over x~N(0,I):")
    for t, d in diag["eps_bias"].items():
        print(f"  t={t:4d}  rms {d['bias_rms']:.4f}  max {d['bias_max']:.4f}")
    print("excess eps-MSE over abar_t:")
    for lo, hi, m, ab in diag["excess_mse"]:
        print(f"  [{lo:4d},{hi:4d})  {m-ab:+.4f}")

    draws = {}
    for name, d in cfg.draws.items():
        n = d.n if args.draw_n is None else args.draw_n
        res = sample_ptheta(model, sched, n=n, seed=d.seed, cfg=cfg, device=device, verbose=False)
        save_draw(cfg, name, res)
        draws[name] = res
        print(f"draw {name}: n={n:,} rejected {res.n_rejected} max|z|={np.abs(res.z).max():.2f} "
              f"{res.seconds/60:.1f} min on {res.device}", flush=True)

    ref = reference_paths(cfg)
    report = run_gate(draws["A"].z, ts.std, cfg, ref=ref)
    print_report(report, columns=True)
    out = summary_dict(report)
    out["diagnostic"] = {"eps_bias": {str(k): v for k, v in diag["eps_bias"].items()},
                         "excess_mse": diag["excess_mse"]}
    out["column_bias_z"] = dict(rms=float(np.sqrt(((draws["A"].z.mean(0) - ts.std.to_z(ref.returns()).mean(0)) ** 2).mean())),
                                max=float(np.abs(draws["A"].z.mean(0) - ts.std.to_z(ref.returns()).mean(0)).max()),
                                sum=float((draws["A"].z.mean(0) - ts.std.to_z(ref.returns()).mean(0)).sum()))
    out["rejections"] = {k: int(v.n_rejected) for k, v in draws.items()}
    out["draw_device"] = {k: v.device for k, v in draws.items()}
    out["run"] = dict(tag=args.tag, epochs=cfg.epochs, snr_weighting=cfg.snr_weighting,
                      gamma=cfg.snr_gamma, lr_decay=cfg.lr_decay, ema=cfg.ema,
                      train_seconds=train_s, device=device, draw_n=args.draw_n)
    with open(os.path.join(cfg.artifact_dir, "gate_report.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nRUNG {args.tag} DONE  gate passed = {report.passed}  "
          f"column bias z: rms {out['column_bias_z']['rms']:.4f} max {out['column_bias_z']['max']:.4f} "
          f"sum {out['column_bias_z']['sum']:+.3f}", flush=True)


if __name__ == "__main__":
    main()
