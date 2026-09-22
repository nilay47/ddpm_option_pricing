"""
Step 4: corrected sampler draw + amortization unit test (+ discretization series).

    ~/.venv/bin/python3 -m taskc.run_smt --level C3 --device mps
"""
import argparse, json, os, pickle, time
from dataclasses import replace
import numpy as np, torch
import taskc  # noqa
from taskc.config import CFG
from taskc.ptheta import load_checkpoint, load_draw, make_schedule, save_draw
from taskc.hnet import load_hnet
from taskc.smt import sample_smt, sample_strided, sliced_wasserstein, reverse_ancestral_strided
from taskc.sampler import reverse_ancestral
from taskc.dual import evaluate_on, baseline_on, TASKB_EXOTICS_Q
from constraints import build, build_heldout
from config import q_params, CONSTRAINT_LEVELS, HELDOUT_TESTFUNS, HELDOUT_VANILLAS, EXOTICS
import evaluation as ev
from projection import wmean, wmean_se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="lrema"); ap.add_argument("--level", default="C3")
    ap.add_argument("--device", default=None); ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=2001); ap.add_argument("--series_n", type=int, default=20_000)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    RUN = replace(CFG, artifact_dir=CFG.run_dir(args.tag))
    out_dir = os.path.join(RUN.artifact_dir, "smt"); os.makedirs(out_dir, exist_ok=True)
    model, std, _, _ = load_checkpoint(os.path.join(RUN.artifact_dir, RUN.ckpt_name), device=device)
    sched = make_schedule(RUN, device=device)
    hnet = load_hnet(os.path.join(RUN.artifact_dir, "hpsi", f"hpsi_{args.level}.pt"), device=device)
    tilt = pickle.load(open(os.path.join(RUN.artifact_dir, "dual", "tilts.pkl"), "rb"))[args.level]
    q = q_params(); spec = CONSTRAINT_LEVELS[args.level]
    A, C = load_draw(RUN, "A"), load_draw(RUN, "C")

    # sanity: strided sampler at full step count reproduces the frozen loop (up to fp rounding)
    torch.manual_seed(0); y1 = reverse_ancestral(model, (256, 21), sched.alphas, sched.alphas_bar, sched.betas, torch.device(device), t_start=sched.t_start)
    torch.manual_seed(0); y2 = reverse_ancestral_strided(model, (256, 21), sched, sched.t_start + 1, torch.device(device))
    print(f"strided(K={sched.t_start+1}) vs frozen loop: max|diff| = {float((y1-y2).abs().max()):.2e}", flush=True)

    # --- SMT draw ----------------------------------------------------------
    S = sample_smt(model, hnet, sched, n=args.n, seed=args.seed, cfg=RUN, device=device, verbose=False)
    save_draw(replace(RUN, artifact_dir=out_dir), f"smt_{args.level}", S)
    print(f"SMT draw: n={args.n:,} seed {args.seed} on {S.device}: rejected {S.n_rejected} of {S.n_drawn:,} "
          f"({S.reject_rate*100:.4f}%; P_theta draws: 0), max|z|={np.abs(S.z).max():.2f}, {S.seconds/60:.1f} min", flush=True)

    pS, pC = std.to_paths(S.z, "SMT"), std.to_paths(C.z, "Ptheta_C")
    csS, csC = build(pS, spec["testfuns"], spec["vanillas"], q), build(pC, spec["testfuns"], spec["vanillas"], q)
    eC = evaluate_on(tilt, pC, q); w = eC["w"]
    # (1) E[g] under SMT vs c, with resampling SE from the SMT draw
    mS, seS = csS.G.mean(0), csS.G.std(0, ddof=1) / np.sqrt(csS.n)
    mW = np.array([wmean(csC.G[:, j], w) for j in range(csC.m)]); seW = np.array([wmean_se(csC.G[:, j], w) for j in range(csC.m)])
    sd = csC.G.std(0)
    devS = (mS - csS.c) / seS; devW = (mW - csC.c) / seW
    print(f"\n(1) constraint means under SMT vs targets c ({csS.m} columns):")
    print(f"    max |E_SMT[g]-c|/SE = {np.abs(devS).max():.2f}   within 2 SE: {int((np.abs(devS)<=2).sum())}/{csS.m}   within 3 SE: {int((np.abs(devS)<=3).sum())}/{csS.m}   "
          f"RMS dev/SE {np.sqrt(np.mean(devS**2)):.2f}   max |E_SMT[g]-c|/sd = {np.abs((mS-csS.c)/sd).max():.4f}")
    print(f"    reference, weighted P_theta (C, weights from A): max |E_w[g]-c|/SE = {np.abs(devW).max():.2f}   within 2 SE: {int((np.abs(devW)<=2).sum())}/{csC.m}   RMS {np.sqrt(np.mean(devW**2)):.2f}")
    dSW = (mS - mW) / np.hypot(seS, seW)
    print(f"    SMT vs weighted-C directly: max |diff|/SE = {np.abs(dSW).max():.2f}   within 2 SE: {int((np.abs(dSW)<=2).sum())}/{csS.m}   RMS {np.sqrt(np.mean(dSW**2)):.2f}")
    worst = np.argsort(-np.abs(devS))[:6]
    for j in worst:
        print(f"      {csS.names[j]:24s} c={csS.c[j]:9.5f}  SMT {mS[j]:9.5f} ({devS[j]:+.2f} SE)  wC {mW[j]:9.5f} ({devW[j]:+.2f} SE)  unw-C {csC.G[:,j].mean():9.5f}")
    # held-out columns: SMT vs weighted C
    hoS, hoC = build_heldout(pS, HELDOUT_TESTFUNS, HELDOUT_VANILLAS, q), build_heldout(pC, HELDOUT_TESTFUNS, HELDOUT_VANILLAS, q)
    mhS, sehS = hoS.G.mean(0), hoS.G.std(0, ddof=1)/np.sqrt(hoS.n)
    mhW = np.array([wmean(hoC.G[:, j], w) for j in range(hoC.m)]); sehW = np.array([wmean_se(hoC.G[:, j], w) for j in range(hoC.m)])
    dh = (mhS - mhW) / np.hypot(sehS, sehW)
    print(f"    held-out columns ({hoS.m}), SMT vs weighted-C: max |diff|/SE = {np.abs(dh).max():.2f}  within 2 SE: {int((np.abs(dh)<=2).sum())}/{hoS.m}")
    hv = ev.heldout_vanilla_errors(hoS, None)
    print(f"    held-out vanilla RMSE: SMT (unweighted draw) {np.sqrt(np.mean(hv[3]**2)):.4f}   weighted-C {eC['ho_van_rmse']:.4f}   Heston-Q floor 0.0122")
    # column sds
    print(f"    column sd ratio SMT/weighted-C: max |ratio-1| = {np.abs(csS.G.std(0)/np.sqrt(np.array([wmean((csC.G[:,j]-mW[j])**2, w) for j in range(csC.m)]))-1).max():.4f}")

    # (2) exotics
    exS, exC_w, exC_u = ev.price_exotics(pS), eC["exotics"], baseline_on(pC, q)["exotics"]
    print("\n(2) exotics: SMT (unweighted draw) vs weighted P_theta vs Heston-Q")
    for k in EXOTICS:
        (a, sa), (b, sb) = exS[k], exC_w[k]
        print(f"    {k:28s} SMT {a:.4f}+-{sa:.4f}   wC {b:.4f}+-{sb:.4f}   diff {(a-b)/np.hypot(sa,sb):+.2f} SE   (P_theta unw {exC_u[k][0]:.4f}, Heston-Q {TASKB_EXOTICS_Q[k]:.4f})")
    # (3) martingale profile
    mpS = ev.martingale_profile(pS, None); mpW = ev.martingale_profile(pC, w)
    print(f"\n(3) max |discounted martingale deviation|: SMT {np.abs(mpS).max():.2e}   weighted-C {np.abs(mpW).max():.2e}   P_theta unw {np.abs(ev.martingale_profile(pC, None)).max():.2e}")

    # (4) sliced Wasserstein in z-space, with null
    t0 = time.time()
    sw_null = sliced_wasserstein(A.z, C.z)                       # two independent P_theta draws
    sw_signal = sliced_wasserstein(C.z, C.z, wX=w)               # unweighted vs weighted: the tilt itself
    sw_test = sliced_wasserstein(C.z, S.z, wX=w)                 # weighted-C vs SMT
    sw_smt_unw = sliced_wasserstein(C.z, S.z)                    # unweighted-C vs SMT
    sw_null2 = sliced_wasserstein(A.z, S.z, wX=evaluate_on(tilt, std.to_paths(A.z), q)["w"])  # weighted-A vs SMT (second null-ish reference)
    print(f"\n(4) sliced Wasserstein (z-space, 256 projections, {time.time()-t0:.0f}s):")
    print(f"    null   SW(P_theta A, P_theta C)          = {sw_null:.5f}")
    print(f"    signal SW(C unweighted, C weighted)      = {sw_signal:.5f}   (size of the tilt)")
    print(f"    test   SW(C weighted, SMT)               = {sw_test:.5f}   <- amortization error; compare to null")
    print(f"           SW(A weighted, SMT)               = {sw_null2:.5f}")
    print(f"           SW(C unweighted, SMT)             = {sw_smt_unw:.5f}   (should be ~ signal if SMT carries the tilt)")

    # (5) discretization series -- NOT P_theta below full steps
    print(f"\n(5) step-count series at n={args.series_n:,} (labelled: K < {sched.t_start+1} is NOT P_theta):")
    print(f"    {'K':>5s} {'P_theta-strided max|E[g]-c|/sd':>32s} {'SMT-strided max|E[g]-c|/sd':>28s} {'SMT RMS dev/SE':>15s} {'rej%':>6s}")
    series = {}
    for K in (sched.t_start + 1, 500, 250, 100, 50):
        dP = sample_strided(model, sched, args.series_n, K, seed=3000 + K, cfg=RUN, device=device)
        dS = sample_strided(model, sched, args.series_n, K, seed=4000 + K, cfg=RUN, device=device, hnet=hnet)
        gP = build(std.to_paths(dP.z), spec["testfuns"], spec["vanillas"], q).G; gS = build(std.to_paths(dS.z), spec["testfuns"], spec["vanillas"], q).G
        gapP = np.abs((gP.mean(0) - csS.c) / sd).max(); gapS = np.abs((gS.mean(0) - csS.c) / sd).max()
        rmsS = np.sqrt(np.mean(((gS.mean(0) - csS.c) / (gS.std(0, ddof=1)/np.sqrt(gS.shape[0])))**2))
        series[K] = dict(gapP=float(gapP), gapS=float(gapS), rmsS=float(rmsS), rej=dS.reject_rate)
        print(f"    {K:5d} {gapP:32.4f} {gapS:28.4f} {rmsS:15.2f} {dS.reject_rate*100:6.3f}")

    json.dump(dict(level=args.level, smt=dict(n=args.n, seed=args.seed, device=S.device, rejected=S.n_rejected, drawn=S.n_drawn),
                   constraints=dict(max_dev_se=float(np.abs(devS).max()), within2=int((np.abs(devS)<=2).sum()), within3=int((np.abs(devS)<=3).sum()),
                                    rms_dev_se=float(np.sqrt(np.mean(devS**2))), max_dev_sd=float(np.abs((mS-csS.c)/sd).max()),
                                    wC_max_dev_se=float(np.abs(devW).max()), wC_within2=int((np.abs(devW)<=2).sum()),
                                    smt_vs_wC_max=float(np.abs(dSW).max()), smt_vs_wC_within2=int((np.abs(dSW)<=2).sum()),
                                    heldout_smt_vs_wC_max=float(np.abs(dh).max()), heldout_within2=int((np.abs(dh)<=2).sum()),
                                    ho_van_rmse_smt=float(np.sqrt(np.mean(hv[3]**2))), ho_van_rmse_wC=float(eC["ho_van_rmse"])),
                   exotics={k: dict(smt=[float(exS[k][0]), float(exS[k][1])], wC=[float(exC_w[k][0]), float(exC_w[k][1])]) for k in EXOTICS},
                   martingale=dict(smt=float(np.abs(mpS).max()), wC=float(np.abs(mpW).max())),
                   sw=dict(null=sw_null, signal=sw_signal, test=sw_test, A_weighted_vs_smt=sw_null2, C_unw_vs_smt=sw_smt_unw),
                   series={str(k): v for k, v in series.items()}),
              open(os.path.join(out_dir, f"amortization_{args.level}.json"), "w"), indent=1)
    print("\nsaved", out_dir)


if __name__ == "__main__":
    main()
