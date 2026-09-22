# Task B — entropy projection on raw Heston paths

Pure finance go/no-go. **No neural networks anywhere in this package.** The only
iterative optimization in the entire codebase is L-BFGS on the convex dual
(`projection.py:120`); nothing else is fitted, regressed, or trained.

## Modules

| file | role |
|---|---|
| `config.py` | Heston P/Q parameters, constraint-level definitions, exotics contract |
| `heston.py` | full-truncation Euler simulator; returns S **and** v, with `without_variance()` to strip v before constraint building |
| `charfn.py` | Heston characteristic function (Albrecher "little trap") + Carr–Madan call pricer, FFT variant as cross-check |
| `constraints.py` | `g(X)` from price paths only; dynamic test functions, vanilla payoffs, coordinate-wise feasibility screen |
| `projection.py` | column standardization → convex dual → L-BFGS; ESS, ‖β‖, gradient residual, max weight, KL, sd(log w) |
| `evaluation.py` | exotic payoffs, held-out residuals, martingale profile, weighted SEs |
| `run_taskb.py` | headline driver: C0→C3 sweep, Q-sample noise floors, multi-seed stability |
| `sweep_volgap.py` | repeats the headline sweep across four P→Q vol gaps |
| `check_heldout_calibrated.py` | one-run diagnostic: under-specified family vs structural trade-off |
| `sweep_testfuns.py` | martingale-family richness curve (ESS vs number of test functions) |

## Rerun

```bash
python run_taskb.py --paths 100000 --seeds 3     # headline sweep
python sweep_volgap.py --paths 100000            # across vol gaps
python check_heldout_calibrated.py --paths 100000
python charfn.py                                 # Carr–Madan vs MC validation
```

## Design points that matter

**The variance path never reaches `g`.** `build_martingale_columns` and
`build_vanilla_columns` both assert `paths.v is None`. A leak is a hard error.

**Standardization is a reparameterization, not a fit.** Columns are centred and
scaled by their empirical mean/sd; the location shift cancels exactly in the
dual, and `beta_raw = beta_std / sd` recovers the identical measure. Reported
both ways.

**The feasibility screen is coordinate-wise only.** It does not establish that
`c` lies in the joint convex hull, and must not be described that way in the
paper.

**Monte Carlo noise floors.** Every held-out diagnostic is also evaluated on an
independent Heston-Q sample of the same size, where the truth is 0 (martingale)
or the Carr–Madan price (vanilla). Without the floor you cannot separate a real
bias from sampling noise — at N=1e5 the held-out vanilla floor is ~0.012, and
several Q\* numbers sit below it.

**The Q benchmark is chosen, not fitted.** `q_from_target_vol` inverts a
closed-form mean-variance identity to pin the 21-day risk-neutral vol level.
v0 is a state variable and is held equal under P and Q; only (κ, θ) move.
The single-parameter convention `q_params_lam` is also provided, but at H=21
it can only move realised vol by ~0.25 points because v0 is shared — a limit of
the restricted kernel family, not an empirical statement. `sweep_volgap.py`
therefore runs the whole sweep at +0.25, +1.0, +2.6 and +4.0 vol points.

## Findings (N = 1e5, 3 seeds)

- Constraints matched: fitted error ≤ 3e-6 in raw price units at every level and
  every vol gap; gradient residual ~3e-7; all solves converged.
- Held-out vanillas: C3 RMSE below the Q-sample MC noise floor at all four gaps;
  43×–195× better than the P baseline; monotone C1→C2→C3.
- ESS: 99.4% (C0) → 93.6% (C3) at the headline gap; 81.8% at the stressed
  +4-point gap. Degrades smoothly, never collapses. Seed-to-seed sd ≤ 0.23 pp.
- Held-out martingale residual rises from C0 to C1–C3 and scales with the size
  of the premium. `check_heldout_calibrated.py` confirms this is **structural**,
  not an under-specified family: promoting the held-out functions into
  calibration leaves the tier-2 residual at 1.70× the noise floor (from 1.98×)
  while held-out vanilla error degrades 1.5× on tier 1 and 6.3× on tier 2.
  Report this quantity; do not gate on it.
- Exotics: the barrier closes 68–89% of the P→Q span at every gap. The lookback
  barely moves at the headline gap — the clean illustration that 10d/21d
  vanillas do not identify the running minimum. Distance to Heston-Q is a
  diagnostic, not a success condition.
