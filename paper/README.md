# paper/ — generated LaTeX

`build_tables.py` reads the artifact files and writes `tables/*.tex` and
`figures/*.pdf`. **Every number comes from an artifact; nothing is retyped.**
Regenerate after any run:

    ~/.venv/bin/python3 paper/build_tables.py              # everything available
    ~/.venv/bin/python3 paper/build_tables.py --only overlay

An emitter whose inputs are missing prints `skipped` and the others still run,
so this is safe to run at any time — including now, before the overlay lands.

Each file is a complete float (`table`/`table*` with `\caption` and `\label`),
so `\input{tables/<name>}` is all the document needs. Preamble requirements:
`booktabs`, `graphicx`, and `amsmath`; `\checkmark` comes from `amssymb`.

| `\input{}` | label | contents | artifact |
|---|---|---|---|
| `tables/gauss_table6.tex` | `tab:gauss_full` | Gaussian: no shift / exact / estimated / retrained-Q, martingale dev., RMSE, MAE, W1, KS | `artifacts_gauss/colab_A100/gauss_appendix.json` |
| `tables/gauss_table7_lambda.tex` | `tab:gauss_lambda` | correction-strength sweep, all eight rows | same |
| `tables/gauss_table9_maturity.tex` | `tab:gauss_maturity` | ATM price vs maturity | same |
| `tables/gauss_table10_terminal.tex` | `tab:gauss_terminal` | terminal distribution at H=21 | same |
| `tables/gauss_table13_2d.tex` | `tab:gauss_2d` | 2-D transport, with the extrapolation-limit note | same |
| `tables/taskb_sweep.tex` | `tab:taskb_sweep` | Task B C0–C3: ESS, beta, KL, held-out diagnostics, Q noise floor | `taskb/full_run.log` |
| `tables/taskb_exotics.tex` | `tab:taskb_exotics` | Task B exotic prices across levels (`table*`) | same |
| `tables/taskc_gate.tex` | `tab:taskc_gate` | the pre-registered acceptance gate for P_theta | `artifacts_taskc/lrema/gate_report.json` |
| `tables/taskc_dual.tex` | `tab:taskc_dual` | dual on P_theta: screen margin, beta_raw, ESS, KL, held-out (`table*`) | `artifacts_taskc/lrema/sweep/sweep.json` |
| `tables/taskc_exotics.tex` | `tab:taskc_exotics` | exotics under the projection with the learned prior (`table*`) | same |
| `tables/taskc_prior_acceptance.tex` | `tab:prior_acceptance` | frozen-prior acceptance: which priors passed, tightest check, drops | `artifacts_taskc/*/gate_report.json` |
| `tables/amort_compute.tex` | `tab:amort_compute` | compute per arm: paths, network evaluations, wall-clock | `artifacts_taskc/lrema/amort77_colab/` (or the older `amort74_colab/`) `amortization.json` |
| `tables/amort_agreement.tex` | `tab:amort_agreement` | weighted vs amortised agreement on the exotics (`table*`) | same |
| **`tables/overlay_spread.tex`** | `tab:overlay_spread` | C3 spread per prior with the retrain band (`table*`) | `artifacts_overlay/overlay.json` |
| **`figures/overlay_identification.pdf`** | — | exotic price vs constraint level, one band per prior | same |

The two overlay outputs are written the moment the overlay results land; the
generator accepts either the notebook's `overlay.json` (tag → sweep) or
per-prior `artifacts_overlay/<tag>/sweep/sweep.json` files.

Figures are included the usual way:

    \begin{figure*}[t]\centering
      \includegraphics[width=\textwidth]{figures/overlay_identification}
      \caption{...}\label{fig:overlay}
    \end{figure*}
