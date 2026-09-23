# Overleaf drop-in

Unzip `overleaf_tables_figures.zip` at the **root of the Overleaf project**. It contains
exactly two folders and no other files:

    <project root>/
      main.tex                     (yours)
      tables/                      <- from the zip (14 .tex files)
      figures/                     <- from the zip (1 .pdf)

Preamble (all standard on Overleaf, nothing to install):

    \usepackage{booktabs}      % \toprule \midrule \bottomrule
    \usepackage{graphicx}      % \includegraphics
    \usepackage{amsmath,amssymb}   % \checkmark, math in captions

Each table file is a complete float with its own `\caption` and `\label`, so the document
needs only:

    \input{tables/taskc_gate}

Do not add `\begin{table}` around it. Files whose contents are wide are already
`table*` (two-column spanning): `taskb_exotics`, `taskc_dual`, `taskc_exotics`,
`amort_agreement`, `overlay_spread`. In a one-column document `table*` behaves as `table`;
in a two-column ICLR layout they span both columns, which is what they need.

The figure is included the usual way (note: no file extension, so Overleaf picks the PDF):

    \begin{figure*}[t]\centering
      \includegraphics[width=\textwidth]{figures/overlay_identification}
      \caption{Exotic price against the amount of vanilla information, one line per
               physical prior. Bands are $\pm 2$ standard errors.}
      \label{fig:overlay}
    \end{figure*}

## What is in the zip

| `\input{}` / `\includegraphics{}` | `\label` |
|---|---|
| `tables/gauss_table6` | `tab:gauss_full` |
| `tables/gauss_table7_lambda` | `tab:gauss_lambda` |
| `tables/gauss_table9_maturity` | `tab:gauss_maturity` |
| `tables/gauss_table10_terminal` | `tab:gauss_terminal` |
| `tables/gauss_table13_2d` | `tab:gauss_2d` |
| `tables/taskb_sweep` | `tab:taskb_sweep` |
| `tables/taskb_exotics` | `tab:taskb_exotics` |
| `tables/taskc_gate` | `tab:taskc_gate` |
| `tables/taskc_dual` | `tab:taskc_dual` |
| `tables/taskc_exotics` | `tab:taskc_exotics` |
| `tables/taskc_prior_acceptance` | `tab:prior_acceptance` |
| `tables/amort_compute` | `tab:amort_compute` |
| `tables/amort_agreement` | `tab:amort_agreement` |
| `tables/overlay_spread` | `tab:overlay_spread` |
| `figures/overlay_identification.pdf` | (figure, label it yourself) |

## Regenerating

Never edit these files by hand — each carries a `% source:` comment naming the artifact it
came from, and the next regeneration overwrites it. After any new run:

    python3 paper/build_tables.py
    python3 paper/make_overleaf_zip.py

and re-upload. `taskc_gate` reports the amended G2b (studentised, §20) and also prints the
superseded flat bound as `G2b-old`, which is deliberate: the appendix subsection in
`docs/08_DRAFT_APPENDIX_GATE_AMENDMENT.md` discusses both.
