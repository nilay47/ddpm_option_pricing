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

The figure now ships with its own float wrapper, so it is used exactly like a table:

    \input{figures/overlay_identification}

That file carries `width=\textwidth,height=4.3cm,keepaspectratio`. **The 4.3 cm height is
the page budget and must not be relaxed.** `width=\textwidth` sits alongside it so graphicx
scales to whichever constraint binds first: the height can never exceed the cap, and the
figure can never overrun the column. The figure's aspect ratio (3.28) is chosen so that at a
5.5 in column the height lands at 4.26 cm — just inside the cap, using the budget rather
than wasting it. On a wider text block (6.27 in, a4 with 1 in margins) the height constraint
binds instead and the figure is centred at 4.3 cm; nothing overflows either way.

For reference: `height=4.3cm,keepaspectratio` *alone* would imply a 5.55 in width, which
overflows a 5.5 in ICLR column. The dual constraint is what prevents that.

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
| `tables/taskc_gate` (alias: `tables/app_gate`) | `tab:taskc_gate` |
| `tables/taskc_dual` | `tab:taskc_dual` |
| `tables/taskc_exotics` | `tab:taskc_exotics` |
| `tables/taskc_prior_acceptance` | `tab:prior_acceptance` |
| `tables/amort_compute` | `tab:amort_compute` |
| `tables/amort_agreement` | `tab:amort_agreement` |
| `tables/overlay_spread` | `tab:overlay_spread` |
| `figures/overlay_identification` (via `\input{figures/overlay_identification}`) | `fig:overlay` |

## Regenerating

Never edit these files by hand — each carries a `% source:` comment naming the artifact it
came from, and the next regeneration overwrites it. After any new run:

    python3 paper/build_tables.py
    python3 paper/make_overleaf_zip.py

and re-upload. `taskc_gate` (and its alias `app_gate`) reads the **re-gate** artifact
`gate_report_g2bz.json` and reports the amended studentised G2b, with the superseded flat
bound shown beneath it as `G2b-old`, marked *not gated*. Both rows are deliberate: the
appendix subsection in `docs/08_DRAFT_APPENDIX_GATE_AMENDMENT.md` discusses the amendment,
and the table is its evidence. If the re-gate artifact is ever missing the builder falls
back to the pre-amendment report and prints a WARNING — do not ship a table built that way.
