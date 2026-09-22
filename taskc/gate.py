"""
Step-2 hard gate: does P_theta reproduce Heston-P in the ways the projection
cannot repair?

Thresholds are the pre-registered numbers in DECISIONS.md section 7 and live in
`GateThresholds`; this module computes statistics and compares. It never
adjusts a threshold, and it reports every check whether or not an earlier one
failed. Draw A is compared against an INDEPENDENT Heston-P sample (seed
20260922), not the training set, so the Monte Carlo floor is honest.

Checks
  G1-tail  max |E_A[g] - E_ref[g]| / sd_ref over the 21 vanilla columns   <= 0.02
  G1-mean  same over the 56 martingale-type columns                        <= 0.05
  G1-info  uniform 0.02 over all 77 columns                                reported only
  G2a      max |sd_A/sd_ref - 1| over martingale columns                   <= 0.05
  G2b      same over vanilla columns                                       <= 0.10
  G3       |p_A - p_ref| / sqrt(se_A^2 + se_ref^2), three exotics          <= 5
  G4a      CV(RV_21)                                                       in [0.385, 0.435]
  G4b      lag-1 autocorrelation of squared returns                        in [0.030, 0.050]
  G4c      corr(Y_s, mean_{u=s+1..s+5} Y_u^2), s = 0..15                   in [-0.129, -0.077]
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np

import taskc  # noqa: F401
from config import (CALIB_TESTFUNS, VANILLA_C3, HELDOUT_TESTFUNS, HELDOUT_VANILLAS,  # taskb
                    EXOTICS)
from constraints import build          # taskb
from heston import Paths               # taskb
import evaluation as ev                # taskb
from taskc.config import TaskCConfig, CFG
from taskc.data import PathStandardizer, reference_paths


# --------------------------------------------------------------------------
# thresholds (DECISIONS.md section 7) -- do not edit here without a log entry
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class GateThresholds:
    g1_tail: float = 0.02          # vanilla column means, sd units
    g1_mean: float = 0.05          # martingale column means, sd units
    g1_info_uniform: float = 0.02  # reported only
    g2_mart: float = 0.05          # |sd ratio - 1|
    g2_van: float = 0.10
    g3_se: float = 5.0             # combined-SE units
    g4a_cv: Tuple[float, float] = (0.385, 0.435)
    g4b_acf1: Tuple[float, float] = (0.030, 0.050)
    g4c_lev5: Tuple[float, float] = (-0.129, -0.077)


THRESH = GateThresholds()

CV_NULL = 0.309   # iid-Gaussian null for CV(RV_21) at H = 21, matched variance (DECISIONS.md section 7)


def thresholds_for(cfg: TaskCConfig, ref: Paths) -> GateThresholds:
    """
    DECISIONS.md section 14 gate rule. For a prior trained on Heston-P the
    pre-registered section-7 numbers apply unchanged. For any other simulator
    the G4 bands are rebuilt from the prior's OWN reference sample by the same
    construction: G4a [null + 0.75 excess, null + 1.25 excess]; G4b, G4c
    ref x [0.75, 1.25] with the reference's sign, half-width floor 0.005.
    G1-G3 are relative to the reference already and keep their thresholds.
    """
    from config import P_PARAMS
    if cfg.heston == P_PARAMS:
        return THRESH
    sv = sv_stats(ref.returns(), cfg.dt)
    ex = sv["cv_rv"] - CV_NULL
    g4a = (CV_NULL + 0.75 * ex, CV_NULL + 1.25 * ex)

    def band(v):
        hw = max(0.25 * abs(v), 0.005)
        return (v - hw, v + hw)
    return GateThresholds(g4a_cv=tuple(sorted(g4a)), g4b_acf1=band(sv["acf1_sq"]), g4c_lev5=band(sv["lev5"]))


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def all_columns(paths: Paths):
    """The 53 C3 calibration columns followed by the 24 held-out columns."""
    cs = build(paths, CALIB_TESTFUNS, VANILLA_C3)
    ho = build(paths, HELDOUT_TESTFUNS, HELDOUT_VANILLAS)
    G = np.column_stack([cs.G, ho.G])
    names = cs.names + ho.names
    kinds = cs.kinds + ho.kinds
    calibrated = np.array([True] * cs.m + [False] * ho.m)
    return G, names, np.array(kinds), calibrated


def sv_stats(Y: np.ndarray, dt: float) -> Dict[str, float]:
    """Stochastic-volatility structure from the return path only."""
    Y = np.asarray(Y, dtype=np.float64)
    Y2 = Y ** 2
    rv = Y2.mean(axis=1) / dt
    cv = float(rv.std() / rv.mean())
    acf1 = float(np.corrcoef(Y2[:, :-1].ravel(), Y2[:, 1:].ravel())[0, 1])
    lev1 = float(np.corrcoef(Y[:, :-1].ravel(), Y2[:, 1:].ravel())[0, 1])
    H = Y.shape[1]
    S_max = H - 5                               # s = 0..H-6 so that s+5 <= H-1
    fw = np.stack([Y2[:, s + 1:s + 6].mean(axis=1) for s in range(S_max)], axis=1)
    lev5 = float(np.corrcoef(Y[:, :S_max].ravel(), fw.ravel())[0, 1])
    return dict(cv_rv=cv, acf1_sq=acf1, lev1=lev1, lev5=lev5,
                rv_mean=float(rv.mean()), rv_sd=float(rv.std()))


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

@dataclass
class Check:
    id: str
    value: float
    bound: str
    passed: bool
    detail: str = ""


@dataclass
class GateReport:
    checks: List[Check]
    passed: bool
    n_A: int
    n_ref: int
    names: List[str]
    kinds: np.ndarray
    calibrated: np.ndarray
    mean_A: np.ndarray
    mean_ref: np.ndarray
    sd_A: np.ndarray
    sd_ref: np.ndarray
    dmean_sd: np.ndarray            # |E_A - E_ref| / sd_ref
    sd_ratio: np.ndarray            # sd_A / sd_ref
    exotics_A: dict
    exotics_ref: dict
    sv_A: dict
    sv_ref: dict
    mc_floor_sd: float              # MC floor for a mean difference, sd units
    thr: GateThresholds = None      # thresholds actually used (section 14 rule)


def run_gate(zA: np.ndarray, std: PathStandardizer, cfg: TaskCConfig = CFG,
             thr: GateThresholds = None, ref: Paths = None) -> GateReport:
    pA = std.to_paths(zA, world="Ptheta_A")
    ref = reference_paths(cfg) if ref is None else ref
    thr = thresholds_for(cfg, ref) if thr is None else thr

    GA, names, kinds, calibrated = all_columns(pA)
    GR, names_r, kinds_r, _ = all_columns(ref)
    assert names == names_r

    mean_A, mean_ref = GA.mean(0), GR.mean(0)
    sd_A, sd_ref = GA.std(0), GR.std(0)
    dmean_sd = np.abs(mean_A - mean_ref) / sd_ref
    sd_ratio = sd_A / sd_ref
    van = kinds == "vanilla"
    mart = ~van

    checks = []

    def worst(mask, arr):
        j = int(np.argmax(np.where(mask, arr, -np.inf)))
        return j, float(arr[j])

    # G1
    j, v = worst(van, dmean_sd)
    checks.append(Check("G1-tail", v, f"<= {thr.g1_tail}", v <= thr.g1_tail,
                        f"worst {names[j]} (E_A {mean_A[j]:.5f} vs E_ref {mean_ref[j]:.5f})"))
    j, v = worst(mart, dmean_sd)
    checks.append(Check("G1-mean", v, f"<= {thr.g1_mean}", v <= thr.g1_mean,
                        f"worst {names[j]} (E_A {mean_A[j]:.5f} vs E_ref {mean_ref[j]:.5f})"))
    j, v = worst(np.ones_like(van), dmean_sd)
    n_fail = int((dmean_sd > thr.g1_info_uniform).sum())
    checks.append(Check("G1-info", v, f"(uniform {thr.g1_info_uniform}, not gated)",
                        v <= thr.g1_info_uniform,
                        f"{n_fail}/77 columns above {thr.g1_info_uniform}; worst {names[j]}"))
    # G2
    j, v = worst(mart, np.abs(sd_ratio - 1))
    checks.append(Check("G2a", v, f"<= {thr.g2_mart}", v <= thr.g2_mart,
                        f"worst {names[j]} (sd_A/sd_ref = {sd_ratio[j]:.4f})"))
    j, v = worst(van, np.abs(sd_ratio - 1))
    checks.append(Check("G2b", v, f"<= {thr.g2_van}", v <= thr.g2_van,
                        f"worst {names[j]} (sd_A/sd_ref = {sd_ratio[j]:.4f})"))
    # G3
    exA, exR = ev.price_exotics(pA), ev.price_exotics(ref)
    for k in EXOTICS:
        (pa, sa), (pr, sr) = exA[k], exR[k]
        z = abs(pa - pr) / np.hypot(sa, sr)
        checks.append(Check(f"G3:{k}", float(z), f"<= {thr.g3_se}", z <= thr.g3_se,
                            f"A {pa:.4f}+-{sa:.4f}  ref {pr:.4f}+-{sr:.4f}  diff {pa-pr:+.4f} "
                            f"({(pa-pr)/pr*100:+.2f}%)"))
    # G4
    svA, svR = sv_stats(pA.returns(), cfg.dt), sv_stats(ref.returns(), cfg.dt)
    for cid, key, (lo, hi) in (("G4a", "cv_rv", thr.g4a_cv), ("G4b", "acf1_sq", thr.g4b_acf1),
                               ("G4c", "lev5", thr.g4c_lev5)):
        v = svA[key]
        checks.append(Check(cid, v, f"in [{lo}, {hi}]", lo <= v <= hi,
                            f"ref {svR[key]:.4f}"))

    gated = [c for c in checks if c.id != "G1-info"]
    return GateReport(checks=checks, passed=all(c.passed for c in gated),
                      n_A=pA.n, n_ref=ref.n, names=names, kinds=kinds, calibrated=calibrated,
                      mean_A=mean_A, mean_ref=mean_ref, sd_A=sd_A, sd_ref=sd_ref,
                      dmean_sd=dmean_sd, sd_ratio=sd_ratio, exotics_A=exA, exotics_ref=exR,
                      sv_A=svA, sv_ref=svR, mc_floor_sd=float(np.sqrt(1 / pA.n + 1 / ref.n)), thr=thr)


def print_report(r: GateReport, columns: bool = True):
    print("=" * 78)
    print(f"STEP-2 GATE  P_theta draw A (n={r.n_A:,}) vs independent Heston-P (n={r.n_ref:,})")
    print(f"MC floor for a mean difference: {r.mc_floor_sd:.4f} sd")
    print(f"G4 bands in use: CV {r.thr.g4a_cv}, ACF1 {r.thr.g4b_acf1}, lev5 {r.thr.g4c_lev5} "
          f"({'section-7 Heston-P numbers' if r.thr == THRESH else 'rebuilt from the prior\'s own simulator, section 14'})")
    print("=" * 78)
    for c in r.checks:
        tag = "PASS" if c.passed else "FAIL"
        if c.id == "G1-info":
            tag = "info"
        print(f"{tag:4s} {c.id:28s} {c.value:9.4f}  {c.bound:22s} {c.detail}")
    print("-" * 78)
    print(f"sv structure   {'':10s} {'P_theta A':>10s} {'Heston-P':>10s}")
    for k in ("cv_rv", "acf1_sq", "lev1", "lev5", "rv_mean"):
        print(f"  {k:24s} {r.sv_A[k]:10.4f} {r.sv_ref[k]:10.4f}")
    print("-" * 78)
    print(f"*** GATE {'PASSED' if r.passed else 'FAILED'} ***")
    if columns:
        print("\ncolumns (calibrated C3 first, then held-out):")
        print(f"{'column':26s} {'kind':8s} {'E_ref':>9s} {'E_A':>9s} {'dE/sd':>7s} {'sd_ref':>8s} "
              f"{'sd_A/sd':>8s}")
        for j, n in enumerate(r.names):
            flag = ""
            lim = THRESH.g1_tail if r.kinds[j] == "vanilla" else THRESH.g1_mean
            if r.dmean_sd[j] > lim:
                flag += " <G1"
            lim2 = THRESH.g2_van if r.kinds[j] == "vanilla" else THRESH.g2_mart
            if abs(r.sd_ratio[j] - 1) > lim2:
                flag += " <G2"
            ho = "" if r.calibrated[j] else " (held-out)"
            print(f"{n:26s} {r.kinds[j]:8s} {r.mean_ref[j]:9.5f} {r.mean_A[j]:9.5f} "
                  f"{r.dmean_sd[j]:7.4f} {r.sd_ref[j]:8.4f} {r.sd_ratio[j]:8.4f}{flag}{ho}")


def summary_dict(r: GateReport) -> dict:
    out = {c.id: dict(value=float(c.value), passed=bool(c.passed), bound=c.bound, detail=c.detail)
           for c in r.checks}
    out.update(passed=bool(r.passed), n_A=int(r.n_A), n_ref=int(r.n_ref),
               thresholds=dict(g4a_cv=list(r.thr.g4a_cv), g4b_acf1=list(r.thr.g4b_acf1), g4c_lev5=list(r.thr.g4c_lev5),
                               own_simulator_bands=bool(r.thr != THRESH)),
               sv_A={k: float(v) for k, v in r.sv_A.items()},
               sv_ref={k: float(v) for k, v in r.sv_ref.items()})
    return out
