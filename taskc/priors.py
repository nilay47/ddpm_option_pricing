"""Task D priors (DECISIONS.md section 14.3). Every prior uses the frozen recipe; only
the simulator parameters or the fit seed change."""
from dataclasses import dataclass, replace
import taskc  # noqa
from config import P_PARAMS, Heston  # taskb


@dataclass(frozen=True)
class Prior:
    tag: str
    heston: Heston
    init_seed: int
    note: str


PRIORS = {
    "base":     Prior("base",     P_PARAMS,                       0, "baseline (= lrema)"),
    "retrain1": Prior("retrain1", P_PARAMS,                       1, "same spec, init/data-order seed 1"),
    "retrain2": Prior("retrain2", P_PARAMS,                       2, "same spec, init/data-order seed 2"),
    "retrain1b": Prior("retrain1b", P_PARAMS,                     3, "one-retry rule: replaces retrain1 (seed 1 failed G2b), seed 3"),
    "mu25":     Prior("mu25",     replace(P_PARAMS, drift=0.25),  0, "corrupted drift mu=0.25 (vs 0.10)"),
    "rho02":    Prior("rho02",    replace(P_PARAMS, rho=-0.2),    0, "leverage rho=-0.2 (vs -0.7)"),
    "kappa6":   Prior("kappa6",   replace(P_PARAMS, kappa=6.0),   0, "mean reversion kappa=6 (vs 3)"),
}
