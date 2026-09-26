"""
Task C: path DDPM P_theta + entropy projection + learned h_psi amortization.

`taskb/` is a flat package that imports its own modules by bare name
(`from config import ...`), so it has to be on sys.path as a directory. We put
the repo root and taskb/ on the path here, once, so every taskc module and
notebook can do `from taskc import ...` and `from heston import Paths` alike.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_HERE)
TASKB_DIR = os.path.join(REPO_ROOT, "taskb")

for _p in (REPO_ROOT, TASKB_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Cap CPU intra-op threads (requested 2026-09-20): 8 threads on 12 cores pinned the
# machine for a small wall-clock gain. Override with TASKC_THREADS=<n>.
try:
    import torch as _torch
    _torch.set_num_threads(int(os.environ.get("TASKC_THREADS", "4")))
except Exception:  # torch absent or already initialised with a fixed pool
    pass

# Bumped whenever a notebook's expectations of this package change; the Colab
# notebooks assert against it so a stale cached notebook fails at cell 0.
__version__ = "taskc-2026.09.25d"   # + Bates fitter and panel dynamics (section 27)
