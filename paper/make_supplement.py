"""
Build the anonymised supplementary zip.

Included: taskb/, taskc/ (code + tests), notebooks/ with every clone/checkout cell
stripped, paper/build_tables.py and its notes, a README.
Excluded: .git/, docs/, taskc/DECISIONS.md, artifacts_*/, models/, logs, caches.
Then scanned for author names, local paths and credentials; the build ABORTS on a hit.

    ~/.venv/bin/python3 paper/make_supplement.py
"""
import argparse, io, json, os, re, subprocess, sys, zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "paper", "supplementary_anonymous.zip")
# data/aapl/ is the frozen market surface; it CANNOT be regenerated (yfinance returns
# live expiries), so the market experiment does not reproduce without it.
INCLUDE_DATA = ["data/aapl"]

INCLUDE_DIRS = ["taskb", "taskc"]
INCLUDE_FILES = ["paper/build_tables.py", "paper/make_overleaf_zip.py", "paper/OVERLEAF.md", "paper/README.md"]
EXCLUDE_NAMES = {"DECISIONS.md", ".DS_Store"}
EXCLUDE_EXT = {".log", ".pyc", ".npz", ".pt", ".pkl", ".zip", ".pdf", ".png", ".jpg"}
EXCLUDE_DIR_PARTS = {".git", "__pycache__", ".ipynb_checkpoints", "docs", "artifacts_taskc",
                     "artifacts_gauss", "artifacts_overlay", "manuscript_floats", "tables", "figures"}

# patterns that must not appear anywhere in the bundle
FORBIDDEN = [
    ("author name", r"(?i)\bnilay\b|\btiwari\b|\bsanchari\b|nilay47"),
    ("local path", r"/Users/[A-Za-z0-9_.-]+|/home/[A-Za-z0-9_.-]+"),
    ("credential", r"github_pat[A-Za-z0-9_]*|gho_[A-Za-z0-9]{10,}|ghp_[A-Za-z0-9]{10,}|AKIA[A-Z0-9]{10,}|BEGIN (?:RSA|OPENSSH) PRIVATE KEY"),
    # real TLDs only: a bare [A-Za-z]{2,} also matches things like "@torch.no_grad"
    ("email", r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.(?:com|org|net|edu|gov|io|ai|co\.uk|ac\.uk|de|fr)\b"),
    ("venue/review", r"(?i)\bopenreview\b|\brebuttal\b|submission\s*3319|\bdP8B\b|\bcViC\b|\baGVW\b"),
    ("repo url", r"github\.com/[A-Za-z0-9_.-]+/"),
    ("git remote", r"\borigin/[A-Za-z0-9_.\-]+"),
    ("venue", r"(?i)\bICLR\b|\bNeurIPS\b|\bICML\b|\bAISTATS\b"),
    ("drive path", r"MyDrive|/content/drive"),
]
CLONE_MARKERS = ("git clone", "%cd /content", "PINNED_COMMIT", "git checkout", "drive.mount")


def strip_notebook(path):
    """Drop clone/checkout cells; neutralise the Drive mount and the pinned-commit guard."""
    nb = json.load(open(path))
    out, dropped = [], 0
    for c in nb["cells"]:
        src = "".join(c["source"])
        if c["cell_type"] == "code" and ("git clone" in src or "%cd /content" in src):
            dropped += 1
            continue
        if c["cell_type"] == "code":
            src = re.sub(r'^PINNED_COMMIT\s*=.*$', 'PINNED_COMMIT    = ""   # removed for anonymous review', src, flags=re.M)
            src = re.sub(r'^EXPECT_TASKC\s*=.*$', 'EXPECT_TASKC     = None  # version guard disabled for anonymous review', src, flags=re.M)
            src = re.sub(r'.*assert HEAD\.startswith.*\n(?:.*\n)?', '', src)
            src = re.sub(r'.*assert taskc\.__version__ == EXPECT_TASKC.*\n(?:\s+f?".*\n)?', '', src)
            src = re.sub(r'.*assert NOTEBOOK_VERSION in open\(.*\n(?:\s+".*\n)*', '', src)
            src = re.sub(r'https://github\.com/\S+', 'https://<anonymised>/', src)
            src = re.sub(r'"/content/drive/MyDrive/[^"]*"', '"artifacts"', src)
            # two mount idioms across the notebooks: one-line, and import/mount on separate
            # lines. Match the mount call itself rather than the surrounding form.
            src = src.replace('from google.colab import drive; drive.mount("/content/drive")',
                              'raise ImportError("Colab Drive mount removed for anonymous review")')
            src = re.sub(r'drive\.mount\(\s*"/content/drive"\s*\)',
                         'raise ImportError("Colab Drive mount removed for anonymous review")', src)
            # the AAPL notebook fetches prices_yf.py from another branch; the bundle vendors it
            src = re.sub(r'\s*src = subprocess\.run\(\["git", "show", "origin/[^\]]*\],\n.*\n.*\n',
                         '\n', src)
            src = src.replace('open("/tmp/prices_yf.py", "w").write(src)\n', '')
            src = src.replace('spec = importlib.util.spec_from_file_location("prices_yf", "/tmp/prices_yf.py")',
                              'spec = importlib.util.spec_from_file_location("prices_yf", "src_real/data/prices_yf.py")')
            src = re.sub(r'origin/[A-Za-z0-9_.\-]+', '<branch>', src)
            src = re.sub(r'(?i)\b(ICLR|NeurIPS|ICML|AISTATS)\b\s*\d{0,4}', 'the venue', src)
            src = re.sub(r'\b(HEAD|subprocess\.run\(\["git".*)\b.*\n', '', src) if "rev-parse" in src else src
            c = dict(c, source=src)
        out.append(c)
    nb["cells"] = out
    md = nb.get("metadata", {})
    md.pop("colab", None)
    nb["metadata"] = md
    return json.dumps(nb, indent=1), dropped


def scrub_text(txt):
    """Applied to EVERY text file in the bundle, not just notebooks: the venue and
    remote-branch names appear in module docstrings too, which the notebook-only
    scrubbing missed on the first run."""
    txt = re.sub(r'(?i)\b(ICLR|NeurIPS|ICML|AISTATS)\b(\s+\d{4})?', '<venue>', txt)
    txt = re.sub(r'\borigin/[A-Za-z0-9_.\-]+', '<branch>', txt)
    txt = re.sub(r'https://github\.com/\S+', 'https://<anonymised>/', txt)
    txt = re.sub(r'github\.com/[A-Za-z0-9_.-]+/', '<anonymised>/', txt)
    return txt


def vendored_prices_yf():
    """The AAPL notebook pulls this from another branch at run time, which leaves a remote
    reference in the bundle and does not work without the repository. Vendor the file."""
    try:
        return subprocess.run(["git", "-C", REPO, "show", "origin/v2_aapl:src_real/data/prices_yf.py"],
                              capture_output=True, text=True, check=True).stdout.encode()
    except Exception as e:
        print(f"  WARNING: could not vendor prices_yf.py ({e}); the AAPL notebook will fall back "
              f"to the committed frozen history, which is included")
        return None


def collect():
    files = {}
    for d in INCLUDE_DIRS:
        for root, dirs, fs in os.walk(os.path.join(REPO, d)):
            dirs[:] = [x for x in dirs if x not in EXCLUDE_DIR_PARTS]
            for f in fs:
                if f in EXCLUDE_NAMES or os.path.splitext(f)[1] in EXCLUDE_EXT:
                    continue
                p = os.path.join(root, f)
                files[os.path.relpath(p, REPO)] = open(p, "rb").read()
    for d in INCLUDE_DATA:
        for root, dirs, fs in os.walk(os.path.join(REPO, d)):
            dirs[:] = [x for x in dirs if x not in EXCLUDE_DIR_PARTS]
            for f in fs:
                if f in EXCLUDE_NAMES or os.path.splitext(f)[1] in EXCLUDE_EXT:
                    continue
                pth = os.path.join(root, f)
                files[os.path.relpath(pth, REPO)] = open(pth, "rb").read()
    py = vendored_prices_yf()
    if py is not None:
        files["src_real/data/prices_yf.py"] = py
        files["src_real/__init__.py"] = b""
        files["src_real/data/__init__.py"] = b""
    for rel in INCLUDE_FILES:
        p = os.path.join(REPO, rel)
        if os.path.exists(p):
            files[rel] = open(p, "rb").read()
    for f in sorted(os.listdir(os.path.join(REPO, "notebooks"))):
        if f.endswith(".ipynb") and not f.startswith("."):
            txt, dropped = strip_notebook(os.path.join(REPO, "notebooks", f))
            files[f"notebooks/{f}"] = txt.encode()
            print(f"  notebooks/{f}: {dropped} clone cell(s) removed")
    return files


def _is_text(data):
    try:
        data.decode()
        return True
    except UnicodeDecodeError:
        return False


def scan(files):
    hits = []
    for rel, data in sorted(files.items()):
        try:
            txt = data.decode()
        except UnicodeDecodeError:
            continue
        for label, pat in FORBIDDEN:
            for m in re.finditer(pat, txt):
                line = txt[:m.start()].count("\n") + 1
                hits.append((rel, line, label, m.group(0)[:60]))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    out_path = os.path.expanduser(args.out)
    files = collect()
    files["README.md"] = (
        "# Supplementary code\n\n"
        "Anonymised for review: repository URLs, clone cells, author paths and the\n"
        "project's internal decision log are removed.\n\n"
        "  taskb/    entropy projection on simulated paths (no neural networks)\n"
        "  taskc/    the path diffusion model, acceptance gate, dual, learned h-transform,\n"
        "            corrected sampler, identification sweep\n"
        "  notebooks/ the runs, in the order they are executed\n"
        "  data/aapl/ the frozen option surface and daily bars for the market experiment.\n"
        "            The surface CANNOT be regenerated: the vendor API returns whatever\n"
        "            expiries are live when it is called, so the file itself is the data.\n"
        "  src_real/ the pinned-date return loader used by the market experiment\n"
        "  paper/build_tables.py  emits every table and figure in the paper from the\n"
        "            artifact files; no number in the paper is typed by hand\n\n"
        "Artifacts (checkpoints, sampled paths, result JSONs) are omitted for size; every\n"
        "script regenerates them. Runtimes are in the paper's reproducibility statement.\n"
    ).encode()
    files = {rel: (scrub_text(data.decode()).encode()
                   if _is_text(data) else data) for rel, data in files.items()}
    hits = scan(files)
    if hits:
        print("\nABORTED -- forbidden content found:")
        for rel, line, label, frag in hits[:40]:
            print(f"  {rel}:{line}  [{label}]  {frag!r}")
        print(f"\n{len(hits)} hit(s). Nothing written.")
        sys.exit(1)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        for rel, data in sorted(files.items()):
            z.writestr(rel, data)
    print(f"\nclean: {len(files)} files -> {out_path} ({os.path.getsize(out_path)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
