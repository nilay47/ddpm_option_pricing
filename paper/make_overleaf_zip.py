"""Bundle paper/tables/*.tex and paper/figures/* into one Overleaf-ready zip."""
import os, zipfile, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "paper", "overleaf_tables_figures.zip")
n = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for sub, exts in (("tables", (".tex",)), ("figures", (".pdf", ".png"))):
        d = os.path.join(REPO, "paper", sub)
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if f.endswith(exts):
                z.write(os.path.join(d, f), f"{sub}/{f}"); n += 1
                print(f"  {sub}/{f}")
print(f"{n} files -> {OUT} ({os.path.getsize(OUT)/1024:.0f} KB)")
