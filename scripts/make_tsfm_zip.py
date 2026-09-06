"""Package the TS-FM suite deliverables + audit + scripts into one zip."""
import zipfile, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
OUT = REPO/"results/iclr/tsfm_deliver"
ZIP = REPO/"tsfm_suite_deliverables.zip"
base = "tsfm_suite_deliverables"

def add(z, src, dst=None):
    dst = dst or src
    z.write(src, f"{base}/{dst}")

with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    # README first
    z.write("/tmp/zip_readme.md", f"{base}/README.md")
    # core results
    for f in ["metrics.csv","config.json","SUMMARY.md"]:
        z.write(OUT/f, f"{base}/{f}")
    # per-window npz
    for npz in sorted((OUT/"perwindow").glob("*.npz")):
        z.write(npz, f"{base}/perwindow/{npz.name}")
    # audit
    for f in sorted((OUT/"audit").glob("*")):
        z.write(f, f"{base}/audit/{f.name}")
    # scripts (this round)
    for s in ["iclr_tsfm_deliver.py","iclr_tsfm_oracleC.py","rebuild_metrics_from_pw.py",
              "make_tsfm_deliver_md.py","tsfm_audit_artifacts.py"]:
        z.write(REPO/"scripts"/s, f"{base}/scripts/{s}")
    # original E4 routing scripts (family-label provenance)
    for s in ["iclr_e4_balanced.py","iclr_e4_balanced3.py","iclr_e4_balanced3_natural.py",
              "iclr_e4_balanced_full.py","iclr_learned_router.py"]:
        z.write(REPO/"scripts"/s, f"{base}/scripts/original_E4/{s}")
print("wrote", ZIP)
import os
print("size MB:", round(os.path.getsize(ZIP)/1e6,2))
