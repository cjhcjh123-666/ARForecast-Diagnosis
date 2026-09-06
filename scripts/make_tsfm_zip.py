"""Package the v2 TS-FM suite deliverables + audit + scripts into one zip."""
import zipfile
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
V2 = REPO/"results/iclr/tsfm_deliver_v2"
V1 = REPO/"results/iclr/tsfm_deliver"
ZIP = REPO/"tsfm_suite_deliverables_v2.zip"
base = "tsfm_suite_deliverables_v2"
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for f in ["README.md","metrics.csv","config.json","SUMMARY.md","optimizer_config.json","DEPENDENCIES.md","windows_manifest.json"]:
        z.write(V2/f, f"{base}/{f}")
    for npz in sorted((V2/"perwindow").glob("*.npz")):
        z.write(npz, f"{base}/perwindow/{npz.name}")
    for f in sorted((V2/"audit").glob("*")):
        z.write(f, f"{base}/audit/{f.name}")
    for f in sorted((V2/"init_audit").glob("*.json")):
        z.write(f, f"{base}/init_audit/{f.name}")
    for s in ["iclr_tsfm_deliver.py","iclr_tsfm_oracleC.py","iclr_tsfm_init_audit.py",
              "rebuild_metrics_from_pw.py","make_tsfm_deliver_md.py","tsfm_audit_artifacts.py",
              "run_frozen_probe.py","tsfm_models.py","make_tsfm_zip.py"]:
        z.write(REPO/"scripts"/s, f"{base}/scripts/{s}")
    for s in ["iclr_e4_balanced.py","iclr_e4_balanced3.py","iclr_e4_balanced3_natural.py",
              "iclr_e4_balanced_full.py","iclr_learned_router.py"]:
        z.write(REPO/"scripts"/s, f"{base}/scripts/original_E4/{s}")
    for s in ["analysis/linear_probe.py","data/dynamics.py","models/experts.py"]:
        z.write(REPO/s, f"{base}/scripts/deps/{s}")
    # legacy v1 (selective-reset diagnostic) note + metrics
    z.writestr(f"{base}/legacy_v1/README.txt",
               "v1 = selective module-reset random branches (results/iclr/tsfm_deliver). "
               "Kept as diagnostic; v2 (this zip) uses architecture-native from-config random. "
               "See v2 init_audit/ for the tensor-level proof of no pretrained residual.")
    z.write(V1/"metrics.csv", f"{base}/legacy_v1/metrics.csv")
print("wrote", ZIP)
import os
print("size MB:", round(os.path.getsize(ZIP)/1e6,2))
