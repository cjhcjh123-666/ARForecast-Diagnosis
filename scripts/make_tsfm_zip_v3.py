"""Package v3 deliverable zip + separate Qwen-features zip."""
import zipfile
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
V2 = REPO/"results/iclr/tsfm_deliver_v2"
ZIP = REPO/"tsfm_suite_deliverables_v3.zip"
FZIP = REPO/"tsfm_qwen_features.zip"
base="tsfm_suite_deliverables_v3"

# --- main compact zip ---
core = ["README.md","metrics.csv","config.json","SUMMARY.md","REVIEWER_FOLLOWUP_analyses.md",
        "optimizer_config.json","DEPENDENCIES.md","windows_manifest.json","features_manifest.json"]
scripts = ["iclr_tsfm_deliver.py","iclr_tsfm_oracleC.py","iclr_tsfm_init_audit.py",
           "rebuild_metrics_from_pw.py","make_tsfm_deliver_md.py","tsfm_audit_artifacts.py",
           "run_frozen_probe.py","tsfm_models.py","make_tsfm_zip_v3.py",
           "tsfm_qwen_features.py","tsfm_dim_matching.py","tsfm_scale_sweep.py",
           "tsfm_mlp_router.py","tsfm_10seed_summary.py","tsfm_gen_seed10_windows.py",
           "tsfm_expertbank_stability.py","tsfm_expertbank_ood.py"]
e4 = ["iclr_e4_balanced.py","iclr_e4_balanced3.py","iclr_e4_balanced3_natural.py",
      "iclr_e4_balanced_full.py","iclr_learned_router.py"]
deps = ["analysis/linear_probe.py","data/dynamics.py","models/experts.py"]
with zipfile.ZipFile(ZIP,"w",zipfile.ZIP_DEFLATED) as z:
    for f in core: z.write(V2/f, f"{base}/{f}")
    for npz in sorted((V2/"perwindow").glob("*.npz")): z.write(npz, f"{base}/perwindow/{npz.name}")
    for f in sorted((V2/"audit").glob("*")): z.write(f, f"{base}/audit/{f.name}")
    for f in sorted((V2/"init_audit").glob("*.json")): z.write(f, f"{base}/init_audit/{f.name}")
    # 10-seed OOD windows (seeds 37..97)
    for s in [37,47,57,67,77,87,97]:
        for tag,src in [("bal3",REPO/f"results/iclr/e4_balanced3/windows/bal3_s{s}.npz"),
                        ("bal3n",REPO/f"results/iclr/e4_balanced3_natural/windows/bal3n_s{s}.npz")]:
            z.write(src, f"{base}/windows_10seed/{tag}_s{s}.npz")
    if (V2/"seed10_delta.npy").exists(): z.write(V2/"seed10_delta.npy", f"{base}/seed10_delta.npy")
    for s in scripts: z.write(REPO/"scripts"/s, f"{base}/scripts/{s}")
    for s in e4: z.write(REPO/"scripts"/s, f"{base}/scripts/original_E4/{s}")
    for s in deps: z.write(REPO/s, f"{base}/scripts/deps/{s}")
    z.writestr(f"{base}/legacy_v1/README.txt",
               "v1 = selective module-reset random branches (results/iclr/tsfm_deliver). "
               "v3 (this zip) uses architecture-native from-config random. init_audit shows 0 pretrained residual.")
    z.write(V2.parent.parent/ "iclr" if False else REPO/"results/iclr/tsfm_deliver/metrics.csv",
            f"{base}/legacy_v1/metrics.csv")

# --- features zip ---
with zipfile.ZipFile(FZIP,"w",zipfile.ZIP_DEFLATED) as z:
    for d in ["qwen_feats","qwen_feats_10s"]:
        for f in sorted((REPO/d).glob("*.npz")):
            z.write(f, f"tsfm_qwen_features/{d}/{f.name}")
    z.write(V2/"features_manifest.json","tsfm_qwen_features/features_manifest.json")
import os
for zz in [ZIP,FZIP]:
    print(zz.name, round(os.path.getsize(zz)/1e6,1),"MB")
