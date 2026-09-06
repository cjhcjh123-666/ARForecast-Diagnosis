"""Generate a self-contained markdown summary of results/iclr/tsfm_deliver."""
import csv, json
from pathlib import Path
import numpy as np

OUT = Path("results/iclr/tsfm_deliver")
rows = list(csv.DictReader(open(OUT/"metrics.csv")))
cfg = json.loads((OUT/"config.json").read_text())

def g(m, i, s, t, key, test=None):
    for r in rows:
        if r["model"]==m and r["init"]==i and r["seed"]==s and r["task"]==t and (test is None or r["test_set"]==test):
            v = r.get(key)
            return float(v) if v not in ("", None) else None
    return None

def mean(key, m, i, t, test=None):
    v = [float(r[key]) for r in rows if r["model"]==m and r["init"]==i and r["task"]==t and r[key] and (test is None or r["test_set"]==test)]
    return float(np.mean(v)) if v else None

MODELS = ["qwen3_8b_base","chronos_t5-small","chronos_t5-base","chronos_bolt-small",
          "timesfm_2.5-200m","moment-1-large","moirai-1.1-R-small"]
SEEDS = ["7","17","27"]
INITS = ["pretrained","random"]

L = []
add = L.append
add("# TS-Foundation-Model Cross-Model Suite — Results Summary")
add("")
add("`results/iclr/tsfm_deliver/` — generated `2026-09-06` by `scripts/iclr_tsfm_deliver.py` + `scripts/rebuild_metrics_from_pw.py`.")
add("")
add("**One-line result:** on the clean 5-class structure-recognition task almost every model (pretrained *or* random) is strong, but only the **language-pretrained Qwen3-8B** representation shows a large pretraining advantage on **zero-shot compositional routing** (bal3 balanced-acc `0.833` vs its random control `0.558`). No time-series foundation model shows such an advantage (pretrained <= random for Bolt/TimesFM/MOMENT/Moirai on routing).")
add("")
add("## 1. Protocol")
add("")
add("- Context **C=64**, horizon **H=16**, seeds **7/17/27**; windows are context-standardized (mean/std of the 64-length context).")
add("- **A. Structure recognition**: 5-class family probe (trend/periodic/local/mixture/regime), 450 train / 300 test windows per seed; frozen backbone + full-batch softmax probe (2,000 steps, lr=0.5, L2=1e-3). `A_recog_shuf` = per-window independent random time-permutation, probe retrained on shuffled features.")
add("- **B. Numerical readout**: frozen rep + linear head → H=16; AdamW lr=1e-3, wd=0, full-batch 300 epochs. MSE in context-normalized space; `head_params = dim*16+16`.")
add("- **C. Compositional routing**: router = softmax probe fit on the 270 clean trend/periodic/local windows (family labels; identical to the E4 reference protocol), tested on two balanced 3-class OOD sets (**bal3**, **bal3n**; 120 windows each, T/P/L = 40/40/40; oracle = argmin future MSE over the 3 experts). Balanced accuracy / macro-F1 / per-class recall / routed MSE.")
add("- **native**: each model's own forecasting interface on the 300 synth5 test windows. Point extraction: Chronos-T5 = median of 20 samples; Chronos-Bolt = 0.5 quantile; TimesFM = official decode point index 5 (GPU port verified bit-identical to `forecast_naive`); Moirai = median of 50 distribution samples, fixed patch 16; **MOMENT = n/a** (official forecasting requires a learned head); **Qwen = n/a** (its native numerical generation is the separate recognition-vs-generation experiment in the paper).")
add("")
add("## 2. Deliverables")
add("")
add("| File | Content |")
add("|---|---|")
add(f"| `metrics.csv` | **{len(rows)} rows** — one per (model, init, seed, task, test_set); fields: accuracy / balanced_accuracy / macro_f1 / recall_trend / recall_periodic / recall_local / mse / head_params |")
add("| `perwindow/pw_<model>_<init>_s<seed>.npz` (42 files) | per-window arrays: `a_true/a_pred`, `as_true/as_pred`, `b_future/b_pred`, `c_{bal3,bal3n}_{oracle,pred,err}` (err = 3 experts' per-window MSE), `n_future/n_pred` |")
add("| `config.json` | checkpoints, families, dims, pooling, normalization, training settings, native point-extraction, and the **Moirai pipeline note** |")
add("")
add("## 3. Models")
add("")
add("| model | family | dim | pooling | native interface |")
add("|---|---|---:|---|---|")
add("| qwen3_8b_base | LLM (decoder-only, text tok.) | 4096 | last-token hidden | n/a (LM generation) |")
add("| chronos_t5-small | T5 seq2seq TS FM | 512 | encoder last token (EOS) | median of 20 samples |")
add("| chronos_t5-base | T5 seq2seq TS FM | 768 | encoder last token (EOS) | median of 20 samples |")
add("| chronos_bolt-small | patched T5 TS FM (Chronos-Bolt) | 512 | encoder [REG] token | 0.5 quantile |")
add("| timesfm_2.5-200m | decoder-only patched TS FM | 1280 | last patch hidden | decode point idx 5 |")
add("| moment-1-large | encoder-only patch TS FM (flan-t5-large) | 1024 | MOMENT embed mean | n/a (learned head) |")
add("| moirai-1.1-R-small | patch×variate encoder TS FM (uni2ts 1.x) | 384 | last patch-token hidden (p=16) | median of 50 samples (p=16) |")
add("")
add("> **Moirai note**: Moirai's attention operates on (patch × variate) tokens and cannot be fed a raw `(B,64)` tensor; inputs are constructed exactly like `MoiraiForecast._convert` (target `(B, n_patch, max_patch=128)` + sample/time/variate ids + masks). Env workarounds (jaxtyping shim, einops/dynamo skip, uni2ts `__init__` bypass) are in `config.json`.")
add("")
add("## 4. Mean over seeds (pretrained | random)")
add("")
add("| model | A_recog_orig | A_recog_shuf | B MSE | C bal3 balacc | C bal3n balacc | C bal3 macro-F1 | native MSE |")
add("|---|---:|---:|---:|---:|---:|---:|---:|")
for m in MODELS:
    a0=mean("accuracy",m,"pretrained","A_recog_orig"); a1=mean("accuracy",m,"random","A_recog_orig")
    s0=mean("accuracy",m,"pretrained","A_recog_shuf"); s1=mean("accuracy",m,"random","A_recog_shuf")
    b0=mean("mse",m,"pretrained","B_readout"); b1=mean("mse",m,"random","B_readout")
    c0=mean("balanced_accuracy",m,"pretrained","C_routing",test=lambda x: True) if False else mean("balanced_accuracy",m,"pretrained","C_routing","bal3_s7")
    def cm(key,mi,tag):
        return mean(key,m,mi,"C_routing",tag)
    cbal0 = np.nanmean([cm("balanced_accuracy","pretrained",f"bal3_s{s}") for s in SEEDS])
    cbal1 = np.nanmean([cm("balanced_accuracy","random",f"bal3_s{s}") for s in SEEDS])
    cn0 = np.nanmean([cm("balanced_accuracy","pretrained",f"bal3n_s{s}") for s in SEEDS])
    cn1 = np.nanmean([cm("balanced_accuracy","random",f"bal3n_s{s}") for s in SEEDS])
    f0 = np.nanmean([cm("macro_f1","pretrained",f"bal3_s{s}") for s in SEEDS])
    f1 = np.nanmean([cm("macro_f1","random",f"bal3_s{s}") for s in SEEDS])
    n0=mean("mse",m,"pretrained","native_forecast"); n1=mean("mse",m,"random","native_forecast")
    f=lambda x: f"{x:.3f}" if x is not None else "n/a"
    add(f"| {m} | {f(a0)} / {f(a1)} | {f(s0)} / {f(s1)} | {f(b0)} / {f(b1)} | {f(cbal0)} / {f(cbal1)} | {f(cn0)} / {f(cn1)} | {f(f0)} / {f(f1)} | {f(n0)} / {f(n1)} |")
add("")
add("### 4.1 C-routing per-class recall (bal3, mean over seeds, pretrained | random)")
add("")
add("| model | recall_trend | recall_periodic | recall_local |")
add("|---|---:|---:|---:|")
for m in MODELS:
    def r3(mi,k):
        return np.nanmean([(lambda v: v if v is not None else float("nan"))( (lambda r: float(r[k]) if r.get(k) else None)(next((x for x in rows if x["model"]==m and x["init"]==mi and x["seed"]==s and x["task"]=="C_routing" and x["test_set"]==f"bal3_s{s}"), {})) ) for s in SEEDS])
    row=[]
    for k in ["recall_trend","recall_periodic","recall_local"]:
        row.append(f"{r3('pretrained',k):.3f} / {r3('random',k):.3f}")
    add(f"| {m} | {row[0]} | {row[1]} | {row[2]} |")
add("")
add("## 5. Per-seed detail (all rows)")
add("")
add("| model | init | seed | task | test_set | n | acc | balacc | macroF1 | recT | recP | recL | mse | head_params |")
add("|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for m in MODELS:
    for i in INITS:
        for s in SEEDS:
            for t in ["A_recog_orig","A_recog_shuf","B_readout"]:
                r=next((x for x in rows if x["model"]==m and x["init"]==i and x["seed"]==s and x["task"]==t),None)
                if r: add(f"| {m} | {i} | {s} | {t} | synth5 | {r['n_test']} | {r['accuracy'] or '-'} | | | | | | {r['mse'] or '-'} | {r['head_params'] or '-'} |")
            for tag in ["bal3","bal3n"]:
                r=next((x for x in rows if x["model"]==m and x["init"]==i and x["seed"]==s and x["task"]=="C_routing" and x["test_set"]==f"{tag}_s{s}"),None)
                if r: add(f"| {m} | {i} | {s} | C_routing | {tag} | {r['n_test']} | | {r['balanced_accuracy']} | {r['macro_f1']} | {r['recall_trend']} | {r['recall_periodic']} | {r['recall_local']} | {r['mse']} | |")
            r=next((x for x in rows if x["model"]==m and x["init"]==i and x["seed"]==s and x["task"]=="native_forecast"),None)
            if r: add(f"| {m} | {i} | {s} | native_forecast | synth5 | {r['n_test']} | | | | | | | {r['mse']} | |")
add("")
add("## 6. Key observations (descriptive)")
add("")
add("- **Recognition is easy for everyone**: pretrained A_recog_orig ≈ 0.98–1.00 for all models; even random weights give ≥0.90 for the patch-based TS FMs (chronos-T5 random ~0.61 is the exception). Clean-family separability alone does **not** indicate transfer.")
add("- **Order sensitivity varies strongly**: A_recog_shuf drops most for MOMENT (0.993→0.231), Bolt (1.0→0.37), TimesFM (1.0→0.56), Moirai (0.997→0.46); Qwen drops 0.981→0.64.")
add("- **Compositional routing (the key test)**: only **Qwen3-8B pretrained** beats its random control by a wide margin (0.833 vs 0.558 bal3). For every TS FM the pretrained router is ≤ its random control (Bolt 0.683 vs 0.675, TimesFM 0.653 vs 0.694, MOMENT 0.631 vs 0.725, Moirai 0.564 vs 0.656). No TS-FM representation reproduces the language-pretrained routing transfer.")
add("- **Numerical readout**: pretrained helps for Qwen (0.588 vs 0.679), Bolt (0.405 vs 1.049) and TimesFM (0.404 vs 0.895); MOMENT random ≈ pretrained; Moirai random is close to pretrained (0.436 vs 0.394).")
add("- **Native forecasting**: TS-FM native interfaces (pretrained) give 0.53–0.67 MSE; random-weight native is 5–25× worse (TimesFM random 16.9, Chronos-T5-base random 12.4), as expected. MOMENT/Qwen native are not defined under this protocol (documented).")
add("")
add("> Caveat: these are diagnostic accessibility/routing numbers on controlled synthetic windows; no OOD test set was used for any model/training selection. See `config.json` for exact checkpoint paths and the Moirai/native caveats.")
(OUT/"SUMMARY.md").write_text("\n".join(L)+"\n")
print("wrote", OUT/"SUMMARY.md", f"({len(L)} lines)")
