"""Audit artifacts for the TS-FM suite deliverable.

Produces results/iclr/tsfm_deliver/audit/:
  clean270_label_crosstab.json/.md   family x oracle-expert on the 270 clean windows (per seed)
  test_class_counts.json             oracle class counts on bal3/bal3n (per seed)
  confusion_matrices.json            per (model, init, seed, test_set) routing confusion (C_routing, C_routing_oracle)
  paired_bootstrap.json              95% percentile bootstrap (2000, window-level, stratified by seed) of
                                     pretrained - random delta in balanced accuracy and routed MSE
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
from data.dynamics import build_labeled_windows
from models.experts import LocalExpert, PeriodicExpert, TrendExpert
KINDS=["trend","periodic","local","mixture","regime"]
EXPERTS=[TrendExpert(),PeriodicExpert(),LocalExpert()]
OUT=Path("results/iclr/tsfm_deliver"); AUD=OUT/"audit"; AUD.mkdir(parents=True,exist_ok=True)
MODELS=["qwen3_8b_base","chronos_t5-small","chronos_t5-base","chronos_bolt-small",
        "timesfm_2.5-200m","moment-1-large","moirai-1.1-R-small"]
SEEDS=[7,17,27]
RNG=np.random.default_rng(0); NB=2000

def expert_errors(ctx,fut):
    err=np.zeros((len(ctx),3))
    for j,e in enumerate(EXPERTS):
        for i in range(len(ctx)):
            err[i,j]=float(np.mean((e.predict(ctx[i],16)-fut[i])**2))
    return err

def balacc(pred,oracle,ncls=3):
    cm=np.zeros((ncls,ncls),int)
    for t,pp in zip(oracle,pred):
        if t<ncls and pp<ncls: cm[t,pp]+=1
    rec=np.array([cm[i,i]/(cm[i].sum()+1e-12) for i in range(ncls)])
    return float(rec.mean())

# ---------- 1. clean270 family x oracle crosstab ----------
ct={}
for s in SEEDS:
    ctx,fut,kk=build_labeled_windows(KINDS,64,16,150,s)
    n=150; ntr=90
    clean=np.concatenate([np.arange(k*n,k*n+ntr) for k in range(3)])
    err=expert_errors(ctx[clean],fut[clean]); oracle=err.argmin(1); fam=kk[clean]
    cm=np.zeros((3,3),int)
    for f,o in zip(fam,oracle): cm[f,o]+=1
    agree=float(np.mean(fam==oracle))
    ct[str(s)]={"family_counts":[int((fam==k).sum()) for k in range(3)],
                "oracle_counts":[int((oracle==k).sum()) for k in range(3)],
                "family_x_oracle":cm.tolist(),"agreement":round(agree,4)}
(AUD/"clean270_label_crosstab.json").write_text(json.dumps(ct,indent=2))
md=["# family x oracle-expert labels on the 270 clean training windows","",
    "Rows=family (trend/periodic/local), cols=oracle-expert (argmin future-MSE expert).",""]
for s in SEEDS:
    d=ct[str(s)]; cm=d["family_x_oracle"]
    md.append(f"## seed {s} (agreement {d['agreement']:.1%})")
    md.append("| fam\\oracle | T | P | L | total |")
    md.append("|---|---:|---:|---:|---:|")
    tot=[sum(cm[i]) for i in range(3)]
    for i,nm in enumerate(["trend","periodic","local"]):
        md.append(f"| {nm} | {cm[i][0]} | {cm[i][1]} | {cm[i][2]} | {tot[i]} |")
    md.append("")
(AUD/"clean270_label_crosstab.md").write_text("\n".join(md)+"\n")

# ---------- 2. test class counts ----------
tc={}
for s in SEEDS:
    tc[str(s)]={}
    for tag in ["bal3","bal3n"]:
        z=np.load(OUT/f"results_nonexist.npz") if False else np.load(f"results/iclr/e4_balanced3/windows/bal3_s{s}.npz") if tag=="bal3" else np.load(f"results/iclr/e4_balanced3_natural/windows/bal3n_s{s}.npz")
        o=z["oracle"]; tc[str(s)][tag]=[int((o==k).sum()) for k in range(3)]; tc[str(s)][tag+"_n"]=int(len(o))
(AUD/"test_class_counts.json").write_text(json.dumps(tc,indent=2))

# ---------- 3. confusion matrices + 4. paired bootstrap ----------
def load_pw(model, init, s, oracle_prefix=False):
    stem=f"pw_oracle_{model}_{init}_s{s}" if oracle_prefix else f"pw_{model}_{init}_s{s}"
    return np.load(OUT/"perwindow"/f"{stem}.npz")

conf={}; boot={}
for variant,opre in [("C_routing",False),("C_routing_oracle",True)]:
    boot[variant]={}
    for m in MODELS:
        boot[variant][m]={}
        for tag in ["bal3","bal3n"]:
            per_seed={}
            deltas_ba=[]; deltas_mse=[]
            for s in SEEDS:
                zpre=load_pw(m,"pretrained",s,opre); zran=load_pw(m,"random",s,opre)
                o=zpre[f"c_{tag}_oracle"]; ppre=zpre[f"c_{tag}_pred"]; pran=zran[f"c_{tag}_pred"]
                err=zpre[f"c_{tag}_err"]
                # confusion matrices
                for ini,z in [("pretrained",zpre),("random",zran)]:
                    pred=z[f"c_{tag}_pred"]; oracle=z[f"c_{tag}_oracle"]
                    cm=np.zeros((3,3),int)
                    for t,pp in zip(oracle,pred):
                        if t<3 and pp<3: cm[t,pp]+=1
                    conf[f"{m}|{ini}|s{s}|{tag}|{variant}"]=cm.tolist()
                # point delta
                mse_pre=float(np.mean(err[np.arange(len(ppre)),ppre])); mse_ran=float(np.mean(err[np.arange(len(pran)),pran]))
                ba_pre=balacc(ppre,o); ba_ran=balacc(pran,o)
                per_seed[str(s)]={"ba_pre":round(ba_pre,4),"ba_rand":round(ba_ran,4),
                                  "mse_pre":round(mse_pre,4),"mse_rand":round(mse_ran,4),
                                  "n":int(len(o))}
                # bootstrap within this seed (stratified)
                n=len(o)
                for _ in range(NB):
                    idx=RNG.integers(0,n,n)
                    bb_pre=balacc(ppre[idx],o[idx]); bb_ran=balacc(pran[idx],o[idx])
                    mm_pre=float(np.mean(err[idx, ppre[idx]])); mm_ran=float(np.mean(err[idx, pran[idx]]))
                    deltas_ba.append(bb_pre-bb_ran); deltas_mse.append(mm_pre-mm_ran)
            ba_delta_mean=float(np.mean([per_seed[str(s)]["ba_pre"]-per_seed[str(s)]["ba_rand"] for s in SEEDS]))
            mse_delta_mean=float(np.mean([per_seed[str(s)]["mse_pre"]-per_seed[str(s)]["mse_rand"] for s in SEEDS]))
            # delta is a mean over seeds of seed-level metrics; percentile over resample distribution
            lo=np.percentile(deltas_ba,2.5); hi=np.percentile(deltas_ba,97.5)
            lom=np.percentile(deltas_mse,2.5); him=np.percentile(deltas_mse,97.5)
            boot[variant][m][tag]={"ba_delta_mean_over_seeds":round(ba_delta_mean,4),
                                   "ba_delta_ci95":[round(lo,4),round(hi,4)],
                                   "mse_delta_mean_over_seeds":round(mse_delta_mean,4),
                                   "mse_delta_ci95":[round(lom,4),round(him,4)],
                                   "per_seed":per_seed,"n_bootstrap":NB}
(AUD/"confusion_matrices.json").write_text(json.dumps(conf,indent=2))
(AUD/"paired_bootstrap.json").write_text(json.dumps(boot,indent=2))
print("wrote audit artifacts to", AUD)
# quick console summary
print("\n=== pretrained - random delta (bal3, family-label) ===")
for m in MODELS:
    d=boot["C_routing"][m]["bal3"]
    print(f"{m:<20} BA {d['ba_delta_mean_over_seeds']:+.4f} [{d['ba_delta_ci95'][0]:+.4f},{d['ba_delta_ci95'][1]:+.4f}]  MSE {d['mse_delta_mean_over_seeds']:+.4f}")
