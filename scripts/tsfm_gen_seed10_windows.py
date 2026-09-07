"""Generate bal3/bal3n balanced OOD windows for extra seeds (10-seed run).

Replicates iclr_e4_balanced3.py (local = AR_weak_sine) and
iclr_e4_balanced3_natural.py (local = periodic+AR), mixture -> T/P oracle.
Writes results/iclr/e4_balanced3/windows/bal3_s{seed}.npz and
results/iclr/e4_balanced3_natural/windows/bal3n_s{seed}.npz (canonical dirs).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
from data.dynamics import generate_window
from models.experts import LocalExpert, PeriodicExpert, TrendExpert
EXPERTS=[TrendExpert(),PeriodicExpert(),LocalExpert()]
SEEDS=[37,47,57,67,77,87,97]; PC=40

def gen_ar_weak_sine(rng):
    total=80; t=np.arange(total,dtype=float); x=np.empty(total); x[0]=rng.normal(0,0.2)
    for k in range(1,total): x[k]=0.9*x[k-1]+rng.normal(0,0.3)
    x=x+0.15*np.sin(2*np.pi*t/24)
    m=x[:64].mean(); s=x[:64].std()+1e-6
    return (x[:64]-m)/s,(x[64:]-m)/s

def gen_per_ar(rng):
    total=80; t=np.arange(total,dtype=float); x=np.empty(total); x[0]=rng.normal(0,0.2)
    for k in range(1,total): x[k]=0.9*x[k-1]+rng.normal(0,0.28)
    x=x+0.4*np.sin(2*np.pi*t/24)
    m=x[:64].mean(); s=x[:64].std()+1e-6
    return (x[:64]-m)/s,(x[64:]-m)/s

def build(local_fn, outdir, tag):
    outdir.mkdir(parents=True,exist_ok=True)
    for s in SEEDS:
        w=outdir/f"{tag}_s{s}.npz"
        if w.is_file(): print("exists",w); continue
        rng=np.random.default_rng(s); ctx,fut,oracle=[],[],[]
        for _ in range(5000):
            cw,fw=generate_window("mixture",64,16,rng)
            errs=[float(np.mean((e.predict(cw,16)-fw)**2)) for e in EXPERTS]
            j=int(np.argmin(errs))
            if j<2: ctx.append(cw); fut.append(fw); oracle.append(j)
        for _ in range(8000):
            cw,fw=local_fn(rng)
            errs=[float(np.mean((e.predict(cw,16)-fw)**2)) for e in EXPERTS]
            if int(np.argmin(errs))==2: ctx.append(cw); fut.append(fw); oracle.append(2)
        ctx=np.stack(ctx); fut=np.stack(fut); oracle=np.array(oracle)
        keep=np.concatenate([np.where(oracle==k)[0][:PC] for k in range(3)])
        ctx,fut,oracle=ctx[keep],fut[keep],oracle[keep]
        np.savez(w,ctx=ctx,fut=fut,oracle=oracle)
        print(tag,s,int((oracle==0).sum()),int((oracle==1).sum()),int((oracle==2).sum()),flush=True)

if __name__=="__main__":
    build(gen_ar_weak_sine, REPO/"results/iclr/e4_balanced3/windows","bal3")
    build(gen_per_ar, REPO/"results/iclr/e4_balanced3_natural/windows","bal3n")
    print("done")
