"""Unified frozen TS-foundation-model representation extractors."""
import sys, numpy as np, torch
from pathlib import Path
TIMESFM_SRC="/public/chenjiahui/SAFER-TS/code/baselines_external/timesfm/src"
UNI2TS_SRC="/public/chenjiahui/SAFER-TS/code/baselines_external/uni2ts/src"
if TIMESFM_SRC not in sys.path: sys.path.insert(0,TIMESFM_SRC)
if UNI2TS_SRC not in sys.path: sys.path.insert(0,UNI2TS_SRC)

# ---------------- TimesFM 2.5 ----------------
class TimesFMRep:
    """context: (B, L) standardized -> rep (B, d). last patch hidden."""
    def __init__(self, path, device="cuda:0", random_init=False, seed=0):
        from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch_module
        from timesfm.torch import util
        self.util=util
        if random_init:
            # architecture-native random: construct from config with default torch
            # initializers and NO pretrained weights (the module has no auto-load).
            torch.manual_seed(seed)
            self.m=TimesFM_2p5_200M_torch_module()
        else:
            self.m=TimesFM_2p5_200M_torch_module()
            self.m.load_checkpoint(str(path), torch_compile=False)
        self.m=self.m.to(device).eval(); self.p=32; self.device=device
    def embed(self, x):
        x=torch.from_numpy(x).float().to(self.device)
        B,L=x.shape
        mask=torch.zeros_like(x)
        patched=x.reshape(B,-1,self.p); pm=mask.reshape(B,-1,self.p)
        n=torch.zeros(B,device=self.device); mu=torch.zeros(B,device=self.device); sig=torch.zeros(B,device=self.device)
        mus=[]; sigs=[]
        for i in range(patched.shape[1]):
            (n,mu,sig),_=self.util.update_running_stats(n,mu,sig,patched[:,i],pm[:,i])
            mus.append(mu); sigs.append(sig)
        cmu=torch.stack(mus,1); csig=torch.stack(sigs,1)
        normed=self.util.revin(patched,cmu,csig,reverse=False)
        normed=torch.where(pm.bool(),0.0,normed)
        with torch.no_grad():
            out=self.m(normed,pm)[0]
        return out[1][:,-1,:].float().cpu().numpy()  # output_embeddings last patch

# ---------------- Moirai (MoiraiModule) ----------------
class MoiraiRep:
    def __init__(self, path, device="cuda:0", random_init=False, seed=0):
        from uni2ts.model.moirai.module import MoiraiModule
        self.mod=MoiraiModule.from_pretrained(str(path))
        self.mod=self.mod.to(device).eval()
        self.patch_sizes=list(self.mod.patch_sizes)
        self.p=16 if 16 in self.patch_sizes else self.patch_sizes[0]
        self.device=device
    def embed(self, x):
        import torch as T
        B,L=x.shape; p=int(self.p); n_patch=(L+p-1)//p
        xp=T.zeros(B,n_patch*p,device=self.device); xp[:,:L]=T.from_numpy(x).float().to(self.device)
        target=xp.reshape(B,n_patch,p)
        seq_len=n_patch
        obs=T.ones_like(target,dtype=T.bool)
        sid=T.zeros(B,seq_len,dtype=T.long,device=self.device)
        tid=T.arange(seq_len,dtype=T.long,device=self.device).unsqueeze(0).repeat(B,1)
        vid=T.zeros(B,seq_len,dtype=T.long,device=self.device)
        pmsk=T.zeros(B,seq_len,dtype=T.bool,device=self.device)
        psz=T.full((B,seq_len),p,dtype=T.long,device=self.device)
        loc,scale=self.mod.scaler(target,obs*~pmsk.unsqueeze(-1),sid,vid)
        scaled=(target-loc)/scale
        with T.no_grad():
            reprs=self.mod.in_proj(scaled,psz)
            reprs=self.mod.encoder(reprs,None,time_id=tid,var_id=vid)
        return reprs[:,-1,:].float().cpu().numpy()

def get_rep(name, ckpt, device, random_init=False, seed=0):
    if name.startswith("timesfm"):
        return TimesFMRep(ckpt, device, random_init, seed)
    elif name.startswith("moirai"):
        return MoiraiRep(ckpt, device, random_init, seed)
    raise ValueError(name)

