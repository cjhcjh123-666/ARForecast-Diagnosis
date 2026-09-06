"""ICLR TS-foundation-model suite deliverable runner (aligned protocol).

For each model x {pretrained, random} x seed {7,17,27}:
  A: 5-class structural recognition (synth5; original + per-window shuffled)
  B: frozen-rep -> linear H=16 readout MSE (synth5)
  C: 3-class balanced OOD routing (router fit on 270 clean trend/periodic/local
     windows, family labels, same protocol as E4 reference; tested on the two
     balanced sets bal3 / bal3n)
  native: model's own forecasting interface on the synth5 test windows
     (Chronos T5/Bolt, TimesFM; MOMENT = N/A needs learned head; qwen = N/A
     LM text generation is a separate interface studied in the paper)

Writes metrics.csv rows + per-window npz + config.json into --out.
Env-robust: qwen/chronos/timesfm/moment run under wavellm; moirai under chatts
(with jaxtyping shim + einops/dynamo workaround + package-__init__ stubs).
"""
from __future__ import annotations
import argparse, json, os, sys, types, importlib, math
from pathlib import Path
import numpy as np, torch, torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))

# ---------------------------------------------------------------- env shims
def _install_jaxtyping_shim_if_needed():
    try:
        import jax  # noqa: F401
        return
    except Exception:
        pass
    from typing import Any
    class _Fake:
        def __class_getitem__(cls, item): return Any
    class _Mod(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith('__') and name.endswith('__'):
                raise AttributeError(name)
            return _Fake
    _m = _Mod('jaxtyping')
    _m.__file__ = '<jaxtyping-shim>'
    sys.modules['jaxtyping'] = _m

def _skip_einops_dynamo():
    # einops>=0.7 imports torch._dynamo on first torch use (broken in chatts);
    # pretend torch>=2.8 so einops skips the registration entirely.
    try:
        import einops
        if tuple(int(x) for x in einops.__version__.split(".")[:2]) >= (0, 7):
            import torch as _t
            if tuple(int(x) for x in _t.__version__.split("+")[0].split(".")) < (2, 8):
                _t.__version__ = "2.8.0"
    except Exception:
        pass

def _register_moirai_pkg(uni2ts_src):
    """Skip uni2ts.model.moirai/__init__ (imports lightning -> broken)."""
    base = os.path.join(uni2ts_src, "uni2ts")
    if "uni2ts" in sys.modules and hasattr(sys.modules["uni2ts"], "__path__"):
        return  # already a real package
    def reg_pkg(name, path):
        m = types.ModuleType(name); m.__path__ = [path]; m.__package__ = name
        sys.modules[name] = m
    reg_pkg("uni2ts", base)
    reg_pkg("uni2ts.model", os.path.join(base, "model"))
    reg_pkg("uni2ts.model.moirai", os.path.join(base, "model", "moirai"))
    for sub in ["finetune", "forecast", "pretrain"]:
        sys.modules[f"uni2ts.model.moirai.{sub}"] = types.ModuleType(f"uni2ts.model.moirai.{sub}")

# ---------------------------------------------------------------- core data
from analysis.linear_probe import accuracy, softmax_predict, softmax_regression
from data.dynamics import build_labeled_windows
from models.experts import LocalExpert, PeriodicExpert, TrendExpert
KINDS = ["trend", "periodic", "local", "mixture", "regime"]
EXPERTS = [TrendExpert(), PeriodicExpert(), LocalExpert()]

def shuffled_windows(ctx: np.ndarray, seed: int) -> np.ndarray:
    out = np.empty_like(ctx)
    for i, w in enumerate(ctx):
        rng = np.random.default_rng(100000 * seed + i)
        out[i] = rng.permutation(w)
    return out

def randomize(model: nn.Module, seed: int):
    torch.manual_seed(seed)
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            try:
                m.reset_parameters()
            except Exception:
                pass

def recog_metrics3(pred, oracle):
    n = len(pred); cm = np.zeros((3, 3), int)
    for t, pp in zip(oracle, pred):
        if t < 3 and pp < 3:
            cm[t, pp] += 1
    rec = np.array([cm[i, i] / (cm[i].sum() + 1e-12) for i in range(3)])
    prec = np.array([cm[:, i][i] / (cm[:, i].sum() + 1e-12) if cm[:, i].sum() > 0 else 0 for i in range(3)])
    f1 = float(np.mean([2 * p * r / (p + r + 1e-12) for p, r in zip(prec, rec)]))
    return float(rec.mean()), f1, rec

# ---------------------------------------------------------------- extractors
def make_extractor(name, init, seed, device):
    """Frozen-representation extractor for one (model, init, seed).

    init == "pretrained": load the released checkpoint.
    init == "random": construct the SAME architecture from its config with
    native initializers, seed-controlled, WITHOUT loading any released weights
    (no pretrained residual; see init_audit/). Custom norms/projections that
    are initialized to their architectural constants are reported, not treated
    as pretrained leakage.
    """
    if name == "qwen3_8b_base":
        sys.path.insert(0, str(REPO))
        import transformers.modeling_utils as _mu
        if not isinstance(_mu.ALL_PARALLEL_STYLES, (set, frozenset)):
            _mu.ALL_PARALLEL_STYLES = frozenset({"tp","block","sharded","pp","sequence","rowwise","colwise","naive","serial","manual","flex","ddp"})
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        mpath = str(REPO / "models/qwen3-8b-base")
        tok = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        if init == "random":
            torch.manual_seed(seed)
            cfg = AutoConfig.from_pretrained(mpath, trust_remote_code=True)
            model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
            model = model.to(torch.bfloat16)
        else:
            devidx = int(device.split(":")[1]) if ":" in device else 0
            model = AutoModelForCausalLM.from_pretrained(
                mpath, trust_remote_code=True, torch_dtype=torch.bfloat16,
                attn_implementation="sdpa", device_map={"": devidx})
        model = model.to(device).eval()
        from scripts.run_frozen_probe import extract_last_hidden
        def embed(ctx, _m=model, _t=tok):
            return extract_last_hidden(_m, _t, ctx, device)
        return embed, 4096, None

    if name.startswith("chronos_t5") or name.startswith("chronos_bolt"):
        CH = str(REPO / "third_party/chronos-forecasting/src")
        if CH not in sys.path: sys.path.insert(0, CH)
        from transformers import AutoConfig, AutoModelForSeq2SeqLM
        from chronos import ChronosConfig, ChronosModel, ChronosPipeline, ChronosBoltPipeline
        tag = name.replace("chronos_", "")
        mpath = f"/public/chenjiahui/SAFER-TS/code/model_cache/amazon__chronos-{tag}"
        if name.startswith("chronos_t5"):
            if init == "random":
                cfg = AutoConfig.from_pretrained(mpath)
                cc = ChronosConfig(**cfg.chronos_config)
                torch.manual_seed(seed)
                inner = AutoModelForSeq2SeqLM.from_config(cfg)
                pipe = ChronosPipeline(tokenizer=cc.create_tokenizer(),
                                       model=ChronosModel(config=cc, model=inner))
            else:
                pipe = ChronosPipeline.from_pretrained(mpath)
            def embed(ctx, _p=pipe):
                out = []
                with torch.no_grad():
                    for st in range(0, len(ctx), 64):
                        b = torch.from_numpy(ctx[st:st+64]).float()
                        emb, _ = _p.embed(b)
                        out.append(emb[:, -1, :].numpy())
                return np.concatenate(out)
            dim = int(pipe.model.model.config.d_model)
        else:  # chronos_bolt
            from chronos.chronos_bolt import ChronosBoltModelForForecasting
            if init == "random":
                cfg = AutoConfig.from_pretrained(mpath)
                torch.manual_seed(seed)
                model = ChronosBoltModelForForecasting(cfg)
                pipe = ChronosBoltPipeline(model=model)
            else:
                pipe = ChronosBoltPipeline.from_pretrained(mpath)
            def embed(ctx, _p=pipe):
                out = []
                with torch.no_grad():
                    for st in range(0, len(ctx), 64):
                        b = torch.from_numpy(ctx[st:st+64]).float()
                        emb, _ = _p.embed(b)
                        out.append(emb[:, -1, :].numpy())
                return np.concatenate(out)
            dim = int(pipe.model.config.d_model)
        pipe.model = pipe.model.to(device).eval()
        def native(ctx, _p=pipe):
            torch.manual_seed(seed)   # deterministic sampling for reproducible random-native rows
            b = torch.from_numpy(ctx).float()
            outs = []
            with torch.no_grad():
                for st in range(0, len(b), 16):
                    chunk = b[st:st+16]
                    if name.startswith("chronos_t5"):
                        fc = _p.predict(chunk, prediction_length=16, num_samples=20)
                        fc = fc.median(dim=1).values.numpy()
                    else:
                        fc = _p.predict(chunk, prediction_length=16)
                        fc = fc[:, _p.quantiles.index(0.5), :].numpy()
                    outs.append(fc)
            return np.concatenate(outs)
        return embed, dim, native

    if name.startswith("timesfm"):
        sys.path.insert(0, '/public/chenjiahui/SAFER-TS/code/baselines_external/timesfm/src')
        from scripts.tsfm_models import TimesFMRep
        # TimesFMRep(random_init=True) now constructs from config WITHOUT loading weights
        r = TimesFMRep("/public/chenjiahui/SAFER-TS/code/model_cache/google__timesfm-2.5-200m-pytorch/model.safetensors",
                       device, init == "random", seed)
        r.m.device = torch.device(device); r.m = r.m.to(device)
        def native(ctx):
            import math as _m
            p_, o_, q_, aridx_ = r.m.p, r.m.o, r.m.q, r.m.aridx
            util = r.util
            B = len(ctx); x = torch.from_numpy(np.asarray(ctx, dtype=np.float64)).float()
            npatch = x.shape[1] // p_
            x = x.to(device).reshape(B, -1, p_)
            masks = torch.zeros_like(x, dtype=torch.bool)
            n = torch.zeros(B, device=device); mu = torch.zeros(B, device=device); sig = torch.zeros(B, device=device)
            p_mu, p_sig = [], []
            for i in range(npatch):
                (n, mu, sig), _ = util.update_running_stats(n, mu, sig, x[:, i], masks[:, i])
                p_mu.append(mu); p_sig.append(sig)
            cmu = torch.stack(p_mu, 1); csig = torch.stack(p_sig, 1)
            normed = util.revin(x, cmu, csig, reverse=False)
            normed = torch.where(masks, 0.0, normed)
            with torch.no_grad():
                (_, _, out_ts, _), _ = r.m(normed, masks)
                renormed = util.revin(out_ts, cmu, csig, reverse=True).reshape(B, -1, o_, q_)
            return renormed[:, -1, :16, aridx_].float().cpu().numpy()
        return r.embed, 1280, native

    if name.startswith("moment"):
        sys.path.insert(0, '/public/chenjiahui/波数据时序基座大模型.bak_public/moment-main/Inference')
        import json as _json
        from momentfm.models.moment import MOMENT
        wdir = '/public/chenjiahui/波数据时序基座大模型.bak_public/WaveFormer/models/external/moment'
        cfg = _json.load(open(f"{wdir}/config.json")); cfg['d_model'] = cfg['t5_config']['d_model']
        if init == "random":
            torch.manual_seed(seed)
            m = MOMENT(cfg, model_kwargs={})          # arch-native init, no pretrained load
        else:
            m = MOMENT(cfg, model_kwargs={})
            sd = torch.load(f"{wdir}/pytorch_model.bin", map_location="cpu", weights_only=False)
            m.load_state_dict(sd, strict=False)
        m = m.to(device).eval()
        def embed(ctx):
            x = torch.from_numpy(ctx).float().unsqueeze(1).to(device)
            out = []
            with torch.no_grad():
                for st in range(0, len(ctx), 32):
                    o = m.embed(x_enc=x[st:st+32]); out.append(o.embeddings.cpu().numpy())
            return np.concatenate(out)
        return embed, cfg['d_model'], None

    if name.startswith("moirai"):
        _install_jaxtyping_shim_if_needed()
        _skip_einops_dynamo()
        UNI2TS = "/public/chenjiahui/SAFER-TS/code/baselines_external/uni2ts/src"
        if UNI2TS not in sys.path: sys.path.insert(0, UNI2TS)
        _register_moirai_pkg(UNI2TS)
        from uni2ts.model.moirai.module import MoiraiModule, decode_distr_output
        import json as _json
        ckpt_dir = "/public/chenjiahui/SAFER-TS/code/model_cache/Salesforce__moirai-1.1-R-small"
        cfgd = _json.load(open(f"{ckpt_dir}/config.json"))
        if init == "random":
            torch.manual_seed(seed)
            distr = decode_distr_output(cfgd["distr_output"])
            mod = MoiraiModule(distr_output=distr, d_model=cfgd["d_model"],
                               num_layers=cfgd["num_layers"], patch_sizes=cfgd["patch_sizes"],
                               max_seq_len=cfgd["max_seq_len"], attn_dropout_p=cfgd["attn_dropout_p"],
                               dropout_p=cfgd["dropout_p"], scaling=cfgd["scaling"])
        else:
            mod = MoiraiModule.from_pretrained(ckpt_dir)
        mod = mod.to(device).eval()
        p = 16
        max_patch = max(mod.patch_sizes)
        def _ctx_pack(x, device):
            B, L = x.shape
            np_ = int(math.ceil(L / p))
            xp = torch.zeros(B, np_ * p, device=device)
            xp[:, :L] = torch.from_numpy(x).float().to(device)
            seq = xp.reshape(B, np_, p)
            om = torch.ones_like(seq, dtype=torch.bool)
            sid = torch.arange(B, device=device).unsqueeze(1).repeat(1, np_)
            tid = torch.arange(np_, device=device).unsqueeze(0).repeat(B, 1)
            vid = torch.zeros(B, np_, dtype=torch.long, device=device)
            pm = torch.zeros(B, np_, dtype=torch.bool, device=device)
            psz = torch.full((B, np_), p, dtype=torch.long, device=device)
            target = torch.nn.functional.pad(seq, (0, max_patch - p))
            obs = torch.nn.functional.pad(om, (0, max_patch - p))
            return target, obs, sid, tid, vid, pm, psz
        def embed(ctx):
            B = len(ctx); out = []
            for st in range(0, B, 32):
                target, obs, sid, tid, vid, pm, psz = _ctx_pack(ctx[st:st+32], device)
                with torch.no_grad():
                    loc, scale = mod.scaler(target, obs * ~pm.unsqueeze(-1), sid, vid)
                    scaled = (target - loc) / scale
                    reprs = mod.in_proj(scaled, psz)
                    reprs = mod.encoder(reprs, None, time_id=tid, var_id=vid)
                out.append(reprs[:, -1, :].float().cpu().numpy())
            return np.concatenate(out)
        def native(ctx):
            H = 16; nf = int(math.ceil(H / p)); B = len(ctx)
            outs = []
            for st in range(0, B, 16):
                batch = ctx[st:st+16]; b = len(batch)
                xp = torch.zeros(b, (4 + nf) * p, device=device)
                xp[:, :64] = torch.from_numpy(np.asarray(batch)).float().to(device)
                target = xp.reshape(b, 4 + nf, p)
                om = torch.ones_like(target, dtype=torch.bool)
                sid = torch.arange(b, device=device).unsqueeze(1).repeat(1, 4 + nf)
                tid = torch.arange(4 + nf, device=device).unsqueeze(0).repeat(b, 1)
                vid = torch.zeros(b, 4 + nf, dtype=torch.long, device=device)
                pm = torch.zeros(b, 4 + nf, dtype=torch.bool, device=device); pm[:, 4:] = True
                psz = torch.full((b, 4 + nf), p, dtype=torch.long, device=device)
                target = torch.nn.functional.pad(target, (0, max_patch - p))
                obs = torch.nn.functional.pad(om, (0, max_patch - p))
                with torch.no_grad():
                    distr = mod(target, obs, sid, tid, vid, pm, psz)
                    samples = distr.sample(torch.Size([50]))
                fut = samples[:, :, 4:, :p]
                fut = fut.permute(1, 0, 2, 3).reshape(b, 50, nf * p)[:, :, :H]
                outs.append(fut.median(dim=1).values.cpu().numpy())
            return np.concatenate(outs)
        return embed, mod.d_model, native
    raise ValueError(name)

# ---------------------------------------------------------------- readout / router
def lin_readout(Xtr, Ytr, Xte, seed):
    torch.manual_seed(seed)
    net = nn.Linear(Xtr.shape[1], 16)
    Xt = torch.from_numpy(Xtr).float(); Yt = torch.from_numpy(Ytr).float()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01)  # explicit; matches executed default
    for _ in range(300):
        opt.zero_grad(); l = nn.functional.mse_loss(net(Xt), Yt); l.backward(); opt.step()
    with torch.no_grad():
        pred = net(torch.from_numpy(Xte).float()).numpy()
    return pred, Xtr.shape[1] * 16 + 16

def expert_errors(ctx, fut):
    err = np.zeros((len(ctx), 3))
    for j, e in enumerate(EXPERTS):
        for i in range(len(ctx)):
            err[i, j] = float(np.mean((e.predict(ctx[i], 16) - fut[i]) ** 2))
    return err

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="moirai-1.1-R-small")
    ap.add_argument("--seeds", default="7,17,27")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, default=Path("results/iclr/tsfm_deliver"))
    ap.add_argument("--only-tasks", default="A,B,C,native")
    ap.add_argument("--inits", default="pretrained,random")
    a = ap.parse_args()
    a.inits = [x.strip() for x in a.inits.split(",") if x.strip()]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "perwindow").mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in a.seeds.split(",")]
    tasks = set(a.only_tasks.split(","))
    rows = []
    config_records = {}
    fields = ["model", "init", "seed", "task", "test_set", "n_test", "accuracy",
              "balanced_accuracy", "macro_f1", "recall_trend", "recall_periodic",
              "recall_local", "mse", "head_params"]

    for name in a.models.split(","):
        for seed in seeds:
            # shared windows
            n, ntr = 150, 90
            ctx, fut, kk = build_labeled_windows(KINDS, 64, 16, n, seed)
            tr_idx = np.concatenate([np.arange(k * n, k * n + ntr) for k in range(5)])
            te_idx = np.concatenate([np.arange(k * n + ntr, (k + 1) * n) for k in range(5)])
            tr_lab, te_lab = kk[tr_idx], kk[te_idx]
            clean270 = np.concatenate([np.arange(k * n, k * n + ntr) for k in range(3)])
            # balanced OOD windows
            bal = {}
            for tag, root in [("bal3", f"results/iclr/e4_balanced3/windows/bal3_s{seed}.npz"),
                              ("bal3n", f"results/iclr/e4_balanced3_natural/windows/bal3n_s{seed}.npz")]:
                if Path(root).is_file():
                    z = np.load(root); bal[tag] = (z["ctx"], z["fut"], z["oracle"])
            for init in a.inits:
                rec = {"model": name, "init": init, "seed": seed}
                try:
                    embed, dim, native = make_extractor(name, init, seed, a.device)
                except Exception as e:
                    print(f"[{name}/{init}/s{seed}] extractor FAIL {type(e).__name__}: {e}", flush=True)
                    continue
                H = embed(ctx)
                pw = {"model": name, "init": init, "seed": seed}
                # ---------- A ----------
                if "A" in tasks:
                  try:
                    for cond, Hc, lbl in [("orig", H[te_idx], te_lab), ]:
                        _, pA = softmax_predict(Hc, *softmax_regression(H[tr_idx], tr_lab, n_iter=2000))
                        rows.append({**rec, "task": f"A_recog_{cond}", "test_set": "synth5",
                                     "n_test": len(pA), "accuracy": round(accuracy(pA, lbl), 4)})
                        pw["a_true"] = lbl; pw["a_pred"] = pA
                    ctx_shuf = shuffled_windows(ctx, seed)
                    Hs = embed(ctx_shuf)
                    _, pAs = softmax_predict(Hs[te_idx], *softmax_regression(Hs[tr_idx], tr_lab, n_iter=2000))
                    rows.append({**rec, "task": "A_recog_shuf", "test_set": "synth5",
                                 "n_test": len(pAs), "accuracy": round(accuracy(pAs, te_lab), 4)})
                    pw["as_true"] = te_lab; pw["as_pred"] = pAs
                  except Exception as e:
                    print(f"[{name}/{init}/s{seed}] A FAIL {type(e).__name__}: {e}", flush=True)
                # ---------- B ----------
                if "B" in tasks:
                  try:
                    pred_b, hp = lin_readout(H[tr_idx], fut[tr_idx], H[te_idx], seed)
                    mse_b = float(np.mean((pred_b - fut[te_idx]) ** 2))
                    rows.append({**rec, "task": "B_readout", "test_set": "synth5",
                                 "n_test": len(te_idx), "mse": round(mse_b, 4), "head_params": hp})
                    pw["b_future"] = fut[te_idx]; pw["b_pred"] = pred_b
                  except Exception as e:
                    print(f"[{name}/{init}/s{seed}] B FAIL {type(e).__name__}: {e}", flush=True)
                # ---------- C ----------
                if "C" in tasks:
                  try:
                    for tag in sorted(bal):
                        bctx, bfut, bor = bal[tag]
                        Hb = embed(bctx)
                        _, rp = softmax_predict(Hb, *softmax_regression(H[clean270], kk[clean270], n_iter=2000))
                        err = expert_errors(bctx, bfut)
                        ba, mf1, rec3 = recog_metrics3(rp, bor)
                        mse_c = float(np.mean(err[np.arange(len(rp)), rp]))
                        rows.append({**rec, "task": "C_routing", "test_set": f"{tag}_s{seed}",
                                     "n_test": len(rp), "balanced_accuracy": round(ba, 4),
                                     "macro_f1": round(mf1, 4), "recall_trend": round(rec3[0], 4),
                                     "recall_periodic": round(rec3[1], 4), "recall_local": round(rec3[2], 4),
                                     "mse": round(mse_c, 4)})
                        pw[f"c_{tag}_oracle"] = bor; pw[f"c_{tag}_pred"] = rp; pw[f"c_{tag}_err"] = err
                  except Exception as e:
                    print(f"[{name}/{init}/s{seed}] C FAIL {type(e).__name__}: {e}", flush=True)
                # ---------- native ----------
                if "native" in tasks and native is not None:
                    try:
                        fc = native(ctx[te_idx])
                        mse_n = float(np.mean((fc - fut[te_idx]) ** 2))
                        rows.append({**rec, "task": "native_forecast", "test_set": "synth5",
                                     "n_test": len(te_idx), "mse": round(mse_n, 4)})
                        pw["n_future"] = fut[te_idx]; pw["n_pred"] = fc
                        print(f"[{name}/{init}/s{seed}] native mse={mse_n:.4f}", flush=True)
                    except Exception as ne:
                        print(f"[{name}/{init}/s{seed}] native FAIL {type(ne).__name__}: {ne}", flush=True)
                print(f"[{name}/{init}/s{seed}] done A/B/C dim={dim}", flush=True)
                np.savez(a.out / "perwindow" / f"pw_{name}_{init}_s{seed}.npz", **pw)
                config_records[f"{name}|{init}|{seed}"] = {"dim": dim, "n_pw_windows": 750}
        print(f"[{name}] seed loop finished", flush=True)

    # merge into existing metrics.csv if present
    csv_path = a.out / "metrics.csv"
    old_rows = []
    if csv_path.is_file():
        import csv as _csv
        with open(csv_path) as f:
            old_rows = list(_csv.DictReader(f))
    all_rows = old_rows + rows
    import csv
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in fields})
    # config.json
    cfg_path = a.out / "config.json"
    cfg = {}
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text())
    cfg["generated_by"] = "scripts/iclr_tsfm_deliver.py"
    cfg["protocol"] = {
        "context": 64, "horizon": 16, "seeds": [7, 17, 27],
        "synth5": {"n_per_kind": 150, "train_per_kind": 90, "test_per_kind": 60,
                   "kinds": ["trend", "periodic", "local", "mixture", "regime"]},
        "A_recognition": "5-class family probe, softmax full-batch 2000 steps (analysis.linear_probe), "
                         "frozen backbone; shuffled = per-window independent time permutation (seed 100000*s+i)",
        "B_readout": "linear head H=16 on frozen rep, AdamW lr=1e-3 weight_decay=0.01, full-batch 300 epochs",
        "C_routing": "router = softmax probe on 270 clean trend/periodic/local windows (family labels), "
                     "test = e4_balanced3 (bal3) and e4_balanced3_natural (bal3n), oracle = argmin expert future MSE",
        "native": "model-native forecasting interface; point prediction = median (chronos 20 samples / bolt 0.5 quantile / timesfm decode_index 5 / moirai 50 samples median)",
    }
    cfg["model_records"] = config_records
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"saved metrics rows={len(all_rows)} to {csv_path}; config to {cfg_path}")

if __name__ == "__main__":
    main()
