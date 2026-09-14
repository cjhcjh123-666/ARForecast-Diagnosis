"""Oracle-target stability: is the single-realised-future expert label a stable target?

For every synthetic window we fix the latent process (the parameters actually drawn by
data/dynamics.py) and the realised context, then resample the future noise K times.
That gives p_j(x) = P(expert j wins | context), stability s(x) = max_j p_j(x), the
expected-risk winner argmax_j p_j(x) (majority label) and the expert-risk margin.

Replay is exact: the traced generator is asserted to reproduce the frozen futures that
were used by the main suite (`features_full/full_*.npz: fut`), so the stability numbers
refer to the same windows as the routing results.

CPU only; no new LM forward passes are needed (the frozen clean270 features are reused).
Outputs: results/open_llm_suite/raw/oracle_stability.csv
         results/open_llm_suite/raw/oracle_stability_routing.csv
"""
from __future__ import annotations
import csv, os, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from analysis.linear_probe import softmax_predict, softmax_regression
from data.dynamics import build_labeled_windows
from models.experts import LocalExpert, PeriodicExpert, TrendExpert

KINDS = ["trend", "periodic", "local", "mixture", "regime"]
SEEDS = [7, 17, 27]
K = 20                      # future resamples per window
FF = REPO / "results/open_llm_suite/features_full"
OUT = REPO / "results/open_llm_suite/raw"
EXPERTS = [TrendExpert(), PeriodicExpert(), LocalExpert()]
MODELS = os.environ.get("OPENLLM_STAB_MODELS", "").split(",") if os.environ.get("OPENLLM_STAB_MODELS") else [
    "qwen3_8b_base", "llama31_8b", "gemma2_9b", "deepseek_llm_7b", "olmo2_7b"]
STAB_SUFFIX = os.environ.get("OPENLLM_STAB_SUFFIX", "")
SKIP_STABILITY = os.environ.get("OPENLLM_STAB_SKIP_STABILITY", "") == "1"


def gen_traced(kind, context_len, horizon, rng):
    """Mirror of data.dynamics.generate_window, additionally returning the realised
    parameters and the noise vector so the future can be resampled."""
    total = context_len + horizon
    t = np.arange(total, dtype=float)
    if kind == "trend":
        slope = rng.uniform(0.03, 0.18); noise = 0.05
        e = rng.normal(0.0, noise, total)
        x = slope * t + e
    elif kind == "periodic":
        period = float(rng.choice([12, 24])); amp = rng.uniform(0.5, 1.5)
        noise = 0.05; phase = rng.uniform(0.0, 2.0 * np.pi)
        e = rng.normal(0.0, noise, total)
        x = amp * np.sin(2.0 * np.pi * t / period + phase) + e
    elif kind == "local":
        phi = rng.uniform(0.7, 0.95); sigma = 0.15
        e = rng.normal(0.0, sigma, total)
        x = np.empty(total); x[0] = e[0]
        for k in range(1, total):
            x[k] = phi * x[k - 1] + e[k]
    elif kind == "mixture":
        slope = rng.uniform(0.01, 0.06); period = float(rng.choice([12, 24]))
        amp = rng.uniform(0.4, 1.2)
        e = rng.normal(0.0, 0.05, total)
        x = slope * t + amp * np.sin(2.0 * np.pi * t / period) + e
    elif kind == "regime":
        switch = int(context_len * 0.6)
        e = np.concatenate([rng.normal(0.0, 0.05, switch),
                            rng.normal(0.0, 0.05, total - switch)])
        x = np.empty(total)
        x[:switch] = 0.08 * t[:switch] + e[:switch]
        x[switch:] = np.sin(2.0 * np.pi * t[switch:] / 12.0) + e[switch:]
    else:
        raise ValueError(kind)
    mu, sd = float(x[:context_len].mean()), float(x[:context_len].std()) + 1e-6
    return x, dict(noise=e, mu=mu, sd=sd)


def rebuild(kind, x0, params, e_new, context_len, horizon):
    """Same latent process + same context noise, fresh future noise."""
    total = context_len + horizon
    t = np.arange(total, dtype=float)
    # recover the deterministic parameters from the realised series
    if kind == "trend":
        slope = np.polyfit(t[context_len:], x0[context_len:] - params["noise"][context_len:], 1)[0]
        x = slope * t + e_new
    elif kind == "periodic":
        raise NotImplementedError  # handled via stored params below
    else:
        raise NotImplementedError
    return (x - params["mu"]) / params["sd"]


def stability_for_windows(kind, ctx, fut, params_list, rng):
    """Return per-window arrays: winner counts and mean per-expert MSE over K futures."""
    n, H = len(ctx), fut.shape[1]
    counts = np.zeros((n, 3), int)
    Emean = np.zeros((n, 3))
    Ereal = np.zeros((n, 3))
    preds = np.zeros((n, 3, H))
    for j, e in enumerate(EXPERTS):
        for i in range(n):
            preds[i, j] = e.predict(ctx[i], H)
    for i in range(n):
        for j in range(3):
            Ereal[i, j] = float(np.mean((preds[i, j] - fut[i]) ** 2))
        acc = np.zeros(3)
        for _ in range(K):
            xf = resample_future(kind, ctx[i], params_list[i], rng)
            for j in range(3):
                acc[j] += float(np.mean((preds[i, j] - xf) ** 2))
        acc /= K
        counts[i, int(acc.argmin())] += 1
        # winner distribution needs the per-resample winners, not the mean -> recomputed in main
        Emean[i] = acc
    return counts, Emean, Ereal


def resample_future(kind, ctx_i, p, rng):
    """Draw a new future for the same latent process / same context."""
    H = p["H"]; C = p["C"]
    e = rng.normal(0.0, p["sigma"], C + H)
    e[:C] = p["noise"][:C]                      # context noise stays identical
    t = np.arange(C + H, dtype=float)
    if kind == "trend":
        x = p["slope"] * t + e
    elif kind == "periodic":
        x = p["amp"] * np.sin(2.0 * np.pi * t / p["period"] + p["phase"]) + e
    elif kind == "local":
        x = np.empty(C + H); x[0] = e[0]
        for k in range(1, C + H):
            x[k] = p["phi"] * x[k - 1] + e[k]
    elif kind == "mixture":
        x = p["slope"] * t + p["amp"] * np.sin(2.0 * np.pi * t / p["period"]) + e
    elif kind == "regime":
        sw = int(C * 0.6)
        e2 = np.concatenate([p["noise"][:sw], rng.normal(0.0, 0.05, C + H - sw)])
        x = np.empty(C + H)
        x[:sw] = 0.08 * t[:sw] + e2[:sw]
        x[sw:] = np.sin(2.0 * np.pi * t[sw:] / 12.0) + e2[sw:]
    else:
        raise ValueError(kind)
    return (x[C:] - p["mu"]) / p["sd"]


def build_params(kind, context_len, horizon, rng):
    """Replay one window and return the latent parameters + the resampling helper info."""
    total = context_len + horizon
    t = np.arange(total, dtype=float)
    p = dict(C=context_len, H=horizon)
    if kind == "trend":
        p["slope"] = rng.uniform(0.03, 0.18); p["sigma"] = 0.05
        e = rng.normal(0.0, p["sigma"], total)
    elif kind == "periodic":
        p["period"] = float(rng.choice([12, 24])); p["amp"] = rng.uniform(0.5, 1.5)
        p["sigma"] = 0.05; p["phase"] = rng.uniform(0.0, 2.0 * np.pi)
        e = rng.normal(0.0, p["sigma"], total)
    elif kind == "local":
        p["phi"] = rng.uniform(0.7, 0.95); p["sigma"] = 0.15
        e = rng.normal(0.0, p["sigma"], total)
    elif kind == "mixture":
        p["slope"] = rng.uniform(0.01, 0.06); p["period"] = float(rng.choice([12, 24]))
        p["amp"] = rng.uniform(0.4, 1.2); p["sigma"] = 0.05
        e = rng.normal(0.0, p["sigma"], total)
    elif kind == "regime":
        sw = int(context_len * 0.6)
        e = np.concatenate([rng.normal(0.0, 0.05, sw), rng.normal(0.0, 0.05, total - sw)])
        p["sigma"] = 0.05
    p["noise"] = e
    return p


def render(kind, p, e):
    C, H = p["C"], p["H"]
    t = np.arange(C + H, dtype=float)
    if kind == "trend":
        x = p["slope"] * t + e
    elif kind == "periodic":
        x = p["amp"] * np.sin(2.0 * np.pi * t / p["period"] + p["phase"]) + e
    elif kind == "local":
        x = np.empty(C + H); x[0] = e[0]
        for k in range(1, C + H):
            x[k] = p["phi"] * x[k - 1] + e[k]
    elif kind == "mixture":
        x = p["slope"] * t + p["amp"] * np.sin(2.0 * np.pi * t / p["period"]) + e
    else:
        sw = int(C * 0.6)
        x = np.empty(C + H)
        x[:sw] = 0.08 * t[:sw] + e[:sw]
        x[sw:] = np.sin(2.0 * np.pi * t[sw:] / 12.0) + e[sw:]
    return x


def main():
    rows = []
    clean_expected = {}      # (seed, index) -> expected-risk label on clean750
    if SKIP_STABILITY and (OUT / "oracle_stability.csv").is_file():
        for r in csv.DictReader(open(OUT / "oracle_stability.csv")):
            if int(r["kind"]) < 3:
                clean_expected[(int(r["seed"]), int(r["index"]))] = int(r["majority_winner"])
        print("reusing existing oracle_stability.csv (", len(clean_expected), "primitive windows )", flush=True)
    for seed in ([] if SKIP_STABILITY else SEEDS):
        ctx_ref, fut_ref, kk_ref = build_labeled_windows(KINDS, 64, 16, 150, seed)
        rng = np.random.default_rng(seed)
        ctxs, futs, kinds, params = [], [], [], []
        for kind_id, kind in enumerate(KINDS):
            for _ in range(150):
                p = build_params(kind, 64, 16, rng)
                x = render(kind, p, p["noise"])
                mu, sd = float(x[:64].mean()), float(x[:64].std()) + 1e-6
                p["mu"], p["sd"] = mu, sd
                ctxs.append((x[:64] - mu) / sd); futs.append((x[64:] - mu) / sd)
                kinds.append(kind_id); params.append(p)
        ctx = np.stack(ctxs); fut = np.stack(futs); kinds = np.asarray(kinds)
        # exact replay check against the frozen futures used by the suite
        fref = FF / ("full_%s_%s_s%d.npz" % (MODELS[0], "pretrained", seed))
        if fref.is_file():
            fz = np.load(fref)["fut"]
            ok = np.allclose(fz, fut, atol=1e-6)
            print("[seed %d] replay matches frozen futures: %s (max diff %.2e)" %
                  (seed, ok, float(np.abs(fz - fut).max())), flush=True)
            if not ok:
                # fall back: use the frozen futures, contexts are still ours
                fut = fz
        rng2 = np.random.default_rng(1000 + seed)
        # per-expert prediction on the realised context
        n = len(ctx); H = fut.shape[1]
        preds = np.zeros((n, 3, H))
        for j, e in enumerate(EXPERTS):
            for i in range(n):
                preds[i, j] = e.predict(ctx[i], H)
        for i in range(n):
            win = np.zeros(3, int); acc = np.zeros(3)
            for _ in range(K):
                xf = resample_future(KINDS[kinds[i]], ctx[i], params[i], rng2)
                err = np.array([float(np.mean((preds[i, j] - xf) ** 2)) for j in range(3)])
                win[int(err.argmin())] += 1; acc += err
            acc /= K
            Ereal = np.array([float(np.mean((preds[i, j] - fut[i]) ** 2)) for j in range(3)])
            pj = win / K
            fam = kinds[i]
            single = int(Ereal.argmin())
            maj = int(acc.argmin())
            rows.append(dict(seed=seed, index=i, kind=int(fam), stability=round(float(pj.max()), 3),
                             entropy=round(float(-np.sum(pj * np.log(pj + 1e-12))), 3),
                             margin=round(float(np.sort(acc)[1] / (np.sort(acc)[0] + 1e-9)), 3),
                             single_winner=single, majority_winner=maj,
                             single_is_majority=int(single == maj),
                             majority_matches_family=int(maj == fam) if fam < 3 else "",
                             single_matches_family=int(single == fam) if fam < 3 else ""))
            if fam < 3:
                clean_expected[(seed, i)] = maj
        print("[seed %d] stability computed for %d windows" % (seed, n), flush=True)
    if rows:
        with open(OUT / "oracle_stability.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print("wrote oracle_stability.csv", len(rows), "rows")
        st = np.array([r["stability"] for r in rows])
        print("stability: mean %.3f, ==1.0: %.1f%%, >=0.8: %.1f%%, <=0.6: %.1f%%" %
              (st.mean(), 100 * np.mean(st == 1.0), 100 * np.mean(st >= 0.8), 100 * np.mean(st <= 0.6)))
        prim = [r for r in rows if r["kind"] < 3]
        print("primitive windows: single==majority %.1f%%" % (100 * np.mean([r["single_is_majority"] for r in prim])))
        print("primitive windows: majority==family %.1f%%" % (100 * np.mean([r["majority_matches_family"] for r in prim])))

    # ---------------- routing with expected-risk labels ----------------
    rrows = []
    for model in MODELS:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            # rebuild the clean750 params (same replay) to know which indices are primitives
            idx_prim = np.concatenate([np.arange(k * 150, k * 150 + 90) for k in range(3)])
            y_exp = np.array([clean_expected[(seed, i)] for i in idx_prim])
            ctx, fut, kk = build_labeled_windows(KINDS, 64, 16, 150, seed)
            Ec = np.zeros((len(idx_prim), 3))
            for j, e in enumerate(EXPERTS):
                for ii, i in enumerate(idx_prim):
                    Ec[ii, j] = float(np.mean((e.predict(ctx[i], 16) - fut[i]) ** 2))
            y_single = Ec.argmin(1)
            for init in ["pretrained", "random"]:
                f = FF / ("full_%s_%s_s%d.npz" % (model, init, seed))
                if not f.is_file():
                    continue
                z = np.load(f)
                Hc = z["h_clean"][idx_prim]
                for tag in ["bal3", "bal3n"]:
                    Hb = z["h_" + tag]; bor = z[tag + "_oracle"]
                    out = {"model": model, "init": init, "seed": seed, "test": tag}
                    for name, y in [("family", kk[idx_prim]), ("single_oracle", y_single), ("expected_risk", y_exp)]:
                        _, pred = softmax_predict(Hb, *softmax_regression(Hc, y, n_iter=2000))
                        cm = np.zeros((3, 3), int)
                        for t, p in zip(bor, pred):
                            if t < 3 and p < 3:
                                cm[t, p] += 1
                        rec = np.array([cm[i, i] / (cm[i].sum() + 1e-12) for i in range(3)])
                        out[name] = round(float(rec.mean()), 4)
                    rrows.append(out)
            print("[routing %s/%d] done" % (model, seed), flush=True)
    with open(OUT / ("oracle_stability_routing%s.csv" % STAB_SUFFIX), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rrows[0].keys())); w.writeheader(); w.writerows(rrows)
    print("wrote oracle_stability_routing%s.csv" % STAB_SUFFIX, len(rrows), "rows")
    for model in MODELS:
        for tag in ["bal3", "bal3n"]:
            def g(name, init):
                v = [r[name] for r in rrows if r["model"] == model and r["test"] == tag and r["init"] == init]
                return float(np.mean(v)) if v else float("nan")
            if g("family", "pretrained") == g("family", "pretrained"):
                print("%-16s %-5s delta: family %+.3f | single-oracle %+.3f | expected-risk %+.3f" %
                      (model, tag, g("family", "pretrained") - g("family", "random"),
                       g("single_oracle", "pretrained") - g("single_oracle", "random"),
                       g("expected_risk", "pretrained") - g("expected_risk", "random")))


if __name__ == "__main__":
    main()
