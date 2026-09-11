# Open-LLM Suite — Failures & Blockers (rolling, 2026-09-11)

| # | item | stage | exception / evidence | attempted fix | status |
|---|---|---|---|---|---|
| 1 | huggingface.co direct | download | DNS→31.13.96.194 timeout; proxy 127.0.0.1:7890 CONNECT 200 but TLS `UNEXPECTED_EOF` | switched to `HF_ENDPOINT=https://hf-mirror.com` with all proxies unset; verified HTTP 200 + snapshot download | RESOLVED |
| 2 | Llama-3.1-8B / Llama-3.2-3B / Gemma-2-9B | download | gated repos; `HF_TOKEN` unset, no token file | none (no non-official mirror substitution allowed) | **BLOCKED_GATED_ACCESS — needs user HF_TOKEN** |
| 3 | `hf_transfer` acceleration | download | all 4 jobs silently die after printing resolved sha | reverted to standard HTTP; `aria2c -x16` also ERR against mirror | RESOLVED (slower) |
| 4 | transformers `ALL_PARALLEL_STYLES=None` | load | Qwen3 from-config/from_pretrained raised `TypeError: argument of type 'NoneType' is not iterable` (wavellm/transformers 4.52) | patch module-level frozenset in `openllm_suite.load_lm` (same as `run_frozen_probe`) | RESOLVED |
| 5 | Qwen3-8B full suite (P+R) | run | process killed with no traceback after 2× shard loading (suspected host-RAM peak during 8B from-config fp32 construction) | split `--inits pretrained` / `random`; pretrained pass relaunched; random pass will use a memory-capped construction | MITIGATING |
| 6 | `api.txt` key location | API | key was in repo root (untracked) | added `api.txt` to `.gitignore`; key only read via `AIGCBEST_API_KEY` fallback for local runs | RESOLVED |

All negative results and killed runs are kept (logs/ + this file); nothing was silently dropped.

## Update 2026-09-11 (after user provided HF token)

- Token `hf_***` verified: `whoami` = **five6667 (user)**; `model_info` succeeds for
  `meta-llama/Llama-3.1-8B` (sha `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`),
  `meta-llama/Llama-3.2-3B` (`13afe5124825b4f3751f836b40dafda64c1ed062`),
  `google/gemma-2-9b` (`33c193028431c2fde6c6e51f29e6f17b60cbfac6`), `google/gemma-2-2b`.
- **However hf-mirror.com does not proxy gated files**: download fails with
  "Access to model ... is restricted and you are not in the authorized list" even with the token in env.
  HuggingFace direct remains unreachable (TLS EOF / DNS poisoning). So the token alone does not unblock the
  weights on this cluster's network path.
- Fallback in use: **ModelScope official-org mirrors** (`LLM-Research/Meta-Llama-3.1-8B`,
  `LLM-Research/Llama-3.2-3B`, `LLM-Research/gemma-2-9b`, `LLM-Research/gemma-2-2b`) via
  `scripts/openllm_download_modelscope.py`, provenance recorded in `results/open_llm_suite/raw/modelscope_*.json`
  (source=ModelScope, sha256 of config/index). **Measured throughput ≈0.18–0.24 MB/s** → ~16–19 GB per model
  ⇒ tens of hours; `aria2c`/multi-connection is rejected by the ModelScope endpoint.
- Practical unblock options (pick one):
  1. provide a network path to huggingface.co (working proxy/VPN endpoint), or
  2. pre-stage the four gated checkpoints into a cluster path we can read, or
  3. accept the slow ModelScope transfer (runs in background; ~1–2 days for all four).
- Public P0/P1 models (DeepSeek-LLM-7B, DeepSeek-V2-Lite, Mistral-7B-v0.3, OLMo-2-7B, OLMo-2-13B) continue
  at ≈7.5 MB/s aggregate via hf-mirror and will be run first.
