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
