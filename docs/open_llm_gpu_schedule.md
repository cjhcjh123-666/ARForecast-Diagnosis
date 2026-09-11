# Open-LLM Suite — GPU Schedule & Memory Strategy (2026-09-11)

Hardware (auto-detected): **8 × NVIDIA A800-SXM4-80GB**, 81920 MiB each, driver 535.261.03, CUDA 12.2.
At start each card held ~9.1–9.5 GB from other users → ≈72 GB free/card.
Policy: BF16 only for attribution models (no INT4/INT8); 1 GPU per 7–9B dense model; MoE/13B+ scheduled
after measuring peak; 70B (if unblocked) 2–4 GPUs.

| job | model | init | GPUs | batch | peak VRAM (GB) | wall time | status |
|---|---|---|---|---|---|---|---|
| smoke | Qwen3-8B (local) | P/R | 0 | 16 | ~33 | ~6 min | crashed (device-compare bug), fixed in code |
| validation | Qwen3-8B (local) | pretrained | 0 | 16 | TBD | running | in progress |
| download | DeepSeek-LLM-7B | – | – | – | – | – | downloading |
| download | DeepSeek-V2-Lite (MoE) | – | – | – | – | – | downloading |
| download | Mistral-7B-v0.3 | – | – | – | – | – | downloading |
| download | OLMo-2-7B | – | – | – | – | – | downloading |
| download | OLMo-2-13B | – | – | – | – | – | downloading (P1) |

Download: `HF_ENDPOINT=https://hf-mirror.com`, proxies disabled, 5 parallel streams,
`snapshot_download(max_workers=8)`; measured aggregate ≈7.5 MB/s (per-stream ≈1.2–1.9 MB/s).
`hf_transfer` and `aria2c` were tested and are **incompatible with the mirror** (silent failure / ERR),
so the plain HTTP path is used. Downloads resume from partial blobs.
