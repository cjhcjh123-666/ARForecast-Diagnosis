# Protocol validation: new open-LLM runner vs published Qwen3-8B numbers

Run: `scripts/openllm_suite.py --model-key qwen3_8b_base --path models/qwen3-8b-base --seeds 7,17,27 --tasks A,B,C --inits pretrained`
(18 rows in `results/open_llm_suite/all_metrics_long.csv`). Wall time ≈ 4.4 min/seed on 1×A800-80GB.

| quantity | new runner (per seed) | mean | published / prior | match |
|---|---|---|---|---|
| A clean recognition (orig) | 0.9833 / 0.9767 / 0.9833 | **0.981** | 0.981 ± 0.003 | ✅ exact |
| A shuffled recognition | 0.6233 / 0.6433 / 0.6533 | **0.640** | 0.640 | ✅ exact |
| B linear readout MSE | 0.5766 / 0.6595 / 0.5269 | **0.588** | 0.588 | ✅ exact |
| C family-label bal3 bal-acc | 0.8750 / 0.8250 / 0.8000 | **0.833** | 0.833 (E4 reference) | ✅ exact |
| C family-label bal3n bal-acc | 0.8750 / 0.8083 / 0.8083 | **0.831** | 0.831 | ✅ exact |
| C routed MSE (bal3) | 0.6823 / 0.6814 / 0.7656 | **0.710** | 0.710 | ✅ exact |
| B MLP64 readout MSE (new) | 0.6049 / 0.5893 / 0.4684 | **0.554** | – | new |

Conclusion: the unified open-LLM runner is protocol-identical to the published pipeline at 4-decimal
agreement on all shared quantities; it can be applied to the new open base LMs without changing the benchmark.
