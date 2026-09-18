# Interface A reproducibility addendum (locked before A target results)

Parent protocol SHA-256:
`9369c5678c6f834455d2b2cea04da923d4e3e97b4b45ac16cabc2a368aac028c`.

The parent protocol fixed Interface A as a task-specific linear readout over
frozen final-layer mean-nonpadding representations, with seeds 7/17/27, but did
not state the optimizer budget or normalization. No Interface A target-model
result had been run when this addendum was created. This document fills that
reproducibility gap without changing benchmark items, splits, gold, options,
metrics, B/C prompts, parsers, or completed B/C results.

For each task type, one multiclass linear head maps the frozen temporal vector
to the sorted semantic gold labels observed in train. The task identity chooses
the head; no natural-language question, candidate option, gold label, or hidden
source label enters the temporal representation. Each vector receives
parameter-free per-sample layer normalization.

The head uses AdamW, learning rate 0.001, weight decay 0.01, batch size 256,
100 epochs, and no early stopping. Epoch `e` uses the permutation from
`numpy.random.default_rng(seed + e)`. P and R for a given seed share the same
head initialization, class order, minibatch order, and optimization budget.
The pretrained temporal cache is the official checkpoint cache extracted with
seed 7; the matched-random cache uses the paired seed.

Evaluation uses the locked test set. Accuracy is reported for Track R and Track
I and follows family → group → task → domain → domain-macro aggregation. P/R
bootstrap comparisons use identical source-group samples, 2,000 iterations,
seed 20260918. This interface measures task-conditioned linear accessibility;
it does not measure native question answering.

The addendum hash is stored in
`results/tsqa_evidence/real_v1/interface_a_addendum_sha256.txt`.
