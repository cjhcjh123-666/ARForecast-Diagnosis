# Scientific conclusions — locked matrix complete

The completed evidence supports the benchmark's measurement validity and three
limited observations. First, Qwen3-8B, Llama-3.1-8B, and Gemma-2-9B show large
free-versus-constrained differences in output validity and/or item accuracy, so
native output interfaces must be reported separately. Gemma and DeepSeek free
generation currently have no strictly valid outputs; this is an interface
failure rather than a zero ability estimate.

Second, the ten complete A/B model families do not support a universal
pretraining gain. Llama-3.2 has a consistent positive P−R difference for Track R
under B, while Gemma-2-2B and Qwen intervals cross zero. Llama-3.1 has a positive
Track R P−R difference under B but a negative difference under A on the same
locked sample. Its intervention differences cross zero, and its B four-cell
joint score is zero in both branches. The observed-real benefit is therefore
selective by task and interface.

Third, accessibility and use cannot be collapsed into one score. Gemma-2-2B
shows a positive Track R P−R difference under A but not B; Llama-3.1 shows the
opposite direction. Gemma-2-9B has a positive Track I difference under A and a
negative one under B. Mistral has a positive target-update difference under A
but not B. These are paired, same-sample comparisons, yet they still do not
establish why an interface succeeds or fails. The newly closed families preserve
this heterogeneity: DeepSeek-V2-Lite has positive A intervention metrics but a
negative B Track R difference; OLMo-2-7B and OLMo-2-13B have positive A TUS and
positive B Track R differences, while their Track I intervals cross zero;
DeepSeek-LLM-7B's selected A/B intervals cross zero.

The complete locked-matrix evidence does not support claims that the models understand the
signals, that shuffled performance identifies language-prior dominance, that a
probe/readout gain transfers to native QA, or that a shared causal mechanism
explains representation and QA results. Cross-seed and cross-model runs remain
required.
