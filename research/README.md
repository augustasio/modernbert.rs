# Research notes

Web and literature research, done by research agents on 2026-11-07, while a benchmark ran.
Nothing here was measured on this Mac unless it says so. Treat each figure as a lead to
check, not a result.

- `inference_optimisations.md`: ways to make zero-shot inference faster, smaller or
  cheaper in energy, ranked for this setting (batch 1, M4 Air, CPU).
- `fine_tuning_on_m4.md`: whether fine-tuning ModernBERT on this Mac is feasible and
  worth it, and how it would fit the write-up (a separate track, not a rung).
- `serving.md`: dynamic batching designs, off-the-shelf servers that run ModernBERT, and
  how to load-test a server fairly (open-loop, Poisson arrivals).
- `cloud_hardware.md`: AWS instance families, their bf16 and int8 matrix throughput, and
  estimated cost per million classifications.

Direction since 2026-11-07: CPU first, with optimizations that transfer to cloud CPUs
(x86 with AMX, Graviton). No Mac-only work. GPUs come later.

## A correction that applies to both

The agents were told the prompts are about 60 tokens. That came from an early estimate in
PLAN.md, and it is wrong. Measured from `data/agnews_test_prompt_ids.tsv`:

- the prompt length is 119 to 348 tokens, median 152;
- the fixed template alone (the content-free prompts) is 103 to 105 tokens, so about two
  thirds of a typical prompt is the template.

So the matmul work per example is about 2.5 times the agents' figure: about 104 GFLOP,
not 41. At the M4's measured SME fp32 peak of about 2 TFLOPS (one unit shared by the four
P-cores), the matmul floor is about 52 ms against rung 13's p50 of 105 ms, not 20 ms. And
a shorter template is a larger lever than the first report estimates: halving it would cut
about a third of the tokens.
