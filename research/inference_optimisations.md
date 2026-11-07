# Faster, smaller, cheaper: options for rung 14 onward

Research agent report, 2026-11-07. Not measured on this Mac. **Read `README.md` first:**
this report assumed 60-token prompts; the real median is 152 tokens, of which 103 are the
template. The arithmetic in "Measure first" and the estimate for a shorter template are
corrected there.

## Measure first

Rough arithmetic for one forward pass at 60 tokens (see the correction above):
- The encoder linear layers hold about 342M weights, which is 1.37 GB in fp32.
- They do about 41 GFLOP of matmul work (about 104 GFLOP at the real 152 tokens).
- M4 SME fp32 peak is about 2 TFLOPS. Jena measured this, and they found that **all four
  P-cores share one SME unit**. That gives a matmul floor of about 20 ms (about 52 ms at
  152 tokens).
- Streaming the fp32 weights at about 120 GB/s gives a floor of about 11 ms.

So either Accelerate's sgemm runs below peak at these shapes, or the code outside the
matmuls still takes much of the 105 ms. **Profile how much of rung 13's time is sgemm
before you pick a rung.** Every estimate below depends on that split.

The shared SME unit also explains a past result: extra threads cannot speed up the
matmuls. They only help the work around them (rung 11). Source:
[Jena SME microbenchmarks](https://scalable.uni-jena.de/opt/sme/micro.html), measured.

Second check: raw accuracy is 71.6%, and rung 0's raw predictions never say Tech
(`results/rung0_stats.json`, `raw/predicted_counts`). Low-margin examples will flip under
any lossy change. Log the margin distribution now. Then a lossy rung can report how many
examples flipped and the accuracy change, with the McNemar test already in place.

## Ranked table

| # | Technique | Expected speedup here | Accuracy risk | Effort | One-variable rung? |
|---|---|---|---|---|---|
| 1 | Shorter prompt template | 1.2 to 1.8x as estimated; larger at 152 tokens | Changes accuracy, must report | Low | y |
| 2 | Last layer: compute Q, attention output, Wo and FFN for the mask row only | 1.02 to 1.03x | None (exact up to summation order) | Low | y |
| 3 | int8 dynamic (W8A8), own SME `SMOPA` kernel | Matmul up to 2x; 1.2 to 1.5x end to end if matmul is about half the time | Medium | High | y |
| 4 | Own SME GEMM, pre-packed weights, plus the E-cluster SME unit | 1.1 to 1.25x | None to tiny | High | y |
| 5 | GPU via MLX or Metal | Speculative 1.5 to 3x | Low (fp32) or medium (fp16) | Medium | y (as a compute-unit rung) |
| 6 | Neural Engine via Core ML, fp16 | Speculative 2 to 4x, best energy | High: fp16 overflow, known wrong outputs | High | y |
| 7 | fp16/bf16 weights via BNNS, fp32 accumulate | 0 to 1.1x speed; 2x less memory | Low (fp16) to medium (bf16) | Medium | y |
| 8 | int8 weight-only, dequantize to fp32 | About 1x, or slower; 4x less weight memory | Low | Medium | y |
| 9 | Attention-head pruning without retraining | 1.05 to 1.1x | Medium to high | Medium | y |
| 10 | Layer dropping without retraining | Linear in layers dropped (4 of 28 = 1.17x) | High | Low | y |
| 11 | int4 (GPTQ/AWQ style) | About 1x (compute-bound) | Medium to high | High | y |
| 12 | ModernBERT-base zero-shot | About 3x fewer FLOPs | Unknown; no instruct checkpoint exists | High (needs training) | n |
| 13 | Distillation, FFN pruning, early exit | Varies | Needs training | High | n |
| 14 | 2:4 or unstructured sparsity | None on this CPU | Needs retraining | High | n |
| 15 | Caching the template prefix | Not possible | n/a | n/a | n/a |

## Notes per technique

**1. Shorter template.** Almost all the cost is the linear layers, and they scale linearly
with token count. The template is about two thirds of a typical prompt (see `README.md`),
so this is the largest single lever. It is a task-design rung, not an engine rung. Report
accuracy with intervals and keep rung 0's template as the oracle. The authors do not study
template sensitivity ([paper](https://arxiv.org/abs/2502.03793)).

**2. Mask row in the last layer.** Layer 27 needs K and V for every token, but only the
mask row continues after attention. Wo plus the FFN are 8.1M of each layer's 12.2M
weights. Skipping all rows but one there saves about 2.4% in total. The agent's own
arithmetic from the config.

**3. int8 W8A8.** Measured on M4: `SMOPA` i8 to i32 runs at 4020 GOPS against 2008 GFLOPS
for fp32 `FMOPA` ([Jena](https://scalable.uni-jena.de/opt/sme/micro.html)). There is no
int8 GEMM in cblas. A Rust crate that probed BNNS reports that `BNNSMatMul` rejects int8
([rlx-cpu source](https://docs.rs/crate/rlx-cpu/0.2.17/source/src/intrinsics/apple_amx/bnns.rs);
third-party, not Apple docs). So this means writing our own SME kernel.

Accuracy: plain W8A8 post-training quantization lost more than 10 points on BERT-base, and
ZeroQuant's per-token and group-wise scheme recovered it
([ZeroQuant](https://arxiv.org/pdf/2206.01861)). Our residual reaches about 28,000, a sign
of large outlier activations. Known BERT outliers sit in specific channels
([Bondarenko et al. 2021](https://arxiv.org/abs/2109.12948)). Use per-token activation
scales and per-channel weight scales. Keep the residual and the norms in fp32.

**4. Own SME GEMM.** MpGEMM reports an average 1.23x over Accelerate on an M4 Pro, measured
([arXiv 2512.21473](https://arxiv.org/abs/2512.21473)). Jena's generated kernels
"outperform the vendor-optimized BLAS" in almost all tested configurations
([arXiv 2409.18779](https://arxiv.org/abs/2409.18779)). One P thread plus one E thread
reached 2371 GFLOPS against 2008 for P alone, so the E-cluster SME unit adds about 18%
peak. Each switch into and out of streaming mode has a cost: kernels with only 8 `FMOPA`
reach 434 GFLOPS. Pre-packing the weights once at load time removes Accelerate's packing on
every call (speculative: the size of that cost is not public).

**5. GPU (MLX/Metal).** No measured ModernBERT-large batch-1 numbers exist. An AI-generated
review reports BERT-base at 8 ms on an M2 Max with MLX
([arXiv 2510.18921](https://arxiv.org/html/2510.18921v1)); unverified and on a larger GPU.
`mlx-embeddings` supports ModernBERT ([repo](https://upd.dev/Blaizzy/mlx-embeddings)). The
M4's 10-core GPU has more fp32 throughput than one SME unit, but kernel launches cost
latency at batch 1 across 28 layers. The agent guessed 30 to 60 ms (speculation).

**6. Neural Engine.** Apple's ANE-optimised DistilBERT ran 2.84x faster at sequence 128 and
batch 1 ([ane-transformers](https://pypi.org/project/ane-transformers),
[Apple](https://machinelearning.apple.com/research/neural-engine-transformers)). Known
problems with ModernBERT-family models:
- An mmBERT-small conversion gave wrong outputs on the ANE, so its author pinned it to CPU
  and GPU ([card](https://huggingface.co/znaat/mmbert-small-coreml)).
- Conversion needs transformers 4.48.3 ([card](https://huggingface.co/znaat/modernbert-coreml)).
- A 130M Japanese ModernBERT ran on the ANE with a mean difference of 0.021 from fp32
  ([card](https://huggingface.co/umiyosh/modernbert-ja-130m-coreml)).

Hypothesis (speculation): our residual of about 28,000 is close to the fp16 maximum of
65,504, and fp16's step size at that scale is 16. This could cause overflow or drift.
Expect the 1e-2 gate to fail, and report the accuracy instead. Energy per example is likely
the best of all options.

**7. fp16/bf16.** Measured: on M4 SME, the widening `FMOPA` fp16 to fp32 and `BFMOPA` bf16 to
fp32 reach the same GOPS as fp32. Jena calls the reduced-precision throughput
"underwhelming". So there is no compute gain. The gain is half the weight bytes and memory
(1.5 GB to 0.75 GB). M4 lacks FEAT_SME_F16F16
([forum](https://forums.macrumors.com/threads/apple-silicon-in-sciences.2374458/page-15)).
Use BNNS for half-precision matmul. Keep activations and the residual in fp32: bf16 has a
step size of 128 at 28,000.

**8, 11. Weight-only int8 and int4.** These win when a workload is limited by memory
bandwidth, like LLM decoding. llama.cpp shows quantization barely changes prompt-processing
speed ([llama.cpp #4167](https://github.com/ggml-org/llama.cpp/discussions/4167)). Our
workload is compute-bound. Treat these as memory rungs, not speed rungs. A Core ML int8
ModernBERT-base doubled latency, 10 ms to 23 ms
([card](https://huggingface.co/znaat/modernbert-coreml)).

**9, 10. Pruning without retraining.** Many heads can be removed at test time with little
loss ([Michel et al. 2019](https://arxiv.org/abs/1905.10650)). Heads only cover the QKV and
Wo share, about a third of the weights. Layer dropping keeps quality only with fine-tuning
([Poor Man's BERT](https://arxiv.org/pdf/2004.03844v1.pdf)). LayerDrop gives sub-networks
without fine-tuning only if the model was trained that way
([Fan et al.](https://arxiv.org/abs/1909.11556)). An MLM head that reads the final layer is
fragile. Structured FFN pruning and 2:4 sparsity need retraining. No sparse path was found
in Accelerate's dense GEMM or in M4 SME. A custom sparse ternary kernel on M-series exists,
but for a different problem ([arXiv 2510.06957](https://arxiv.org/abs/2510.06957)).

**12, 13. Base model, distillation, early exit.** No ModernBERT-base-instruct exists. The
paper trained only Large ([paper](https://arxiv.org/html/2502.03793v2),
[Answer.AI](https://answer.ai/posts/2025-02-10-modernbert-instruct.html)). The paper reports
no zero-shot AG News numbers, only fine-tuned ones: 95.24 for ModernBERT-Large-Instruct. All
three options need training, which ends the zero-shot premise.

**15. Prefix caching.** This does not work in a bidirectional encoder. From layer 1 onward,
every template token attends to the article tokens, so its hidden state changes with every
input. Only the token embeddings are reusable, and they are a lookup. Decoders can cache
because of causal masking.

**Local attention.** The report said that at 60 tokens the 128-token window covers the whole
sequence. At the real 119 to 348 tokens that is not true for most prompts, so local layers
do less work than global ones, as rung 8 already exploits.

## Recommended order of next rungs

1. **Profile rung 13** (sgemm share, plus time spent switching streaming mode). Log the
   verbalizer margins. Not a rung.
2. **Rung 14: mask row only in the last layer.** Cheap and passes the exact gate.
3. **Rung 15: Metal GPU or MLX, fp32.** One variable: the compute unit. Report energy.
4. **Rung 16: own SME fp32 GEMM with pre-packed weights.** Add the E-cluster unit if the
   profile shows matmul dominates.
5. **Rung 17: int8 W8A8 with SME `SMOPA`.** Report flips and the accuracy change.
6. **Memory rung: fp16 weights via BNNS.** Halves the weights at almost no speed change.
7. **Optional: Core ML on the ANE, fp16.** Only as an honest result, which may be a failure.
8. **Task rung: shorter template.** Do it last so the engine rungs stay comparable. Report
   it as an accuracy trade.
