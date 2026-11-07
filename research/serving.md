# Serving ModernBERT-Large-Instruct over HTTP

Research agent report, 2026-11-07, revised to cover cloud targets. Web research only;
nothing here was measured. Tags: **[verified]** read in source code or official docs,
**[claimed]** docs, papers or posts not checked in code, **[estimate]** the agent's own
reasoning. Prompt lengths use the corrected figures in `README.md` (119 to 348, median 152).

## Summary

- **Batching design.** Encoder servers all use one design: a queue feeds one model loop,
  each batch is capped by a **token budget**, and the loop takes whatever is waiting when
  it becomes free. TEI uses no wait window and Triton defaults to none; Ray Serve and
  Infinity wait 5 to 10 ms. Continuous batching gives nothing extra for encoders, because
  each request needs one forward pass.
- **Best exact-task baseline: vLLM `ModernBertForTokenClassification`** [verified in the
  registry]. Its head is dense, act, norm, linear: the HF MLM head's structure. A checkpoint
  whose 4-label classifier is the 4 verbalizer rows of the decoder, read at [MASK], gives
  the real task, unpadded, on CUDA. On x86 and Graviton CPU it may also work [not verified].
- **TEI** supports ModernBERT on CPU (x86 and arm64 images), Metal and CUDA (T4
  experimental, A10G, L4). It has no MLM head, and its classifier pools at CLS, so it is a
  speed-only baseline here. Batches are padded everywhere except on CUDA with fp16 and
  FlashAttention-2.
- **Batching gain depends on hardware** [estimate]: about 1.3x to 2x on CPU, 5x to 20x on
  GPU from batch 1. Defaults to start from: `max_wait = 0`, then a token-budget sweep.
- **Load testing:** open-loop Poisson load, timed from the scheduled send time. A small
  Rust generator is best; oha with `-q` and `--latency-correction` is the cross-check.

## 1. Dynamic batching design

- **TEI** [verified, [queue.rs](https://github.com/huggingface/text-embeddings-inference/blob/main/core/src/queue.rs)]:
  a FIFO queue on its own thread behind a channel. Each batch drains the queue head until
  the token budget would overflow; the overflowing entry goes back to the front. Cost is
  `max_len * n` when padded and `sum(len)` when unpadded. No delay window. Requests whose
  client disconnected are dropped. Defaults: `--max-batch-tokens 16384` ("largest possible
  until the model is compute bound"), `--max-batch-requests` unset,
  `--max-concurrent-requests 512` (refuse beyond it), `--max-client-batch-size 32`.
- **Triton** [verified, [batcher docs](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html)]:
  `max_queue_delay_microseconds` defaults to no delay. Advice: raise the max batch size and
  the delay until the latency budget is hit; avoid `preferred_batch_size` for most models.
  The queue policy has `max_queue_size` and timeouts. `instance_group` runs several model
  copies in parallel [claimed]; on CPU that is often a better use of cores than one large
  batch.
- **Ray Serve** [verified, [docs](https://docs.ray.io/en/latest/serve/advanced-guides/dyn-req-batch.html)]:
  `max_batch_size=10`, `batch_wait_timeout_s=0.01`; `batch_size_fn` caps by tokens.
- **TorchServe** [verified, [docs](https://docs.pytorch.org/serve/batch_inference_with_ts.html)]:
  `batchSize` plus `maxBatchDelay`; no documented defaults; no longer maintained.
- **Infinity** [verified, [repo](https://github.com/michaelfeil/infinity), `batch_handler.py`]:
  `batch_delay = 5 ms`, a length-prioritised queue, and tokenize, forward and postprocess
  on separate threads joined by bounded queues.
- **vLLM** [claimed]: the scheduler packs tokens up to `max_num_batched_tokens`, unpadded,
  at each step. An encoder request finishes in one step, so this reduces to adaptive
  batching.

**Continuous batching** [estimate]: it pays off when sequences can join between decode
steps. An encoder has one step per request, so the right policy is "form a new batch
whenever the model is free". Overlap tokenization and batch formation with the forward
pass, as TEI's prefetch and Infinity's thread pipeline do.

**GPU against CPU** [estimate]:
- A GPU at batch 1 is bound by kernel launches and weight reads. Throughput grows nearly
  linearly to about 2k to 10k tokens per batch (15 to 60 of our prompts). Large budgets
  (8k to 16k) and a short wait (1 to 5 ms) can pay off at low load.
- CPU gains plateau early, because a batch-1 matmul already has about 150 rows. Budgets of
  1k to 4k tokens suffice. Several model instances, each with fewer threads, often beat one
  large batch on many-core cloud CPUs.

**Padding** [estimate]: a padded batch of 16 of our prompts wastes about 40% of the FLOPs.
Unpadded (varlen) batching removes this waste for the linears, which are most of the work.
With varlen, FIFO order is fine; length sorting helps only padded engines and costs tail
latency.

## 2. Off-the-shelf servers that run ModernBERT

**TEI 1.9** [verified, [supported models](https://huggingface.co/docs/text-embeddings-inference/supported_models) and source]:

| Target | Image or install | Backend | Batching |
|---|---|---|---|
| Mac | `brew install text-embeddings-inference` | Metal via candle | padded |
| x86 CPU | `cpu-1.9` (ort, candle and MKL; `MKL_ENABLE_INSTRUCTIONS=AVX512_E4`) | ort or candle | padded |
| Graviton | `cpu-arm64-1.9` (ort and candle, no MKL) | ort or candle | padded |
| T4 | `turing-1.9` (experimental, flash off by default for precision) | candle | padded |
| A10G | `86-1.9` | candle; FlashModernBert with fp16 and FA2 | varlen |
| L4 | `89-1.9` | candle; FlashModernBert with fp16 and FA2 | varlen |

- On CUDA, `FlashModernBertModel` is used only with fp16 and FA2; otherwise it falls back to
  the padded model. Which of ort and candle wins on CPU when both are present was not checked.
- ModernBERT has a classifier head (CLS or mean pooling), no MaskedLM head; SPLADE is
  rejected. As a speed baseline: a classifier checkpoint built from the 4 decoder rows.
  FLOPs match ours; accuracy does not.
- A PR "Enable ModernBert on metal" (#562) was closed, and #561 touched the local attention
  mask for MPS [claimed]. Check that Metal ModernBERT runs correctly before relying on it.

**vLLM** [verified in `registry.py` and `modernbert.py`]: `ModernBertModel`,
`ModernBertForSequenceClassification`, `ModernBertForTokenClassification` (pooling ALL,
encoder-only attention) and `ColBERTModernBertModel`. The TokenClassification remap gives
exact per-token logits: the best exact-task baseline on T4, L4 and A10G. CPU
[verified, [docs](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/)]: x86
with AVX-512 (AMD Genoa or newer) has wheels since 0.17 and the `vllm/vllm-openai-cpu`
image; Arm and Graviton 3/4 with bf16; macOS experimental, source only. Whether pooling or
encoder models run on CPU is not stated [not verified]. vllm-metal lists no ModernBERT
[claimed].

**Triton with ONNX Runtime or TensorRT:** `optimum-onnx` has a `ModernBertOnnxConfig`
[verified], needing transformers 4.48 or newer. An export failure is reported when
flash_attn's Triton rotary kernel is installed
([forum](https://discuss.huggingface.co/t/modernbert-unable-to-convert-to-onnx-model/134489),
claimed); export with SDPA or eager attention. ONNX models are normally padded; Triton
ragged batching needs a model built for it. No published ModernBERT TensorRT or Triton
numbers were found.

**MLX (Mac only):** `mlx-embeddings` supports ModernBERT including MLM [claimed,
[repo](https://github.com/Blaizzy/mlx-embeddings)]. A library, so it needs a thin server.

## 3. Measuring serving performance

- **Closed-loop** tools wait for each reply before sending more, so load drops when the
  server stalls: coordinated omission (Gil Tene;
  [wrk2 README](https://github.com/giltene/wrk2)). Closed-loop: locust, hey, wrk, k6 VU
  executors.
- **Open-loop** tools follow a fixed schedule and time each request from its intended send
  time: wrk2; k6 `constant-arrival-rate` and `ramping-arrival-rate` (watch
  `dropped_iterations`, [docs](https://grafana.com/docs/k6/latest/using-k6/scenarios/concepts/open-vs-closed/));
  [oha](https://github.com/hatoo/oha) with `-q`, `--latency-correction`, `-Z` bodies and
  JSON output, but no Poisson; vegeta at constant rate [claimed].
- **Use Poisson arrivals.** A constant rate hides the bursts that a batcher absorbs.

Method [estimate]:
1. Estimate capacity in a closed loop.
2. Sweep the offered rate from 10% to 120% of that capacity.
3. At each rate, warm up 15 to 30 s, then measure 60 to 120 s. That covers thermal steady
   state on the Air, and CPU credit or turbo effects in the cloud. Avoid burstable instances.
4. Report achieved against offered throughput, p50, p90, p99 and the reject rate.
5. Plot p99 against achieved throughput; mark saturation where achieved falls below offered.
6. Headline: the maximum throughput under a p99 SLO.
7. Fairness: replay one fixed-seed schedule of the same prompts against every engine, run
   the load generator on a separate machine or cores, and keep PLAN.md's replication rules.

## 4. Expected batching gain

- **Nearest measured neighbour** [claimed, [arXiv 2510.18921](https://www.arxiv.org/pdf/2510.18921)]:
  MLX on an M2 Max, BERT-base 8.0 ms at batch 1 and 70.5 ms at batch 32, about 3.6x
  throughput, at an unknown sequence length.
- **Intel AMX** [claimed, [Next Platform](https://www.nextplatform.com/2023/04/05/why-ai-inference-will-remain-largely-on-the-cpu/)]:
  two Sapphire Rapids sockets reach about 90% of an A10 on BERT-Large, but much of that
  gain is bf16 against fp32; like-for-like comparisons give about 1.5x to 2x [forum claim].
  **AMX has no fp32 path**, so an fp32 sgemm engine gets nothing from it; taking AMX needs
  bf16 (oneDNN, MKL bf16 GEMM, ONNX Runtime or PyTorch). Graviton 3/4 likewise speed up
  bf16 (BFMMLA), not fp32.

Estimates:
- **M4 CPU fp32:** weight streaming is about 13 ms per forward (1.6 GB at about 120 GB/s),
  the matmul floor about 52 ms per example against a p50 of 105 ms. Ceiling about 2x;
  realistic 1.3x to 1.7x.
- **Cloud CPU:** similar batching gains. Larger gains come from bf16 (AMX or BFMMLA, about
  2x or more) and from several instances across cores.
- **Small GPUs in fp16,** at about 104 GFLOP per example: T4 (65 TFLOPS peak), A10G and L4
  (about 120 TFLOPS dense peak) should go from about 5 to 15 ms per example at batch 1 to
  about 2 to 4 ms per example at batch 32 or more: 5x to 20x. TEI on T4 is padded and loses
  about 30 to 40% to padding.
- **M4 GPU:** about 4 TFLOPS gives a floor of about 25 ms per example, sharing the thermal
  budget with the CPU.

## 5. Rust building blocks

- `tiny_http` (synchronous, small, thread-per-connection) is the best fit. std-only
  HTTP/1.1 is feasible but fiddly. `hyper` 1.x needs tokio; only for thousands of
  connections. `axum` is too much for one route.
- Server shape:
  1. HTTP threads tokenize their requests.
  2. They send `(ids, reply Sender)` into a bounded `std::sync::mpsc::sync_channel`; when it
     is full, return 503.
  3. One model thread owns the weights and BLAS, and loops: `recv()` blocking; optionally
     wait until the deadline; drain with `try_recv` up to the token budget, carrying the
     overflowing item over; run the varlen forward; reply to each request.
  4. For multi-instance CPU serving, N model threads share one channel, with BLAS threads
     split between them.

## Recommended design

**Batcher:** adaptive, unpadded, token budget, FIFO, bounded queue (256, then 503). Sweep
`max_batch_tokens` over 1k, 2k, 4k, 8k and 16k (to 32k on GPU), and `max_wait` over 0, 2,
5 and 10 ms (the wait starts at the first queued request). On cloud CPU, also sweep
instances x threads, for example 1x16, 2x8 and 4x4. Pick `max_batch_tokens` at the
throughput knee of the offline sweep.

**Load test:** a Rust open-loop Poisson generator with a fixed seed and real prompts, timed
from the schedule, with replies checked. Cross-check with `oha -q R --latency-correction`.
Run the rate sweep from section 3.

**Baselines, by target:**
- Mac: the Rust batch-1 engine; the Rust varlen batched engine; TEI Metal (speed-only); an
  mlx-embeddings wrapper (exact task).
- Cloud CPU: the Rust engine in fp32, plus bf16 if AMX or BFMMLA is used; TEI `cpu-1.9` or
  `cpu-arm64-1.9` (speed-only); vLLM CPU with the TokenClassification remap (exact, if it
  runs).
- Cloud GPU: vLLM with the TokenClassification remap (exact, varlen); TEI `86-1.9` or
  `89-1.9` (FlashModernBert, speed-only); optionally Triton with an ORT or TensorRT export.
- Match token budgets across engines. Report each engine at its defaults and at the
  matched setting.
