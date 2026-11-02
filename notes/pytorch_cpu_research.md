# Why rung 0 beats rung 8: a PyTorch source reading

Date: 2026-11-03. Versions read: torch **2.15.0** and transformers **5.22.1**, as installed
in `python/.venv` (read from the `dist-info` names, without importing torch). All PyTorch
links point at tag `v2.15.0`. Nothing was built, run or profiled for this note, because a
benchmark session needed the machine idle. Every timing split below is an estimate and is
marked as one.

Workload: ModernBERT-Large-Instruct, fp32, batch 1, about 150 tokens, M4 Air, 4 threads.
Rung 0 p50 137 ms, rung 5 167 ms, rung 8 220 ms.

## Summary

The matmuls are the same on both sides. PyTorch calls Accelerate's `sgemm_`, with no
prepacking and no oneDNN, as rung 8 does. The gap is in the other work, and it has two
causes:

1. **PyTorch runs all the non-matmul work on 4 threads.** Attention is split into 80
   tasks (16 heads x 5 query blocks). LayerNorm is split by rows, and GELU and the adds by
   chunks. Rung 8 runs all of this on one thread. This matches the P-cluster clock that
   stayed near 3.97 GHz in rung 7.
2. **PyTorch does the inner arithmetic in f32 NEON with polynomial `exp`, and it sends the
   attention products to sgemm.** Rung 8 uses f64 libm `exp` per element, and its
   attention products are scalar-order loops.

Rung 8 already does better than PyTorch in four places: it skips the keys outside the
local window, it has no Python dispatch, it makes no allocations, and it fuses RoPE and
GELU-gate. So once the work is threaded and vectorised, rung 8 should beat rung 0.

## 1. Matmul: Accelerate `sgemm_`, no prepacking

- `nn.Linear` with no bias (ModernBERT has `attention_bias = mlp_bias = False`) goes to
  `at::matmul(input, weight.t())`. That folds `[1, S, 1024]` into one 2-D `mm`.
  [Linear.cpp#L88-L135](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/Linear.cpp#L88-L135)
- `mm` reaches `addmm_impl_cpu_`. The oneDNN/ACL branch is compiled only when
  `AT_MKLDNN_ACL_ENABLED()`. `python/BLAS.md` shows `USE_MKLDNN=OFF`, so the code falls
  through to `cpublas::gemm`.
  [LinearAlgebra.cpp#L1497-L1536](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/LinearAlgebra.cpp#L1497-L1536)
- The `float` overload of `cpublas::gemm` calls the Fortran `sgemm_`, which resolves to
  Accelerate (`BLAS.md`, check 2). The `cblas_sgemm` branch is for iOS only.
  `use_blas_gemm` checks only the strides, so it applies no size threshold.
  [CPUBlas.cpp#L110-L122](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/CPUBlas.cpp#L110-L122),
  [CPUBlas.cpp#L207-L250](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/CPUBlas.cpp#L207-L250)
- **No weight prepacking for fp32.** The flash kernel has a `with_pack` path, but it is
  only for Half and BFloat16 (`could_pack`, below). Linear has no packed path on this
  build.
- **Threads.** `torch.set_num_threads` sets the OpenMP pool. It does not set Accelerate's
  thread count. Accelerate picks its own, capped by `VECLIB_MAXIMUM_THREADS`, which the
  harness sets to the same value for every rung (`crates/mb-harness/src/bench.rs:180`).
  So both sides get the same matmul threading. Rung 8's `cblas_sgemm(RowMajor, NoTrans,
  Trans)` and PyTorch's column-major `sgemm_` describe the same product.

**Conclusion: there is no matmul lever here.** Both rungs pay the same sgemm cost.

## 2. Attention: SDPA with the CPU flash kernel, in every layer

### What transformers does

- Rung 0 loads with no `attn_implementation`, so transformers picks `"sdpa"`
  (`modeling_utils.py`, `get_correct_attn_implementation`, near line 1845).
- Global layers get **no mask** (`None`): there is no padding, and `local_size` is `None`
  (`masking_utils.py`, `_ignore_bidirectional_mask_sdpa`, lines 311-339).
- Local layers get a **boolean `[1, 1, S, S]` mask**, `|q - k| <= 64`. The skip check
  requires `kv_length < 64`, so every AG News prompt (119 tokens or more) builds the mask
  (`masking_utils.py`, lines 137-155 and 500, `sliding_window = config.sliding_window =
  64`). This mask is rebuilt on every forward pass, in Python and small tensor ops.
- RoPE in HF is about 10 small elementwise ops per layer (`.float()`, `cat`, mul, add)
  (`modeling_modernbert.py`, `apply_rotary_pos_emb`).

### What PyTorch does with it

- `scaled_dot_product_attention` asks `_fused_sdp_choice_cpp`. The CPU order is flash,
  then math. Flash accepts fp32, `head_dim` 64, a 4-D mask, last-dim stride 1 and no
  dropout, so **flash is chosen for both layer types**.
  [sdp_utils_cpp.cpp#L38-L59](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/transformers/sdp_utils_cpp.cpp#L38-L59),
  [attention.cpp#L815-L847](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/transformers/attention.cpp#L815-L847)
- The boolean mask becomes an additive float mask, `where(mask, 0, -inf)`. That is one
  more `[S, S]` tensor per local layer.
  [attention.cpp#L573-L586](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/transformers/attention.cpp#L573-L586)
- **Block sizes.** For `q_seq_len < 192`, `qSplitSize = 32` and `kvSplitSize = 512`. So at
  about 150 tokens, one KV block holds every key, and the online-softmax rescale never
  runs. From 192 tokens it is 64 x 512.
  [FlashAttentionKernel.cpp#L1173-L1200](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/cpu/FlashAttentionKernel.cpp#L1173-L1200)
- **Parallelism.** One `at::parallel_for` runs over `batch x heads x qSlice`, with grain 1.
  At S = 150 that is 16 x 5 = **80 independent tasks** on 4 OpenMP threads, and each
  thread has its own scratch buffer.
  [FlashAttentionKernel.cpp#L557-L566](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/cpu/FlashAttentionKernel.cpp#L557-L566)
- **Both products are sgemm.** `cpublas::gemm` computes `K^T Q` for a 32-query block
  (`[kv, 64] x [64, 32]`), then `V P` (`[64, kv] x [kv, 32]`). That is the same `sgemm_`
  to Accelerate as above, called from inside the OpenMP region.
  [FlashAttentionKernel.cpp#L616-L630](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/cpu/FlashAttentionKernel.cpp#L616-L630),
  [FlashAttentionKernel.cpp#L747-L762](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/cpu/FlashAttentionKernel.cpp#L747-L762)
- **The mask does not skip work.** `num_keys = kvSize` unless `is_causal`. A local layer
  computes all S scores per query, adds the `-inf` mask row by row
  (`_scale_attn_mask_fusion_kernel`), and takes `exp` of `-inf`. Rung 8's key-range loop
  does about 129/150 of that work at 150 tokens, and less on longer prompts.
  [FlashAttentionKernel.cpp#L580](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/cpu/FlashAttentionKernel.cpp#L580),
  [FlashAttentionKernel.cpp#L644-L675](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/cpu/FlashAttentionKernel.cpp#L644-L675)
- **exp.** `_exp_reduce_sum_fusion_kernel` computes `exp(x - max)` and the row sum in one
  pass, with `Vectorized<float>::exp_u20()` and an **f32 vector accumulator**.
  [FlashAttentionKernel.cpp#L84-L148](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/cpu/FlashAttentionKernel.cpp#L84-L148)
  On Apple NEON (not SVE), `exp_u20` is `vexpq_f32_u20`. That is a 5-term polynomial
  after range reduction `x = n ln2 + r`, with ln2 split into hi and lo parts, and the
  exponent built by an integer shift. It is 4 lanes and has no Sleef call. It falls back
  to the full `exp()` only when some `|x| >= 87.3`.
  [vec128_float_neon.h#L389-L448](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/cpu/vec/vec128/vec128_float_neon.h#L389-L448)
- The output is normalised by `1 / sum` and written straight into the `[B, S, H, D]`
  layout, so no transpose copy is needed before `Wo`.
  [FlashAttentionKernel.cpp#L773-L776](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/cpu/FlashAttentionKernel.cpp#L773-L776),
  [attention.cpp#L1018-L1029](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/transformers/attention.cpp#L1018-L1029)
- `could_pack` is true only for Half and BFloat16, so fp32 takes the plain gemm path.
  [FlashAttentionKernel.cpp#L1170-L1171](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/cpu/FlashAttentionKernel.cpp#L1170-L1171)

The standalone softmax kernel (`SoftMaxKernel.cpp`) is not on rung 0's hot path, because
SDPA fuses the softmax.

## 3. LayerNorm, GELU, elementwise

- **The parallel_for rule.** A loop goes parallel when `numiter > grain_size`, more than 1
  thread is set, and it is not already inside a parallel region. The OpenMP backend then
  splits the range into `min(threads, ceil(n / grain))` equal chunks.
  [Parallel-inl.h#L10-L31](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/Parallel-inl.h#L10-L31),
  [ParallelOpenMP.h#L17-L45](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/ParallelOpenMP.h#L17-L45)
- **Elementwise ops** (the residual add, `act(input) * gate`) go through TensorIterator
  `cpu_kernel_vec`, with `GRAIN_SIZE = 32768` elements. `[150, 1024]` is 153,600
  elements, so each add is split 4 ways. Each op is NEON-vectorised.
  [TensorIterator.h#L76](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/TensorIterator.h#L76),
  [Loops.h#L304-L318](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/cpu/Loops.h#L304-L318)
- **GELU (exact erf).** Above 16,384 elements the grain is `numel / threads`, so
  `[150, 2624]` is split 4 ways.
  [Activation.cpp#L289-L341](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/cpu/Activation.cpp#L289-L341)
  The vector path is `x * 0.5 * (1 + (x * M_SQRT1_2).erf())`
  ([Gelu.h#L73](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/cpu/Gelu.h#L73)).
  `Vectorized<float>::erf` on NEON is **the same Abramowitz and Stegun 7.1.26
  approximation as our `ops::erf`, with the same coefficients**. The difference is that
  it runs in f32, 4 lanes, with `vexpq_f32_u20` for `exp(-x^2)`.
  [vec128_float_neon.h#L830-L861](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/cpu/vec/vec128/vec128_float_neon.h#L830-L861)
  This assumes `AT_VEC_CUSTOM_MATH` is not defined in the wheel. That is likely, but it
  was not checked.
  PyTorch makes three passes here (`chunk`, `gelu`, `mul`). Rung 8 fuses them into one.
- **LayerNorm.** `parallel_for(0, M, 1, ...)` runs over rows, so 150 rows give 4 chunks of
  about 38 rows. Each row's moments are a vectorised f32 Welford with cascade summation
  (`RowwiseMoments`, chunk 16). Rung 8 uses two passes in f64.
  [layer_norm_kernel.cpp#L57-L95](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/cpu/layer_norm_kernel.cpp#L57-L95),
  [moments_utils.h#L112-L140](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/cpu/moments_utils.h#L112-L140)
  ModernBERT has no norm bias, so `beta_null` is true and the second pass takes the
  **scalar** `for` loop, not `vec::map3`. The compiler may still auto-vectorise it.
  [layer_norm_kernel.cpp#L35-L42](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/cpu/layer_norm_kernel.cpp#L35-L42)

## 4. Other things on the hot path

- **Threadpool.** The pool is OpenMP (`libomp` in the wheel, 4 threads). Each
  `parallel_for` is one `#pragma omp parallel` region. libomp keeps its workers alive,
  and they spin briefly before they sleep (`KMP_BLOCKTIME` defaults to 200 ms). So at
  about 700 ops per forward pass, the 4 threads are always warm.
- **Fusion.** The only fused kernel is flash attention (scale, mask, softmax and both
  products). Everything else is one ATen op at a time, with an allocation per output
  from the CPU caching allocator.
- **Python overhead.** Roughly 25 ops per layer x 28 layers, plus building the mask. At
  about 5 to 10 µs of dispatch each, that is an estimated 4 to 7 ms of rung 0's 137 ms.
  Rung 8 does not pay this.
- **Memory format.** Q, K and V stay strided views of the `Wqkv` output, `[B, H, S, D]`
  with last-dim stride 1. The flash kernel reads them through strides, with no copy.

## Comparison with rung 8

| Work | Rung 0 (PyTorch) | Rung 8 (ours) |
|---|---|---|
| Linear layers | Accelerate `sgemm_`, Accelerate's threads | same, `cblas_sgemm` |
| `QK^T`, `PV` | Accelerate sgemm on 32-query blocks | scalar-order NEON loops, 1 thread |
| Attention parallelism | 80 tasks on 4 threads | 1 thread |
| Local window | computes all S keys, masks with `-inf` | key range, skips masked keys |
| Softmax exp | f32 NEON polynomial, f32 vector sum | f64 libm `exp` per element, f64 sum |
| GELU erf | same A&S formula, f32 NEON, 4 threads | same formula, f64 scalar libm `exp`, 1 thread |
| LayerNorm | f32 Welford, rows on 4 threads | f64 two-pass, 1 thread |
| Adds | NEON, 4 threads | auto-vectorised, 1 thread |
| RoPE, GELU-gate | about 10 ops and 3 ops | fused into the scatter, one pass |
| Allocation, dispatch | about 700 ops, allocator, Python | zero, none |

### An estimated time budget

This is not measured, so a profile must confirm it before it goes into the blog.

- **Matmul FLOPs at S = 150:** 3.68 GFLOP per layer x 28 = about 103 GFLOP. `BLAS.md`
  measured 1,168 GFLOP/s at 2048^3. These skinny shapes (M = 150) reach less, so the
  matmuls likely take **about 90 to 120 ms** on both sides.
- That leaves about **20 to 45 ms of non-matmul work in rung 0** and about **100 to 130 ms
  in rung 8**.
- Rung 7 was 367 ms, with 52% (about 190 ms) in the two attention loops. Rung 8 took off
  147 ms with layout alone, so rung 8's attention is likely about 40 to 50 ms. `exp`
  (softmax and GELU) is 17%, about 37 ms.
- **What follows from this:** rung 8 can at best reach its matmul floor, which is about
  the same as rung 0's. To beat rung 0, rung 8's non-matmul work must cost less than
  PyTorch's 20 to 45 ms. That looks reachable, because PyTorch does more attention work
  on local layers and also pays for dispatch.

## Candidate future rungs, ranked by likely gain

Each rung changes one variable (hard rule 2). The numbers are estimates from the budget
above. Each rung must still pass the in-run logit gate (1e-2, same argmax) and Tier 3
(100% agreement over all 7,600 examples). The tier-2 bounds stay at 1e-4 for activations,
1e-3 for logits, and cosine >= 1 - 1e-6.

### Rung 9: thread the non-matmul work (parallelism only)

Split attention over `(head, query block)` and the norms, GELU-gate, adds and scatter over
rows, on 4 threads. The arithmetic inside each row or `(head, query)` stays the same, so
**the result stays bit-identical to rung 8**. Test it the same way rung 8 was tested
against rung 7.

- **Expected gain:** the largest. If 100 to 130 ms scales about 3.3x on 4 P-cores, rung 8
  drops from 220 ms to an estimated **130 to 150 ms**, about level with rung 0.
- **Correctness risk:** none numerically. The risks are practical:
  - A thread pool must not allocate in the forward pass, so it needs persistent workers.
    `std::thread::scope` spawns allocate. Rayon would be a new dependency, so it must be
    discussed first (CLAUDE.md).
  - AMX is shared per cluster. Our threads must never run while Accelerate's sgemm runs.
    That is already true, because the ops run one after another.
- **Option:** if the blog wants attention measured alone, make this two rungs: 9a
  attention only, 9b everything else.

### Rung 10: f32 vectorised `exp` (the exp function only)

Replace f64 libm `exp` with an f32 NEON polynomial like `vexpq_f32_u20`, in softmax and in
`erf`'s `exp(-x^2)`. Keep the f64 softmax sum and the f64 `erf` arithmetic, so only one
thing changes.

- **Expected gain:** most of the 37 ms (17%) on one thread. After rung 9 it is about 10
  ms, so the order of rungs 9 and 10 changes how the blog splits the credit. Doing 10
  first gives a cleaner single-thread story.
- **Correctness risk:** low. PyTorch itself uses this f32 `exp` (u20 means an error under
  about 2 ULP), so the gap to rung 0 may get smaller. Tier 2 must be measured again.
  Rung 10 is no longer bit-identical to rung 8.
- **Option:** Accelerate's vForce `vvexpf` instead of our own polynomial. Accelerate is
  already a justified dependency, but owning the polynomial is closer to the project's
  point.

### Rung 11: attention products through `cblas_sgemm` (the attention matmul only)

Per head and query block, compute `Q K^T` and `P V` with `cblas_sgemm`, as PyTorch's flash
kernel does. Use a key range per block for local layers, plus a `-inf` edge mask inside
the block. That keeps rung 8's skipping, which PyTorch does not have.

- **Expected gain:** medium. Attention is only about 2.6 GFLOP in total, so on AMX the
  products cost almost nothing. The gain is most of the remaining NEON loop time, about
  10 to 30 ms on one thread before rung 9, or less after it. Many small sgemm calls
  (`[~150, 64] x [64, 32]`) have a fixed call cost, so the block size needs tuning: try 32
  as PyTorch does, and also one call per head.
- **Correctness risk:** low to medium. The summation order changes, so rung 11 is not
  bit-identical. Expect differences of order 1e-6 relative on the scores, well inside
  1e-4. Watch the sensitive example, index 4100. Do not use `beta = 1` accumulation onto
  large values, for the reason in `fused.rs`. Here it is safe only for `P V` across key
  blocks, which this layout does not need.

### Rung 12: f32 vectorised erf and LayerNorm statistics (precision of the remaining f64 math)

Move `erf` to the same f32 formula as PyTorch, and LayerNorm to f32 vector sums (or f32
Welford).

- **Expected gain:** small, about 3 to 8 ms on one thread, and less after rung 9.
- **Correctness risk:** medium for LayerNorm. f64 was chosen on purpose to stay inside
  1e-4 (`ops.rs`). The residual stream reaches about 28,000, so an f32 mean over 1,024
  values loses low bits. Use Welford, or a pairwise sum like PyTorch's cascade, and not a
  naive f32 sum. The risk for erf is low, because it would match PyTorch's arithmetic.
- This should probably be two rungs, erf and LayerNorm, if each is measured on its own.

### Not rungs, but worth one measurement

- **The Accelerate thread count for the skinny matmuls.** `BLAS.md` found 4 threads slower
  than 1 at 2048^3. The 1-thread session now running will show this for M = 150 shapes. It
  is a harness setting that applies equally to all rungs, not a rung.
- **One profile of rung 8 after the benchmark session**, to replace the estimated budget
  above with a measured split between matmul and the rest. Do this before choosing
  between the orders 9 then 10, or 10 then 9.

## What PyTorch has that we should not copy

- The materialised `[S, S]` local mask, and computing all keys on local layers. Rung 8's
  key range is strictly less work.
- Separate passes for `chunk`, GELU and mul, and for RoPE. Rung 8's fusions are already
  better.
- Weight prepacking. fp32 does not use it here, so there is nothing to gain from it.
