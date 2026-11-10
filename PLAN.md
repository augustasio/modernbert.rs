# Plan

## The question

Take a modern BERT-like encoder, run it on a laptop CPU, and find out where inference
time actually goes. Then rewrite the forward pass from scratch in Rust and measure
whether it helps, and which part of "it helps" is the language, which part is the BLAS
library, and which part is simply work the stock pipeline was doing for no reason.

The deliverable is an attribution, not a speed record: a ladder of implementations where
each rung changes exactly one variable, so every delta can be pinned on something.

Spirit: Karpathy's llm.c and minbpe. Small, readable, human-understandable code.
No framework abstractions, no code that handles every case. A clear loop beats a clever
abstraction. Dependencies only where they clearly earn their place.

## Model and task

`answerdotai/ModernBERT-Large-Instruct`, 395,881,664 parameters, classified zero-shot
through its masked-language-model head. AG News topic classification, 4 classes,
7,600 test examples, texts of roughly 40 to 60 tokens.

The checkpoint ships **bf16** (755 MiB). There is no fp32 download for the Instruct
model; fp32 means upcasting on load, and every variant must upcast identically or the
numerical comparison is meaningless.

### Two corrections from the research pass

**AG News does not use A/B/C/D letters.** The reference harness uses semantic
single-token verbalizers, and remaps the dataset's `Sci/Tech` label to `Tech` because
`Sci/Tech` is three tokens while the others are one:

| class | token | id |
|---|---|---|
| Business | `ĠBusiness` | 10518 |
| World | `ĠWorld` | 3645 |
| Sports | `ĠSports` | 15000 |
| Tech | `ĠTech` | 18447 |

**There is no published zero-shot AG News number.** The 95.24 in the paper is a two-epoch
full fine-tune, in a section titled "Full-Finetune". The zero-shot table covers only MMLU,
MMLU-Pro, ADEv2, NIS and OSE. So "accuracy matches the paper" is not available as a
correctness criterion, and we should not pretend otherwise.

This does not hurt the project. The correctness criterion that matters is **agreement
between implementations**, which is strictly stronger than matching a published number.
Zero-shot AG News accuracy is whatever it is; we report it, we require every rung to
produce it to four decimal places, and we note that it is not comparable to the
fine-tuned figure.

### Prompt template

The sources disagree, so pin one and say which. Use the AG News template from the
reference harness (`train.py`), which is the one paired with the verbalizers above.
Note the four trailing spaces after `** Instructions **`, confirmed real:

```
** Instructions **    
Given a short news report, determine which news category it belongs to.
** Content **
What is the category of the following news report?
News Report: {txt}
** Options **
- Business: It is about business or financial news.
- World: It's about world news.
- Sports: It's about sports.
- Tech: It's about Science or Technology, or related companies.
** Answer **
Answer: [unused0] {label}
```

At inference the trailing `{label}` becomes `[MASK]`, and the four verbalizer logits are
compared at the mask position. `[CLS]` and `[SEP]` are never written into the prompt;
the tokenizer's post-processor adds them.

See `docs/research/prompt-template.md` for the MMLU/letter-based variant and for the
places where the model card and the harness disagree.

## The ladder

Each rung changes one variable against the rung above it.

| # | Rung | Isolates |
|---|---|---|
| 0 | HF transformers, stock pipeline, fp32 | baseline, and the correctness oracle for everything else |
| 1 | + MLM head at the mask position only, four verbalizer ids only | wasted work in the stock pipeline |
| 2 | + `torch.compile` (inductor, CPU) | free compiler wins |
| 3 | ONNX Runtime | a different runtime over the same graph |
| 4 | candle, default backend | language and framework, with no BLAS help |
| 5 | candle + `accelerate` feature | the BLAS contribution, alone |
| 6 | hand-written Rust, naive matmul | our own code, without BLAS |
| 7 | hand-written Rust + Accelerate `cblas_sgemm` | our own code, with BLAS |
| 8 | hand-written Rust + fusion, no-copy, arena allocation | what is only possible once you own the code |

Rung 1 is worth its own step because the stock pipeline projects 1024 to 50368 at every
position when only the mask position is needed. On a 60-token prompt that is roughly 10
to 15 percent of total FLOPs discarded, and it is available in Python in a few lines. It
belongs here rather than bundled into the Rust work, where it would inflate Rust's
apparent gain.

Rungs 6 and 7 must stay separate. That split is what distinguishes "Rust is faster" from
"Apple's matrix units are faster", and it is the single most informative measurement in
the project.

Rungs 2 and 3 are the ones expected to fight back: inductor on CPU is inconsistent, and
ModernBERT's alternating local/global attention plus RoPE does not always export to ONNX
without rewriting the mask construction. Both are cheap to attempt and cheap to drop.
"We tried ONNX and it would not export without rewriting attention" is a fine result.

Reordering as we learn things is fine. The invariant to protect is one variable per rung.

**The tokenizer is a sidebar, not a rung.** Tokenizing a 60-token prompt takes
microseconds against a forward pass of roughly 100 ms, so swapping HF `tokenizers` for
our own will not move the end-to-end number. It gets its own benchmark in tokens/sec over
the full test set, where the difference is real and interesting, and the main ladder feeds
every rung byte-identical pre-computed token IDs so tokenizer differences cannot
contaminate model timings.

## What we measure

### Time

- **Batch 1 latency as a distribution**: p50, p90, p99 over ~500 runs. Not the mean;
  means hide exactly the thermal spikes this machine produces.
- **Throughput sweep**: batch sizes 1, 2, 4, 8, 16, 32, 64 in examples/sec.
- **The headline plot**: p99 latency against throughput, one line per rung, points being
  batch sizes. This answers the real question, which is what throughput fits under a
  given p99 budget.
- **Three timings kept separate** on every run: tokenizer, model, end-to-end.
- **Padding waste**: fraction of FLOPs spent on pad tokens per batch size. This is what
  decides whether ModernBERT-style unpadding is worth implementing, as a number rather
  than an argument.
- **Cold start**: process start to first completed inference. Rust should win heavily,
  it matters for serverless, and nobody measures it.

### Memory

Six quantities, each catching something the others miss. All from `getrusage` where
possible, since the kernel's own high-water mark cannot miss a spike the way sampling can.
Note that `ru_maxrss` is in **bytes** on Darwin and kilobytes on Linux.

1. **Peak RSS**, whole process, exec to exit.
2. **Floor RSS**, after process start and before loading any weights. Isolates pure
   runtime overhead. Importing torch costs hundreds of MB before a single parameter
   exists; a Rust binary costs single-digit MB. This is the most quotable number here.
3. **Post-load RSS**, weights resident, before the first forward pass. Catches accidental
   weight duplication, which transiently doubles peak and is otherwise invisible.
4. **Activation delta**, peak during inference minus post-load, swept over batch size.
   A curve, so it needs a sampling thread polling `task_info` every ~10 ms. Answers
   "largest batch that fits in 8 GB".
5. **Allocations per forward pass**, count and bytes, via a wrapping `GlobalAlloc` in
   Rust and `torch.profiler(profile_memory=True)` in Python. Rung 8's target is **zero
   allocations per forward pass**, asserted as a unit test. That is a far stronger claim
   than any peak-RSS table.
6. **Major page faults** (`ru_majflt`), because of the mmap hazard below.

### Energy

`powermetrics` is already being sampled for thermal state, so joules per 1000
classifications is nearly free. For a project about running models on a laptop it is
arguably a better number than milliseconds.

### Three measurement hazards

**mmap makes RSS lie.** Mapped safetensors pages count toward RSS only once touched and
can be evicted under pressure, so a variant that mmaps looks artificially lean beside one
that reads into owned memory, and the figure is not even stable run to run. Every variant
reads into owned memory for the headline comparison. mmap gets its own sidebar, since
lazy loading is a real cold-start optimization that deserves measuring rather than hiding.

**Allocator high-water marks never shrink.** Both the system malloc and PyTorch's caching
allocator retain freed memory, so peak RSS taken after a batch sweep reflects the largest
batch, not the current one. Every batch size needs a fresh process.

**Memory runs and timing runs cannot be the same runs.** Clean peak memory needs a fresh
process per variant per batch size. Trustworthy timing needs the interleaved round-robin
below, which is long-lived and shares allocator state. These requirements contradict each
other, so the matrix gets traversed twice.

## Benchmark protocol (fanless M4 Air)

The machine throttles. The fix is not to fight it but to make the comparison immune to it.

1. **Interleave instead of running to completion.** Never variant A for ten minutes then
   variant B. Alternate in roughly 10-second slices, 30 cycles, order randomized within
   each cycle, so thermal drift hits both about equally instead of punishing whichever
   ran last. This is the most important single change. Memory allows two variants
   resident at once, so each process pairs one variant with the ruler; see the changelog.
2. **A ruler variant.** Re-measure rung 0 in every cycle and report every other rung as a
   ratio to the concurrently measured ruler. Ratios survive throttling in a way absolute
   milliseconds do not. Absolute numbers are still reported; the conclusions rest on the
   ratios.
3. **Log thermal state with every sample.** On Apple silicon the unprivileged signal is
   `ProcessInfo.thermalState` (nominal, fair, serious, critical); `pmset -g therm` records
   nothing here. `powermetrics` gives frequency and power, but only as root. A sample
   without its thermal state is not reproducible. Flag samples taken above nominal, and
   report results with and without them. See the changelog.
4. **Burst and sustained, always both.** Burst is a 30-second run after two minutes idle.
   Sustained is the last 60 seconds of a ten-minute continuous run. One number is a lie on
   a fanless machine.
5. **Pin thread counts explicitly** at 1, 4 and 10 rather than letting each framework
   choose. With 4 performance and 6 efficiency cores, 10 threads is often slower than 4
   for latency-bound work, and that deserves its own small experiment.
6. **Control and record the environment**: AC versus battery (macOS throttles
   differently), Low Power Mode off, Spotlight quiet, nothing else running, lid open,
   macOS version.

**Check before building anything**: `torch.__config__.show()`. If PyTorch on this machine
is linked against its bundled OpenBLAS rather than Accelerate, then rungs 0 through 3 run
on a slower BLAS than rungs 5 and 7, and the headline comparison is indefensible. Thirty
seconds of work that determines whether the project's main claim holds.

The harness is therefore real software with its own correctness requirements, not a
script. It is a first-class crate with its own tests, and it is built before the rungs
exist, because otherwise there will be eight implementations and no trustworthy way to
compare them.

## Verification

The automated gates are what carry correctness while the code runs ahead of review.

### Tier 0: hygiene

`cargo clippy -D warnings`, `cargo fmt --check`, `cargo test`, `ruff`, `pytest`. Nothing
lands red. Environment problems get fixed rather than routed around.

### Tier 1: tokenizer equality

- A committed golden file of HF token IDs for all 7,600 AG News test examples, asserted
  exactly, printing the first mismatch as example index, char offset, expected, actual.
- The splitter alone against HF's ByteLevel regex via `fancy-regex`, so a lexer bug can
  never be confused with a merge bug.
- Exhaustive single-codepoint sweep: all ~1.1M code points in several contexts.
- Proptest fuzzing weighted toward whitespace runs, `\r\n`, NBSP, emoji, CJK, combining
  marks, digits and apostrophes.
- Invariants: pieces concatenate back to the input; `decode(encode(x)) == x` modulo the
  known `add_prefix_space` asymmetry.

### Tier 2: per-layer numerical equivalence

This is the tier that will actually save the project. Do not test only final logits.

PyTorch forward hooks dump intermediate activations into a `.safetensors` file, one tensor
per named checkpoint: embedding output, `embeddings.norm` output, the output of every one
of the 28 layers, RoPE'd q and k for one local and one global layer, post-softmax
attention weights for one local and one global layer, `final_norm` output, head output at
the mask position, and the four verbalizer logits. A Rust test loads that file and asserts
each checkpoint independently, so a RoPE bug fails at the RoPE checkpoint instead of
surfacing as a mysterious 1e-2 at the end.

Three metrics per checkpoint, because each catches what the others wave through:
max absolute difference, max relative difference, and cosine similarity (which catches a
systematic scale error that absolute difference on small activations would miss).
Thresholds in fp32: activations <= 1e-4, logits <= 1e-3 after accumulation, cosine
>= 1 - 1e-6.

**The fixture set must contain long inputs.** ModernBERT's local attention window is 128
total, 64 each side, inclusive, so 129 keys per query.

Measured, once the prompts existed: AG News *texts* are 19 to 248 tokens with a median of
52, but the *prompt* around them runs 119 to 348 tokens with a median of 152, and **96.8%
of the 7,600 prompts exceed 129 tokens**. So the real workload does engage the sliding
window, which an earlier draft of this plan got wrong by reasoning from the text length
instead of the prompt length.

The window still only bites gently at that size: a query in the middle of a 152-token
sequence sees 129 of the 152 keys, so a wrong mask shifts the result by little and may
hide inside the tolerance. The fixtures therefore still include synthetic inputs of 300+
and 600+ tokens, where a query sees a fifth of the sequence and a mask bug cannot hide.
The reason is signal strength, not coverage.

Eight fixtures: shortest real, longest real, 300+ tokens, 600+ tokens, non-ASCII,
digit-and-punctuation heavy, multi-space (to exercise the whitespace added tokens), and
one near-empty.

**Two per-layer tests, because they answer different questions.** The full-stack test
runs all 28 layers and compares each output, so its gap includes drift carried up from
every layer below. A layer that builds a very large residual value amplifies that drift:
on one fixture the gap steps up about 40x at layer 19, where a value of about -20,800
forms. The isolated test feeds each layer PyTorch's own input from the layer below and
compares only that layer's output, so a gap there belongs to that layer alone. The
isolated test carries the tight bound and is the bug detector. The full-stack test
carries a bound for accumulated drift, set from measurement, and the final test of the
stack is the logits and the prediction.

### Tier 3: behavioural equivalence on the full set

- Prediction agreement against rung 0 asserted at exactly 100% over all 7,600, printing
  disagreeing indices on failure.
- Accuracy equal to four decimal places.
- Per-class confusion matrix identity, which catches a verbalizer-ordering bug that
  overall accuracy would mask.
- Argmax margin distribution, because two implementations can agree on every prediction
  while their logit margins differ materially, and that difference is a bug.

### Tier 4: benchmark integrity

The gates that keep the numbers honest, which is a separate problem from keeping the code
correct.

- Hash each variant's input token sequences and assert they are identical. A variant that
  got fast by tokenizing differently fails.
- Assert each variant's predictions in the **benchmarked** run, not only in a separate
  correctness run. Getting fast by computing something else must fail loudly.
- Assert the BLAS backend actually linked, per variant. A silently missing Accelerate link
  invalidates the whole comparison and is otherwise invisible.
- Coefficient of variation per measurement, run rejected above ~5%. Too noisy to report is
  a result, not a number to publish.
- Assert thermal state was captured for every reported sample, and that the ruler variant
  is within tolerance of its historical value, else the machine is in a different state and
  the run is void.
- Assert post-load RSS is within ~1.3x of theoretical weight size.
- Assert zero allocations per forward pass for rung 8, as a unit test.

### Tier 5: regression tracking

The results JSON is committed. CI diffs against the previous commit and flags anything
over 10%, on time and on peak RSS alike, so the optimization rungs stay honest as the code
moves underneath them.

## Statistics

Every reported number carries an interval. A bare number cannot be compared with
another bare number, and this project is a list of comparisons.

This section is the analysis plan. It is written before the benchmark harness exists, so
the choices below are not tuned to the results. A change made after seeing data goes in the
changelog at the end of this file, with the reason.

### Accuracy

- 95% percentile bootstrap interval over the 7,600 test examples, 10,000 resamples, fixed
  seed. The Wilson interval is reported beside it as a cross-check. At n = 7,600 and an
  accuracy near 0.72, the half-width is about 1.0 percentage point.
- Per-class recall and macro-F1, each with a bootstrap interval. Overall accuracy hides a
  class that is never predicted; rung 0 never predicts Tech.
- **Raw and calibrated, both, for every rung.** Calibration is contextual calibration (Zhao
  et al. 2021, "Calibrate Before Use"). It is fitted only on content-free inputs, never on
  test labels, so nothing leaks from the test set. It is a fixed transform of the four
  verbalizer logits, applied identically to every rung's own logits. So it is part of the
  evaluation protocol, not a rung, and it does not touch the one-variable-per-rung rule.
- Two numbers on the same examples are compared **paired**: a paired bootstrap interval on
  the difference, and an exact McNemar test on the discordant pairs. An unpaired test
  would throw away most of the information.

### Agreement between rungs

- Prediction agreement with rung 0, with a Wilson interval. Zero disagreements on 7,600
  examples does not mean the disagreement rate is zero. It means the rate is below 3 in
  7,600, about 0.04%, at 95% confidence (the rule of three). Report it that way.
- The logit difference against rung 0 as a distribution over all 7,600 examples: median,
  p99 and maximum. The maximum alone is one example.
- Agreement is always measured on raw logits. Calibration is applied after.

### Latency and throughput

- **The unit of replication is the process, not the iteration.** Iterations inside one
  process share warm caches, allocator state and thermal state, so they are not
  independent, and treating them as independent gives intervals that are too narrow. Each
  configuration runs in at least 5 separate processes.
- Inside each process: the interleaved round-robin protocol above, a recorded number of
  warm-up iterations discarded, then p50, p90 and p99 from the rest.
- Across processes: the median of the per-process figures, with a bootstrap interval over
  the processes.
- Speedups are ratios against the ruler variant measured in the same cycle, with a
  bootstrap interval over the paired cycles. A ratio of two independent intervals is
  wider and less honest.
- A configuration whose coefficient of variation across processes exceeds 5% is reported
  as too noisy, not averaged into a number.
- Throughput in examples per second follows the same replication at each batch size.

### The A/A floor

Every benchmark session starts with an **A/A run**: the ruler against an identical copy
of itself, with the full protocol and at least 5 replicates. Its speedup should be 1, and
its interval is the smallest difference the session can detect. A variant whose interval
overlaps the A/A interval is reported as "no detectable difference", whatever its point
estimate.

This is needed because macOS gives no way to pin a process to a core. One of two
identical workers can land on efficiency cores and run slower, and that bias would hide
inside every comparison. Replicates in separate processes average it out; the A/A run
measures what is left.

The analysis uses a two-stage bootstrap: resample replicates, then cycles inside each.
Examples within a cycle share a thermal state and replicates share nothing, so resampling
single examples would give intervals that are too narrow. The speedup is the geometric
mean of paired per-example ratios, because ratios multiply.

### Memory

Peak RSS and the other memory figures are nearly deterministic, so report the range over
the 5 fresh processes rather than an interval.

### Many comparisons

The ladder makes many comparisons: rungs, thread counts, batch sizes. Effect sizes with
intervals are the main result. Where a p-value is reported, it is Holm-corrected within a
named family of comparisons, and the family is stated.

## Tokenizer

Byte-level BPE, written here, with HF `tokenizers` used only as a test oracle.

The pre-tokenization regex is **pure GPT-2 and applies unchanged**. The splitter is
already written and verified byte-identical against `fancy-regex` over a 238 MB corpus in
the sibling `bpe-lexer` project, and gets copied in rather than depended on by path, with
the o200k and cl100k state machines trimmed out. `bpe-lexer` remains the home of the big
corpus and the regex oracle; this repo carries only the AG News equality test.

HF's hardcoded ByteLevel pattern, confirmed unchanged across `tokenizers` versions
0.15 through 1.0.0-rc.3:

```
's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

Four things wrap around that regex, and they are the actual work:

1. **An NFC normalizer.** Not null. No lowercasing, no accent stripping.
2. **`add_prefix_space: true`**, which is conditional and per-piece: a space is prepended
   only when the piece does not already start with one, and it runs per span surviving
   added-token extraction, not once per input string. GPT-2's own config has this `false`,
   so this is a real divergence.
3. **A 116-entry added-token layer extracted before the regex.** This includes 23
   whitespace-run tokens (ids 50254 to 50276, for runs of 24 spaces down to 2), which
   change the output even though the regex is byte-identical: `"  A"` gives `[50276, 329]`
   where pure GPT-2 would give `[209, 329]`. `[MASK]` has `lstrip: true` and swallows
   preceding whitespace. This pass is a genuine new component, not a detail.
4. **A `TemplateProcessing` post-processor** wrapping the sequence as `[CLS] $A [SEP]`.
   `type_id` is 0 for sequence B as well, and `token_type_ids` is not a model input.

Merge settings: `ignore_merges: false`, so the standard pair-ranking loop applies with no
whole-pretoken shortcut. 50,280 vocab entries and 50,009 merges, with merges stored as
two-element arrays rather than space-joined strings. `vocab_size` is 50,368 including
added tokens. This is a different algorithm from tiktoken's, which ranks by the
concatenation's vocab id; it needs a `HashMap<(u32,u32), u32>` pair-rank table.

`add_prefix_space: true` has one consequence worth stating loudly: `"A"` and `" A"` encode
identically, and the bare single-letter ids 34 to 37 are unreachable by encoding a letter
on its own. Any code that resolves verbalizer ids by looking up the raw vocab string
instead of encoding through the tokenizer will read logits the model was never trained to
raise. Resolve verbalizer ids by encoding, and assert them against the expected values.

Full details in `docs/research/tokenizer-json.md`.

## Model

28 layers, hidden 1024, intermediate 2624, 16 heads of head_dim 64, vocab 50368,
max position 8192. 173 tensors in a single safetensors shard; there is no
`model.safetensors.index.json`.

```
h = embeddings.norm(tok_embeddings[ids])
for each layer:
    h = h + Attn(attn_norm(h))      # attn_norm absent for layer 0
    h = h + MLP(mlp_norm(h))
h = final_norm(h)
logits = decoder(head.norm(gelu_erf(head.dense(h))))   # mask position only
```

Components to implement: embedding lookup, LayerNorm (weight only, no bias), linear
layers, GeGLU feed-forward, RoPE, and attention alternating global and local.

### Traps, from the research pass

Full list of 30 in `docs/research/architecture.md`. The ones that will actually bite:

- **Layer 0 is global**, not local. "Every third layer" invites reading it as 2, 5, 8.
  Global layers are `layer_id % 3 == 0`, so 0, 3, 6, 9, 12, 15, 18, 21, 24, 27. Layer 27,
  the last, is also global.
- **Layer 0 has no `attn_norm`.** The tensor is absent from the checkpoint. A loader that
  defaults a missing norm weight to ones silently double-normalizes. Make it an `Option`
  and skip the LayerNorm entirely.
- **The window has three off-by-ones.** `local_attention: 128` is the **total** window, so
  64 each side, and the predicate is **inclusive**, giving 129 keys per query. Current
  transformers sets `sliding_window = config.sliding_window + 1` and then the flash
  attention path subtracts it back out. Do not copy the `+1`. The paper does not
  disambiguate total versus per-side; only the code does.
- **Two RoPE thetas, paired counterintuitively**: local layers get 10000, global layers
  get 160000. Full 64 dims, `rotate_half` half-split convention, not interleaved. An
  interleaved implementation produces garbage with plausible norms and never crashes.
- **GELU is the exact erf form, not the tanh approximation.** This hits both the MLP and
  `head.act` across 28 layers. In candle that is `gelu_erf()`; candle's bare `gelu()` is
  the tanh approximation.
- **GeGLU activates the FIRST half.** `input, gate = Wi(x).chunk(2, -1)` then
  `act(input) * gate`. Contiguous halves of 2624 rows each, not interleaved. This is
  backwards from the LLaMA/SwiGLU habit.
- **LayerNorm, not RMSNorm**, despite config docstrings suggesting otherwise. The mean must
  be subtracted, and because `norm_bias: false` gives one tensor per norm, the state dict
  shape does not disambiguate this for you.
- **`decoder.bias` exists and is the only bias in the file**, while `head.dense` has none.
  Dropping it shifts every MLM logit and changes the argmax, which is the entire inference
  protocol. A loader that zero-fills missing biases gets every bias right except the one
  that matters.
- **`decoder.weight` is absent**, tied to `tok_embeddings.weight`.
- **`Wqkv` is three contiguous 1024-row blocks** `[Q|K|V]`, head-major within each, not
  per-head interleaved. No QKV bias. Scale 0.125.
- **The padding mask uses `finfo(dtype).min`, not `-inf`**, shaped `[bs,1,1,seq]`, masking
  padded keys only and never padded queries, so pad rows emit finite junk. Harmless if pad
  outputs are ignored, fatal under mean pooling. True `-inf` risks NaN on a fully masked
  row.
- **Local layers need the padding mask OR'd into the window mask.** Build both masks once
  outside the layer loop and select by layer type.
- **Softmax in fp32 then cast back**, and cos/sin tables built in fp32. Accumulate in f32
  even with bf16 weights.
- **`vocab_size` 50368 exceeds the real vocabulary** (50,285 real plus 83 unused, padded to
  a multiple of 64). An unrestricted argmax can return an undecodable id.
- **Vestigial config fields**: `position_embedding_type: "absolute"` is never read and no
  such tensor exists; `layer_norm_eps` is ignored in favour of `norm_eps` (1e-5).
- **Name prefix asymmetry**: encoder tensors are `model.layers.*`, `model.embeddings.*`,
  `model.final_norm.*`, but head tensors are not prefixed (`head.dense.weight`,
  `decoder.bias`).

Estimated size: roughly 600 to 1000 lines of Rust.

## Repository layout

```
crates/mb-tokenizer/   byte-level BPE: added tokens, NFC, splitter, merges, post-processor
crates/mb-model/       safetensors loader and the forward pass
crates/mb-harness/     the benchmark harness: interleaving, ruler, thermal, memory, energy
python/                rungs 0 to 3, reference activation dumps, plots
docs/research/         verified findings from the research pass, with sources
```

## Phase 2: serving throughput on cloud CPUs

Decided on 2026-11-07 and 08, after rung 14. The goal moves from batch-1 latency on this
Mac to serving: an HTTP server with dynamic batching on a cloud CPU, not the Mac. GPUs
are out of scope for now.

**The target metric, set on 2026-11-09.** The most examples per second per dollar while
the p99 request latency stays at or under **200 ms**. A request's latency runs from its
arrival at the server to its reply, so it includes the wait in the batching queue. Every
cloud result is read against this budget. Throughput above the budget, for example at
batch 16 or 64, is reported only as a ceiling, never as the result.

Why it changed: the first Genoa rough checks (rung 16, 4 cores) took 460 ms per request
at batch 1 and 6.0 s per batch at batch 16. Batch 16 and 64 are useless under any
interactive budget, and on 4 cores even batch 1 in fp32 misses it.

What follows from it:

- **Small batches only.** Cloud runs sweep batch 1, 2, 4 and 8. Larger batches are run
  once per host, as the ceiling.
- **Cores per request are a lever.** Batch-1 latency falls as one request gets more
  threads, until the matmuls are too small to split. So each host gets a thread sweep at
  batch 1 (4, 8, 16, 32 threads) to find the fewest cores that meet the budget. The
  serving question is then how many such workers fit on one box, and what that box costs.
- **Precision matters more.** bf16 and int8 (rungs 17 and 20) shorten each request, and
  that is what makes the budget reachable on few cores.
- **The server rung is the judge.** Its load test raises the Poisson arrival rate until p99 passes
  200 ms, and reports the throughput at the last rate that met it, and that per dollar.

**Rules for this phase.**

- No Mac-only work: no own SME kernels, no Neural Engine, no Metal or MLX, no tuning that
  only helps Accelerate. Every rung must also build and run on Linux x86.
- One variable per rung still holds, and rung 0 stays the ruler and the oracle.
- The Mac stays the machine for paired benchmarks, because it is the one quiet machine we
  control. Cloud hosts give rough checks only (see below).

**Target hosts.** A probe of RunPod hosts on 2026-11-08 found no AMX (Intel Sapphire
Rapids or later). It found two CPUs worth targeting:

| Host CPU | Matrix instructions | Same CPU on AWS |
|---|---|---|
| AMD EPYC 9254 (Zen 4, Genoa) | AVX-512 with bf16 and VNNI (int8) | c7a, m7a |
| Intel Xeon Gold 6342 (Ice Lake) | AVX-512 with VNNI (int8), no bf16 | c6i, m6i |

So bf16 is measured on Genoa, int8 on both. AMX (AWS c7i) is a later check, if an AWS
host becomes available; the same bf16 and int8 code paths should use it through the
library, with no change of ours.

**The hosts after 2026-11-09.** Those two CPUs came on GPU pods, whose containers are capped
near 5 CPUs, too few for the thread sweep. A probe of RunPod's CPU-only pods on
2026-11-09 found:

| RunPod flavor | Host CPU | Matrix instructions | Price, secure cloud |
|---|---|---|---|
| `cpu5c` (compute-optimized) | AMD EPYC 9575F (Zen 5, Turin) | AVX-512 with bf16 and VNNI, no AMX | $0.035 per vCPU-hour, up to 32 vCPUs ($1.12/h) |
| `cpu3c` | AMD EPYC 7713 (Zen 3, Milan) | no AVX-512 | $0.03 per vCPU-hour |

So `cpu5c` is the main host from here: a current server CPU (Zen 5 is the CPU of AWS
c8a and GCP C4D), on the RunPod account and tools we already use, with no setup. `cpu3c`
is skipped. A vCPU is likely a hardware thread, not a core; the first run records which.

AMX (Intel Xeon 4th generation and later: AWS c7i and c8i, GCP C3 and C4) is not on
RunPod. It is the second host, for one rough check of rungs 16 to 18 once they are done,
on a spot instance. Aggregators put spot prices near $0.02 to $0.05 per vCPU-hour, close
to RunPod's. It needs an AWS or GCP account, which only the user can set up. PyTorch's
bf16 path also uses AMX through oneDNN, so rung 0 gains from it as well, and the
ratios there may be smaller.

**Rough checks on cloud hosts.** A RunPod pod shares its host with other renters. There is
no `thermalState` and no `powermetrics` on Linux. So a cloud run records, with each slice,
the CPU clock from `/proc/cpuinfo` and the steal time from `/proc/stat` in place of hard
rule 3's thermal state, and still pairs every slice with rung 0 in the same process. Its
results are labelled rough checks, never mixed into the Mac tables.

**Proposed rungs.** The order may change as data comes in.

| # | Rung | Isolates |
|---|---|---|
| 15 | Last layer at the mask row only | work the head never reads: the last layer is global, so all keys are still needed, but only the mask query |
| 16 | The same code on Linux x86, with MKL in place of Accelerate, fp32 | the platform; the first rung that builds on Linux |
| 17 | bf16 weights and bf16 matmul, f32 accumulation | the precision of the matmuls |
| 18 | MKL serial inside the attention threads | a bug: each thread of ours opened its own OpenMP team through MKL, so t threads became about t squared |
| 19 | Our threads taken from the OpenMP pool that oneDNN and MKL use | two thread pools competing on the same cores |
| 20 | int8 weights and activations, dynamic per-row scales | the precision again, one step further |
| 21 | HTTP server with dynamic batching (max batch at most 8, max wait chosen to fit the 200 ms budget) | serving overhead; load-tested open-loop, with Poisson arrivals, judged by throughput at p99 of 200 ms |

Rungs 17 and 20 change the numbers, so Tier 3 for them is the
agreement rate with rung 0 and the accuracy, not a logit bound; the bound they must meet
is set before they are measured.

**Rung 17's bound, set on 2026-11-09 before any bf16 logit was seen.** Over all 7,600
examples: agreement with rung 0 of at least 99.0% (at most 76 changed predictions), and
calibrated accuracy within 1.0 point of rung 0's 84.78%. The logit differences are
reported, with no bound. The design: the checkpoint is bf16, so bf16 weights are exactly
the checkpoint's values and lose nothing; only the activations entering each linear
layer are rounded to bf16 (round to nearest even, as PyTorch rounds), and the products
accumulate in f32. The attention products, the norms, the softmax and GELU stay f32. The
in-run gate (hard rule 5) for this rung is set from its Tier 3 run, as the fp32 gate
was: 4x the worst logit difference seen, and at most twice the disagreement rate seen.

**Rung 20's bound, set on 2026-11-10 before any int8 logit was seen.** Over all 7,600
examples: agreement with rung 0 of at least 97.0% (at most 228 changed predictions), and
calibrated accuracy within 1.5 points of rung 0's 84.78%. The logit differences are
reported, with no bound. The design: the same five linear layers as rung 17 (Wqkv, Wo,
Wi, the MLP's Wo, head.dense). Their weights are s8 with one scale per output column.
Their activations are s8 with one scale per row, computed from that row's largest value
on each call, so nothing is calibrated in advance. The products accumulate in s32 and are
scaled back to f32 by the two scales. The attention products, the norms, the softmax
and GELU stay f32. The in-run gate is set from its Tier 3 run, by the same rule as rung 17.

**Rung 22's bound, set on 2026-11-10 before any rung 22 logit was seen.** Over all 7,600
examples: agreement with rung 0 of at least 99.0% (at most 76 changed predictions), and
calibrated accuracy within 1.0 point of rung 0's 84.78%: rung 17's bound, since rung 22
must be as safe to serve as bf16. The design: rung 20's s8 matmul only for the MLP's Wi
and the attention's Wo; Wqkv, the MLP's Wo and head.dense stay bf16. The two kinds were
chosen by putting each kind alone in s8 on the first 760 examples and comparing logits
with rung 0, with no labels read: Wi alone changed 0 of 760 predictions (p99 logit
difference 0.22) and the attention's Wo 1 (0.10), against 3 for Wqkv (0.41) and 4 for
the MLP's Wo (0.31). The in-run gate is set from its Tier 3 run, by rung 17's rule.

**Libraries: the ones PyTorch uses on each platform.** Agreed on 2026-11-08. Each rung
uses the same matmul library as rung 0 on the same machine, so a comparison never mixes
libraries:

| Platform | fp32 matmul | bf16 and int8 matmul |
|---|---|---|
| macOS (this Mac) | Accelerate, as now | (not measured here) |
| Linux x86 (Genoa, Ice Lake) | Intel MKL, `cblas_sgemm` | oneDNN, which also uses AMX when present |

Both are C libraries reached through FFI, as Accelerate is now. So rung 16 is the same
code built for Linux with MKL, compared with rung 0 on the same host, which also uses MKL.
Rungs 16 to 20 are measured only on cloud hosts, as rough checks: MKL does not run on
Apple silicon, and oneDNN is weak there. One known bias: MKL has taken slower code paths
on AMD CPUs. Rung 0 shares it, so the ratios stay fair, but absolute Genoa numbers may
understate the CPU.

## Open questions

- **Benchmark machine.** Answered for phase 2: the target is a cloud CPU, and the M4 Air
  stays the measurement machine. See "Phase 2" above.
- **Batching.** Answered by rung 14: unpadding. Rung 0 padded loses throughput above
  batch 4, and rung 14 unpadded gains up to batch 64. Under the 200 ms budget of phase 2,
  only batches up to 8 count.
- **Precision.** fp32 up to rung 16. bf16 and int8 are rungs 17 and 20, because on cloud
  CPUs fp32 cannot use the matrix instructions.
- **GitHub.** Local git now; push to `augustasio/modernbert.rs` once the tokenizer matches
  HF exactly, so the first public commit is not a skeleton.

## Changelog

Changes to the plan made after seeing data, with the reason.

- **Fixture justification.** An earlier draft said AG News examples are too short to engage
  the 129-key window. That reasoned from text length. The prompts have a median of 152
  tokens and 96.8% exceed 129, so the window is engaged. The long fixtures stay, for signal
  strength.
- **Per-layer bounds.** The first full-stack bound, 1e-5 relative to peak, was set before
  measuring. Two fixtures exceeded it after layer 19. The isolated per-layer test showed
  every layer correct on its own, at 7e-8 for layer 19, so the excess is amplified drift and
  not a bug. The tests now separate the two, as described in Tier 2.
- **Calibrated accuracy added.** Rung 0 never predicts Tech because of a class prior the
  model has before it reads the article. Calibrated accuracy is now reported beside raw
  accuracy for every rung. Rung 0's raw logits stay the oracle.
- **Thermal state, not `CPU_Speed_Limit`.** `pmset -g therm` records no levels on Apple
  silicon, so `CPU_Speed_Limit` does not exist here. The unprivileged signal that does
  exist is macOS's own `ProcessInfo.thermalState`: nominal, fair, serious or critical. It is
  coarse, so the ruler variant measured in every cycle stays the main defence against
  throttling, and the thermal state is recorded with every sample to flag and discard.
- **Energy and CPU frequency need root.** `powermetrics` refuses to run without it. They
  are measured only if the benchmark can call `powermetrics` with elevated rights; if not,
  they are reported as not measured rather than estimated.
- **Two variants resident, not all of them.** Each variant holds the model, about 1.6 GB,
  so all of them at once would need about 11 GB on a 16 GB machine. Each benchmark process
  therefore holds the ruler and one variant and interleaves the two in randomized slices.
  Every variant is still measured against a ruler run in the same cycles, which is the
  property the interleaving existed for, and variants are compared with each other
  through their ratios to the ruler.
- **Rung 1 is one change, not two.** The ladder listed eval and no-grad hygiene with the
  mask-only head, but rung 0 already runs in `eval()` under `torch.inference_mode()`, as HF
  pipelines do. So rung 1 changes only the head, which keeps it one variable. Measuring
  the cost of autograd would be its own rung.
- **The A/A floor added.** The first smoke run, rung 7 against an identical copy of
  itself, gave a speedup of 0.956 with an interval of [0.933, 0.980], which excludes 1.
  It was one replicate of 6 examples on a machine running another job, so it proves
  nothing, but it shows why a same-against-same run must set the detection floor of
  every session. See "The A/A floor" in Statistics.
- **Cold start needs an empty compiler cache.** Rung 2's first compile took 32.6 s in a
  fresh run and 14.5 s in a later process, because inductor caches compiled kernels on
  disk across processes. A cold-start figure for rung 2 is therefore taken with a fresh,
  empty `TORCHINDUCTOR_CACHE_DIR` per process, and the warm-cache figure is reported
  beside it. Latency runs are unaffected: compiling happens before the worker reports
  ready, outside every timed slice.
- **The benchmark's logit gate is 1e-2, not 1e-3.** The full-set comparison found one
  example (index 4100, 173 tokens) where rung 4, candle with its default `gemm` matmul,
  differs from rung 0 by 2.55e-3. The prediction is the same, with a margin of about one
  logit. Rungs 2, 5 and 7 differ on that example by 2.7e-4 to 2.8e-4, so it is a sensitive
  example, and rungs 4 and 5 run the same candle graph with only the matmul library
  changed, so the cause is summation order, not a model difference. The 1e-3 tier-2 bound
  on the short fixtures stays. The in-run gate of hard rule 5 must not fail a correct
  rung, so it is now 4x the worst difference seen across all rungs and all 7,600
  examples, and it still requires the same argmax.
- **Energy and CPU frequency are now measured.** The README's sudo rule lets the harness
  run `sudo -n /usr/bin/powermetrics` with no password. One sampler runs for the whole
  benchmark at 100 ms and the harness records its raw samples on the same clock as the
  slices. The analysis uses only samples wholly inside a slice, because a sample that
  straddles an edge mixes two workers' power; a synthetic test showed the bias that
  splitting such samples by overlap gives. So a slice that feeds an energy figure must be
  several seconds long. The figure is whole-CPU energy, including `powermetrics` itself
  and the idle worker, and the A/A run measures that floor.
- **Rung 8 keeps rung 7's arithmetic exactly.** Profiling rung 7 put 52% of its time
  in the two attention loops, which read K and V with a stride of 1024 floats and sum
  each `q . k` one add at a time. Rung 8 (`crates/mb-model/src/fused.rs`, feature
  `fused`) stores each head's Q, K and V contiguously, with K transposed, so the vector
  runs across keys and every score still adds its 64 products in rung 7's order. The
  local window is a key range instead of an additive mask, RoPE is applied while
  scattering the heads, the residual add and the next norm share one pass, and every
  buffer lives in a reused workspace, so a forward pass makes zero allocations (a test
  asserts it with a counting allocator; Accelerate's own `malloc` calls are outside
  what it can see). The result is bit-identical to rung 7, which a test asserts at
  every checkpoint, so rung 8's speedup is layout alone. One fusion was tried and
  dropped: adding the output projections onto the residual stream inside `cblas_sgemm`
  (`beta = 1`) made the drift against PyTorch five times worse, because the residual
  holds values near 28,000. Two levers remain and are not in rung 8, because each
  changes something other than layout: `exp` (17% of rung 8's time, f64 libm, used by
  softmax and GELU) and the attention products, which could go to `cblas_sgemm` too.
- **Rung 9 changes only `exp`.** It replaces f64 libm `exp` in the attention softmax and
  in `erf` with an f32 polynomial (`crates/mb-model/src/expf.rs`, feature `expf`), the
  Cephes `expf` method, written as plain branch-free Rust so LLVM vectorises it on NEON.
  It is within 2 ulp of f64 `exp` over a 35-million-value sweep, and exactly 0 below
  the f32 range, so a masked key still adds nothing. The f64 softmax sum and the f64
  `erf` arithmetic stay, so only `exp` changes. On 400 examples it saves 16 ms of 209
  (8 ms each from the softmax and `erf`), less than the 17% that `exp` took in the rung 8
  profile: the ordered f64 sum and the f64 `erf` arithmetic around it remain. Rung 9 is
  no longer bit-identical to rung 8, but it agrees 7600/7600 with rung 0, with Tier 2
  errors no larger than rung 8's. The 1-thread session showed that threading explains
  little of rung 8's gap to rung 0 (1.34x on one core, 1.47x on four), so the per-core
  work goes first.
- **Rung 10 moves the attention products into `cblas_sgemm`.** Per head, `scale * Q K^T`
  is one `[seq, 64] x [64, seq]` product and `P V` one `[seq, seq] x [seq, 64]` product
  written straight into the head's columns of the context (feature `attention-sgemm`,
  `attend_sgemm` in `fused.rs`, `sgemm_strided` in `accelerate.rs`). This is what
  PyTorch's CPU attention does. A local layer computes every score, as PyTorch does, and
  the keys outside the window get probability exactly 0. Accelerate sums in its own
  order, so rung 10 is not bit-identical to rung 9 and the fused-equals-plain test is
  off for it. Tier 2: worst stack error 8.2e-5 of peak (rung 9: 6.1e-5; bound 3e-4).
  Tier 3: 7600/7600 agreement, max logit difference 4.6e-4 (rung 9: 2.5e-4; bound 1e-3).
- **Rung 11 splits the work outside the big matmuls across threads.** Attention is split
  by head: each thread scatters, rotates and attends its own heads, with its own score
  buffer. The norms, the residual adds and the GELU gate are split by blocks of rows. The
  thread count is `MB_THREADS`, the same one every rung gets (feature `threads`,
  `fused/threads.rs`). This is how PyTorch spreads the same work over its OpenMP pool. The
  threads are `std::thread::scope`, started and joined five times a layer, because a pool
  kept alive between calls needs a dependency or `unsafe`. So with more than one thread a
  forward pass allocates again (thread spawns); with one it still allocates nothing. Each
  head now writes its own `[seq, 64]` block, copied into the context afterwards, because
  the heads' columns of the context interleave. The arithmetic is rung 10's: every layer
  is bit-identical to rung 10's at 1, 3 and 4 threads, and all 7,600 Tier 3 logits are
  bit-identical to rung 10's.
- **Rung 12 moves the arithmetic still in f64 to f32.** That is `erf` in GELU (the same
  Abramowitz and Stegun formula, now `x * 0.5 * (1 + erf(x * M_SQRT1_2))` in f32, as
  PyTorch writes it), the LayerNorm mean and variance, and the softmax total (feature
  `f32-math`, `f32math.rs`). One variable, the precision of the non-matmul arithmetic,
  so not split into two rungs. The sums keep 16 lane totals added pairwise at the end,
  because the residual stream reaches about 28,000 and a plain f32 sum would lose the
  small terms; PyTorch's `RowwiseMoments` cascades in the same spirit. Tier 2: worst stack
  error 1.16e-4 of peak (rung 11: 8.2e-5; bound 3e-4), worst verbalizer gap 4.6e-5.
  Tier 3: 7600/7600 agreement, max logit difference 7.0e-4 (rung 11: 4.6e-4; bound 1e-3).
- **Rung 13 moves the weights into the model instead of copying them.** Until rung 12,
  each component's loader cloned its tensors out of the `Checkpoint`, which then dropped
  at the end of `ModernBert::load`. So two f32 copies of the weights (1,507 MiB each)
  existed at once, and the peak was about 3,160 MiB, the highest of any rung. The memory
  session also showed 750 to 1,040 MiB more than the weights still resident after load.
  `vmmap` names it: 911 to 963 MiB of dirty "Malloc Large (empty)", which is macOS's
  allocator keeping the freed checkpoint copy cached in the process. Rung 13 (feature
  `owned-weights`, `Checkpoint::take`) removes each tensor from the checkpoint and moves
  its values into the model, so only one copy ever exists, and `ModernBert::load` now
  refuses a checkpoint with any tensor left unread. One variable, the load path; the
  forward pass is rung 12's. All 7,600 Tier 3 logits are bit-identical to rung 12's.
- **Phase 2: serving on cloud CPUs.** After rung 14 the goal moved from batch-1 latency
  on this Mac to serving throughput on a cloud CPU. A RunPod probe found no AMX, but
  found Zen 4 (AVX-512 bf16) and Ice Lake (AVX-512 VNNI) hosts, which match AWS c7a and
  c6i. So bf16 and int8 target those, and the Mac stays the paired-benchmark machine.
  Matmul libraries follow PyTorch per platform: Accelerate on macOS, MKL and oneDNN on
  Linux x86. See "Phase 2".
- **Rung 15 runs the last layer at the mask rows only** (feature `mask-rows`,
  `ModernBert::last_layer_at_masks` in `fused/threads.rs`). The last layer is global, so
  the attention norm and `Wqkv` still run on every row, for the keys and values. But only
  the mask query is scored, and `Wo`, the residual add, the MLP norm and the MLP run on
  one row per sequence: about three quarters of the last layer's matmul work, or about 3%
  of the forward pass. The mask query's scores are dot products in a loop instead of an
  `sgemm`, so the sums run in a different order: the logits differ from rung 14's by at
  most 5.1e-5. Tier 2: worst verbalizer gap 4.9e-5. Tier 3: 7600/7600 agreement, max
  logit difference 7.1e-4 (bound 1e-3).
- **Rung 16 is rung 15 built on Linux x86 with MKL**, measured on a RunPod Genoa host
  (`results/bench/cloud_genoa_t4/SUMMARY.md`). `accelerate.rs` links `mkl_rt` on Linux;
  nothing else changes. Tier 3: 7600/7600 agreement, max logit difference 3.7e-4.
  Rough check: 1.10x rung 0 at batch 1 and 1.51x at batch 16, against an A/A floor of
  1.004 [1.000, 1.008]. Two harness facts learned there: cloud slices are 16 examples,
  because fp32 takes about 0.5 s an example on 4 Genoa cores; and steal time is always
  0, because a RunPod pod is a container on bare metal, so the paired design and the
  A/A floor, not steal, are the defence against other renters' load.
- **Rung 17 passes its bound.** bf16 linear layers through oneDNN 3.13.3, on Genoa:
  7598/7600 agreement, calibrated accuracy 84.72% against 84.78%, worst logit difference
  0.179. The in-run gate is 0.72 and a share of 2 x 2/7600.
- **Phase 2 gets a latency budget: p99 at or under 200 ms per request** (2026-11-09).
  The goal was throughput with "some latency is fine". The Genoa rough checks showed
  what that hides: 460 ms per request at batch 1 and 6.0 s per batch at batch 16, on 4
  cores. So the target is now throughput per dollar within the budget; cloud runs sweep
  batch 1 to 8 and threads per request; and the main host moves from GPU pods (about 5
  CPUs) to RunPod CPU pods on Zen 5 (`cpu5c`, up to 32 vCPUs). AMX on AWS or GCP is a
  later check that needs the user's account.
- **Rung 17 rough check on Genoa, 4 threads** (`results/bench/cloud_genoa_t4_r17/SUMMARY.md`):
  2.04x rung 0 at batch 1 and 3.10x at batch 16, against rung 16's 1.10x and 1.51x. No
  prediction changed in the in-run check. Batch-1 p99 is 339 ms, so 4 cores miss the
  200 ms budget even in bf16. The A/A floor on that pod was 0.993 [0.988, 0.998]: read
  ratios within about 1% of 1 as no difference.
- **Two new rungs before int8: 18 and 19** (2026-11-09). The thread sweep on Zen 5 found
  rung 17 getting slower with more threads: 723 ms at 8 and 1,110 ms at 16 at batch 1,
  against rung 0's 300. On the pod, `MKL_NUM_THREADS=1` alone gave 163 ms at 8 threads, and
  turning off OpenMP's spin-wait helped further at 32. So two problems: each attention
  thread of ours opened its own OpenMP team through MKL (rung 18 fixes it with MKL's
  per-thread setting), and our spawned threads compete with the OpenMP pool (rung 19 runs
  them on it). All rung 16 and 17 figures so far carry the first problem. int8 becomes
  rung 20 and the server rung 21. Rungs 18 and 19 must meet rung 17's bound, set before
  they are measured: at least 99.0% agreement with rung 0 and calibrated accuracy within
  1.0 point of 84.78%.
- **Rungs 18 and 19 pass rung 17's bound** (Zen 5, `cpu5c`, batches of 16 at 16
  threads). Both: 7598/7600 agreement, calibrated accuracy 84.72%, worst logit difference
  0.179, the same two changed predictions as rung 17 (examples 401 and 6503). They change
  threads, not arithmetic, and the logits are bit-identical to each other. Tier 3 wall
  clock, not a benchmark: 0.083 s an example for rung 18 and 0.0545 s for rung 19.
- **Rung 20 fails its bound** (2026-11-10, Zen 5, oneDNN `brg_matmul:avx512_core_vnni`).
  Agreement 7566/7600 (99.55%) passes the 97.0% floor, but calibrated accuracy is 83.12%
  [82.28, 83.97], 1.66 points below 84.78%, outside the 1.5 points allowed. Worst logit
  difference 1.17. It is also slower than rung 19: 0.088 s an example at batch 16 (Tier
  3 wall clock) and 0.096 s against 0.076 s at batch 1. So it is not benchmarked and gets
  no in-run gate. The bound stays as set. A later int8 attempt (for example, keeping
  head.dense in bf16, or static scales) is a new rung with its own bound, set first.
- **A bug in rung 19's pool, fixed** (`omp.rs`, 2026-11-10). `parallel_for` asked
  libgomp for one thread per block. Attention has one block per sequence and head, so
  batches of 2 to 8 asked for 32 to 128 threads on 16 cores, and 4 or 8 threads asked
  for 16. The extra threads in libgomp's pool spun against the matmuls: at batch 2,
  about 7% of calls took 1.5 s against a 150 ms median. The team is now capped at
  `omp_get_max_threads()`. Batch 1 at 16 threads asked for exactly 16 and is unchanged;
  the other rung 19 sweep jobs were rerun.
- **The Zen 5 sweep answers the budget question for batch 1**
  (`results/bench/cloud_turin_sweep/SUMMARY.md`). Rung 19 at 16 threads: p99 115 ms,
  15.4 examples/s, 4.4x rung 0, about 49,500 examples per dollar on a $1.12/h pod. Rung 0
  misses 200 ms at every thread count. Batches of 2 to 8 miss it too and add at most 13%
  throughput; a few of their calls still stall for about 1.5 s after the team cap, cause
  not yet found. 4 threads give 8.6 examples/s on a quarter of the cores, so rung 21
  should test several workers per host before batching.
- **Four workers of 4 threads beat one of 16** (`results/bench/cloud_turin_multi/SUMMARY.md`,
  2026-11-10, a closed-loop rough check, not the paired harness). Rung 19 at batch 1 on 16
  Zen 5 cores: 36.7 examples/s at p99 160 ms, against 25.5/s for one 16-thread worker:
  about 118,000 examples per dollar. Rung 0 meets 200 ms in no layout (best 5.8/s at
  p99 216 ms). Threads must be bound through OpenMP (`OMP_PLACES`, `OMP_PROC_BIND=close`):
  that removed every 1.5 s stall seen, while `taskset` caused them. So rung 21 serves with
  four bound 4-thread workers, and its open-loop test adds the queue wait this check
  leaves out.
- **Why rung 20 failed, and rung 22** (2026-11-10). A profile of rung 19 at batch 1 and 4
  threads on Zen 5: 123 ms an example, 103 ms of it the bf16 matmuls (the MLP's Wi alone
  45.5 ms). So the matmuls are the lever. Rung 20's s8 matmuls took only 30 ms, but its
  quantize and dequantize loops ran on one thread and ate the gain; they now run on the
  pool. Its accuracy loss came mostly from Wqkv and the MLP's Wo (measured one kind at a
  time against rung 0, no labels). Rung 22 keeps s8 only for Wi and the attention's Wo,
  with rung 17's bound, set above before measuring.
- **Rung 22 passes its bound** (Zen 5, batches of 16 at 16 threads). Agreement 7583/7600
  (99.78%), calibrated accuracy 84.72% [83.92, 85.53] against 84.78%, worst logit
  difference 0.427. In-run gate: 1.707 and a share of 2 x 17/7600. Tier 3 wall clock, not
  a benchmark: 0.0417 s an example against rung 19's 0.0545. At batch 1 on 4 threads, in
  three interleaved rounds on a busy host (load 113 to 144): rung 19 0.186 to 0.194 s,
  rung 20 (with its loops on the pool) 0.145 to 0.148 s, rung 22 0.169 to 0.175 s. The
  open-loop server test on that host was too noisy to read and is redone
  (`results/bench/cloud_turin_serve22/`). It also showed that a load test which stops
  early checks too few replies to hold a share set over 7,600; `mb-load` now allows as
  many changed predictions as the gate does over one full pass.
