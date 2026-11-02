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

## Open questions

- **Benchmark machine.** The M4 Air is the target. Is the PC also in scope? x86 needs a
  different BLAS (OpenBLAS or MKL) as the Accelerate equivalent.
- **Batching.** Padding with an attention mask, or ModernBERT-style unpadding. The padding
  waste measurement decides this rather than an argument. Start with padding.
- **Precision.** fp32 first, from the upcast bf16 checkpoint. bf16 end to end later, as its
  own rung, if it is worth it.
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
