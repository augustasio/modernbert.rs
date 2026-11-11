# Conventions

Read `PLAN.md` first. It is the agreed plan and the source of truth for what this project
is measuring and why.

## Code style

Written to be read. The reader is someone learning how a transformer actually works, not
someone integrating a library.

- A clear loop beats a clever abstraction. If a `for` loop over indices is the most
  obvious thing, write that.
- No generics, traits or type parameters unless they remove real duplication. Concrete
  `f32` everywhere until there is a measured reason not to be.
- Functions do one recognisable thing and are short enough to read without scrolling.
- Name things after the paper or the reference implementation, so a reader can grep the
  transformers source and find the same word.
- Comments explain *why*, and flag the traps. The trap list in `PLAN.md` is not
  documentation to be read once; each trap gets a comment at the line where it would
  otherwise be silently violated.
- No `unsafe` outside the FFI boundary to Accelerate.
- No `.unwrap()` on anything that depends on input or files. Loader errors say which
  tensor, which shape, and what was expected.

## Dependencies

Only where they clearly earn their place. Current justified set:

- `unicode-general-category` in the tokenizer, for exact general-category tables.
  `char::is_alphabetic()` is **not** regex `\p{L}` and must never stand in for it. Nor is
  `unicode-properties` exact: it reports a letter or number for 17,480 code points that
  `\p{L}` and `\p{N}` reject, because its ranges swallow the unassigned gaps inside letter
  blocks. A corpus test cannot see this, because real text holds no unassigned code
  points. Any replacement must pass the exhaustive sweep in `tests/split_oracle.rs`.
- `unicode-normalization` for NFC, which `tokenizer.json` asks for. The decomposition
  tables and the canonical ordering rules are large and fiddly, so writing them here
  would be transcription rather than insight.
- `memchr`, `serde_json` where parsing JSON by hand would be noise rather than insight.
- `fancy-regex` and HF `tokenizers` as **dev-dependencies only**, as test oracles, and
  `flate2` to read the gzipped AG News golden file. These must never appear in a non-test
  build path.
- Native libraries reached through FFI, the ones PyTorch itself uses on each platform
  (agreed 2026-11-08, see `PLAN.md`): Apple Accelerate on macOS; on Linux x86, Intel
  MKL's `cblas_sgemm` for fp32, oneDNN (`libdnnl`) for bf16 and int8 matmuls, and
  `libgomp` for the OpenMP pool those two share (rung 19). They are matmul kernels and a
  thread pool only; everything else in the forward pass stays ours.

Anything else gets discussed before it lands. In particular: no tensor framework, no
`ndarray`, no autodiff. The point is that the forward pass is ours.

## The hard rules

These exist because violating them silently invalidates the project's results rather than
breaking a test.

1. **Every rung consumes byte-identical token IDs.** Tokenizer differences never
   contaminate model timings. The harness hashes the inputs and asserts this.
2. **One variable per rung.** If a change touches two things, it is two rungs.
3. **A measurement without its thermal state is not a measurement.** Never report a
   timing that was not captured alongside `ProcessInfo.thermalState`. (`CPU_Speed_Limit`
   does not exist on Apple silicon; `pmset -g therm` records nothing here.)
4. **Never report a mean latency.** Percentiles or nothing.
5. **Correctness is asserted in the benchmarked run**, not only in a separate test run. A
   variant that got fast by computing something else must fail loudly.
6. **HF libraries are oracles, never implementation.** If HF code is in the hot path of a
   rung that claims to be from-scratch, that rung is a lie.
7. **fp32 means the same upcast everywhere.** The checkpoint is bf16; every variant
   upcasts identically or the numerical comparison is meaningless.
8. **No test loads the whole checkpoint per test function.** It is 791 MB on disk and
   1.58 GB as f32, and `cargo test` runs test functions in parallel across all ten cores,
   so a per-test load asks for roughly 20 GB on a 16 GB machine. It did, and the machine
   went down. Structural checks use `safetensors::Header`, which reads no values at all;
   a single tensor uses `Checkpoint::load_one`, which seeks to its range. Only the
   forward pass calls `Checkpoint::from_file`, and a test that needs it shares one
   instance across the binary.
9. **This is a 16 GB fanless laptop, and the work has to fit on it.** Before running
   several heavy things at once, add up what they want: the Python side holds the model
   in fp32 (about 1.6 GB plus runtime), a release build with `lto = "fat"` and
   `codegen-units = 1` is memory-hungry on its own, and parallel agents multiply both.
   Swap is capped at 3 GB, so there is no slack to absorb a mistake.

## Verification

`PLAN.md` has the five tiers in full. The habit that matters: when implementing a model
component, add its tier-2 reference checkpoint in the same commit. A component without a
per-layer reference comparison is unverified, regardless of whether the end-to-end logits
happen to match.

Reference activations live in `python/` and are dumped by forward hooks on rung 0. Rust
tests load them from safetensors and compare per checkpoint.

## Commits

Small and individually readable; they are meant to be read after the fact, in order.
One logical change each. Message says what changed and why, not what file was touched.
