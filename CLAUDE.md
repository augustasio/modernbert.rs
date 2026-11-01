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
- `fancy-regex` and HF `tokenizers` as **dev-dependencies only**, as test oracles. They
  must never appear in a non-test build path.

Anything else gets discussed before it lands. In particular: no tensor framework, no
`ndarray`, no autodiff. The point is that the forward pass is ours.

## The hard rules

These exist because violating them silently invalidates the project's results rather than
breaking a test.

1. **Every rung consumes byte-identical token IDs.** Tokenizer differences never
   contaminate model timings. The harness hashes the inputs and asserts this.
2. **One variable per rung.** If a change touches two things, it is two rungs.
3. **A measurement without its thermal state is not a measurement.** Never report a
   timing that was not captured alongside `CPU_Speed_Limit`.
4. **Never report a mean latency.** Percentiles or nothing.
5. **Correctness is asserted in the benchmarked run**, not only in a separate test run. A
   variant that got fast by computing something else must fail loudly.
6. **HF libraries are oracles, never implementation.** If HF code is in the hot path of a
   rung that claims to be from-scratch, that rung is a lie.
7. **fp32 means the same upcast everywhere.** The checkpoint is bf16; every variant
   upcasts identically or the numerical comparison is meaningless.

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
