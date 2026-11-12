# modernbert.rs

A from-scratch Rust forward pass for ModernBERT-Large-Instruct, built as a ladder of small
steps, each measured against stock PyTorch, to find out where CPU inference time goes.

The task is zero-shot AG News topic classification through the model's masked-language
head: 395M parameters, 7,600 test examples, no fine-tuning. Every step ("rung") changes one
thing against the step before it, and must keep its predictions within a stated bound of
PyTorch's. So each speedup can be pinned on one cause, and none of them comes from
computing something else.

Written in the spirit of llm.c: plain loops, no tensor framework, no autodiff. The only
native code is the matmul kernels PyTorch itself uses on each platform (Accelerate on
macOS; MKL, oneDNN and libgomp on Linux x86), reached through FFI.

[PLAN.md](PLAN.md) is the full plan and the changelog of every rung, with its bound and its
result. [CLAUDE.md](CLAUDE.md) holds the conventions.

## Results

### Phase 1: batch-1 latency on a fanless MacBook Air M4 (4 threads)

Speedup is rung 0's time over the rung's, measured in the same process, interleaved, 5
replicates of 30 cycles each. The A/A floor (rung 0 against itself) is about ±1%.
Intervals are 95% bootstrap. Full table: [results/bench/t4/SUMMARY.md](results/bench/t4/SUMMARY.md).

| Rung | What changes | Speedup vs rung 0 | p50 ms |
|---|---|---|---|
| 0 | HF transformers, stock pipeline, fp32 | 1 | 137 |
| 1 | MLM head at the mask position only | 1.12 | 117 |
| 2 | + `torch.compile` | 1.12 | 118 |
| 3 | ONNX Runtime | 0.98 | 140 |
| 5 | candle + Accelerate | 0.78 | 167 |
| 6 | our Rust, naive matmul | 0.008 | 23,800 |
| 7 | our Rust + Accelerate `sgemm` | 0.41 | 338 |
| 8 | + fused layout, zero allocations (bit-identical to 7) | 0.68 | 220 |
| 9 | + f32 `exp` | 0.73 | 189 |
| 10 | + attention products in `sgemm` | 0.97 | 146 |
| 11 | + threads outside the matmuls | 1.14 | 116 |
| 12 | + f32 `erf`, norms and softmax sums | 1.22 | 110 |
| 13 | + weights moved into the model, not copied | 1.22 | 105 |
| 15 | + last layer at the mask rows only | 1.24 | 104 |

What it says:

- **The matmul library is the floor, not the gap.** With the same Accelerate `sgemm`,
  our first forward pass (rung 7) took 2.5x as long as PyTorch. The difference was all in
  the code between the matmuls: attention, softmax, norms and GELU.
- **Layout and threading close it.** Contiguous heads with K transposed (rung 8, same
  logits bit for bit), f32 `exp` (rung 9), attention through `sgemm` (rung 10), and
  threads for the work between matmuls (rung 11) take a request from 338 ms to 116 ms.
- **Memory falls more than time.** Rung 13 peaks at 1.7 GB against PyTorch's 2.6 GB, and
  starts in 0.54 s against 1.93 s ([results/memory/t4_quiet/SUMMARY.md](results/memory/t4_quiet/SUMMARY.md)).

### Phase 2: serving throughput on a cloud CPU

The goal: the most examples per second per dollar while the p99 latency, queue included,
stays at or under 200 ms. Host: RunPod `cpu5c`, AMD EPYC 9575F (Zen 5), 16 cores, $1.12/h.
An HTTP server (`mb-serve`) runs several bound workers; an open-loop Poisson load test
(`mb-load`) raises the rate until the p99 passes 200 ms, and checks every reply against
rung 0's logits as it goes.

| Implementation | Best rate within 200 ms | Examples per dollar |
|---|---|---|
| Rung 0, PyTorch fp32 | none (p99 415 ms even at 1/s) | 0 |
| Rung 19, our bf16 matmuls + OpenMP pool | 20/s | 64,000 |
| TODO-BASELINE PyTorch int8 (dynamic, per channel) | | |
| TODO-BASELINE ONNX Runtime int8 (dynamic, per channel) | | |
| Rung 23, int8 with SmoothQuant-style scales | 55/s | 177,000 |
| Rung 24, int8 with u8 input (VNNI) | 60/s | 193,000 |
| **Rung 28, + bf16 attention products** | **60 to 65/s** | **193,000 to 209,000** |

Every int8 rung agrees with rung 0 on at least 99% of the 7,600 examples, with calibrated
accuracy within one point (rung 28: 7565/7600, 84.92% against 84.78%). The way there, and
what did not work, is in the changelog in [PLAN.md](PLAN.md).

What it says:

- **int8 is what makes the budget reachable.** A request at about 90 ms on two cores fits
  eight workers on 16 cores with room for the queue.
- **The int8 matmuls are now near the hardware.** In a request of about 87 ms they take
  57 ms, at about 77% of the VNNI peak. The last rungs found 0 to 2 ms each elsewhere.
- **The cloud host is a noisy neighbour.** A renter's burst can halve the best rate for a
  round, so every comparison here interleaves the two rungs in the same rounds.

## Correctness

Five tiers, in [PLAN.md](PLAN.md): token ids byte-identical to HF's tokenizer (an
exhaustive Unicode sweep included), per-layer numerical checks against reference
activations, behaviour on all 7,600 examples against rung 0, and an in-run gate that fails
a benchmark if a reply drifts. The tokenizer (`crates/mb-tokenizer`) matches HF
`tokenizers` exactly.

## Layout

| Path | What |
|---|---|
| `crates/mb-tokenizer` | the tokenizer, from scratch |
| `crates/mb-model` | the forward pass; each rung is a cargo feature |
| `crates/mb-harness` | the paired benchmark, the server and the load generator |
| `crates/mb-candle` | rungs 4 and 5, the candle baseline |
| `python/` | rungs 0 to 3, the int8 baselines, the oracles and the statistics |
| `results/` | every measurement, raw lines and summaries |

## Running it

Fetch the checkpoint and the data as [python/README.md](python/README.md) describes. Then,
for example, rung 28 on Linux x86 with MKL and oneDNN installed:

```bash
cargo build --release -p mb-model --features scatter-bf16 --bin eval_agnews
```

```bash
cargo build --release -p mb-harness
```

`eval_agnews` scores the test set and writes the logits; `python/rust_rung.py` checks
them against rung 0. `mb-serve` and `mb-load` run the serving test.

Energy on the Mac comes from `powermetrics`, which needs root. Benchmarks that record it
need a sudo rule for `/usr/bin/powermetrics` only:

```bash
echo "$USER ALL=(root) NOPASSWD: /usr/bin/powermetrics" > /tmp/powermetrics.sudoers
sudo visudo -cf /tmp/powermetrics.sudoers && sudo install -m 0440 -o root -g wheel /tmp/powermetrics.sudoers /etc/sudoers.d/powermetrics
```

Remove it afterwards with `sudo rm /etc/sudoers.d/powermetrics`.

## License

MIT.
