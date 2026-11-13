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
| PyTorch int8, dynamic, per channel (fails accuracy: 71.1%) | 30 to 40/s | (not usable) |
| ONNX Runtime int8, dynamic, per channel | 40/s | 129,000 |
| OpenVINO int8, NNCF static, SmoothQuant | 55 to 60/s (same rounds as rung 28's 60) | 177,000 to 193,000 |
| Rung 23, int8 with SmoothQuant-style scales | 55/s | 177,000 |
| Rung 24, int8 with u8 input (VNNI) | 60/s | 193,000 |
| Rung 28, + bf16 attention products | 60 to 70/s | 193,000 to 225,000 |
| **Rung 31, + exp and int8 rounding without scalar casts** | **75 to 80/s** (rung 30's 70 in the same rounds) | **241,000 to 257,000** |

The rates move with the host: the same rung 28 build reached 70/s on a quiet host and 60/s
on a slower one. Rung 31 was measured on a quiet host only. So each baseline was run in the same rounds as rung 28, and the table
gives each one's range across those rounds
([results/bench/cloud_turin_baseline_int8_quiet2/SUMMARY.md](results/bench/cloud_turin_baseline_int8_quiet2/SUMMARY.md)).
Tier 3 for the baselines: PyTorch's dynamic int8 agrees with rung 0 on 89.1% of examples
and loses 13.7 points of calibrated accuracy; ONNX Runtime 95.0% and 85.0%; OpenVINO
98.6% and 86.1%.

Every int8 rung agrees with rung 0 on at least 99% of the 7,600 examples, with calibrated
accuracy within one point (rungs 28 to 31: 7565/7600, 84.92% against 84.78%). The way there, and
what did not work, is in the changelog in [PLAN.md](PLAN.md).

What it says:

- **int8 is what makes the budget reachable.** A request at about 90 ms on two cores fits
  eight workers on 16 cores with room for the queue.
- **The int8 matmuls are now near the hardware.** At two threads they take about 57 ms of
  a request, at about 77% of the VNNI peak. The last rungs found 0 to 5 ms each elsewhere.
- **Stock OpenVINO was within one rate step of rung 28; rung 31 pulls ahead.** OpenVINO's
  static int8 (NNCF, SmoothQuant, calibrated on train prompts) was about 4% slower per
  request than rung 28. In the same rounds as rung 31 it is about 15% slower per request
  (worker p50 90 ms against 75 at 8x2), and its 1x16 p99 at 10/s is about twice rung
  31's ([results/bench/cloud_turin_r31_openvino/SUMMARY.md](results/bench/cloud_turin_r31_openvino/SUMMARY.md)).
  Rung 28 against OpenVINO: An op-by-op
  profile shows the matmuls tie; OpenVINO wins on a fused GELU gate and loses on layout
  copies ([results/bench/cloud_turin_profile_openvino/SUMMARY.md](results/bench/cloud_turin_profile_openvino/SUMMARY.md)).
  Rungs 30 and 31 then closed the gate gap.
- **Saturating float-to-int casts were the hidden cost.** On x86, Rust's `as i8` and
  `as i32` compile to scalar code inside an otherwise vector loop. Rung 30 rounds to int8
  by adding 1.5 * 2^23 instead (gate rounding 5.1 to 1.3 thread-ms, a tie behind the
  server). Rung 31 builds exp's 2^n the same way: the GELU gate falls from 5.4 to 2.8 ms
  at two threads, about OpenVINO's 2.5, and the softmax from 9.3 to 4.7 thread-ms. Both
  keep the logits bit-identical; rung 31 serves 75 to 80/s against rung 30's 70
  ([results/bench/cloud_turin_serve31/SUMMARY.md](results/bench/cloud_turin_serve31/SUMMARY.md)).
- **For the lowest latency, give one request all the cores.** Rungs 28 and 31 as one
  worker of 16 threads answer in about 35 ms (p50) with a p99 of 64 to 108 ms at 5
  requests per second (OpenVINO: 46 to 49 ms and 116 to 133 ms), but serves only up to 20/s within 200 ms. Eight workers of two threads serve
  60 to 70/s with p99 near 140 to 160 ms below their limit.
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
