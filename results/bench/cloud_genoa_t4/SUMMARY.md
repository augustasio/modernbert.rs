# Cloud rough check: rung 16 on Genoa, 4 threads

A rough check (PLAN.md, phase 2), not a Mac measurement: never mix these figures into
the Mac tables.

Run on 2026-11-08 and 09, 23:08 to 01:11 UTC, at commit 8cfc2f1, on a RunPod secure
pod (L4 GPU type, GPU unused). Host CPU: AMD EPYC 9254 (Zen 4, Genoa), 48 CPUs on the
host, the container capped at 5.1 CPUs by its cgroup quota. Ubuntu 22.04, Python 3.13,
PyTorch 2.15.0 CPU wheel (MKL 2024.2, GNU OpenMP), rung 16 linked to pip `mkl` 2024.2.2
through `mkl_rt` with `MKL_THREADING_LAYER=GNU`. The weights' SHA-256 matches the Mac's.

Each row is 5 replicates (separate processes) of 30 cycles of 16 examples, rung 0 as
the ruler in every cycle. Slices are 16 examples, not the Mac's 64, because fp32 here
takes about 0.5 s an example and there are no energy samples that need long slices.
All 15 replicates passed the in-run correctness check (worst logit difference 1.9e-4).
Intervals are 95%, from the two-stage bootstrap in `python/analyze_bench.py`.

| Config | Speedup vs rung 0 | Rung 0 examples/s | Variant examples/s | Variant p50 ms | Variant p99 ms |
|---|---|---|---|---|---|
| A/A: rung 0 against itself | 1.004 [1.000, 1.008] | 1.94 [1.91, 1.98] | 1.95 [1.92, 1.98] | 506 [498, 514] | 705 [674, 719] |
| 16, batch 1 | 1.10 [1.10, 1.11] | 1.95 [1.93, 1.96] | 2.14 [2.12, 2.16] | 460 [458, 464] | 641 [602, 672] |
| 16, batch 16 | 1.51 [1.45, 1.56] | 1.74 [1.67, 1.82] | 2.66 [2.63, 2.68] | 5990 [5940, 6060] | 6610 [6420, 6630] |

At batch 16 the latencies are per batch.

## Notes

- **Rung 16 beats rung 0 on Linux too, by less than on the Mac.** At batch 1 it is
  1.10x here against rung 15's 1.24x on the Mac; at batch 16, 1.51x against 1.66x.
  Both rungs use MKL here, so the gap is our code against PyTorch's on this CPU, with
  the same matmul library underneath.
- **The A/A floor is 1.004 [1.000, 1.008].** So on this shared host a ratio further
  than about 1% from 1 is real, and both rung 16 figures are well past it.
- **fp32 here is about 6x slower than on the Mac.** Rung 16 scores 2.1 examples/s at
  batch 1 and 2.7 at batch 16, against rung 15's 9.4 and 13.6 on the Mac. The M4 runs
  sgemm on its matrix unit; Genoa has none for fp32, and 4 Zen 4 cores peak near 450
  GFLOP/s. This is the case for rung 17's bf16, which Genoa's AVX-512 runs natively.
- **Batching helps rung 0 less here than on the Mac.** Padding costs it 11% of its
  batch-1 throughput at batch 16 (1.74 against 1.95 examples/s), while unpadded rung 16
  gains 24% (2.66 against 2.14).
- **Steal time was 0 in every slice.** A RunPod pod is a container on a bare-metal
  host, not a VM, so there is no hypervisor to steal time and the counter shows
  nothing. The noise source is other renters' load on the same host, which the paired
  design and the A/A floor are there for. The mean host clock varied from 2.55 to 3.42
  GHz between configs (from `/proc/cpuinfo`, all 48 CPUs), a sign of that load.
- **p99 is noisy** across replicates for rung 16 at batch 1 and for rung 0 at batch
  16, so `analyze_bench.py` flags them. Trust p50 and the speedups more.
- **Cost:** the pod ran 3 h 06 min at $0.59/h, about $1.83, for setup, Tier 3 and this
  run.
