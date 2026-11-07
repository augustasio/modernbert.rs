# Cloud hardware and cost for serving

Research agent report, 2026-11-07. Prices are for us-east-1 on that day, from the DoiT
price tracker (doit.com/compute/spot/us-east-1/<type>), which matches AWS list prices; spot
prices are the lowest single-AZ price and move. **Every throughput figure for our model is
an estimate.** Measured numbers are for other encoders (BERT-large, RoBERTa-large), scaled
by FLOPs: our model needs about 104 GFLOP per example at the median 152 tokens, about 1.3x
BERT-large at sequence 128 (79 GFLOP).

The project's current direction (2026-11-07) is CPU first, GPUs later. The GPU and
Inferentia rows are kept for the eventual comparison.

## Summary

- **Best cost per million with steady traffic of several hundred examples/s:**
  inf2.xlarge in bf16 (about $0.35/M on-demand, $0.05/M at the day's spot price; ModernBERT
  fill-mask is listed as supported by Optimum Neuron but no published compile was found),
  then g6.xlarge (L4) or g5.xlarge (A10G) at about $0.4 to $0.5/M.
- **Best for low or bursty traffic (under about 25 examples/s):** c8i.xlarge or c7i.xlarge
  with AMX bf16, about $135/month for one instance against about $580/month for the
  cheapest always-on L4. Per million it is 4 to 5x dearer than a busy GPU, but cheaper in
  absolute money until traffic fills a GPU.
- **Best for the hand-written Rust engine:** Intel AMX (c8i, c7i). Replace Accelerate with
  oneDNN or MKL bf16 GEMM; the rest of the engine carries over. Graviton4 (c8g) is second.
  AMD and Graviton3 are weak for this model.
- **fp32 on cloud CPUs is not competitive:** about 2 examples/s on c7i.xlarge with AVX-512
  (estimate). The CPU route goes through bf16 or int8 matrix units.

## 1. Instances and prices

| Instance | Hardware | vCPU (physical cores) | On-demand $/h | Spot $/h (lowest) |
|---|---|---|---|---|
| c7i.xlarge | Sapphire Rapids, AMX | 4 (2) | 0.1785 | 0.0495 |
| c8i.xlarge | Xeon 6 (Granite Rapids), AMX | 4 (2) | 0.1874 | 0.0751 |
| c7a.xlarge | Zen 4, AVX-512 BF16 | 4 (4) | 0.2053 | 0.0718 |
| c8a.xlarge | Zen 5, AVX-512 BF16 | 4 (4) | 0.2155 | 0.0657 |
| c7g.xlarge | Graviton3, BF16/I8MM | 4 (4) | 0.1450 | 0.0672 |
| c8g.xlarge | Graviton4 | 4 (4) | 0.1595 | 0.0850 |
| g4dn.xlarge | T4 16 GB | 4 | 0.526 | 0.225 |
| g5.xlarge | A10G 24 GB | 4 | 1.006 | 0.417 |
| g6.xlarge | L4 24 GB | 4 | 0.805 | 0.434 |
| g6e.xlarge | L40S 48 GB | 4 | 1.861 | 1.603 |
| inf2.xlarge | 1 Inferentia2 chip | 4 | 0.758 | 0.106 |

- m7i costs about 13% more than c7i with the same cores; it only helps if RAM is needed.
- CPU prices scale linearly with size, so cost per million does not depend on size.
- On Intel, 2 vCPUs share one physical core and its one AMX unit, so a c7i or c8i.xlarge
  has only 2 AMX units.
- Elsewhere: Hetzner GEX44 (RTX 4000 SFF Ada 20 GB, dedicated) at EUR 184/month plus a
  setup fee ([hetzner.com](https://www.hetzner.com/dedicated-rootserver/gex44/)); RunPod L4
  at about $0.39 to $0.44/h ([computeprices.com](https://computeprices.com/providers/runpod/gpus/l4)).
  Hetzner's Arm CAX (Neoverse N1) has no BF16 or I8MM, so skip it.

## 2. Matrix throughput and measured encoder benchmarks

**Intel AMX** (vendor): 1024 bf16 and 2048 int8 ops per cycle per core
([AWS Compute blog](https://aws.amazon.com/blogs/compute/accelerate-cpu-based-ai-inference-workloads-using-intel-amx-on-amazon-ec2)),
about 3.3 TFLOPS bf16 per core at 3.2 GHz, against about 0.2 TFLOPS fp32 with AVX-512.
Libraries: oneDNN (PyTorch CPU uses it for bf16; `ONEDNN_DEFAULT_FPMATH_MODE=bf16` turns
fp32 matmuls into bf16), MKL `cblas_gemm_bf16bf16f32`, OpenBLAS `sbgemm` (SAPPHIRERAPIDS
target), ONNX Runtime through oneDNN or MLAS.
- Measured: bf16 with AMX cut RoBERTa-large latency by 55 to 67% against fp32 on m7i and
  m8i (AWS blog above). Hugging Face reports about 3x with bf16 on Sapphire Rapids for
  bert-base-sized models ([HF blog](https://huggingface.co/blog/intel-sapphire-rapids-inference)).

**AMD Zen 4 and 5** (vendor and arithmetic): Zen 4 about 32 fp32 and 64 bf16 FLOP per cycle
per core, Zen 5 twice that. No matrix unit: roughly 1/10 to 1/20 of AMX per core.

**Graviton** (vendor): Graviton3 and later have BFMMLA (bf16) and SMMLA (int8), used through
oneDNN plus Arm Compute Library (`DNNL_DEFAULT_FPMATH_MODE=BF16`), ArmPL, and KleidiAI
microkernels. The Apple M4 has the same instructions (FEAT_BF16 and FEAT_I8MM, checked on
this Mac), so NEON-side kernels written here transfer.
- Measured: about 1.47x from bf16 fast-math on c7g.xlarge
  ([PyTorch Graviton tutorial](https://docs.pytorch.org/tutorials/recipes/inference_tuning_on_aws_graviton.html)).
- Vendor: up to 65% more throughput for BERT and RoBERTa with ONNX Runtime bf16
  ([AWS ML blog](https://aws.amazon.com/blogs/machine-learning/accelerate-nlp-inference-with-onnx-runtime-on-aws-graviton-processors));
  c7g fp32 at 1.8x c6i on BERT-large
  ([Arm blog](https://community.arm.com/arm-community-blogs/b/infrastructure-solutions-blog/posts/machine-learning-inference-on-aws-graviton3)).

**GPUs** (vendor dense peaks): T4 65 fp16 TF and 130 int8 TOPS, 70 W, no bf16; A10G roughly
70 to 125 fp16 TF (sources disagree), 150 W; L4 121 fp16/bf16 TF and 242 fp8/int8, 72 W,
power-limited; L40S 362 fp16/bf16 TF.
- Measured (NVIDIA): BERT-large at sequence 128 on A100 with TensorRT: batch 128 at 48.7 ms
  fp16 (about 2,630 sequences/s) and 25.0 ms int8; batch 1 at 1.54 ms fp16
  ([TensorRT demo/BERT](https://github.com/NVIDIA/TensorRT/blob/release/10.0/demo/BERT/README.md)).
- Measured (NVIDIA): T4 BERT-large 449 samples/s within a 10 ms limit, 525/s at maximum
  batch ([NVIDIA blog](https://developer.nvidia.com/blog/?p=31638)).
- Measured (third party): L4 MLPerf v3.1 BERT-99 (int8, sequence 384, unpadded) about 950
  to 973 samples/s ([StorageReview](https://storagereview.com/?p=129732)).

**Inferentia2** (measured, AWS, Neuron 2.2x, sequence 128, bf16 matmuls,
[Neuron inf2 performance](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.0/general/benchmarks/inf2/inf2-performance.html)):
bert-large 950 inferences/s at batch 4 with p99 8.85 ms; 739/s at batch 1 with p99 2.78 ms.
ModernBERT fill-mask is listed in
[Optimum Neuron supported architectures](https://huggingface.co/docs/optimum-neuron/supported_architectures);
export uses fixed batch size and sequence length, so length buckets are needed.

## 3. Estimated throughput and cost for our model

Method (estimate): examples/s = effective TFLOPS / 0.104; effective = peak x efficiency
(CPU 40 to 50%, power-limited GPU about 45%); GPU measurements scaled by 79/104; cost per
million = ($/h) / (examples/s x 3600) x 1e6.

| Instance | Precision | Est. ex/s | $/M on-demand | $/M spot |
|---|---|---|---|---|
| c7i.xlarge | bf16 AMX | ~25 | 1.98 | 0.55 |
| c8i.xlarge | bf16 AMX | ~30 | 1.7 | 0.7 |
| c7i/c8i | int8 AMX | ~40 to 50 | ~1.1 | ~0.4 |
| c7i.xlarge | fp32 AVX-512 | ~2 | ~25 | |
| c8a.xlarge | bf16 | ~11 | 5.4 | 1.7 |
| c7a.xlarge | bf16 | ~4.5 | 12.7 | |
| c8g.xlarge | bf16 | ~5 | 8.9 | 4.7 |
| c7g.xlarge | bf16 | ~2.8 | 14 | |
| g4dn.xlarge | fp16 (int8) | ~250 (~400) | 0.58 (0.37) | 0.25 |
| g5.xlarge | bf16 | ~550 | 0.51 | 0.21 |
| g6.xlarge | bf16 (fp8) | ~520 (~850) | 0.43 | 0.23 |
| g6e.xlarge | bf16 | ~1,700 | 0.30 | 0.26 |
| inf2.xlarge | bf16 | ~600 | 0.35 | 0.05 |

For reference, the M4 runs our fp32 engine at about 9.5 examples/s at batch 1 (measured).

Latency (estimate): c7i.xlarge at batch 1 about 50 ms; a 16-core AMX instance about 10 ms;
L4 at batch 1 about 4 to 6 ms; inf2 at batch 4 about 12 ms at p99.

Accuracy risk:
- **bf16 weights cost nothing:** the checkpoint ships in bf16, so only activation rounding
  matters. Keep the residual stream and LayerNorm in fp32 (bf16 spacing at 28,000 is 128).
- **fp16 (T4 only):** 28,000 fits under 65,504, but x squared in the LayerNorm variance
  overflows, so the norm must run in fp32.
- **int8 and fp8:** the residual outliers make per-tensor activation scales poor. Use
  per-token scales, SmoothQuant-style smoothing, or keep the FFN-down projection and the
  final layers in bf16. Measure label flips against fp32.

## 4. What this means for the Rust engine

GEMM backend per platform:
- **Intel:** oneDNN matmul (bf16 and int8 with AMX; pre-pack weights once with a reorder),
  or MKL `cblas_gemm_bf16bf16f32` as the drop-in for the current cblas calls; OpenBLAS
  `sbgemm` as a fallback.
- **AMD:** oneDNN or AOCL-BLIS.
- **Graviton:** oneDNN plus ACL, ArmPL, or KleidiAI bf16 and int8 microkernels.
- **GPU:** cuBLASLt, plus our own CUDA kernels for attention, RoPE, GeGLU and LayerNorm.

Is the hand-written engine still worth it? On CPU with AMX, yes: frameworks lose time to
padding, allocation and dispatch, the GEMM library does the heavy lifting, and the engine
controls everything around it. On GPU, probably not for production: TensorRT, ONNX Runtime
with TensorRT, or PyTorch with FlashAttention-2 and unpadding is near peak for much less
work. Inferentia has no practical hand-written route.

What transfers from the Mac work: unpadding (the biggest win at 119 to 348 tokens, on every
platform), fused QKV and gate-up projections, weights pre-converted and pre-packed at load,
no allocation in the hot path, computing the last layer's FFN and the head at the [MASK]
only, and banded local attention.

## 5. Running a fair cloud benchmark

- **Tenancy:** C, M and G instances are not oversubscribed; each vCPU maps to a fixed
  hardware thread. Neighbours share L3 and memory bandwidth, so measure on a whole-socket or
  `.metal` size, and repeat on 2 or more instances and AZs.
- **Frequency:** record it with `turbostat` or `perf stat`; AMX code lowers the clock.
- **Threads:** one per physical core, pinned (`taskset`, `OMP_PROC_BIND=close`,
  `OMP_PLACES=cores`). Do not use hyperthread siblings with AMX, because they share the
  unit. On 2-socket sizes, `numactl --cpunodebind --membind` or one process per socket.
- **GPU:** persistence mode, log clocks and power, optionally lock clocks, warm up, include
  host-to-device copies.
- **Load:** sustained runs of 5 minutes or more, open-loop Poisson at several rates, the
  real length distribution; log instance ID, AZ, kernel and library versions.
- **Energy:** RAPL is not exposed in VMs, so report cost per million instead.
- **Quotas:** new accounts often have a quota of 0 for G and Inf instances.
- **Session cost:** four CPU 4xlarge instances together about $3/h; the five GPU and inf2
  xlarge instances together about $5/h; 2 hours of each about $15 to $20. Spot cuts this by
  50 to 70% at the risk of interruption.
