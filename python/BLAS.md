# Which BLAS is torch linked against?

`PLAN.md` calls this the thirty seconds of work that decides whether the project's main
claim holds. If rungs 0 to 3 ran on bundled OpenBLAS while rungs 5 and 7 ran on Apple
Accelerate, the headline comparison would be measuring the BLAS library and calling it
Rust.

**Answer: Apple Accelerate. The comparison is defensible.**

Checked 2026-11-01 on macOS 27.0.1 (26A434), arm64, M4 Air, with the wheel installed by
`python/requirements.txt`.

| | |
|---|---|
| torch | 2.15.0 (PyPI wheel, `cp313` macOS arm64) |
| transformers | 5.22.1 |
| BLAS | Apple Accelerate |
| LAPACK | Apple Accelerate |
| MKL | absent (`USE_MKL=OFF`, `torch.backends.mkl.is_available()` is `False`) |
| oneDNN / MKLDNN | absent (`USE_MKLDNN=OFF`, `parallel_info()` prints "MKLDNN not found") |
| bundled OpenBLAS | absent |
| ATen parallel backend | OpenMP 202011, `omp_get_max_threads()` 4 |

## Evidence

Three independent checks, because the build string alone is a claim about how the wheel
was configured, not proof of what the running process calls.

### 1. Build configuration

`torch.__config__.show()` reports `BLAS_INFO=accelerate`, `LAPACK_INFO=accelerate`,
`USE_MKL=OFF`, `USE_MKLDNN=OFF`. There is no OpenBLAS in the build string at all.

The line `LAPACK is enabled (usually provided by MKL)` is a hardcoded string in torch's
own config printer and is not evidence of MKL. `LAPACK_INFO=accelerate` on the same
output, plus `torch.backends.mkl.is_available() == False`, settles it.

### 2. The dynamic link

`otool -L libtorch_cpu.dylib` lists
`/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate` as a hard (not
weak) dependency. `nm -u` on the same library shows `_sgemm_` and `_dgemm_` as undefined
symbols, so they are resolved at load time from Accelerate.

`torch/lib/` contains only `libc10`, `libomp`, `libshm`, `libtorch`, `libtorch_cpu`,
`libtorch_global_deps` and `libtorch_python`. No `libopenblas`, no `libmkl_*`. There is no
other BLAS present on disk that could serve those symbols.

### 3. Measured sgemm throughput

The decisive check, since it observes the running process rather than the build.

```
1 thread,  2048^3 sgemm: 14.7 ms  ->  1168 GFLOP/s
4 threads, 2048^3 sgemm: 16.7 ms  ->  1030 GFLOP/s
```

1168 GFLOP/s on a *single* thread cannot come from a NEON SIMD kernel. An M4 performance
core at roughly 4.4 GHz with four 128-bit FMA pipes tops out near 140 GFLOP/s in fp32, so
a hand-vectorised OpenBLAS-style kernel would land within an order of magnitude of that.
Getting 1168 requires the AMX matrix coprocessor, which on this platform is reachable only
through Accelerate. Whatever the build string said, the matmul is going through Accelerate.

### A note on `USE_EIGEN_FOR_BLAS=ON`

This appears in the build settings and looks alarming, but it is a legacy Caffe2 flag
governing Caffe2's own math layer, not ATen's. ATen dispatches `float` gemm straight to
`sgemm_`, which check 2 shows resolving to Accelerate and check 3 shows running at AMX
speed. The flag is not a problem here.

## Two things worth carrying into the harness

**More threads is not faster for gemm.** Four threads measured *slower* than one on the
same matmul. AMX is a per-cluster shared unit, so threads contend for it rather than
adding throughput. `PLAN.md` already plans to pin thread counts at 1, 4 and 10 and treat
that as its own experiment; this is a concrete reason to expect 1 to win on gemm-bound
work, and it means a rung that happens to run single-threaded is not self-evidently
handicapped.

**Rung 6 is now a sharper measurement than planned.** Hand-written Rust with a naive
matmul is not competing against a NEON-tuned OpenBLAS, it is competing against AMX. The
rung 6 to rung 7 gap should therefore be large, which is exactly the gap `PLAN.md` wants
to isolate as "Apple's matrix units" rather than "Rust". Expect it to dominate.

## Reproducing

```sh
python/.venv/bin/python python/blas_check.py
```
