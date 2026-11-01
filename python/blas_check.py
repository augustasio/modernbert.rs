"""Which BLAS is torch linked against?

PLAN.md: if torch runs on bundled OpenBLAS while rungs 5 and 7 run on Apple Accelerate,
the headline comparison is measuring the BLAS library and calling it Rust. So this runs
before anything else, and the finding is written up in BLAS.md.

Three checks, because the build string is a claim about how the wheel was configured, not
proof of what the running process calls:

  1. the build configuration torch reports,
  2. the dynamic libraries libtorch_cpu actually links against,
  3. measured sgemm throughput, which is the only one that observes the running process.

Usage:
    python/.venv/bin/python python/blas_check.py
"""

import os
import subprocess
import time

import torch

# An M4 performance core at roughly 4.4 GHz with four 128-bit fp32 FMA pipes peaks near
# 140 GFLOP/s. Anything far above that had help from the AMX matrix coprocessor, which on
# this platform is reachable only through Accelerate.
NEON_PEAK_GFLOPS = 140.0

GEMM_SIZE = 2048
GEMM_REPEATS = 5


def report_build_config() -> None:
    print("=" * 72)
    print("1. Build configuration")
    print("=" * 72)
    print(torch.__config__.show())
    print(torch.__config__.parallel_info())
    print(f"torch.backends.mkl.is_available()    : {torch.backends.mkl.is_available()}")
    print(f"torch.backends.openmp.is_available() : {torch.backends.openmp.is_available()}")
    print()


def report_dynamic_links() -> None:
    """What libtorch_cpu actually links against, and what BLAS sits next to it on disk."""
    print("=" * 72)
    print("2. Dynamic links")
    print("=" * 72)
    lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
    libtorch_cpu = os.path.join(lib_dir, "libtorch_cpu.dylib")

    if not os.path.exists(libtorch_cpu):
        print(f"not found: {libtorch_cpu}")
        print()
        return

    linked = subprocess.run(
        ["otool", "-L", libtorch_cpu], capture_output=True, text=True, check=True
    ).stdout
    for line in linked.splitlines():
        if any(name in line for name in ("Accelerate", "openblas", "mkl", "blas", "omp")):
            print(f"  linked: {line.strip()}")

    undefined = subprocess.run(
        ["nm", "-u", libtorch_cpu], capture_output=True, text=True, check=False
    ).stdout
    gemm_symbols = sorted(
        {line.strip() for line in undefined.splitlines() if "gemm" in line.lower()}
    )
    print(f"  undefined gemm symbols: {gemm_symbols}")

    print(f"  libraries shipped in {lib_dir}:")
    for name in sorted(os.listdir(lib_dir)):
        print(f"    {name}")
    print()


def time_gemm(threads: int) -> float:
    """Best-of-N wall time for one fp32 GEMM_SIZE^3 matmul. Best, not mean: PLAN.md's
    rule is percentiles or nothing, and for a floor estimate the minimum is the honest
    statistic, since noise only ever adds time."""
    torch.set_num_threads(threads)
    a = torch.randn(GEMM_SIZE, GEMM_SIZE)
    b = torch.randn(GEMM_SIZE, GEMM_SIZE)
    a @ b  # warm up: first call pays allocator and library init

    best = float("inf")
    for _ in range(GEMM_REPEATS):
        start = time.perf_counter()
        a @ b
        best = min(best, time.perf_counter() - start)
    return best


def report_measured_throughput() -> float:
    """Returns single-thread GFLOP/s, which is what decides the question."""
    print("=" * 72)
    print("3. Measured sgemm throughput")
    print("=" * 72)
    flops = 2 * GEMM_SIZE**3

    single_thread_gflops = 0.0
    for threads in (1, 4, 10):
        seconds = time_gemm(threads)
        gflops = flops / seconds / 1e9
        print(
            f"  {threads:>2} thread(s), {GEMM_SIZE}^3 sgemm: {seconds * 1e3:7.1f} ms -> {gflops:6.0f} GFLOP/s"
        )
        if threads == 1:
            single_thread_gflops = gflops
    print()
    return single_thread_gflops


def main() -> None:
    report_build_config()
    report_dynamic_links()
    single_thread_gflops = report_measured_throughput()

    build_settings = torch.__config__.show()
    says_accelerate = "BLAS_INFO=accelerate" in build_settings
    beats_neon = single_thread_gflops > 2 * NEON_PEAK_GFLOPS

    print("=" * 72)
    print("Verdict")
    print("=" * 72)
    print(f"  build string says Accelerate : {says_accelerate}")
    print(f"  single-thread GFLOP/s        : {single_thread_gflops:.0f}")
    print(f"  above plausible NEON peak    : {beats_neon} (NEON peak ~{NEON_PEAK_GFLOPS:.0f})")

    if says_accelerate and beats_neon:
        print()
        print("  Apple Accelerate. Rungs 0 to 3 and rungs 5 and 7 share a BLAS, so the")
        print("  headline comparison is defensible. See BLAS.md.")
    else:
        print()
        print("  NOT Accelerate, or not running at Accelerate speed.")
        print("  Rungs 0 to 3 would run on a different BLAS from rungs 5 and 7, and the")
        print("  headline comparison would be indefensible. See BLAS.md.")


if __name__ == "__main__":
    main()
