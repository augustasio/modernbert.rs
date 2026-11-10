//! Rung 7: matrix multiply through Apple Accelerate's `cblas_sgemm`.
//!
//! This is the only `unsafe` in the crate, and it is only the FFI boundary.
//!
//! Accelerate runs single-precision GEMM on the M4's AMX matrix unit. The BLAS
//! check in `python/BLAS.md` measured 1168 GFLOP/s on one thread for a 2048^3
//! sgemm, which no NEON kernel can reach. So the gap between rung 6 (the naive
//! loop in `ops::linear_naive`) and rung 7 is mostly the matrix unit, not the
//! language. That gap is the measurement this project is built around.
//!
//! Rung 16: on Linux the same `cblas_sgemm` comes from Intel MKL instead,
//! the library PyTorch's x86 builds use for fp32 matmul. The CBLAS interface
//! is a standard, so MKL takes the same enum values and, in its default LP64
//! interface, the same 32-bit integers. Nothing else in the crate changes, so
//! rung 16 is rung 15 on another platform. The module keeps its name because
//! it is the one FFI boundary on both.
//!
//! Threads: Accelerate picks its own thread count. `VECLIB_MAXIMUM_THREADS`
//! caps it, and the benchmark harness must set it explicitly per run, or rung
//! 7 and PyTorch (which also calls Accelerate) are not run on equal terms.
//! MKL reads `MKL_NUM_THREADS` the same way, and the harness sets both.

use crate::tensor::Matrix;

// CBLAS enum values, from Accelerate's `cblas.h`. MKL's `mkl_cblas.h` has the
// same ones.
const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;
const CBLAS_TRANS: i32 = 112;

// `mkl_rt` is MKL's single dynamic library, which picks the CPU's code path
// and the threading layer at run time. The harness asks for the GNU OpenMP
// layer, the one PyTorch's Linux builds link.
#[cfg_attr(target_os = "macos", link(name = "Accelerate", kind = "framework"))]
#[cfg_attr(target_os = "linux", link(name = "mkl_rt"))]
extern "C" {
    /// `C = alpha * op(A) @ op(B) + beta * C`.
    ///
    /// The integer arguments are 32-bit, as in the classic CBLAS interface
    /// that Accelerate still exports.
    #[allow(clippy::too_many_arguments)]
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

// Rung 18. MKL's per-thread setting, so that a thread of ours can go serial
// without touching the thread count the main thread's big matmuls use.
#[cfg(all(feature = "serial-blas", target_os = "linux"))]
#[link(name = "mkl_rt")]
extern "C" {
    /// Returns the setting it replaced; 0 means "none", which clears it.
    ///
    /// Trap: the lowercase symbol `mkl_set_num_threads_local` is MKL's Fortran
    /// interface, which takes a pointer. This capitalised one is the C one.
    #[link_name = "MKL_Set_Num_Threads_Local"]
    fn mkl_set_num_threads_local(threads: i32) -> i32;
}

/// Rung 18: while this value lives, BLAS calls on this thread run serial.
///
/// Trap: every thread of ours that enters a threaded MKL gets an OpenMP team
/// of its own, because libgomp's teams belong to the thread that opens them.
/// With `MKL_NUM_THREADS` = t and t attention threads, that is about t squared
/// threads, all spin-waiting. On a 16-core pod it made a batch-1 request 7x
/// slower at 8 threads. The value restores the previous setting when dropped,
/// because one of the attention blocks runs on the main thread, and that
/// thread still wants a threaded MKL for the big matmuls.
#[cfg(feature = "serial-blas")]
pub struct SerialBlas {
    #[cfg(target_os = "linux")]
    previous: i32,
}

#[cfg(feature = "serial-blas")]
impl SerialBlas {
    #[cfg(target_os = "linux")]
    pub fn here() -> Self {
        // SAFETY: MKL's documented setter; it only writes a per-thread integer.
        let previous = unsafe { mkl_set_num_threads_local(1) };
        SerialBlas { previous }
    }

    /// Accelerate manages its own threads, so there is nothing to change.
    #[cfg(not(target_os = "linux"))]
    pub fn here() -> Self {
        SerialBlas {}
    }
}

#[cfg(all(feature = "serial-blas", target_os = "linux"))]
impl Drop for SerialBlas {
    fn drop(&mut self) {
        // SAFETY: as in `here`.
        unsafe { mkl_set_num_threads_local(self.previous) };
    }
}

/// `input @ weight.T`, the same contract as `ops::linear_naive`.
///
/// `input` is `[M, K]` and `weight` is `[N, K]` (safetensors' `[out, in]`), so
/// the weight goes in transposed and nothing is copied.
pub fn linear(input: &Matrix, weight: &Matrix) -> Matrix {
    let mut out = Matrix::zeros(input.rows, weight.rows);
    linear_into(input, weight, &mut out);
    out
}

/// `out = input @ weight.T`, into a buffer the caller owns, for rung 8's
/// workspace.
///
/// `beta` is always 0. Rung 8 tried `beta = 1`, which adds the product onto
/// what `out` holds and so would fuse the residual add into the matmul. See
/// `fused.rs` for why it does not.
pub fn linear_into(input: &Matrix, weight: &Matrix, out: &mut Matrix) {
    assert_eq!(
        input.cols, weight.cols,
        "cannot multiply a {}x{} input by a {}x{} weight",
        input.rows, input.cols, weight.rows, weight.cols
    );
    let (m, k, n) = (input.rows, input.cols, weight.rows);
    assert_eq!(
        out.shape(),
        (m, n),
        "a {m}x{n} product cannot go into a {:?} buffer",
        out.shape()
    );
    if m == 0 || n == 0 {
        return;
    }

    // The sizes must fit the 32-bit CBLAS integers. The largest product in
    // this model is the 50368-row decoder, far inside i32.
    let to_i32 = |v: usize, what: &str| -> i32 {
        i32::try_from(v).unwrap_or_else(|_| panic!("{what} = {v} does not fit a CBLAS int"))
    };
    let (m32, n32, k32) = (to_i32(m, "M"), to_i32(n, "N"), to_i32(k, "K"));

    // SAFETY: the three buffers are live for the call and sized exactly as
    // the arguments describe, which the asserts above check. `input.data`
    // holds M * K values with row stride K, `weight.data` holds N * K with
    // row stride K, and `out.data` holds M * N with row stride N. Accelerate
    // only reads A and B and only writes C. C does not alias A or
    // B: it is borrowed mutably, they are borrowed shared.
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            m32,
            n32,
            k32,
            1.0,
            input.data.as_ptr(),
            k32,
            weight.data.as_ptr(),
            k32,
            0.0,
            out.data.as_mut_ptr(),
            n32,
        );
    }
}

/// `c = alpha * a @ b`, all row-major, none transposed, with explicit row
/// strides, for rung 10's attention products.
///
/// `a` is `[m, k]` with row stride `lda`, `b` is `[k, n]` with row stride
/// `ldb`, and `c` is `[m, n]` with row stride `ldc`. The strides let a product
/// read one head's block of a wider buffer, or write one head's columns of
/// the `[seq, hidden]` context, without a copy.
#[allow(clippy::too_many_arguments)]
pub fn sgemm_strided(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    c: &mut [f32],
    ldc: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    // The last row of each matrix needs only its own width, not a full stride.
    let needs = |rows: usize, cols: usize, stride: usize| (rows - 1) * stride + cols;
    assert!(
        lda >= k && ldb >= n && ldc >= n,
        "a row stride is narrower than its row"
    );
    assert!(
        k == 0 || a.len() >= needs(m, k, lda),
        "a is too short for {m}x{k}"
    );
    assert!(
        k == 0 || b.len() >= needs(k, n, ldb),
        "b is too short for {k}x{n}"
    );
    assert!(c.len() >= needs(m, n, ldc), "c is too short for {m}x{n}");
    let to_i32 = |v: usize, what: &str| -> i32 {
        i32::try_from(v).unwrap_or_else(|_| panic!("{what} = {v} does not fit a CBLAS int"))
    };

    // SAFETY: the asserts above check that every element the arguments
    // describe lies inside its slice. Accelerate only reads `a` and `b` and
    // only writes `c`, which is borrowed mutably, so it aliases neither.
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            to_i32(m, "M"),
            to_i32(n, "N"),
            to_i32(k, "K"),
            alpha,
            a.as_ptr(),
            to_i32(lda, "lda"),
            b.as_ptr(),
            to_i32(ldb, "ldb"),
            0.0,
            c.as_mut_ptr(),
            to_i32(ldc, "ldc"),
        );
    }
}
