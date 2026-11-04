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
//! Threads: Accelerate picks its own thread count. `VECLIB_MAXIMUM_THREADS`
//! caps it, and the benchmark harness must set it explicitly per run, or rung
//! 7 and PyTorch (which also calls Accelerate) are not run on equal terms.

use crate::tensor::Matrix;

// CBLAS enum values, from Accelerate's `cblas.h`.
const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;
const CBLAS_TRANS: i32 = 112;

#[link(name = "Accelerate", kind = "framework")]
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
