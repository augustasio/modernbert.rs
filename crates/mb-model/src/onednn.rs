//! Rung 17: the linear layers as bf16 matmuls through oneDNN.
//!
//! oneDNN is the library PyTorch's x86 builds use for bf16 and int8 matmul
//! (PLAN.md, phase 2, "Libraries"). On Genoa it runs bf16 on AVX-512 BF16's
//! `vdpbf16ps`, which multiplies pairs of bf16 values and adds them into f32.
//! With AMX present it would use that instead, with no change here.
//!
//! What is bf16 and what is not:
//!
//! - **The weights** are the checkpoint's own bf16 values, kept beside the f32
//!   copy by [`Matrix::keep_bf16`]. The f32 copy is an exact upcast, so this
//!   loses nothing.
//! - **The activations** entering the product are rounded from f32 to bf16
//!   here, to nearest even, as PyTorch's `.to(torch.bfloat16)` rounds. This
//!   is the only rounding the rung adds.
//! - **The sums** accumulate in f32, and the output is f32.
//!
//! This is the second FFI boundary in the crate, after `accelerate.rs`, and
//! all of its `unsafe` is here.
//!
//! Primitives: oneDNN builds a kernel for each shape. It keeps a cache of
//! built primitives (1024 by default), keyed by the descriptor, so building
//! the same shape again on every call finds the cached kernel. The prompt
//! lengths vary, so a fixed-shape primitive per layer is not possible anyway.

use crate::tensor::Matrix;
use std::cell::RefCell;
use std::ffi::c_void;
use std::sync::OnceLock;

// From oneDNN's `dnnl_common_types.h` and `dnnl_types.h`.
const DNNL_SUCCESS: i32 = 0;
const DNNL_BF16: i32 = 2;
const DNNL_F32: i32 = 3;
const DNNL_CPU: i32 = 1;
const DNNL_STREAM_IN_ORDER: u32 = 0x1;
const DNNL_ARG_SRC: i32 = 1;
const DNNL_ARG_DST: i32 = 17;
const DNNL_ARG_WEIGHTS: i32 = 33;
const DNNL_MAX_NDIMS: usize = 12;

type Handle = *mut c_void;
type Dims = [i64; DNNL_MAX_NDIMS];

#[repr(C)]
struct ExecArg {
    arg: i32,
    memory: Handle,
}

#[link(name = "dnnl")]
extern "C" {
    fn dnnl_engine_create(engine: *mut Handle, kind: i32, index: usize) -> i32;
    fn dnnl_stream_create(stream: *mut Handle, engine: Handle, flags: u32) -> i32;
    fn dnnl_stream_wait(stream: Handle) -> i32;
    fn dnnl_memory_desc_create_with_strides(
        md: *mut Handle,
        ndims: i32,
        dims: *const i64,
        data_type: i32,
        strides: *const i64,
    ) -> i32;
    fn dnnl_memory_desc_destroy(md: Handle) -> i32;
    fn dnnl_matmul_primitive_desc_create(
        pd: *mut Handle,
        engine: Handle,
        src: Handle,
        weights: Handle,
        bias: Handle,
        dst: Handle,
        attr: Handle,
    ) -> i32;
    fn dnnl_primitive_desc_destroy(pd: Handle) -> i32;
    fn dnnl_primitive_create(primitive: *mut Handle, pd: Handle) -> i32;
    fn dnnl_primitive_destroy(primitive: Handle) -> i32;
    fn dnnl_memory_create(
        memory: *mut Handle,
        md: Handle,
        engine: Handle,
        data: *mut c_void,
    ) -> i32;
    fn dnnl_memory_destroy(memory: Handle) -> i32;
    fn dnnl_primitive_execute(
        primitive: Handle,
        stream: Handle,
        nargs: i32,
        args: *const ExecArg,
    ) -> i32;
}

/// The CPU engine and one in-order stream, made once for the process.
struct Context {
    engine: Handle,
    stream: Handle,
}

// SAFETY: oneDNN's CPU engine may be used from any thread. The stream is used
// only by `linear_bf16_into`, which waits for it before returning, and the
// model calls the linear layers from one thread at a time.
unsafe impl Send for Context {}
unsafe impl Sync for Context {}

fn ok(status: i32, what: &str) {
    assert_eq!(
        status, DNNL_SUCCESS,
        "oneDNN {what} failed with status {status}"
    );
}

fn context() -> &'static Context {
    static CONTEXT: OnceLock<Context> = OnceLock::new();
    CONTEXT.get_or_init(|| {
        let mut engine: Handle = std::ptr::null_mut();
        let mut stream: Handle = std::ptr::null_mut();
        // SAFETY: both out-pointers are valid locals; oneDNN writes a handle
        // into each, which lives for the rest of the process.
        unsafe {
            ok(
                dnnl_engine_create(&mut engine, DNNL_CPU, 0),
                "engine_create",
            );
            ok(
                dnnl_stream_create(&mut stream, engine, DNNL_STREAM_IN_ORDER),
                "stream_create",
            );
        }
        Context { engine, stream }
    })
}

/// f32 to bf16, rounded to nearest, ties to even, as PyTorch rounds. NaN does
/// not occur in this model's activations, so it is not special-cased.
#[inline]
pub fn to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let rounding = 0x7FFF + ((bits >> 16) & 1);
    ((bits + rounding) >> 16) as u16
}

thread_local! {
    /// The bf16 copy of the input, reused across calls so a forward pass
    /// does not allocate once it has grown to the longest input.
    static INPUT_BF16: RefCell<Vec<u16>> = const { RefCell::new(Vec::new()) };
}

/// A 2-D memory descriptor with explicit strides.
fn desc(rows: usize, cols: usize, row_stride: usize, col_stride: usize, data_type: i32) -> Handle {
    let mut dims: Dims = [0; DNNL_MAX_NDIMS];
    let mut strides: Dims = [0; DNNL_MAX_NDIMS];
    dims[0] = rows as i64;
    dims[1] = cols as i64;
    strides[0] = row_stride as i64;
    strides[1] = col_stride as i64;
    let mut md: Handle = std::ptr::null_mut();
    // SAFETY: `dims` and `strides` hold DNNL_MAX_NDIMS values, of which
    // oneDNN reads the first two; `md` is a valid out-pointer.
    unsafe {
        ok(
            dnnl_memory_desc_create_with_strides(
                &mut md,
                2,
                dims.as_ptr(),
                data_type,
                strides.as_ptr(),
            ),
            "memory_desc_create",
        );
    }
    md
}

/// `out = bf16(input) @ weight.T`, summed in f32. `weight` must hold its
/// bf16 copy ([`Matrix::keep_bf16`]).
///
/// `input` is `[M, K]` f32, `weight` is `[N, K]` (safetensors' `[out, in]`),
/// `out` is `[M, N]` f32. The weight goes in as a `[K, N]` view with strides
/// `(1, K)`, so nothing is transposed or copied.
pub fn linear_bf16_into(input: &Matrix, weight: &Matrix, out: &mut Matrix) {
    let (m, k, n) = (input.rows, input.cols, weight.rows);
    assert_eq!(
        k, weight.cols,
        "cannot multiply a {m}x{k} input by a {n}x{} weight",
        weight.cols
    );
    assert_eq!(
        out.shape(),
        (m, n),
        "a {m}x{n} product cannot go into a {:?} buffer",
        out.shape()
    );
    assert_eq!(
        weight.bf16.len(),
        n * k,
        "the {n}x{k} weight has no bf16 copy"
    );
    if m == 0 || n == 0 {
        return;
    }
    let ctx = context();

    INPUT_BF16.with(|cell| {
        let mut src = cell.borrow_mut();
        src.resize(m * k, 0);
        for (b, &x) in src.iter_mut().zip(&input.data[..m * k]) {
            *b = to_bf16(x);
        }

        let src_md = desc(m, k, k, 1, DNNL_BF16);
        let weights_md = desc(k, n, 1, k, DNNL_BF16);
        let dst_md = desc(m, n, n, 1, DNNL_F32);
        let mut pd: Handle = std::ptr::null_mut();
        let mut primitive: Handle = std::ptr::null_mut();
        let mut mems: [Handle; 3] = [std::ptr::null_mut(); 3];
        // SAFETY: every handle is created here and destroyed before return.
        // The three memories wrap buffers that are live for the whole call
        // and sized as their descriptors say: `src` holds M * K bf16 values,
        // `weight.bf16` N * K (checked above), `out.data` M * N f32 (checked
        // above). oneDNN reads src and weights and writes only dst, and the
        // stream wait means it is done with them before they can move. The
        // weights are passed as `*mut` because the C API takes one, but a
        // matmul never writes its weights.
        unsafe {
            ok(
                dnnl_matmul_primitive_desc_create(
                    &mut pd,
                    ctx.engine,
                    src_md,
                    weights_md,
                    std::ptr::null_mut(),
                    dst_md,
                    std::ptr::null_mut(),
                ),
                "matmul_primitive_desc_create",
            );
            ok(
                dnnl_primitive_create(&mut primitive, pd),
                "primitive_create",
            );
            ok(
                dnnl_memory_create(&mut mems[0], src_md, ctx.engine, src.as_mut_ptr().cast()),
                "memory_create src",
            );
            ok(
                dnnl_memory_create(
                    &mut mems[1],
                    weights_md,
                    ctx.engine,
                    weight.bf16.as_ptr() as *mut c_void,
                ),
                "memory_create weights",
            );
            ok(
                dnnl_memory_create(
                    &mut mems[2],
                    dst_md,
                    ctx.engine,
                    out.data.as_mut_ptr().cast(),
                ),
                "memory_create dst",
            );
            let args = [
                ExecArg {
                    arg: DNNL_ARG_SRC,
                    memory: mems[0],
                },
                ExecArg {
                    arg: DNNL_ARG_WEIGHTS,
                    memory: mems[1],
                },
                ExecArg {
                    arg: DNNL_ARG_DST,
                    memory: mems[2],
                },
            ];
            ok(
                dnnl_primitive_execute(primitive, ctx.stream, 3, args.as_ptr()),
                "primitive_execute",
            );
            ok(dnnl_stream_wait(ctx.stream), "stream_wait");
            for mem in mems {
                dnnl_memory_destroy(mem);
            }
            dnnl_primitive_destroy(primitive);
            dnnl_primitive_desc_destroy(pd);
            dnnl_memory_desc_destroy(src_md);
            dnnl_memory_desc_destroy(weights_md);
            dnnl_memory_desc_destroy(dst_md);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding_is_to_nearest_even() {
        assert_eq!(to_bf16(1.0), 0x3F80);
        // 1 + 2^-8 is exactly halfway between two bf16 values; even wins.
        assert_eq!(to_bf16(1.0 + 1.0 / 256.0), 0x3F80);
        // Just above halfway rounds up.
        assert_eq!(to_bf16(f32::from_bits(0x3F80_8001)), 0x3F81);
        // Halfway with an odd lower neighbour rounds up to even.
        assert_eq!(to_bf16(f32::from_bits(0x3F81_8000)), 0x3F82);
        assert_eq!(to_bf16(-2.5), 0xC020);
    }

    /// The bf16 product equals the f32 product of the same rounded values, up
    /// to summation order: each bf16 x bf16 product is exact in f32, so only
    /// the order of the f32 sums can differ.
    #[test]
    fn matches_f32_on_rounded_inputs() {
        let (m, k, n) = (7, 300, 33);
        let value = |i: usize| ((i * 2654435761) % 1000) as f32 / 250.0 - 2.0;
        let input = Matrix::from_vec(m, k, (0..m * k).map(value).collect());
        let mut weight = Matrix::from_vec(n, k, (0..n * k).map(|i| value(i + 17)).collect());
        // Make the weight bf16-exact, as the checkpoint's are.
        for w in &mut weight.data {
            *w = f32::from_bits((to_bf16(*w) as u32) << 16);
        }
        weight.keep_bf16();
        let mut out = Matrix::zeros(m, n);
        linear_bf16_into(&input, &weight, &mut out);

        let rounded: Vec<f32> = input
            .data
            .iter()
            .map(|&x| f32::from_bits((to_bf16(x) as u32) << 16))
            .collect();
        for r in 0..m {
            for c in 0..n {
                let mut sum = 0.0f64;
                for i in 0..k {
                    sum += rounded[r * k + i] as f64 * weight.data[c * k + i] as f64;
                }
                let got = out.at(r, c) as f64;
                assert!(
                    (got - sum).abs() <= 1e-4 * sum.abs().max(1.0),
                    "({r},{c}): {got} vs {sum}"
                );
            }
        }
    }
}
