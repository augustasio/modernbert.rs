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
#[cfg(feature = "int8")]
use std::sync::Mutex;
use std::sync::OnceLock;

// From oneDNN's `dnnl_common_types.h` and `dnnl_types.h`.
const DNNL_SUCCESS: i32 = 0;
const DNNL_BF16: i32 = 2;
const DNNL_F32: i32 = 3;
#[cfg(feature = "int8")]
const DNNL_S32: i32 = 4;
#[cfg(feature = "int8")]
const DNNL_S8: i32 = 5;
#[cfg(feature = "int8-u8")]
const DNNL_U8: i32 = 6;
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
    #[cfg(feature = "attn-bf16")]
    fn dnnl_memory_set_data_handle(memory: Handle, handle: *mut c_void) -> i32;
    #[cfg(feature = "attn-bf16")]
    fn dnnl_stream_destroy(stream: Handle) -> i32;
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
    INPUT_BF16.with(|cell| {
        let mut src = cell.borrow_mut();
        src.resize(m * k, 0);
        for (b, &x) in src.iter_mut().zip(&input.data[..m * k]) {
            *b = to_bf16(x);
        }

        // SAFETY: `src` holds M * K bf16 values, `weight.bf16` N * K (checked
        // above) and `out.data` M * N f32 (checked above).
        unsafe {
            run_matmul(
                (m, k, n),
                (src.as_mut_ptr().cast(), DNNL_BF16),
                (weight.bf16.as_ptr() as *mut c_void, DNNL_BF16),
                (out.data.as_mut_ptr().cast(), DNNL_F32),
            );
        }
    });
}

/// One matmul, with its descriptors, primitive and memories made and destroyed
/// here. Each pair is a buffer and its oneDNN data type.
///
/// # Safety
/// `src` must hold M * K values, `weights` N * K and `dst` M * N of the stated
/// types, live for the whole call. oneDNN reads src and weights and writes only
/// dst. The stream wait means it is done with them before this returns.
unsafe fn run_matmul(
    (m, k, n): (usize, usize, usize),
    src: (*mut c_void, i32),
    weights: (*mut c_void, i32),
    dst: (*mut c_void, i32),
) {
    let ctx = context();
    let src_md = desc(m, k, k, 1, src.1);
    // The weight goes in as a `[K, N]` view with strides `(1, K)`, so nothing
    // is transposed or copied.
    let weights_md = desc(k, n, 1, k, weights.1);
    let dst_md = desc(m, n, n, 1, dst.1);
    let mut pd: Handle = std::ptr::null_mut();
    let mut primitive: Handle = std::ptr::null_mut();
    let mut mems: [Handle; 3] = [std::ptr::null_mut(); 3];
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
        dnnl_memory_create(&mut mems[0], src_md, ctx.engine, src.0),
        "memory_create src",
    );
    ok(
        dnnl_memory_create(&mut mems[1], weights_md, ctx.engine, weights.0),
        "memory_create weights",
    );
    ok(
        dnnl_memory_create(&mut mems[2], dst_md, ctx.engine, dst.0),
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

/// The scale that maps a slice's largest magnitude to 127. An all-zero slice
/// gets 1, so nothing divides by zero.
#[cfg(feature = "int8")]
pub fn int8_scale(values: &[f32]) -> f32 {
    let largest = if cfg!(feature = "lane-max") {
        crate::f32math::lane_max(values, 0.0, f32::abs)
    } else {
        values.iter().fold(0.0f32, |most, x| most.max(x.abs()))
    };
    if largest > 0.0 {
        127.0 / largest
    } else {
        1.0
    }
}

/// Rung 23: the largest `|x * f|` over a row and its smoothing factors. Rung
/// 25 keeps 16 running maxima, as [`crate::f32math::lane_max`] does, so the
/// loop vectorises; the answer is the same, bit for bit.
#[cfg(feature = "int8")]
fn largest_smoothed(row: &[f32], smooth: &[f32]) -> f32 {
    if !cfg!(feature = "lane-max") {
        return row
            .iter()
            .zip(smooth)
            .fold(0.0f32, |most, (&x, &f)| most.max((x * f).abs()));
    }
    const LANES: usize = 16;
    let mut acc = [0.0f32; LANES];
    let pairs = row.chunks_exact(LANES).zip(smooth.chunks_exact(LANES));
    for (x, f) in pairs {
        for i in 0..LANES {
            acc[i] = acc[i].max((x[i] * f[i]).abs());
        }
    }
    let tail = row.len() / LANES * LANES;
    for (i, (&x, &f)) in row[tail..].iter().zip(&smooth[tail..]).enumerate() {
        acc[i] = acc[i].max((x * f).abs());
    }
    acc.iter().copied().fold(0.0f32, f32::max)
}

/// One block of rows to quantize: its first row, its s8 rows, its scales.
#[cfg(feature = "int8")]
type QuantBlock<'a> = (usize, &'a mut [i8], &'a mut [f32]);

/// The data type of the matmul's input. Rung 24: u8, because the VNNI
/// instruction on x86 (`vpdpbusd`) multiplies u8 by s8; an s8 input makes
/// oneDNN add 128 and compensate inside every call.
#[cfg(feature = "int8-u8")]
const INPUT_TYPE: i32 = DNNL_U8;
#[cfg(all(feature = "int8", not(feature = "int8-u8")))]
const INPUT_TYPE: i32 = DNNL_S8;

/// One input value, already scaled, rounded to s8; for rung 24 stored as u8
/// (s8 + 128, which is the s8 bit pattern with its top bit flipped).
#[cfg(feature = "int8")]
#[inline]
fn to_matmul_input(scaled: f32) -> i8 {
    // `as` saturates, and the scale keeps every value in range.
    let q = scaled.round_ties_even() as i8;
    if cfg!(feature = "int8-u8") {
        ((q as u8) ^ 0x80) as i8
    } else {
        q
    }
}

/// Rows per block when a loop over `rows` is split across the pool: about
/// 16 blocks, enough to keep every thread busy as threads claim blocks.
#[cfg(feature = "int8")]
fn rows_per_block(rows: usize) -> usize {
    rows.div_ceil(16).max(1)
}

/// Run `work(0)` to `work(count - 1)`, on the OpenMP pool where rung 19 has
/// one (Linux), else in order on this thread.
#[cfg(feature = "int8")]
fn on_pool(count: usize, work: &(dyn Fn(usize) + Sync)) {
    #[cfg(all(feature = "omp-pool", target_os = "linux"))]
    crate::omp::parallel_for(count, work);
    #[cfg(not(all(feature = "omp-pool", target_os = "linux")))]
    for i in 0..count {
        work(i);
    }
}

#[cfg(feature = "int8")]
thread_local! {
    /// The s8 input, its per-row scales and the s32 product, reused across
    /// calls so a forward pass does not allocate once they have grown.
    static INT8_BUFFERS: RefCell<(Vec<i8>, Vec<f32>, Vec<i32>)> =
        const { RefCell::new((Vec::new(), Vec::new(), Vec::new())) };
}

/// Rung 20: `out = input @ weight.T`, with both sides in s8 and the sums in
/// s32. `weight` must hold its s8 copy ([`Matrix::keep_int8`]).
///
/// Each input row gets its own scale, from its largest value on this call, as
/// dynamic quantization does, so nothing is calibrated in advance. Each output
/// column gets the scale of its weight row. So `s32[r][c]` is
/// `sum(round(x * sx[r]) * round(w * sw[c]))`, and dividing by the two scales
/// gives the product back in f32.
///
/// The two scales are divided out here and not given to oneDNN: its matmul
/// takes per-row scales on neither src nor dst.
#[cfg(feature = "int8")]
pub fn linear_int8_into(input: &Matrix, weight: &Matrix, out: &mut Matrix) {
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
        weight.int8.len(),
        n * k,
        "the {n}x{k} weight has no s8 copy"
    );
    if m == 0 || n == 0 {
        return;
    }
    INT8_BUFFERS.with(|cell| {
        let (src, scales, dst) = &mut *cell.borrow_mut();
        src.resize(m * k, 0);
        scales.resize(m, 1.0);
        // Trap: this loop and the dequantize loop touch every input and every
        // output value of the matmul. Serial, they took longer than the s8
        // matmul itself (rung 20 on Zen 5: 30 ms of s8 matmul in 79 ms an
        // example), so they run in blocks of rows across the pool.
        let per = rows_per_block(m);
        stage(10, || {
            let blocks: Vec<Mutex<QuantBlock>> = src
                .chunks_mut(per * k)
                .zip(scales.chunks_mut(per))
                .enumerate()
                .map(|(b, (q, s))| Mutex::new((b * per, q, s)))
                .collect();
            on_pool(blocks.len(), &|b| {
                let mut block = blocks[b].lock().expect("no block panics while held");
                let (first, q, s) = &mut *block;
                for (i, (q_row, scale)) in q.chunks_mut(k).zip(s.iter_mut()).enumerate() {
                    let row = &input.data[(*first + i) * k..(*first + i + 1) * k];
                    *scale = quantize_row(row, weight, q_row);
                }
            });
            drop(blocks);
        });
        matmul_and_dequantize((m, k, n), src, scales, dst, weight, out);
    });
}

/// One input row rounded for `weight`'s s8 matmul, into `q_row`. Returns the
/// row's scale: 127 over its largest magnitude.
#[cfg(feature = "int8")]
pub fn quantize_row(row: &[f32], weight: &Matrix, q_row: &mut [i8]) -> f32 {
    if weight.int8_smooth.is_empty() {
        let scale = int8_scale(row);
        for (q, &x) in q_row.iter_mut().zip(row) {
            // `as` saturates, and the scale keeps every value in range.
            *q = to_matmul_input(x * scale);
        }
        scale
    } else {
        // Rung 23: the row divided by s first, channel by channel.
        let smooth = &weight.int8_smooth;
        let largest = largest_smoothed(row, smooth);
        let scale = if largest > 0.0 { 127.0 / largest } else { 1.0 };
        for ((q, &x), &f) in q_row.iter_mut().zip(row).zip(smooth) {
            *q = to_matmul_input(x * f * scale);
        }
        scale
    }
}

/// Rung 26: rows already rounded by [`quantize_row`], with their scales,
/// kept between the loop that rounds them and the matmul that reads them.
#[cfg(feature = "int8")]
#[derive(Default)]
pub struct Int8Input {
    q: Vec<i8>,
    scales: Vec<f32>,
    rows: usize,
    cols: usize,
}

#[cfg(feature = "int8")]
impl Int8Input {
    /// Room for `rows` rows of `cols`, returned as the rows and the scales.
    /// The buffer only grows, so it is zeroed once, not on every call.
    pub fn prepare(&mut self, rows: usize, cols: usize) -> (&mut [i8], &mut [f32]) {
        if self.q.len() < rows * cols {
            self.q.resize(rows * cols, 0);
        }
        if self.scales.len() < rows {
            self.scales.resize(rows, 1.0);
        }
        (self.rows, self.cols) = (rows, cols);
        (&mut self.q[..rows * cols], &mut self.scales[..rows])
    }

    /// The rounded rows and their scales, as the last `prepare` sized them.
    pub fn rounded(&self) -> (&[i8], &[f32]) {
        (&self.q[..self.rows * self.cols], &self.scales[..self.rows])
    }
}

/// Rung 26: `out = input @ weight.T` for an input already rounded, the rest
/// of [`linear_int8_into`].
#[cfg(feature = "int8")]
pub fn linear_int8_from(input: &mut Int8Input, weight: &Matrix, out: &mut Matrix) {
    let (m, k, n) = (input.rows, input.cols, weight.rows);
    assert_eq!(
        k, weight.cols,
        "cannot multiply a {m}x{k} input by a {n}x{} weight",
        weight.cols
    );
    assert_eq!(
        weight.int8.len(),
        n * k,
        "the {n}x{k} weight has no s8 copy"
    );
    out.resize_rows(m);
    if m == 0 || n == 0 {
        return;
    }
    INT8_BUFFERS.with(|cell| {
        let (_, _, dst) = &mut *cell.borrow_mut();
        let src = &mut input.q[..m * k];
        matmul_and_dequantize((m, k, n), src, &input.scales[..m], dst, weight, out);
    });
}

/// The s8 matmul of `m` rounded rows, then the s32 sums scaled back to f32
/// into `out`.
#[cfg(feature = "int8")]
fn matmul_and_dequantize(
    (m, k, n): (usize, usize, usize),
    src: &mut [i8],
    scales: &[f32],
    dst: &mut Vec<i32>,
    weight: &Matrix,
    out: &mut Matrix,
) {
    assert_eq!(src.len(), m * k, "the rounded input is not {m}x{k}");
    dst.resize(m * n, 0);
    let per = rows_per_block(m);
    // SAFETY: `src` holds M * K bytes (s8, or u8 for rung 24),
    // `weight.int8` N * K s8 (checked by the callers) and `dst` M * N s32.
    // The sums fit: at most 5248 terms of at most 255 * 127.
    stage(11, || unsafe {
        run_matmul(
            (m, k, n),
            (src.as_mut_ptr().cast(), INPUT_TYPE),
            (weight.int8.as_ptr() as *mut c_void, DNNL_S8),
            (dst.as_mut_ptr().cast(), DNNL_S32),
        );
    });
    stage(12, || {
        let blocks: Vec<Mutex<(usize, &mut [f32])>> = out.data[..m * n]
            .chunks_mut(per * n)
            .enumerate()
            .map(|(b, o)| Mutex::new((b * per, o)))
            .collect();
        on_pool(blocks.len(), &|b| {
            let mut block = blocks[b].lock().expect("no block panics while held");
            let (first, o) = &mut *block;
            for (i, o_row) in o.chunks_mut(n).enumerate() {
                let r = *first + i;
                let unscale_input = 1.0 / scales[r];
                let sums = &dst[r * n..(r + 1) * n];
                #[cfg(not(feature = "int8-u8"))]
                for ((y, &sum), &unscale_weight) in
                    o_row.iter_mut().zip(sums).zip(&weight.int8_unscale)
                {
                    *y = sum as f32 * unscale_input * unscale_weight;
                }
                // Rung 24: the u8 input was s8 + 128, so take 128 * sum(w) off
                // each sum first. Exact in s32, so the logits match rung 23's.
                #[cfg(feature = "int8-u8")]
                for (((y, &sum), &unscale_weight), &shift) in o_row
                    .iter_mut()
                    .zip(sums)
                    .zip(&weight.int8_unscale)
                    .zip(&weight.int8_shift)
                {
                    *y = (sum - shift) as f32 * unscale_input * unscale_weight;
                }
            }
        });
    });
}

/// Rung 27: one head's attention products as bf16 matmuls, on the thread
/// that runs the head.
///
/// Two traps, both from running inside the attention's threads:
///
/// - **One stream per thread.** Several heads run at once, one per thread of
///   the pool, and a oneDNN stream must not be used by two threads at a time.
///   The process-wide stream in [`context`] serves the linear layers, which
///   run from one thread; each attention thread gets its own here.
/// - **A primitive per shape, kept.** Building the matmul primitive for each
///   call costs about as much as the product itself at these sizes (a few
///   hundred thousand multiply-adds), and every head of an example has the
///   same shapes. So each thread keeps the primitives of its last few shapes,
///   with memories whose data pointer is set on each call.
///
/// Inside the pool's parallel region oneDNN runs serially on the calling
/// thread, as rung 18 makes MKL do.
#[cfg(feature = "attn-bf16")]
pub struct HeadMatmuls {
    primitives: Primitives,
    /// The bf16 inputs. Rung 27: the two inputs of the current product.
    /// Rung 28: Q (then P) and K, written by the scatter.
    a: Vec<u16>,
    b: Vec<u16>,
    /// Rung 28: V transposed, `[hd, seq]`, written by the scatter.
    #[cfg(feature = "scatter-bf16")]
    values_t: Vec<u16>,
}

/// One thread's oneDNN stream and its matmul primitives, by shape.
#[cfg(feature = "attn-bf16")]
struct Primitives {
    stream: Handle,
    /// Most recently used last.
    cached: Vec<CachedMatmul>,
}

/// One matmul primitive for one shape `(m, k, n)`, bf16 x bf16 into f32, with
/// its descriptors and memories.
#[cfg(feature = "attn-bf16")]
struct CachedMatmul {
    shape: (usize, usize, usize),
    pd: Handle,
    primitive: Handle,
    mds: [Handle; 3],
    mems: [Handle; 3],
}

/// How many shapes each thread keeps. One example needs two (the scores and
/// `P V`); a batch of several lengths needs two per length.
#[cfg(feature = "attn-bf16")]
const SHAPES_KEPT: usize = 8;

#[cfg(feature = "attn-bf16")]
impl CachedMatmul {
    fn new((m, k, n): (usize, usize, usize)) -> Self {
        let engine = context().engine;
        // The same layouts as `run_matmul`: src `[M, K]`, weights a `[K, N]`
        // view of an `[N, K]` buffer, dst `[M, N]`.
        let mds = [
            desc(m, k, k, 1, DNNL_BF16),
            desc(k, n, 1, k, DNNL_BF16),
            desc(m, n, n, 1, DNNL_F32),
        ];
        let mut pd: Handle = std::ptr::null_mut();
        let mut primitive: Handle = std::ptr::null_mut();
        let mut mems: [Handle; 3] = [std::ptr::null_mut(); 3];
        // SAFETY: every out-pointer is a valid local, and the descriptors
        // live until `drop`. The memories are made with no data; each call
        // sets their data pointers before it executes.
        unsafe {
            ok(
                dnnl_matmul_primitive_desc_create(
                    &mut pd,
                    engine,
                    mds[0],
                    mds[1],
                    std::ptr::null_mut(),
                    mds[2],
                    std::ptr::null_mut(),
                ),
                "matmul_primitive_desc_create (attention)",
            );
            ok(
                dnnl_primitive_create(&mut primitive, pd),
                "primitive_create (attention)",
            );
            for (mem, &md) in mems.iter_mut().zip(&mds) {
                ok(
                    dnnl_memory_create(mem, md, engine, std::ptr::null_mut()),
                    "memory_create (attention)",
                );
            }
        }
        CachedMatmul {
            shape: (m, k, n),
            pd,
            primitive,
            mds,
            mems,
        }
    }
}

#[cfg(feature = "attn-bf16")]
impl Drop for CachedMatmul {
    fn drop(&mut self) {
        // SAFETY: every handle was made in `new` and is destroyed once, here.
        unsafe {
            for mem in self.mems {
                dnnl_memory_destroy(mem);
            }
            dnnl_primitive_destroy(self.primitive);
            dnnl_primitive_desc_destroy(self.pd);
            for md in self.mds {
                dnnl_memory_desc_destroy(md);
            }
        }
    }
}

#[cfg(feature = "attn-bf16")]
impl Primitives {
    fn new() -> Self {
        let mut stream: Handle = std::ptr::null_mut();
        // SAFETY: a valid out-pointer; the stream lives until `drop`.
        unsafe {
            ok(
                dnnl_stream_create(&mut stream, context().engine, DNNL_STREAM_IN_ORDER),
                "stream_create (attention)",
            );
        }
        Primitives {
            stream,
            cached: Vec::with_capacity(SHAPES_KEPT),
        }
    }

    /// `dst = a @ b.T` for the shape `(m, k, n)`: `a` is `[M, K]` and `b`
    /// `[N, K]`, both bf16; `dst` is `[M, N]` f32.
    fn run(&mut self, (m, k, n): (usize, usize, usize), a: &[u16], b: &[u16], dst: &mut [f32]) {
        assert!(
            a.len() >= m * k && b.len() >= n * k,
            "the bf16 inputs are short"
        );
        assert!(dst.len() >= m * n, "the output is short");
        let at = match self.cached.iter().position(|c| c.shape == (m, k, n)) {
            Some(at) => at,
            None => {
                if self.cached.len() == SHAPES_KEPT {
                    self.cached.remove(0);
                }
                self.cached.push(CachedMatmul::new((m, k, n)));
                self.cached.len() - 1
            }
        };
        let c = &self.cached[at];
        // oneDNN takes `void *` for every argument but only writes the dst.
        let pointers: [*mut c_void; 3] = [
            a.as_ptr().cast_mut().cast(),
            b.as_ptr().cast_mut().cast(),
            dst.as_mut_ptr().cast(),
        ];
        let args = [DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST];
        let args: [ExecArg; 3] = std::array::from_fn(|i| ExecArg {
            arg: args[i],
            memory: c.mems[i],
        });
        // SAFETY: the buffers hold at least M * K, N * K and M * N values of
        // the memories' types (asserted above) and outlive the call; the
        // stream wait means oneDNN is done with them before this returns.
        unsafe {
            for (mem, ptr) in c.mems.iter().zip(pointers) {
                ok(
                    dnnl_memory_set_data_handle(*mem, ptr),
                    "memory_set_data_handle",
                );
            }
            ok(
                dnnl_primitive_execute(c.primitive, self.stream, 3, args.as_ptr()),
                "primitive_execute (attention)",
            );
            ok(dnnl_stream_wait(self.stream), "stream_wait (attention)");
        }
    }
}

#[cfg(feature = "attn-bf16")]
impl Drop for Primitives {
    fn drop(&mut self) {
        self.cached.clear();
        // SAFETY: made in `new`, destroyed once, after its primitives.
        unsafe {
            dnnl_stream_destroy(self.stream);
        }
    }
}

#[cfg(feature = "attn-bf16")]
impl HeadMatmuls {
    fn new() -> Self {
        HeadMatmuls {
            primitives: Primitives::new(),
            a: Vec::new(),
            b: Vec::new(),
            #[cfg(feature = "scatter-bf16")]
            values_t: Vec::new(),
        }
    }
}

/// Rung 28: the scatter writes Q, K and V in bf16 straight into these
/// buffers, in the layouts the two products read. Rung 27 wrote them in f32
/// first, then made a second pass to round them (and to transpose K back).
#[cfg(feature = "scatter-bf16")]
impl HeadMatmuls {
    /// The buffers the scatter fills for one head of `seq` tokens: Q (scaled)
    /// and K as `[seq, hd]`, and V transposed, `[hd, seq]`.
    pub fn inputs(&mut self, seq: usize, hd: usize) -> (&mut [u16], &mut [u16], &mut [u16]) {
        self.a.resize(seq * hd, 0);
        self.b.resize(seq * hd, 0);
        self.values_t.resize(seq * hd, 0);
        (&mut self.a, &mut self.b, &mut self.values_t)
    }

    /// `scores = (scale * Q) K^T`, from the buffers of [`Self::inputs`].
    pub fn scores(&mut self, (seq, hd): (usize, usize), scores: &mut [f32]) {
        self.primitives.run(
            (seq, hd, seq),
            &self.a[..seq * hd],
            &self.b[..seq * hd],
            scores,
        );
    }

    /// `out = P V`. P is rounded to bf16 into the buffer Q used, which the
    /// scores are done with.
    pub fn values(&mut self, probs: &[f32], (seq, hd): (usize, usize), out: &mut [f32]) {
        self.a.resize(seq * seq, 0);
        for (b, &p) in self.a.iter_mut().zip(&probs[..seq * seq]) {
            *b = to_bf16(p);
        }
        self.primitives
            .run((seq, seq, hd), &self.a, &self.values_t[..seq * hd], out);
    }
}

#[cfg(feature = "attn-bf16")]
thread_local! {
    static HEAD_MATMULS: RefCell<Option<HeadMatmuls>> = const { RefCell::new(None) };
}

/// Run `work` with this thread's [`HeadMatmuls`], made on first use.
#[cfg(feature = "attn-bf16")]
pub fn with_head_matmuls<T>(work: impl FnOnce(&mut HeadMatmuls) -> T) -> T {
    HEAD_MATMULS.with(|cell| {
        let mut slot = cell.borrow_mut();
        work(slot.get_or_insert_with(HeadMatmuls::new))
    })
}

/// Rung 27: one head's scores, `scores = (scale * Q) K^T`, as a bf16 matmul
/// with f32 sums. `query` is `[seq, hd]`, `key_t` rung 10's transposed K,
/// `[hd, seq]`, and `scores` `[seq, seq]`.
///
/// Q is scaled before it is rounded. With a power-of-two scale (1/8 for a
/// head of 64) that is exact, so it equals scaling the sums afterwards.
#[cfg(all(feature = "attn-bf16", not(feature = "scatter-bf16")))]
pub fn head_scores_bf16(
    query: &[f32],
    key_t: &[f32],
    scale: f32,
    (seq, hd): (usize, usize),
    scores: &mut [f32],
) {
    with_head_matmuls(|t| {
        t.a.resize(seq * hd, 0);
        for (b, &x) in t.a.iter_mut().zip(&query[..seq * hd]) {
            *b = to_bf16(x * scale);
        }
        // K back to `[seq, hd]`, the `[N, K]` layout the weights side takes.
        t.b.resize(seq * hd, 0);
        for i in 0..hd {
            for (token, &x) in key_t[i * seq..(i + 1) * seq].iter().enumerate() {
                t.b[token * hd + i] = to_bf16(x);
            }
        }
        t.primitives.run((seq, hd, seq), &t.a, &t.b, scores);
    });
}

/// Rung 27: one head's output, `out = P V`, as a bf16 matmul with f32 sums.
/// `probs` is `[seq, seq]`, `values` `[seq, hd]`, `out` `[seq, hd]`.
#[cfg(all(feature = "attn-bf16", not(feature = "scatter-bf16")))]
pub fn head_values_bf16(probs: &[f32], values: &[f32], (seq, hd): (usize, usize), out: &mut [f32]) {
    with_head_matmuls(|t| {
        t.a.resize(seq * seq, 0);
        for (b, &p) in t.a.iter_mut().zip(&probs[..seq * seq]) {
            *b = to_bf16(p);
        }
        // V transposed, `[hd, seq]`: the `[N, K]` layout the weights side takes.
        t.b.resize(seq * hd, 0);
        for (token, row) in values[..seq * hd].chunks(hd).enumerate() {
            for (i, &x) in row.iter().enumerate() {
                t.b[i * seq + token] = to_bf16(x);
            }
        }
        t.primitives.run((seq, seq, hd), &t.a, &t.b, out);
    });
}

/// Analysis only: time `work` as stage `index` under `stage-times`.
#[cfg(feature = "int8")]
#[inline(always)]
fn stage<T>(index: usize, work: impl FnOnce() -> T) -> T {
    #[cfg(feature = "stage-times")]
    return crate::stage_times::timed(index, work);
    #[cfg(not(feature = "stage-times"))]
    {
        let _ = index;
        work()
    }
}

/// Rung 27: the attention products must be the exact sums of the
/// bf16-rounded inputs (up to f32 summation order), from several threads at
/// once and across a change of shape.
#[cfg(all(test, feature = "attn-bf16"))]
mod attention_bf16_tests {
    use super::*;

    fn bf16_value(x: f32) -> f64 {
        f32::from_bits((to_bf16(x) as u32) << 16) as f64
    }

    fn varied(len: usize, seed: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (((i * 7919 + seed * 104729) % 2003) as f32 - 1001.0) / 211.0)
            .collect()
    }

    /// Both products, `(scale * Q) K^T` and `P V`, through rung 27's calls.
    #[cfg(not(feature = "scatter-bf16"))]
    fn products(
        query: &[f32],
        key_t: &[f32],
        values: &[f32],
        probs: &[f32],
        scale: f32,
        (seq, hd): (usize, usize),
    ) -> (Vec<f32>, Vec<f32>) {
        let mut scores = vec![0.0f32; seq * seq];
        head_scores_bf16(query, key_t, scale, (seq, hd), &mut scores);
        let mut out = vec![0.0f32; seq * hd];
        head_values_bf16(probs, values, (seq, hd), &mut out);
        (scores, out)
    }

    /// Both products through rung 28's calls, with the bf16 inputs written
    /// in the layouts its scatter writes.
    #[cfg(feature = "scatter-bf16")]
    fn products(
        query: &[f32],
        key_t: &[f32],
        values: &[f32],
        probs: &[f32],
        scale: f32,
        (seq, hd): (usize, usize),
    ) -> (Vec<f32>, Vec<f32>) {
        let mut scores = vec![0.0f32; seq * seq];
        let mut out = vec![0.0f32; seq * hd];
        with_head_matmuls(|t| {
            let (q, k, v_t) = t.inputs(seq, hd);
            for token in 0..seq {
                for i in 0..hd {
                    q[token * hd + i] = to_bf16(query[token * hd + i] * scale);
                    k[token * hd + i] = to_bf16(key_t[i * seq + token]);
                    v_t[i * seq + token] = to_bf16(values[token * hd + i]);
                }
            }
            t.scores((seq, hd), &mut scores);
            t.values(probs, (seq, hd), &mut out);
        });
        (scores, out)
    }

    /// Check both products for one head of `seq` tokens.
    fn check_head(seq: usize, hd: usize, seed: usize) {
        let (query, key_t, values) = (
            varied(seq * hd, seed),
            varied(hd * seq, seed + 1),
            varied(seq * hd, seed + 2),
        );
        let scale = 0.125;
        // Any [seq, seq] matrix will do as P for the check.
        let probs = varied(seq * seq, seed + 3);
        let (scores, out) = products(&query, &key_t, &values, &probs, scale, (seq, hd));
        for r in 0..seq {
            for c in 0..seq {
                let want: f64 = (0..hd)
                    .map(|i| bf16_value(query[r * hd + i] * scale) * bf16_value(key_t[i * seq + c]))
                    .sum();
                let got = scores[r * seq + c] as f64;
                assert!(
                    (got - want).abs() <= 1e-4 * (1.0 + want.abs()),
                    "scores {r},{c}: {got} vs {want}"
                );
            }
        }
        for r in 0..seq {
            for i in 0..hd {
                let want: f64 = (0..seq)
                    .map(|t| bf16_value(probs[r * seq + t]) * bf16_value(values[t * hd + i]))
                    .sum();
                let got = out[r * hd + i] as f64;
                assert!(
                    (got - want).abs() <= 1e-4 * (1.0 + want.abs()),
                    "out {r},{i}: {got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn the_products_are_the_sums_of_the_rounded_inputs() {
        // Two lengths on one thread, then back: the shape cache must follow.
        for seq in [37, 160, 37] {
            check_head(seq, 64, seq);
        }
        // Four threads at once, each with its own stream.
        std::thread::scope(|s| {
            for t in 0..4 {
                s.spawn(move || {
                    for seq in [29, 131] {
                        check_head(seq, 64, 10 * t + seq);
                    }
                });
            }
        });
    }
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

    /// The product equals the same sum done by hand on the same rounded values.
    #[cfg(feature = "int8")]
    #[test]
    #[allow(clippy::needless_range_loop)] // the reference sum reads clearest by index
    fn int8_matches_the_sum_of_rounded_values() {
        let (m, k, n) = (5, 200, 21);
        let value = |i: usize| ((i * 2654435761) % 1000) as f32 / 250.0 - 2.0;
        let input = Matrix::from_vec(m, k, (0..m * k).map(value).collect());
        let mut weight = Matrix::from_vec(n, k, (0..n * k).map(|i| value(i + 17)).collect());
        weight.keep_int8();
        let mut out = Matrix::zeros(m, n);
        linear_int8_into(&input, &weight, &mut out);
        for r in 0..m {
            let row = &input.data[r * k..(r + 1) * k];
            let scale = int8_scale(row);
            for c in 0..n {
                let mut sum = 0i64;
                for i in 0..k {
                    let x = (row[i] * scale).round_ties_even() as i64;
                    sum += x * weight.int8[c * k + i] as i64;
                }
                let want = sum as f32 / scale * weight.int8_unscale[c];
                let got = out.at(r, c);
                assert!(
                    (got - want).abs() <= 1e-5 * want.abs().max(1.0),
                    "({r},{c}): {got} vs {want}"
                );
            }
        }
    }

    /// Rung 23: with one input channel 100x the rest, smoothing that channel
    /// keeps the product close to f32, where one scale per row does not.
    #[cfg(feature = "int8")]
    #[test]
    fn smoothing_shrinks_the_error_of_an_outlier_channel() {
        let (m, k, n) = (4, 256, 16);
        let value = |i: usize| ((i * 2654435761) % 1000) as f32 / 250.0 - 2.0;
        let mut x: Vec<f32> = (0..m * k).map(value).collect();
        for r in 0..m {
            x[r * k + 7] *= 100.0;
        }
        let input = Matrix::from_vec(m, k, x);
        let w: Vec<f32> = (0..n * k).map(|i| value(i + 17) / 10.0).collect();
        let exact =
            |r: usize, c: usize| -> f32 { (0..k).map(|i| input.at(r, i) * w[c * k + i]).sum() };
        let worst = |weight: &Matrix| -> f32 {
            let mut out = Matrix::zeros(m, n);
            linear_int8_into(&input, weight, &mut out);
            let mut worst = 0.0f32;
            for r in 0..m {
                for c in 0..n {
                    worst = worst.max((out.at(r, c) - exact(r, c)).abs());
                }
            }
            worst
        };
        let mut plain = Matrix::from_vec(n, k, w.clone());
        plain.keep_int8();
        let mut smooth_scales = vec![1.0f32; k];
        smooth_scales[7] = 10.0; // sqrt of the 100x range, as alpha 0.5 gives
        let mut smoothed = Matrix::from_vec(n, k, w.clone());
        smoothed.keep_int8_smoothed(&smooth_scales);
        let (e_plain, e_smoothed) = (worst(&plain), worst(&smoothed));
        assert!(
            e_smoothed < e_plain / 2.0,
            "smoothed error {e_smoothed} against plain {e_plain}"
        );
    }
}
