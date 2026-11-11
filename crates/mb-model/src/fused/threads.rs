//! Rung 11: rung 10's layer, with the work outside the big matmuls split
//! across threads.
//!
//! Rung 10 runs everything except `cblas_sgemm` on one thread, so it takes
//! the same time with 4 threads as with 1, while rung 0 gains 1.15x
//! (`results/bench/t1/SUMMARY.md`). PyTorch splits attention into
//! (head, query block) tasks and the row-wise work into chunks of rows
//! (`notes/pytorch_cpu_research.md`). This does the same, with two kinds of
//! split:
//!
//! - **By head**, for attention. Each thread takes a run of heads and, for
//!   each, scatters and rotates that head's Q, K and V, computes its scores,
//!   its softmax and `P V`. The heads share nothing but the read-only `qkv`.
//! - **By rows**, for the norms, the residual adds and the GELU gate. A row
//!   is computed on its own, so a block of rows is too.
//!
//! **The arithmetic is rung 10's, term for term.** Every row and every head
//! is computed by the same code in the same order as before; only which
//! thread runs it changes. A test checks that every layer's output is
//! bit-identical to rung 10's.
//!
//! One thing moves. Rung 10 writes `P V` straight into the head's columns of
//! the `[seq, hidden]` context. Those columns are interleaved with other
//! heads', so two threads cannot each hold a `&mut` to their own part of it.
//! Here each head writes its own `[seq, head_dim]` block, and one pass copies
//! the blocks into the context afterwards.
//!
//! The threads are `std::thread::scope`, started for each piece of work and
//! joined at its end, five times a layer. Spawning a thread allocates, so
//! with more than one thread a forward pass is no longer allocation-free.
//! The alternative, a pool of threads kept alive between calls, needs either
//! a dependency or `unsafe` to hand them borrowed data, and `CLAUDE.md`
//! rules out both for this. With one thread, nothing is spawned: the work
//! runs inline and still allocates nothing.

use super::{gelu_gate_row, Workspace};
use crate::accelerate::sgemm_strided;
use crate::model::ModernBert;
use crate::ops::{layer_norm_row, softmax_in_place};
use crate::rope::RopeTable;
use crate::tensor::Matrix;

/// Run `work` on each block, one thread per block. The last block runs on
/// the calling thread, so one block spawns nothing.
#[cfg(not(all(feature = "omp-pool", target_os = "linux")))]
fn in_parallel<T: Send>(blocks: impl Iterator<Item = T>, work: impl Fn(T) + Sync) {
    let mut blocks = blocks.peekable();
    let Some(first) = blocks.next() else { return };
    if blocks.peek().is_none() {
        work(first);
        return;
    }
    let work = &work;
    std::thread::scope(|s| {
        s.spawn(move || work(first));
        while let Some(block) = blocks.next() {
            if blocks.peek().is_none() {
                work(block);
            } else {
                s.spawn(move || work(block));
            }
        }
    });
}

/// Rung 19: run `work` on each block, on the OpenMP pool that oneDNN and MKL
/// also use. The workers are already alive, and no second set of threads
/// competes with the pool's spin-waiting ones.
#[cfg(all(feature = "omp-pool", target_os = "linux"))]
fn in_parallel<T: Send>(blocks: impl Iterator<Item = T>, work: impl Fn(T) + Sync) {
    use std::sync::Mutex;
    // A block moves out of its slot on whichever thread claims that index.
    let slots: Vec<Mutex<Option<T>>> = blocks.map(|b| Mutex::new(Some(b))).collect();
    crate::omp::parallel_for(slots.len(), &|i| {
        let block = slots[i].lock().expect("no block panics while held").take();
        if let Some(block) = block {
            work(block);
        }
    });
}

/// Run `work` as stage `index` of `stage_times::STAGES`: timed in a
/// `stage-times` build, plainly otherwise.
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

/// How many rows each thread takes, so that `threads` blocks cover `rows`.
fn rows_per_block(rows: usize, threads: usize) -> usize {
    rows.div_ceil(threads.max(1)).max(1)
}

/// [`crate::ops::layer_norm_into`], by blocks of rows.
pub(super) fn layer_norm_rows(
    input: &Matrix,
    weight: &[f32],
    eps: f32,
    out: &mut Matrix,
    threads: usize,
) {
    out.resize_rows(input.rows);
    assert_eq!(out.cols, input.cols, "the norm output is the wrong width");
    let width = input.cols;
    let block = rows_per_block(input.rows, threads) * width;
    let blocks = input.data.chunks(block).zip(out.data.chunks_mut(block));
    in_parallel(blocks, |(x, y)| {
        for (x, y) in x.chunks(width).zip(y.chunks_mut(width)) {
            layer_norm_row(x, weight, eps, y);
        }
    });
}

impl ModernBert {
    /// Rung 11's [`ModernBert::layer_fused`]: rung 10's layer, threaded.
    pub(super) fn layer_threaded(&self, index: usize, ws: &mut Workspace) {
        let c = &self.config;
        let layer = &self.layers[index];
        let rope = if layer.is_global {
            &self.rope_global
        } else {
            &self.rope_local
        };
        self.project_qkv(index, ws);
        let window = if layer.is_global {
            None
        } else {
            Some(c.local_window_each_side())
        };
        let scale = 1.0 / (layer.attn.head_dim as f32).sqrt();
        stage(2, || attend_by_head(ws, rope, window, scale));
        stage(3, || gather_heads(ws));
        self.after_attention(index, ws);
    }

    /// The attention norm and the fused `Wqkv` projection of layer `index`,
    /// on every row, into `ws.qkv`.
    fn project_qkv(&self, index: usize, ws: &mut Workspace) {
        let layer = &self.layers[index];
        // Layer 0 has no attention norm (trap: it is `nn.Identity`, not a
        // norm with unit weight), so its projection reads `h` itself.
        // Rung 26: the norm rounds each row for Wqkv as it writes it.
        #[cfg(feature = "fuse-quant")]
        if let (Some(w), false) = (&layer.attn_norm, layer.attn.wqkv.int8.is_empty()) {
            let eps = self.config.norm_eps;
            let wqkv = &layer.attn.wqkv;
            stage(0, || {
                layer_norm_rows_q8(&ws.h, w, eps, &mut ws.normed, ws.threads, wqkv, &mut ws.q8)
            });
            stage(1, || {
                crate::onednn::linear_int8_from(&mut ws.q8, wqkv, &mut ws.qkv)
            });
            return;
        }
        let attn_in = match &layer.attn_norm {
            Some(w) => {
                let eps = self.config.norm_eps;
                stage(0, || {
                    layer_norm_rows(&ws.h, w, eps, &mut ws.normed, ws.threads)
                });
                &ws.normed
            }
            None => &ws.h,
        };
        let qkv = &mut ws.qkv;
        stage(1, || {
            crate::ops::linear_into(attn_in, &layer.attn.wqkv, qkv)
        });
    }

    /// Rung 15: the last layer, at the mask rows only.
    ///
    /// The last layer is global, so the mask query still attends to the keys
    /// and values of every token, and `Wqkv` still runs on every row. But the
    /// head reads only the mask row of the output, and after attention every
    /// row is computed on its own: `Wo`, the residual add, the norm and the
    /// MLP. So those run on one row per sequence, and only the mask query is
    /// scored. That skips about three quarters of the last layer's matmul work.
    ///
    /// Afterwards `ws.h` holds one row per sequence, its mask row. The scores
    /// are summed in a different order from rung 14's `sgemm`, so the logits
    /// are not bit-identical to rung 14's; Tier 3 bounds the difference.
    #[cfg(feature = "mask-rows")]
    pub(super) fn last_layer_at_masks(
        &self,
        index: usize,
        positions: &[usize],
        ws: &mut Workspace,
    ) {
        let layer = &self.layers[index];
        // A local layer would need the window here; ModernBERT's last layer is
        // global (layer 27, and 27 % 3 == 0), so this is checked, not handled.
        assert!(
            layer.is_global,
            "rung 15 needs the last layer to be global, and layer {index} is local"
        );
        assert!(
            ws.head_dim <= MAX_HEAD_DIM,
            "head_dim {} is above the {MAX_HEAD_DIM} rung 15 has room for",
            ws.head_dim
        );
        self.project_qkv(index, ws);

        let batch = positions.len();
        let hidden = ws.h.cols;
        let longest = ws.longest();
        ws.context.resize_rows(batch);
        let scale = 1.0 / (layer.attn.head_dim as f32).sqrt();
        let (rope, qkv, segments) = (&self.rope_global, &ws.qkv, &ws.segments);
        let (heads, hd) = (ws.num_heads, ws.head_dim);

        // One block of sequences per thread. Each sequence writes its own row
        // of the context, so the blocks are runs of rows.
        let per_thread = batch.div_ceil(ws.threads.max(1)).max(1);
        let blocks = ws
            .context
            .data
            .chunks_mut(per_thread * hidden)
            .zip(ws.scores.chunks_mut(longest.max(1)))
            .enumerate();
        in_parallel(blocks, |(block, (context, scores))| {
            for (i, row) in context.chunks_mut(hidden).enumerate() {
                let b = block * per_thread + i;
                let rows = segments[b].clone();
                let mask = rows.start + positions[b];
                for head in 0..heads {
                    attend_mask_head(
                        qkv,
                        rows.clone(),
                        mask,
                        rope,
                        head,
                        hd,
                        scale,
                        &mut scores[..rows.len()],
                        &mut row[head * hd..(head + 1) * hd],
                    );
                }
            }
        });

        // Keep only the mask rows of the residual stream, in order. Row b
        // moves up from row `mask(b)`, and mask(b) >= b, so no row is
        // overwritten before it is read.
        for (b, &position) in positions.iter().enumerate() {
            let from = ws.segments[b].start + position;
            ws.h.data
                .copy_within(from * hidden..(from + 1) * hidden, b * hidden);
        }
        for m in [
            &mut ws.h,
            &mut ws.normed,
            &mut ws.projection,
            &mut ws.projected,
            &mut ws.gated,
        ] {
            m.resize_rows(batch);
        }
        self.after_attention(index, ws);
    }

    /// Everything in layer `index` after attention, on every row of `ws.h`:
    /// `Wo`, the residual add, the MLP norm, the MLP and its residual add.
    fn after_attention(&self, index: usize, ws: &mut Workspace) {
        let c = &self.config;
        let layer = &self.layers[index];
        let attn = &layer.attn;
        let threads = ws.threads;

        // The projection gets its own buffer. See the `fused` module comment
        // for why it is not added onto h inside the matmul.
        stage(4, || {
            crate::ops::linear_into(&ws.context, &attn.wo, &mut ws.projection)
        });

        // h += projection, then mlp_norm(h), row by row. Pre-norm: the add
        // is onto the un-normed h.
        let width = ws.h.cols;
        let block = rows_per_block(ws.h.rows, threads) * width;
        // Rung 26: the norm rounds each row for Wi as it writes it.
        #[cfg(feature = "fuse-quant")]
        let fuse_wi = !layer.mlp.wi.int8.is_empty();
        #[cfg(not(feature = "fuse-quant"))]
        let fuse_wi = false;
        stage(5, || {
            #[cfg(feature = "fuse-quant")]
            if fuse_wi {
                let q8 = ws.q8.prepare(ws.h.rows, width);
                add_norm_rows_q8(
                    (&mut ws.h, &ws.projection, &mut ws.normed),
                    (&layer.mlp_norm, c.norm_eps),
                    (&layer.mlp.wi, q8),
                    block / width,
                );
                return;
            }
            let blocks =
                ws.h.data
                    .chunks_mut(block)
                    .zip(ws.projection.data.chunks(block))
                    .zip(ws.normed.data.chunks_mut(block));
            in_parallel(blocks, |((h, p), normed)| {
                for ((h, p), normed) in h
                    .chunks_mut(width)
                    .zip(p.chunks(width))
                    .zip(normed.chunks_mut(width))
                {
                    for (x, p) in h.iter_mut().zip(p) {
                        *x += p;
                    }
                    layer_norm_row(h, &layer.mlp_norm, c.norm_eps, normed);
                }
            });
        });

        stage(6, || {
            #[cfg(feature = "fuse-quant")]
            if fuse_wi {
                crate::onednn::linear_int8_from(&mut ws.q8, &layer.mlp.wi, &mut ws.projected);
                return;
            }
            let _ = fuse_wi;
            crate::ops::linear_into(&ws.normed, &layer.mlp.wi, &mut ws.projected)
        });

        // gelu(input) * gate, by blocks of rows.
        // Rung 26: the GELU gate rounds each row for the MLP's Wo.
        #[cfg(feature = "fuse-quant")]
        let fuse_wo = !layer.mlp.wo.int8.is_empty();
        #[cfg(not(feature = "fuse-quant"))]
        let fuse_wo = false;
        stage(7, || {
            ws.gated.resize_rows(ws.projected.rows);
            #[cfg(feature = "fuse-quant")]
            if fuse_wo {
                let q8 = ws.q8.prepare(ws.gated.rows, ws.gated.cols);
                let per = rows_per_block(ws.projected.rows, threads);
                gelu_gate_rows_q8(&ws.projected, &mut ws.gated, (&layer.mlp.wo, q8), per);
                return;
            }
            let (wide, narrow) = (ws.projected.cols, ws.gated.cols);
            let rows = rows_per_block(ws.projected.rows, threads);
            let blocks = ws
                .projected
                .data
                .chunks(rows * wide)
                .zip(ws.gated.data.chunks_mut(rows * narrow));
            in_parallel(blocks, |(projected, gated)| {
                for (p, g) in projected.chunks(wide).zip(gated.chunks_mut(narrow)) {
                    gelu_gate_row(p, g);
                }
            });
        });

        stage(8, || {
            #[cfg(feature = "fuse-quant")]
            if fuse_wo {
                crate::onednn::linear_int8_from(&mut ws.q8, &layer.mlp.wo, &mut ws.projection);
                return;
            }
            let _ = fuse_wo;
            crate::ops::linear_into(&ws.gated, &layer.mlp.wo, &mut ws.projection)
        });

        // h += projection.
        stage(9, || {
            let blocks =
                ws.h.data
                    .chunks_mut(block)
                    .zip(ws.projection.data.chunks(block));
            in_parallel(blocks, |(h, p)| {
                for (x, p) in h.iter_mut().zip(p) {
                    *x += p;
                }
            });
        });
    }
}

/// Rung 26: a block of rows of an s8 matmul's input, and their scales.
#[cfg(feature = "fuse-quant")]
type Q8<'a> = (&'a mut [i8], &'a mut [f32]);

/// Rung 26: [`layer_norm_rows`], each row also rounded for `weight`'s s8
/// matmul right after it is written, while it is still in cache.
#[cfg(feature = "fuse-quant")]
fn layer_norm_rows_q8(
    input: &Matrix,
    norm: &[f32],
    eps: f32,
    out: &mut Matrix,
    threads: usize,
    weight: &Matrix,
    q8: &mut crate::onednn::Int8Input,
) {
    out.resize_rows(input.rows);
    let width = input.cols;
    let per = rows_per_block(input.rows, threads);
    let (q, scales) = q8.prepare(input.rows, width);
    let blocks = input
        .data
        .chunks(per * width)
        .zip(out.data.chunks_mut(per * width))
        .zip(q.chunks_mut(per * width).zip(scales.chunks_mut(per)));
    in_parallel(blocks, |((x, y), (q, scales))| {
        let rows = x.chunks(width).zip(y.chunks_mut(width));
        for (i, (x, y)) in rows.enumerate() {
            layer_norm_row(x, norm, eps, y);
            scales[i] = crate::onednn::quantize_row(y, weight, &mut q[i * width..(i + 1) * width]);
        }
    });
}

/// Rung 26: `h += projection`, then the MLP norm into `normed`, each normed
/// row also rounded for `weight`'s s8 matmul right after it is written.
/// `per` rows to a block.
#[cfg(feature = "fuse-quant")]
fn add_norm_rows_q8(
    (h, projection, normed): (&mut Matrix, &Matrix, &mut Matrix),
    (norm, eps): (&[f32], f32),
    (weight, (q, scales)): (&Matrix, Q8),
    per: usize,
) {
    let width = h.cols;
    let blocks = h
        .data
        .chunks_mut(per * width)
        .zip(projection.data.chunks(per * width))
        .zip(normed.data.chunks_mut(per * width))
        .zip(q.chunks_mut(per * width).zip(scales.chunks_mut(per)));
    in_parallel(blocks, |(((h, p), normed), (q, scales))| {
        let rows = h
            .chunks_mut(width)
            .zip(p.chunks(width))
            .zip(normed.chunks_mut(width));
        for (i, ((h, p), normed)) in rows.enumerate() {
            for (x, p) in h.iter_mut().zip(p) {
                *x += p;
            }
            layer_norm_row(h, norm, eps, normed);
            scales[i] =
                crate::onednn::quantize_row(normed, weight, &mut q[i * width..(i + 1) * width]);
        }
    });
}

/// Rung 26: the GELU gate of every row, each gated row also rounded for
/// `weight`'s s8 matmul right after it is written. `per` rows to a block.
#[cfg(feature = "fuse-quant")]
fn gelu_gate_rows_q8(
    projected: &Matrix,
    gated: &mut Matrix,
    (weight, (q, scales)): (&Matrix, Q8),
    per: usize,
) {
    let (wide, narrow) = (projected.cols, gated.cols);
    let blocks = projected
        .data
        .chunks(per * wide)
        .zip(gated.data.chunks_mut(per * narrow))
        .zip(q.chunks_mut(per * narrow).zip(scales.chunks_mut(per)));
    in_parallel(blocks, |((projected, gated), (q, scales))| {
        let rows = projected.chunks(wide).zip(gated.chunks_mut(narrow));
        for (i, (p, g)) in rows.enumerate() {
            gelu_gate_row(p, g);
            scales[i] =
                crate::onednn::quantize_row(g, weight, &mut q[i * narrow..(i + 1) * narrow]);
        }
    });
}

/// Rung 10's scatter and attention, a run of heads per thread.
///
/// One unit of work is one head of one sequence. The units are numbered
/// sequence by sequence and head by head within a sequence, which is also the
/// order of their blocks in Q, K, V and the per-head context: sequence `s`
/// takes `heads * len(s) * head_dim` floats, head `h` the `h`th
/// `len(s) * head_dim` of them. So a run of units is a run of memory, and each
/// thread gets a contiguous run. With one sequence a unit is a head, and the
/// split is rung 11's.
///
/// Each thread also gets a score buffer of its own, sized for the longest
/// sequence.
fn attend_by_head(ws: &mut Workspace, rope: &RopeTable, window: Option<usize>, scale: f32) {
    let (heads, hd) = (ws.num_heads, ws.head_dim);
    let longest = ws.longest();
    assert!(
        longest <= rope.max_positions(),
        "{longest} positions, but the RoPE table covers only {}",
        rope.max_positions()
    );
    let units = ws.segments.len() * heads;
    let per_thread = units.div_ceil(ws.threads.max(1));
    let segments = &ws.segments;
    let qkv = &ws.qkv;
    let unit_floats = |u: usize| segments[u / heads].len() * hd;

    // Cut each buffer into the threads' runs, front to back.
    let (mut q, mut k, mut v, mut out) = (
        &mut ws.query[..],
        &mut ws.key[..],
        &mut ws.value[..],
        &mut ws.context_heads[..],
    );
    let runs = (0..units).step_by(per_thread.max(1)).map(move |first| {
        let last = (first + per_thread).min(units);
        let floats: usize = (first..last).map(unit_floats).sum();
        let (q_run, q_rest) = std::mem::take(&mut q).split_at_mut(floats);
        let (k_run, k_rest) = std::mem::take(&mut k).split_at_mut(floats);
        let (v_run, v_rest) = std::mem::take(&mut v).split_at_mut(floats);
        let (out_run, out_rest) = std::mem::take(&mut out).split_at_mut(floats);
        (q, k, v, out) = (q_rest, k_rest, v_rest, out_rest);
        (first..last, q_run, k_run, v_run, out_run)
    });
    let blocks = runs.zip(ws.scores.chunks_mut((longest * longest).max(1)));
    in_parallel(blocks, |((run, q, k, v, out), scores)| {
        // Rung 18: this block's sgemms must not open their own OpenMP teams.
        #[cfg(feature = "serial-blas")]
        let _serial = crate::accelerate::SerialBlas::here();
        let mut at = 0;
        for unit in run {
            let rows = segments[unit / heads].clone();
            let head = unit % heads;
            let part = at..at + rows.len() * hd;
            at = part.end;
            let (q, k, v) = (
                &mut q[part.clone()],
                &mut k[part.clone()],
                &mut v[part.clone()],
            );
            let seq = rows.len();
            stage(13, || scatter_head(qkv, rows, rope, head, hd, q, k, v));
            attend_head(q, k, v, &mut out[part], scores, window, scale, seq, hd);
        }
    });
}

/// One head of rung 10's `scatter_heads`, for the sequence in `rows` of
/// `qkv`: Q and V `[seq, head_dim]`, K transposed, `[head_dim, seq]`, with Q
/// and K rotated. Positions count from the sequence's own first token.
///
/// The rotation is the arithmetic of [`crate::rope::apply_rope`], term for
/// term, so it rounds the same.
#[allow(clippy::too_many_arguments)]
fn scatter_head(
    qkv: &Matrix,
    rows: std::ops::Range<usize>,
    rope: &RopeTable,
    head: usize,
    hd: usize,
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
) {
    let hidden = qkv.cols / 3;
    let seq = rows.len();
    let half = hd / 2;
    let src = head * hd;
    for t in 0..seq {
        let fused = qkv.row(rows.start + t);
        let cos = rope.cos_row(t);
        let sin = rope.sin_row(t);

        // Read both halves of each pair before writing either.
        let x = &fused[src..src + hd];
        let q = &mut q[t * hd..(t + 1) * hd];
        for i in 0..half {
            let (lo, hi) = (x[i], x[i + half]);
            q[i] = lo * cos[i] - hi * sin[i];
            q[i + half] = hi * cos[i] + lo * sin[i];
        }

        let x = &fused[hidden + src..hidden + src + hd];
        for i in 0..half {
            let (lo, hi) = (x[i], x[i + half]);
            k[i * seq + t] = lo * cos[i] - hi * sin[i];
            k[(i + half) * seq + t] = hi * cos[i] + lo * sin[i];
        }

        v[t * hd..(t + 1) * hd].copy_from_slice(&fused[2 * hidden + src..2 * hidden + src + hd]);
    }
}

/// One head of rung 10's `attend_sgemm`, into the head's own `[seq,
/// head_dim]` block.
#[allow(clippy::too_many_arguments)]
fn attend_head(
    query: &[f32],
    key_t: &[f32],
    values: &[f32],
    out: &mut [f32],
    scores: &mut [f32],
    window: Option<usize>,
    scale: f32,
    seq: usize,
    hd: usize,
) {
    let scores = &mut scores[..seq * seq];
    stage(14, || {
        sgemm_strided(seq, seq, hd, scale, query, hd, key_t, seq, scores, seq)
    });
    stage(15, || {
        for t in 0..seq {
            let (first, end) = match window {
                None => (0, seq),
                Some(w) => (t.saturating_sub(w), (t + w + 1).min(seq)),
            };
            let row = &mut scores[t * seq..(t + 1) * seq];
            // f64 accumulation, as in rung 7. See `ops::softmax_in_place`.
            softmax_in_place(&mut row[first..end]);
            row[..first].fill(0.0);
            row[end..].fill(0.0);
        }
    });
    stage(16, || {
        sgemm_strided(seq, hd, seq, 1.0, scores, seq, values, hd, out, hd)
    });
}

/// The largest head the stack buffers of [`attend_mask_head`] take.
#[cfg(feature = "mask-rows")]
const MAX_HEAD_DIM: usize = 128;

/// Rung 15: one head's attention for one query, the mask, into `out`
/// (`head_dim` long). The sequence is `rows` of `qkv`, and the mask is row
/// `mask`. Q and K are rotated as in [`scatter_head`], term for term, but K
/// one row at a time instead of into a transposed block.
#[cfg(feature = "mask-rows")]
#[allow(clippy::too_many_arguments)]
fn attend_mask_head(
    qkv: &Matrix,
    rows: std::ops::Range<usize>,
    mask: usize,
    rope: &RopeTable,
    head: usize,
    hd: usize,
    scale: f32,
    scores: &mut [f32],
    out: &mut [f32],
) {
    let hidden = qkv.cols / 3;
    let half = hd / 2;
    let src = head * hd;
    let rotate = |x: &[f32], t: usize, into: &mut [f32]| {
        let (cos, sin) = (rope.cos_row(t), rope.sin_row(t));
        for i in 0..half {
            let (lo, hi) = (x[i], x[i + half]);
            into[i] = lo * cos[i] - hi * sin[i];
            into[i + half] = hi * cos[i] + lo * sin[i];
        }
    };

    let mut q = [0.0f32; MAX_HEAD_DIM];
    let q = &mut q[..hd];
    rotate(&qkv.row(mask)[src..src + hd], mask - rows.start, q);
    let mut k = [0.0f32; MAX_HEAD_DIM];
    let k = &mut k[..hd];
    for (t, score) in scores.iter_mut().enumerate() {
        rotate(
            &qkv.row(rows.start + t)[hidden + src..hidden + src + hd],
            t,
            k,
        );
        let dot: f32 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum();
        *score = dot * scale;
    }
    softmax_in_place(scores);

    out.fill(0.0);
    for (t, &p) in scores.iter().enumerate() {
        let v = &qkv.row(rows.start + t)[2 * hidden + src..2 * hidden + src + hd];
        for (o, x) in out.iter_mut().zip(v) {
            *o += p * x;
        }
    }
}

/// Copy each head's `[seq, head_dim]` block into its columns of the `[seq,
/// hidden]` context, sequence by sequence. A copy, so it changes no value.
fn gather_heads(ws: &mut Workspace) {
    let (heads, hd) = (ws.num_heads, ws.head_dim);
    let mut region = 0;
    for rows in &ws.segments {
        let seq = rows.len();
        for t in 0..seq {
            let row = ws.context.row_mut(rows.start + t);
            for head in 0..heads {
                let from = region + (head * seq + t) * hd;
                row[head * hd..(head + 1) * hd].copy_from_slice(&ws.context_heads[from..from + hd]);
            }
        }
        region += heads * seq * hd;
    }
}

/// Rung 26: the fused loops must round exactly what rung 24's separate pass
/// rounds: the same bytes and the same scales, row for row, whatever the
/// split into blocks.
#[cfg(all(test, feature = "fuse-quant"))]
mod fuse_quant_tests {
    use super::*;
    use crate::onednn::{quantize_row, Int8Input};

    /// A matrix of varied values with no pattern a block split could hide.
    fn varied(rows: usize, cols: usize, seed: usize) -> Matrix {
        let mut m = Matrix::zeros(rows, cols);
        for (i, x) in m.data.iter_mut().enumerate() {
            *x = (((i * 7919 + seed) % 1009) as f32 - 504.0) / 37.0;
        }
        m
    }

    /// Rung 24's rounding of every row of `rows`, one row at a time.
    fn separately(rows: &Matrix, weight: &Matrix) -> (Vec<i8>, Vec<f32>) {
        let mut q = vec![0i8; rows.rows * rows.cols];
        let mut scales = vec![0.0f32; rows.rows];
        for r in 0..rows.rows {
            let q_row = &mut q[r * rows.cols..(r + 1) * rows.cols];
            scales[r] = quantize_row(rows.row(r), weight, q_row);
        }
        (q, scales)
    }

    /// One plain weight and one smoothed, `n` rows of `k`.
    fn weights(n: usize, k: usize) -> [Matrix; 2] {
        let plain = Matrix::zeros(n, k);
        let mut smoothed = Matrix::zeros(n, k);
        smoothed.int8_smooth = (0..k).map(|j| 0.5 + (j % 7) as f32 / 4.0).collect();
        [plain, smoothed]
    }

    #[test]
    fn the_norm_rounds_what_the_separate_pass_rounds() {
        let (rows, width) = (7, 48);
        let norm: Vec<f32> = (0..width).map(|j| 1.0 + j as f32 / 64.0).collect();
        for weight in weights(3, width) {
            for threads in [1, 2, 3, 8] {
                let input = varied(rows, width, threads);
                let mut normed = Matrix::zeros(0, width);
                let mut q8 = Int8Input::default();
                layer_norm_rows_q8(&input, &norm, 1e-5, &mut normed, threads, &weight, &mut q8);
                let (q, scales) = separately(&normed, &weight);
                assert_eq!(q8.rounded(), (&q[..], &scales[..]), "{threads} threads");
            }
        }
    }

    #[test]
    fn the_add_and_norm_round_what_the_separate_pass_rounds() {
        let (rows, width) = (7, 48);
        let norm: Vec<f32> = (0..width).map(|j| 1.0 - j as f32 / 128.0).collect();
        for weight in weights(3, width) {
            for per in [1, 2, 3, 7] {
                let mut h = varied(rows, width, 1);
                let projection = varied(rows, width, 2);
                let mut normed = Matrix::zeros(rows, width);
                let mut q8 = Int8Input::default();
                let blocks = q8.prepare(rows, width);
                add_norm_rows_q8(
                    (&mut h, &projection, &mut normed),
                    (&norm, 1e-5),
                    (&weight, blocks),
                    per,
                );
                let (q, scales) = separately(&normed, &weight);
                assert_eq!(q8.rounded(), (&q[..], &scales[..]), "{per} rows a block");
            }
        }
    }

    #[test]
    fn the_gelu_gate_rounds_what_the_separate_pass_rounds() {
        let (rows, narrow) = (7, 40);
        for weight in weights(3, narrow) {
            for per in [1, 2, 3, 7] {
                let projected = varied(rows, 2 * narrow, 3);
                let mut gated = Matrix::zeros(rows, narrow);
                let mut q8 = Int8Input::default();
                let blocks = q8.prepare(rows, narrow);
                gelu_gate_rows_q8(&projected, &mut gated, (&weight, blocks), per);
                let (q, scales) = separately(&gated, &weight);
                assert_eq!(q8.rounded(), (&q[..], &scales[..]), "{per} rows a block");
            }
        }
    }
}
