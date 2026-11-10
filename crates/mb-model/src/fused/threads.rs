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
        attend_by_head(ws, rope, window, 1.0 / (layer.attn.head_dim as f32).sqrt());
        gather_heads(ws);
        self.after_attention(index, ws);
    }

    /// The attention norm and the fused `Wqkv` projection of layer `index`,
    /// on every row, into `ws.qkv`.
    fn project_qkv(&self, index: usize, ws: &mut Workspace) {
        let layer = &self.layers[index];
        // Layer 0 has no attention norm (trap: it is `nn.Identity`, not a
        // norm with unit weight), so its projection reads `h` itself.
        let attn_in = match &layer.attn_norm {
            Some(w) => {
                layer_norm_rows(&ws.h, w, self.config.norm_eps, &mut ws.normed, ws.threads);
                &ws.normed
            }
            None => &ws.h,
        };
        crate::ops::linear_into(attn_in, &layer.attn.wqkv, &mut ws.qkv);
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
        crate::ops::linear_into(&ws.context, &attn.wo, &mut ws.projection);

        // h += projection, then mlp_norm(h), row by row. Pre-norm: the add
        // is onto the un-normed h.
        let width = ws.h.cols;
        let block = rows_per_block(ws.h.rows, threads) * width;
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

        crate::ops::linear_into(&ws.normed, &layer.mlp.wi, &mut ws.projected);

        // gelu(input) * gate, by blocks of rows.
        ws.gated.resize_rows(ws.projected.rows);
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

        crate::ops::linear_into(&ws.gated, &layer.mlp.wo, &mut ws.projection);

        // h += projection.
        let blocks =
            ws.h.data
                .chunks_mut(block)
                .zip(ws.projection.data.chunks(block));
        in_parallel(blocks, |(h, p)| {
            for (x, p) in h.iter_mut().zip(p) {
                *x += p;
            }
        });
    }
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
            scatter_head(qkv, rows, rope, head, hd, q, k, v);
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
    sgemm_strided(seq, seq, hd, scale, query, hd, key_t, seq, scores, seq);
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
    sgemm_strided(seq, hd, seq, 1.0, scores, seq, values, hd, out, hd);
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
