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
        let attn = &layer.attn;
        let threads = ws.threads;
        let rope = if layer.is_global {
            &self.rope_global
        } else {
            &self.rope_local
        };

        // Layer 0 has no attention norm (trap: it is `nn.Identity`, not a
        // norm with unit weight), so its projection reads `h` itself.
        let attn_in = match &layer.attn_norm {
            Some(w) => {
                layer_norm_rows(&ws.h, w, c.norm_eps, &mut ws.normed, threads);
                &ws.normed
            }
            None => &ws.h,
        };
        crate::ops::linear_into(attn_in, &attn.wqkv, &mut ws.qkv);

        let window = if layer.is_global {
            None
        } else {
            Some(c.local_window_each_side())
        };
        attend_by_head(ws, rope, window, 1.0 / (attn.head_dim as f32).sqrt());
        gather_heads(ws);

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
/// Each thread gets its heads' blocks of Q, K, V and the per-head context,
/// and a `[seq, seq]` score buffer of its own.
fn attend_by_head(ws: &mut Workspace, rope: &RopeTable, window: Option<usize>, scale: f32) {
    let (heads, hd) = (ws.num_heads, ws.head_dim);
    let seq = ws.qkv.rows;
    assert!(
        seq <= rope.max_positions(),
        "{seq} positions, but the RoPE table covers only {}",
        rope.max_positions()
    );
    let per_thread = heads.div_ceil(ws.threads.max(1));
    let head_block = per_thread * seq * hd;
    let qkv = &ws.qkv;

    let blocks = ws
        .query
        .chunks_mut(head_block)
        .zip(ws.key.chunks_mut(head_block))
        .zip(ws.value.chunks_mut(head_block))
        .zip(ws.context_heads.chunks_mut(head_block))
        .zip(ws.scores.chunks_mut(seq * seq))
        .enumerate();
    in_parallel(blocks, |(b, ((((q, k), v), out), scores))| {
        let per_head = seq * hd;
        for i in 0..q.len() / per_head {
            let head = b * per_thread + i;
            let part = i * per_head..(i + 1) * per_head;
            let (q, k, v) = (
                &mut q[part.clone()],
                &mut k[part.clone()],
                &mut v[part.clone()],
            );
            scatter_head(qkv, rope, head, hd, q, k, v);
            attend_head(q, k, v, &mut out[part], scores, window, scale, seq, hd);
        }
    });
}

/// One head of rung 10's `scatter_heads`: Q and V `[seq, head_dim]`, K
/// transposed, `[head_dim, seq]`, with Q and K rotated.
///
/// The rotation is the arithmetic of [`crate::rope::apply_rope`], term for
/// term, so it rounds the same.
fn scatter_head(
    qkv: &Matrix,
    rope: &RopeTable,
    head: usize,
    hd: usize,
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
) {
    let hidden = qkv.cols / 3;
    let seq = qkv.rows;
    let half = hd / 2;
    let src = head * hd;
    for t in 0..seq {
        let fused = qkv.row(t);
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

/// Copy each head's `[seq, head_dim]` block into its columns of the `[seq,
/// hidden]` context. A copy, so it changes no value.
fn gather_heads(ws: &mut Workspace) {
    let (heads, hd) = (ws.num_heads, ws.head_dim);
    let seq = ws.qkv.rows;
    for t in 0..seq {
        let row = ws.context.row_mut(t);
        for head in 0..heads {
            let from = (head * seq + t) * hd;
            row[head * hd..(head + 1) * hd].copy_from_slice(&ws.context_heads[from..from + hd]);
        }
    }
}
