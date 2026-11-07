//! Rung 8: the same forward pass, restructured because we own every line.
//!
//! The weights, the matmul (Accelerate) and the arithmetic of every
//! operation are rung 7's. What changes is how the work is laid out, in four
//! ways that a framework running one operator at a time cannot do for us.
//!
//! 1. **Attention reads contiguous memory.** Rung 7 keeps Q, K and V
//!    token-major, `[seq, heads * head_dim]`, so the inner loop over keys
//!    jumps 1024 floats per step. Profiling rung 7 put 52% of its time in
//!    those two attention loops. Here the QKV projection is scattered once
//!    into per-head buffers, K transposed, so each loop reads the next
//!    floats in memory.
//! 2. **Attention vectorizes without changing its rounding.** Rung 7 sums
//!    each `q . k` one add after another, which the compiler must keep, since
//!    f32 addition is not associative. Here the vector runs across keys
//!    instead: all of a query's scores advance one dimension at a time. Each
//!    score adds the same products in the same order as rung 7, so the result
//!    is the same to the bit. See [`attend`].
//! 3. **Fusion.** RoPE is applied while scattering, not as a pass of its
//!    own. The local window is a key range, not a `seq x seq` additive mask:
//!    keys outside it are skipped instead of being added `f32::MIN` and
//!    exponentiated to zero. The residual add after attention and the
//!    feed-forward norm are one pass over each row. GELU and the gate are one
//!    pass. Layer 0 has no attention norm, so it reads `h` directly instead
//!    of a copy.
//! 4. **No allocation.** Every activation lives in a [`Workspace`] that is
//!    built once and reused, so a forward pass allocates nothing. A test
//!    asserts it, with a counting allocator.
//!
//! Skipping the masked keys is exact, not an approximation. A masked score is
//! `f32::MIN`, and `exp(f32::MIN - max)` in f64 is exactly 0, so it adds
//! nothing to the softmax total and `0 * v` adds nothing to the output.
//!
//! One fusion was tried and taken out. `cblas_sgemm` with `beta = 1` adds the
//! output projection straight onto the residual stream, which saves a buffer
//! and a pass. But the residual stream holds values near 28,000, and it made
//! the drift against PyTorch five times worse: on the eight tier-2 fixtures
//! the worst full-stack gap went from 4.2e-5 to 2.3e-4 of the peak (the bound
//! is 3e-4) and the worst verbalizer logit gap from 5.4e-5 to 3.6e-4. The
//! likely cause is that Accelerate accumulates the products onto the large
//! value already in C, where a small partial sum loses its low bits. So the
//! projection lands in its own buffer, rounded on its own, and is then added,
//! as PyTorch does.
//!
//! **Batches, without padding (rung 14).** [`ModernBert::logits_batch_fused`]
//! puts several sequences one after another in the same rows, as
//! ModernBERT's own unpadding does. Every matmul, norm, add and GELU works row
//! by row, so it runs on all the rows at once and gets one large matmul
//! instead of several small ones. Only attention and RoPE need to know where
//! a sequence starts: each sequence attends to itself alone, with its own
//! positions and its own local window. No row is padding, so no work is
//! wasted. A batch of one is exactly the single-sequence path.

use crate::config::Config;
use crate::model::ModernBert;
use crate::ops::{add_in_place, gelu_erf, layer_norm_into, layer_norm_row, linear_into};
use crate::rope::RopeTable;
use crate::safetensors::TensorError;
use crate::tensor::Matrix;

#[cfg(feature = "threads")]
mod threads;

/// Where in the forward pass an activation was observed. The tier-2 tests
/// turn it into the reference dump's checkpoint name.
///
/// An enum, not a `&str` like [`ModernBert::encode`] takes, because building
/// the name `layer.{i}.out` would allocate on every layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Point {
    Embeddings,
    Layer(usize),
    FinalNorm,
}

impl Point {
    /// The name the reference dump uses for this point.
    pub fn name(self) -> String {
        match self {
            Point::Embeddings => "embeddings.norm".to_string(),
            Point::Layer(i) => format!("layer.{i}.out"),
            Point::FinalNorm => "final_norm".to_string(),
        }
    }
}

/// Every buffer one forward pass needs, sized for the longest sequence seen
/// so far.
pub struct Workspace {
    /// The residual stream, `[seq, hidden]`.
    h: Matrix,
    /// A norm's output, `[seq, hidden]`: the input to `Wqkv`, to `Wi`, and
    /// after the last layer the `final_norm` output.
    normed: Matrix,
    /// The fused projection, `[seq, 3 * hidden]`.
    qkv: Matrix,
    /// Q and V per head, `[heads, seq, head_dim]`, and K per head and
    /// transposed, `[heads, head_dim, seq]`. Flat.
    query: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
    /// Rungs 8 and 9: one query's scores, then its probabilities, at most
    /// `seq` long. Rung 10: one head's scores for every query, `[seq, seq]`.
    /// Rung 11: one `[seq, seq]` block per thread.
    scores: Vec<f32>,
    /// The attention output before `Wo`, `[seq, hidden]`.
    context: Matrix,
    /// Rung 11: each head's attention output, `[heads, seq, head_dim]`,
    /// before it is copied into `context`. Empty before rung 11.
    context_heads: Vec<f32>,
    /// An output projection, `Wo` of attention or of the feed-forward,
    /// before it is added to `h`. `[seq, hidden]`.
    projection: Matrix,
    /// The `Wi` projection, `[seq, 2 * intermediate]`.
    projected: Matrix,
    /// `gelu(input) * gate`, `[seq, intermediate]`.
    gated: Matrix,
    /// The head at the mask position: one row each.
    head_in: Matrix,
    head_dense: Matrix,
    head_normed: Matrix,
    /// The decoder rows of the requested tokens, `[tokens, hidden]`.
    decoder_rows: Matrix,
    /// `[1, tokens]`.
    logits: Matrix,
    /// Where each sequence of the batch lives in the rows: one range per
    /// sequence, in order, covering all the rows. One range for a batch of one.
    segments: Vec<std::ops::Range<usize>>,
    num_heads: usize,
    head_dim: usize,
    /// How many threads rung 11 splits the work across. Earlier rungs
    /// ignore it.
    threads: usize,
}

impl Workspace {
    /// Buffers for sequences up to `max_seq` tokens. A longer one still
    /// works: the buffers grow, once, on that call.
    pub fn new(config: &Config, max_seq: usize) -> Self {
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        let mut ws = Workspace {
            h: Matrix::zeros(0, hidden),
            normed: Matrix::zeros(0, hidden),
            qkv: Matrix::zeros(0, 3 * hidden),
            query: Vec::new(),
            key: Vec::new(),
            value: Vec::new(),
            scores: Vec::new(),
            context: Matrix::zeros(0, hidden),
            context_heads: Vec::new(),
            projection: Matrix::zeros(0, hidden),
            projected: Matrix::zeros(0, 2 * intermediate),
            gated: Matrix::zeros(0, intermediate),
            head_in: Matrix::zeros(1, hidden),
            head_dense: Matrix::zeros(1, hidden),
            head_normed: Matrix::zeros(1, hidden),
            // Four verbalizers for AG News.
            decoder_rows: Matrix::zeros(4, hidden),
            logits: Matrix::zeros(1, 4),
            // Room for a batch of 256 sequences, so a batch never allocates
            // here.
            segments: Vec::with_capacity(256),
            num_heads: config.num_attention_heads,
            head_dim: config.head_dim(),
            threads: 1,
        };
        ws.set_seq_len(max_seq);
        ws
    }

    /// Size the buffers for one sequence of `seq` tokens.
    fn set_seq_len(&mut self, seq: usize) {
        self.segments.clear();
        self.segments.push(0..seq);
        self.set_rows(seq, seq);
    }

    /// Size every per-token buffer for `rows` tokens in all, the longest
    /// sequence being `longest`. Within the capacity already reached, this
    /// allocates nothing.
    fn set_rows(&mut self, seq: usize, longest: usize) {
        for m in [
            &mut self.h,
            &mut self.normed,
            &mut self.qkv,
            &mut self.context,
            &mut self.projection,
            &mut self.projected,
            &mut self.gated,
        ] {
            m.resize_rows(seq);
        }
        let per_head = self.num_heads * seq * self.head_dim;
        self.query.resize(per_head, 0.0);
        self.key.resize(per_head, 0.0);
        self.value.resize(per_head, 0.0);
        // The scores are per sequence, so they are sized by the longest one,
        // not by all the rows: a batch of 64 prompts would otherwise ask for
        // gigabytes.
        let scores = if cfg!(feature = "threads") {
            self.threads * longest * longest
        } else if cfg!(feature = "attention-sgemm") {
            seq * seq
        } else {
            seq
        };
        self.scores.resize(scores, 0.0);
        if cfg!(feature = "threads") {
            self.context_heads.resize(per_head, 0.0);
        }
    }

    /// Split rung 11's work across `threads` threads. Earlier rungs run on
    /// one thread whatever this says. Call it before the first forward pass:
    /// it resizes the score buffers, which allocates.
    pub fn set_threads(&mut self, threads: usize) {
        assert!(threads >= 1, "at least one thread");
        self.threads = threads;
        let longest = self.longest();
        self.set_rows(self.h.rows, longest);
    }

    /// The longest sequence in the current batch.
    fn longest(&self) -> usize {
        self.segments.iter().map(|s| s.len()).max().unwrap_or(0)
    }

    /// Lay out a batch: one range of rows per sequence, one after another.
    fn set_batch(&mut self, batch: &[&[u32]]) {
        self.segments.clear();
        let mut start = 0;
        for ids in batch {
            self.segments.push(start..start + ids.len());
            start += ids.len();
        }
        let longest = self.longest();
        self.set_rows(start, longest);
    }

    /// Put `h` in the residual stream, for a test that runs one layer on
    /// PyTorch's own input with [`ModernBert::layer_fused`].
    pub fn set_residual(&mut self, h: &Matrix) {
        self.set_seq_len(h.rows);
        self.h.data.copy_from_slice(&h.data);
    }

    /// The residual stream, `[seq, hidden]`.
    pub fn residual(&self) -> &Matrix {
        &self.h
    }

    /// The `final_norm` output of the last [`ModernBert::encode_fused`].
    pub fn final_norm(&self) -> &Matrix {
        &self.normed
    }
}

impl ModernBert {
    /// Rung 8's [`ModernBert::encode`]: the `final_norm` output ends up in
    /// `ws`, readable with [`Workspace::final_norm`].
    ///
    /// `observe` sees the same checkpoints as `encode`'s observer. Pass
    /// `|_, _| {}` when nothing needs to look.
    pub fn encode_fused(
        &self,
        ids: &[u32],
        ws: &mut Workspace,
        observe: impl FnMut(Point, &Matrix),
    ) -> Result<(), TensorError> {
        self.encode_batch_fused(&[ids], ws, observe)
    }

    /// [`ModernBert::encode_fused`] for several sequences at once, one after
    /// another in the rows of `ws`, without padding.
    pub fn encode_batch_fused(
        &self,
        batch: &[&[u32]],
        ws: &mut Workspace,
        mut observe: impl FnMut(Point, &Matrix),
    ) -> Result<(), TensorError> {
        let c = &self.config;
        // Only rung 11's threaded layer knows where each sequence starts.
        assert!(
            batch.len() == 1 || cfg!(feature = "threads"),
            "a batch of {} needs the threaded layer of rung 11 or later",
            batch.len()
        );
        ws.set_batch(batch);

        // The embedding lookup goes into `normed` and is normed into `h`,
        // which is where the residual stream starts.
        let table = &self.embeddings.weight;
        let mut row = 0;
        for ids in batch {
            for (position, &id) in ids.iter().enumerate() {
                let id = id as usize;
                if id >= table.rows {
                    return Err(TensorError::new(format!(
                        "token id {id} at position {position} is outside the vocabulary of {}",
                        table.rows
                    )));
                }
                ws.normed.row_mut(row).copy_from_slice(table.row(id));
                row += 1;
            }
        }
        norm_rows(
            &ws.normed,
            &self.embeddings.norm_weight,
            self.embeddings.eps,
            &mut ws.h,
            ws.threads,
        );
        observe(Point::Embeddings, &ws.h);

        for i in 0..self.layers.len() {
            self.layer_fused(i, ws);
            observe(Point::Layer(i), &ws.h);
        }

        norm_rows(
            &ws.h,
            &self.final_norm,
            c.norm_eps,
            &mut ws.normed,
            ws.threads,
        );
        observe(Point::FinalNorm, &ws.normed);
        Ok(())
    }

    /// Rung 8's [`ModernBert::logits_for_tokens`], into `out`, one logit per
    /// entry of `token_ids`.
    pub fn logits_for_tokens_fused(
        &self,
        ids: &[u32],
        position: usize,
        token_ids: &[u32],
        ws: &mut Workspace,
        out: &mut [f32],
    ) -> Result<(), TensorError> {
        self.logits_batch_fused(&[ids], &[position], token_ids, ws, out)
    }

    /// [`ModernBert::logits_for_tokens_fused`] for a batch: sequence `b`'s
    /// mask is at `positions[b]`, and its logits land in
    /// `out[b * token_ids.len()..]`.
    pub fn logits_batch_fused(
        &self,
        batch: &[&[u32]],
        positions: &[usize],
        token_ids: &[u32],
        ws: &mut Workspace,
        out: &mut [f32],
    ) -> Result<(), TensorError> {
        let n = token_ids.len();
        assert_eq!(
            positions.len(),
            batch.len(),
            "one mask position per sequence"
        );
        assert_eq!(
            out.len(),
            batch.len() * n,
            "one output slot per token id per sequence"
        );
        for (ids, &position) in batch.iter().zip(positions) {
            assert!(
                position < ids.len(),
                "position {position} is outside the {} tokens",
                ids.len()
            );
        }
        self.encode_batch_fused(batch, ws, |_, _| {})?;

        // The head's `transform`, on the mask rows only, as in rung 1: one
        // row per sequence.
        let head = &self.head;
        ws.head_in.resize_rows(batch.len());
        for (b, &position) in positions.iter().enumerate() {
            let row = ws.segments[b].start + position;
            ws.head_in.row_mut(b).copy_from_slice(ws.normed.row(row));
        }
        // Trap 3: no bias on `head.dense`.
        linear_into(&ws.head_in, &head.dense, &mut ws.head_dense);
        for v in ws.head_dense.data.iter_mut() {
            *v = gelu_erf(*v);
        }
        ws.head_normed.resize_rows(batch.len());
        for b in 0..batch.len() {
            layer_norm_row(
                ws.head_dense.row(b),
                &head.norm_weight,
                head.eps,
                ws.head_normed.row_mut(b),
            );
        }

        // The decoder is the tied embedding matrix. Gather the requested
        // rows and project onto them, as `MlmHead::logits_for_tokens` does.
        let decoder = &self.embeddings.weight;
        ws.decoder_rows.resize_rows(n);
        for (slot, &id) in token_ids.iter().enumerate() {
            let id = id as usize;
            assert!(
                id < decoder.rows,
                "token id {id} is outside the decoder weight's {} rows",
                decoder.rows
            );
            ws.decoder_rows
                .row_mut(slot)
                .copy_from_slice(decoder.row(id));
        }
        if ws.logits.cols != n {
            ws.logits = Matrix::zeros(1, n);
        }
        linear_into(&ws.head_normed, &ws.decoder_rows, &mut ws.logits);
        for b in 0..batch.len() {
            for (slot, &id) in token_ids.iter().enumerate() {
                // Trap 2: the bias is indexed by token id, not by slot.
                out[b * n + slot] = ws.logits.row(b)[slot] + head.decoder_bias[id as usize];
            }
        }
        Ok(())
    }

    /// Encoder layer `index` on `ws.h`, in place: rung 11's threaded layer
    /// in a rung 11 build, [`ModernBert::layer_serial`] before that.
    ///
    /// Public, like [`ModernBert::layer`], so a test can feed one layer
    /// PyTorch's own input. `ws.h` must already hold that input.
    pub fn layer_fused(&self, index: usize, ws: &mut Workspace) {
        #[cfg(feature = "threads")]
        self.layer_threaded(index, ws);
        #[cfg(not(feature = "threads"))]
        self.layer_serial(index, ws);
    }

    /// Rungs 8 to 10's encoder layer, on one thread apart from the matmuls.
    /// Public so that a test can hold rung 11's layer against it.
    pub fn layer_serial(&self, index: usize, ws: &mut Workspace) {
        let c = &self.config;
        let layer = &self.layers[index];
        let attn = &layer.attn;
        let rope = if layer.is_global {
            &self.rope_global
        } else {
            &self.rope_local
        };

        // Layer 0 has no attention norm (trap: it is `nn.Identity`, not a
        // norm with unit weight), so its projection reads `h` itself.
        let attn_in = match &layer.attn_norm {
            Some(w) => {
                layer_norm_into(&ws.h, w, c.norm_eps, &mut ws.normed);
                &ws.normed
            }
            None => &ws.h,
        };
        linear_into(attn_in, &attn.wqkv, &mut ws.qkv);
        scatter_heads(ws, rope);

        let window = if layer.is_global {
            None
        } else {
            Some(c.local_window_each_side())
        };
        attend(ws, window, 1.0 / (attn.head_dim as f32).sqrt());

        // The projection gets its own buffer. See the module comment for why
        // it is not added onto h inside the matmul.
        linear_into(&ws.context, &attn.wo, &mut ws.projection);

        // h += projection, then mlp_norm(h), row by row in one pass. Pre-norm:
        // the add is onto the un-normed h.
        for row in 0..ws.h.rows {
            let h = ws.h.row_mut(row);
            for (x, p) in h.iter_mut().zip(ws.projection.row(row)) {
                *x += p;
            }
            layer_norm_row(h, &layer.mlp_norm, c.norm_eps, ws.normed.row_mut(row));
        }

        linear_into(&ws.normed, &layer.mlp.wi, &mut ws.projected);
        gelu_gate(&ws.projected, &mut ws.gated);
        linear_into(&ws.gated, &layer.mlp.wo, &mut ws.projection);
        add_in_place(&mut ws.h, &ws.projection);
    }
}

/// Split `ws.qkv` into per-head Q, K and V, rotating Q and K on the way.
///
/// Q and V are `[heads, seq, head_dim]`. K is transposed, `[heads, head_dim,
/// seq]`, so that one dimension of every key is a contiguous run; see
/// [`attend`] for why.
///
/// The rotation is the arithmetic of [`crate::rope::apply_rope`], term for
/// term, so it rounds the same. Only where the result is written differs.
fn scatter_heads(ws: &mut Workspace, rope: &RopeTable) {
    let (heads, hd) = (ws.num_heads, ws.head_dim);
    let hidden = heads * hd;
    let seq = ws.qkv.rows;
    let half = hd / 2;
    assert!(
        seq <= rope.max_positions(),
        "{seq} positions, but the RoPE table covers only {}",
        rope.max_positions()
    );

    for t in 0..seq {
        let fused = ws.qkv.row(t);
        let cos = rope.cos_row(t);
        let sin = rope.sin_row(t);
        for head in 0..heads {
            let src = head * hd;
            let row = (head * seq + t) * hd;

            // Read both halves of each pair before writing either.
            let x = &fused[src..src + hd];
            let q = &mut ws.query[row..row + hd];
            for i in 0..half {
                let (lo, hi) = (x[i], x[i + half]);
                q[i] = lo * cos[i] - hi * sin[i];
                q[i + half] = hi * cos[i] + lo * sin[i];
            }

            let x = &fused[hidden + src..hidden + src + hd];
            let column = |d: usize| (head * hd + d) * seq + t;
            for i in 0..half {
                let (lo, hi) = (x[i], x[i + half]);
                ws.key[column(i)] = lo * cos[i] - hi * sin[i];
                ws.key[column(i + half)] = hi * cos[i] + lo * sin[i];
            }

            ws.value[row..row + hd]
                .copy_from_slice(&fused[2 * hidden + src..2 * hidden + src + hd]);
        }
    }
}

/// Scaled dot-product attention for every head and query, into `ws.context`.
///
/// `window` is how far a local layer's query reaches on each side, inclusive,
/// or `None` for a global layer. The keys a query sees are a contiguous range,
/// so the mask is the loop bounds.
///
/// Both products keep rung 7's summation order exactly, and still vectorize,
/// because the vector runs *across keys* (for the scores) or *across
/// dimensions* (for the output), never along the sum:
///
/// - **Scores.** Rung 7 computes each `q . k` as one sum over the 64
///   dimensions, one add after another. Here every key's sum advances one
///   dimension at a time, all keys together: with K transposed, dimension
///   `d` of every key is one contiguous run, so `score[j] += q[d] * k_d[j]`
///   is a plain vector loop over `j`. Each score still adds its 64 products
///   in order 0 to 63.
/// - **Output.** `out[d] += p[j] * v_j[d]` over keys `j` in order, as in
///   rung 7.
///
/// Both take four steps of the sum per pass, so a running value is loaded
/// and stored once per four products instead of once per product. The
/// order inside a pass is still the sum's order.
fn attend(ws: &mut Workspace, window: Option<usize>, scale: f32) {
    #[cfg(feature = "attention-sgemm")]
    attend_sgemm(ws, window, scale);
    #[cfg(not(feature = "attention-sgemm"))]
    attend_loops(ws, window, scale);
}

/// Rungs 8 and 9's [`attend`].
#[cfg_attr(feature = "attention-sgemm", allow(dead_code))]
fn attend_loops(ws: &mut Workspace, window: Option<usize>, scale: f32) {
    let (heads, hd) = (ws.num_heads, ws.head_dim);
    let seq = ws.h.rows;

    for head in 0..heads {
        let key_t = &ws.key[head * hd * seq..(head + 1) * hd * seq];
        let values = &ws.value[head * seq * hd..(head + 1) * seq * hd];

        for t in 0..seq {
            let (first, end) = match window {
                None => (0, seq),
                Some(w) => (t.saturating_sub(w), (t + w + 1).min(seq)),
            };
            let n = end - first;
            let q = &ws.query[(head * seq + t) * hd..(head * seq + t + 1) * hd];
            let scores = &mut ws.scores[..n];

            // Dimension d of keys first..end.
            let k = |d: usize| &key_t[d * seq + first..d * seq + end];
            scores.fill(0.0);
            let mut d = 0;
            while d + 4 <= hd {
                let (k0, k1, k2, k3) = (k(d), k(d + 1), k(d + 2), k(d + 3));
                let (q0, q1, q2, q3) = (q[d], q[d + 1], q[d + 2], q[d + 3]);
                for j in 0..n {
                    let mut s = scores[j];
                    s += q0 * k0[j];
                    s += q1 * k1[j];
                    s += q2 * k2[j];
                    s += q3 * k3[j];
                    scores[j] = s;
                }
                d += 4;
            }
            // The dimensions left over when head_dim is not a multiple of 4.
            for (d, &qd) in q.iter().enumerate().skip(d) {
                for (score, &kd) in scores.iter_mut().zip(k(d)) {
                    *score += qd * kd;
                }
            }
            for s in scores.iter_mut() {
                *s *= scale;
            }
            // f64 accumulation, as in rung 7. See `ops::softmax_in_place`.
            crate::ops::softmax_in_place(scores);

            // Value row j of the window.
            let v = |j: usize| &values[(first + j) * hd..(first + j + 1) * hd];
            let out = &mut ws.context.row_mut(t)[head * hd..(head + 1) * hd];
            out.fill(0.0);
            let mut j = 0;
            while j + 4 <= n {
                let (v0, v1, v2, v3) = (v(j), v(j + 1), v(j + 2), v(j + 3));
                let (p0, p1, p2, p3) = (scores[j], scores[j + 1], scores[j + 2], scores[j + 3]);
                for d in 0..hd {
                    let mut o = out[d];
                    o += p0 * v0[d];
                    o += p1 * v1[d];
                    o += p2 * v2[d];
                    o += p3 * v3[d];
                    out[d] = o;
                }
                j += 4;
            }
            // The keys left over when the window is not a multiple of 4.
            for (j, &p) in scores.iter().enumerate().skip(j) {
                for (o, &vd) in out.iter_mut().zip(v(j)) {
                    *o += p * vd;
                }
            }
        }
    }
}

/// Rung 10's [`attend`]: both products of each head through `cblas_sgemm`.
///
/// Per head, `S = scale * Q K^T` is one `[seq, head_dim] x [head_dim, seq]`
/// product (K is already stored transposed), the softmax runs over each row
/// of `S`, and `P V` is one `[seq, seq] x [seq, head_dim]` product written
/// straight into that head's columns of `ws.context`. This is what PyTorch's
/// CPU attention does, on blocks of queries (`notes/pytorch_cpu_research.md`).
///
/// The price is the summation order. Accelerate picks its own, so rung 10 is
/// not bit-identical to rung 9, and `scale` is applied inside the product
/// (`alpha`) rather than after it.
///
/// A local layer still computes every score, as PyTorch does, and the keys
/// outside the window get probability exactly 0, so they add nothing to
/// `P V`. With about 150 tokens and a window of 64 each side, that wastes
/// little.
#[cfg(feature = "attention-sgemm")]
fn attend_sgemm(ws: &mut Workspace, window: Option<usize>, scale: f32) {
    use crate::accelerate::sgemm_strided;
    let (heads, hd) = (ws.num_heads, ws.head_dim);
    let seq = ws.h.rows;
    let hidden = heads * hd;
    let scores = &mut ws.scores[..seq * seq];

    for head in 0..heads {
        let query = &ws.query[head * seq * hd..(head + 1) * seq * hd];
        let key_t = &ws.key[head * hd * seq..(head + 1) * hd * seq];
        let values = &ws.value[head * seq * hd..(head + 1) * seq * hd];
        sgemm_strided(seq, seq, hd, scale, query, hd, key_t, seq, scores, seq);

        for t in 0..seq {
            let (first, end) = match window {
                None => (0, seq),
                Some(w) => (t.saturating_sub(w), (t + w + 1).min(seq)),
            };
            let row = &mut scores[t * seq..(t + 1) * seq];
            // f64 accumulation, as in rung 7. See `ops::softmax_in_place`.
            crate::ops::softmax_in_place(&mut row[first..end]);
            row[..first].fill(0.0);
            row[end..].fill(0.0);
        }

        // This head's columns of the context: start at column head * hd,
        // row stride hidden.
        let context = &mut ws.context.data[head * hd..];
        sgemm_strided(seq, hd, seq, 1.0, scores, seq, values, hd, context, hidden);
    }
}

/// `gelu_erf(input) * gate` for each row of the `Wi` projection, in one pass.
///
/// The first half of each row is the input, the second the gate, as in
/// `Mlp::forward`, whose arithmetic this repeats.
fn gelu_gate(projected: &Matrix, gated: &mut Matrix) {
    gated.resize_rows(projected.rows);
    for row in 0..projected.rows {
        gelu_gate_row(projected.row(row), gated.row_mut(row));
    }
}

/// One row of [`gelu_gate`]: `projected` is the input then the gate.
fn gelu_gate_row(projected: &[f32], out: &mut [f32]) {
    let intermediate = out.len();
    let (input, gate) = projected.split_at(intermediate);
    for i in 0..intermediate {
        out[i] = gelu_erf(input[i]) * gate[i];
    }
}

/// LayerNorm of every row: by blocks of rows across threads in rung 11,
/// one row after another before it. The same arithmetic either way.
fn norm_rows(input: &Matrix, weight: &[f32], eps: f32, out: &mut Matrix, threads: usize) {
    #[cfg(feature = "threads")]
    threads::layer_norm_rows(input, weight, eps, out, threads);
    #[cfg(not(feature = "threads"))]
    {
        let _ = threads;
        layer_norm_into(input, weight, eps, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_names_match_the_reference_dump() {
        assert_eq!(Point::Embeddings.name(), "embeddings.norm");
        assert_eq!(Point::Layer(27).name(), "layer.27.out");
        assert_eq!(Point::FinalNorm.name(), "final_norm");
    }
}
