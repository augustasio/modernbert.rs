//! Self-attention: the fused QKV projection, RoPE, the masks, and the output
//! projection. Bidirectional, 16 heads of 64, no bias anywhere.
//!
//! Two tensors per layer, because `attention_bias` is false:
//! `model.layers.{N}.attn.Wqkv.weight` with shape `[3072, 1024]` and
//! `model.layers.{N}.attn.Wo.weight` with shape `[1024, 1024]`.
//!
//! The reference implementation, transformers 4.48
//! `modeling_modernbert.py::eager_attention_forward`:
//!
//! ```python
//! qkv = self.Wqkv(hidden_states)
//! qkv = qkv.view(bs, -1, 3, self.num_heads, self.head_dim)
//! query, key, value = qkv.transpose(3, 1).unbind(dim=2)
//! query, key = apply_rotary_pos_emb(query, key, cos, sin)
//!
//! scale = module.head_dim**-0.5
//! attn_weights = torch.matmul(query, key.transpose(2, 3)) * scale
//!
//! if local_attention != (-1, -1):
//!     attention_mask = sliding_window_mask
//! attn_weights = attn_weights + attention_mask
//!
//! attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
//! attn_output = torch.matmul(attn_weights, value)
//! attn_output = attn_output.transpose(1, 2).contiguous().view(bs, -1, dim)
//! ```
//!
//! So: `Wqkv` -> split into heads -> RoPE on Q and K -> scaled dot product ->
//! **additive** mask -> softmax -> `@ V` -> merge heads -> `Wo`.
//!
//! More of the project's traps land in this file than in any other, and each
//! gets a comment at the line where it would otherwise be silently violated:
//!
//! 1. **`Wqkv` splits into three contiguous 1024-row blocks `[Q | K | V]`,
//!    head-major within each.** The reshape is `view(bs, seq, 3, n_heads,
//!    head_dim)`, so the 3 is the *outer* dimension: rows `0..1024` are all of
//!    Q, `1024..2048` all of K, `2048..3072` all of V. A per-head-interleaved
//!    `[q0 k0 v0 q1 k1 v1 ...]` layout, which some other fused-QKV
//!    checkpoints use, is wrong here. See `Attention::split_qkv`.
//! 2. **The scale is `head_dim ** -0.5`, so 0.125, applied exactly once.** HF
//!    scales the logits after the matmul and `pylate-rs` scales `q` before.
//!    Equivalent in exact arithmetic; this file follows HF and scales after.
//! 3. **RoPE goes on Q and K and never on V.** See [`Attention::forward`].
//! 4. **The local window is 64 each side and the test is inclusive, so 129
//!    keys per query, not 128.** `local_attention: 128` is the *total* window.
//!    Current transformers stores `sliding_window = local_attention / 2 + 1`
//!    and its flash-attention path computes `window_size = (sliding_window -
//!    1, sliding_window - 1)`, subtracting the 1 straight back out, so the
//!    `+1` must not be copied here. The arithmetic lives in
//!    [`crate::config::Config::in_local_window`] rather than in this file.
//! 5. **The padding mask uses [`f32::MIN`], not `-inf`.** See [`MASKED`].
//! 6. **Local layers need the padding mask OR'd into the window mask**, which
//!    is why [`Masks`] builds the window mask *on top of* the padding mask
//!    rather than beside it. See [`Masks::new`].
//! 7. **The softmax accumulates wider than f32.** See
//!    [`crate::ops::softmax_in_place`].
//! 8. **Local layers use true absolute positions**, not positions wrapped
//!    into `0..127`. Nothing in this file touches positions at all: the
//!    [`RopeTable`] it is handed is indexed by the row number, and
//!    [`crate::rope::apply_rope`] uses row `t` for position `t` whatever the
//!    layer type.
//!
//! `attention_dropout` is 0.0, so the `dropout` in the reference is the
//! identity at inference and has no counterpart here.

use crate::config::Config;
use crate::ops::{linear, softmax_in_place};
use crate::rope::{apply_rope, RopeTable};
use crate::safetensors::{Checkpoint, TensorError};
use crate::tensor::Matrix;

/// What an additive mask adds to a hidden key's score: `finfo(float32).min`,
/// **not** `-inf`.
///
/// Trap 5. Two reasons the reference uses the finite value, and both matter:
///
/// - A row that ends up *fully* masked becomes all `-inf`, and `exp(-inf -
///   (-inf))` is `exp(NaN)`. With [`f32::MIN`] the row's maximum is finite,
///   every entry becomes `exp(0) = 1`, and the softmax returns a uniform
///   distribution over the masked keys. That is junk, but it is finite junk,
///   and it is bit-for-bit what PyTorch produces, which is the point. A local
///   layer reaches this case whenever a query sits more than 64 positions
///   away from every real token.
/// - Adding it is lossless in the other direction too: the gap between
///   [`f32::MIN`] and its neighbour is about 4e31, so `score + MASKED` rounds
///   back to exactly `MASKED` for any sane score and never saturates to
///   `-inf`.
///
/// `exp(MASKED - max)` for a finite `max` is 0.0 exactly, so a masked key
/// contributes an exact zero to the output rather than a small residue.
pub const MASKED: f32 = f32::MIN;

/// The two additive attention masks, built once per sequence and shared by
/// all 28 layers.
///
/// Trap 6. The reference builds both outside the layer loop and selects by
/// layer type:
///
/// ```python
/// global_attention_mask = _prepare_4d_attention_mask(attention_mask, self.dtype)
/// # ... rows/distance ...
/// window_mask = (distance <= self.config.local_attention // 2)
/// sliding_window_mask = global_attention_mask.masked_fill(window_mask.logical_not(), min)
/// return global_attention_mask, sliding_window_mask
/// ```
///
/// The `masked_fill` is the part that is easy to lose: the local mask is the
/// *padding* mask with the out-of-window positions filled in, not a window
/// mask on its own. Build the window alone and padding leaks into all 18
/// local layers.
pub struct Masks {
    seq_len: usize,
    /// The reference's `global_attention_mask`, which is `[bs, 1, 1, seq]`:
    /// one value per **key**, broadcast over every query. Stored as the one
    /// row it really is rather than as a `seq * seq` matrix of copies.
    global: Vec<f32>,
    /// The reference's `sliding_window_mask`, `[bs, 1, seq, seq]`. This one
    /// genuinely varies per query, so it is a full matrix.
    ///
    /// It costs `seq_len * seq_len` floats, which is 360 KB at the 300 tokens
    /// the tests use and 90 KB at the ~150 a prompt actually runs to. It is
    /// built once per sequence, not once per layer, so paying it twenty-eight
    /// times over is the thing being avoided here.
    local: Matrix,
}

impl Masks {
    /// Build both masks for a sequence of `seq_len` tokens.
    ///
    /// `padding[i]` is true when position `i` holds a real token. `None`
    /// means every position is real, which is the single-sequence case and
    /// gives an all-zero global mask.
    pub fn new(config: &Config, seq_len: usize, padding: Option<&[bool]>) -> Self {
        if let Some(padding) = padding {
            assert_eq!(
                padding.len(),
                seq_len,
                "the padding flags cover {} positions, not {seq_len}",
                padding.len()
            );
        }

        // Trap 5, and the half of it that is easiest to miss: this masks
        // padded **keys** and never padded **queries**. The reference's
        // `[bs, 1, 1, seq]` shape says so — the missing query axis is a
        // broadcast, so no row of the attention matrix is ever suppressed.
        // Pad positions therefore produce finite junk that flows through the
        // whole network. Harmless while the outputs at pad positions are
        // ignored, which is what this model's MLM path does; fatal under mean
        // pooling, which has to exclude pad positions itself.
        let mut global = vec![0.0f32; seq_len];
        if let Some(padding) = padding {
            for key in 0..seq_len {
                if !padding[key] {
                    global[key] = MASKED;
                }
            }
        }

        // Trap 6: start from the padding mask and fill the out-of-window
        // positions, which is `global_attention_mask.masked_fill(...)`. A key
        // that is both padded and in the window stays masked either way.
        //
        // Trap 4 is delegated: `in_local_window` owns the inclusive
        // comparison and the "total window, so half each side" halving, so
        // the three off-by-ones cannot be reintroduced here by accident.
        let mut local = Matrix::zeros(seq_len, seq_len);
        for query in 0..seq_len {
            let row = local.row_mut(query);
            for key in 0..seq_len {
                row[key] = if config.in_local_window(query, key) {
                    global[key]
                } else {
                    MASKED
                };
            }
        }

        Masks {
            seq_len,
            global,
            local,
        }
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// The additive mask one query row sees, `seq_len` values.
    ///
    /// For a global layer that is the padding mask, the same slice for every
    /// query. For a local layer it is that query's row of the window mask.
    /// Selecting here rather than in `forward` is the whole reason the masks
    /// are built in pairs.
    pub fn additive_row(&self, query: usize, is_global: bool) -> &[f32] {
        if is_global {
            &self.global
        } else {
            self.local.row(query)
        }
    }
}

/// One layer's attention weights.
pub struct Attention {
    /// `Wqkv`, shaped `[3 * hidden_size, in_features]`. Rows `0..hidden_size`
    /// project Q, the next `hidden_size` project K, the last project V.
    pub(crate) wqkv: Matrix,
    /// `Wo`, shaped `[hidden_size, hidden_size]`.
    pub(crate) wo: Matrix,
    pub(crate) num_heads: usize,
    /// `hidden_size / num_heads`, so 64. Kept rather than recomputed because
    /// it is the stride every loop below steps by.
    pub(crate) head_dim: usize,
}

impl Attention {
    /// Load layer `layer`'s `Wqkv` and `Wo`.
    ///
    /// `num_attention_heads` is here because the head count is not derivable
    /// from the weight shapes: a `[3072, 1024]` `Wqkv` is 16 heads of 64 or 8
    /// of 128 and the file cannot tell you which. Both shapes are required
    /// exactly, so a checkpoint that disagrees names the tensor and both
    /// shapes rather than producing a wrongly shaped matrix that fails much
    /// later.
    pub fn load(
        checkpoint: &mut Checkpoint,
        layer: usize,
        hidden_size: usize,
        num_attention_heads: usize,
    ) -> Result<Self, TensorError> {
        if num_attention_heads == 0 || hidden_size % num_attention_heads != 0 {
            return Err(TensorError::new(format!(
                "model.layers.{layer}.attn: hidden_size {hidden_size} does not divide \
                 into {num_attention_heads} heads"
            )));
        }

        let wqkv_name = format!("model.layers.{layer}.attn.Wqkv.weight");
        let wo_name = format!("model.layers.{layer}.attn.Wo.weight");

        // Trap 1, stated as a shape: one fused projection with *three* rows
        // per output feature, in the order Q, K, V.
        let wqkv = checkpoint.take(&wqkv_name, &[3 * hidden_size, hidden_size])?;
        let wo = checkpoint.take(&wo_name, &[hidden_size, hidden_size])?;

        // No bias is loaded, and none exists: `attention_bias` is false, so
        // the checkpoint holds no `attn.Wqkv.bias` or `attn.Wo.bias`. The
        // only bias in the whole file is `decoder.bias`.
        Ok(Attention {
            wqkv: Matrix::from_vec(3 * hidden_size, hidden_size, wqkv),
            wo: Matrix::from_vec(hidden_size, hidden_size, wo),
            num_heads: num_attention_heads,
            head_dim: hidden_size / num_attention_heads,
        })
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn hidden_size(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// `head_dim ** -0.5`, which is 0.125 for a head_dim of 64.
    ///
    /// Trap 2. It is applied once, in [`Self::forward`], after the dot
    /// product. Scaling `q` before the matmul instead is equivalent; doing
    /// both is the mistake.
    pub fn scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }

    /// Split the fused projection into Q, K and V, each `[seq, hidden_size]`.
    ///
    /// Trap 1. `view(bs, seq, 3, num_heads, head_dim)` puts the 3 outside the
    /// head axis, so the 3072 columns are `[Q (1024) | K (1024) | V (1024)]`
    /// and each 1024 is `[head0 (64) | head1 (64) | ... | head15 (64)]`. The
    /// slices below are therefore contiguous thirds, and head-major inside.
    ///
    /// Reading the same weight as per-head-interleaved `[q0 k0 v0 q1 k1 v1
    /// ...]` gives three matrices of plausible magnitude that are wrong in
    /// every value, and nothing downstream notices.
    fn split_qkv(&self, qkv: &Matrix) -> (Matrix, Matrix, Matrix) {
        let hidden = self.hidden_size();
        assert_eq!(
            qkv.cols,
            3 * hidden,
            "the fused projection has {} columns, expected {}",
            qkv.cols,
            3 * hidden
        );

        let mut query = Matrix::zeros(qkv.rows, hidden);
        let mut key = Matrix::zeros(qkv.rows, hidden);
        let mut value = Matrix::zeros(qkv.rows, hidden);

        for row in 0..qkv.rows {
            let fused = qkv.row(row);
            query.row_mut(row).copy_from_slice(&fused[0..hidden]);
            key.row_mut(row).copy_from_slice(&fused[hidden..2 * hidden]);
            value
                .row_mut(row)
                .copy_from_slice(&fused[2 * hidden..3 * hidden]);
        }
        (query, key, value)
    }

    /// One attention block: `[seq, hidden_size]` in, `[seq, hidden_size]` out.
    ///
    /// `rope` must be the table for this layer's theta, and `is_global` must
    /// be [`crate::config::Config::is_global_layer`] for this layer. The two
    /// travel together: a global layer uses theta 160000 and the padding
    /// mask, a local layer theta 10000 and the window mask. Passing one
    /// without the other is a silent wrong answer, which is why both are
    /// parameters rather than one being inferred from the other here.
    pub fn forward(&self, x: &Matrix, rope: &RopeTable, masks: &Masks, is_global: bool) -> Matrix {
        let hidden = self.hidden_size();
        assert_eq!(
            x.cols, hidden,
            "the input has {} features, expected {hidden}",
            x.cols
        );
        assert_eq!(
            x.rows,
            masks.seq_len(),
            "the input has {} positions but the masks were built for {}",
            x.rows,
            masks.seq_len()
        );

        // One projection gives all three: `[seq, 3 * hidden_size]`.
        let qkv = linear(x, &self.wqkv);
        let (mut query, mut key, value) = self.split_qkv(&qkv);

        // Trap 3: RoPE rotates Q and K, and never V. Position is a property
        // of the *comparison* between a query and a key, so it belongs in the
        // dot product; V is the payload being averaged and is used as it is.
        // Rotating V as well still produces finite output of the right
        // magnitude, so `tests::rope_reaches_q_and_k_but_not_v` is what
        // catches it.
        //
        // Trap 8: `apply_rope` uses row `t` for position `t` in both layer
        // types. The reference sizes a local layer's cached table to
        // `local_attention`, which bounds the cache and nothing else;
        // positions are never wrapped into `0..127`.
        apply_rope(&mut query, rope, self.num_heads, self.head_dim);
        apply_rope(&mut key, rope, self.num_heads, self.head_dim);

        let seq_len = x.rows;
        let scale = self.scale();
        let mut context = Matrix::zeros(seq_len, hidden);
        // One row of attention probabilities, reused across heads and
        // queries. The full `[heads, seq, seq]` tensor is never materialised.
        let mut probabilities = vec![0.0f32; seq_len];

        for head in 0..self.num_heads {
            // The stride written out: head `h` owns columns
            // `h * head_dim .. (h + 1) * head_dim` of every row.
            let offset = head * self.head_dim;

            for q in 0..seq_len {
                attention_row(
                    query.row(q),
                    &key,
                    offset,
                    self.head_dim,
                    masks.additive_row(q, is_global),
                    scale,
                    &mut probabilities,
                );

                // `attn_weights @ value`, for this head and this query.
                let out = &mut context.row_mut(q)[offset..offset + self.head_dim];
                for (k, &p) in probabilities.iter().enumerate() {
                    let v = &value.row(k)[offset..offset + self.head_dim];
                    for d in 0..self.head_dim {
                        out[d] += p * v[d];
                    }
                }
            }
        }

        // Merging the heads is a no-op here: `context` already holds the
        // heads side by side in the row, which is what
        // `transpose(1, 2).contiguous().view(bs, -1, dim)` produces.
        linear(&context, &self.wo)
    }
}

/// The attention probabilities for one query and one head.
///
/// Scaled dot products against every key, plus the additive mask, through the
/// softmax. Writes `scores`, one value per key.
///
/// This is `softmax(q @ k.T * scale + mask)` with the three steps in the
/// reference's order. It is a free function so the tests can drive the mask
/// and the softmax without going through a whole layer.
fn attention_row(
    query_row: &[f32],
    key: &Matrix,
    head_offset: usize,
    head_dim: usize,
    mask_row: &[f32],
    scale: f32,
    scores: &mut [f32],
) {
    debug_assert_eq!(scores.len(), key.rows);
    debug_assert_eq!(mask_row.len(), key.rows);

    let q = &query_row[head_offset..head_offset + head_dim];

    for (k, score) in scores.iter_mut().enumerate() {
        let kv = &key.row(k)[head_offset..head_offset + head_dim];
        let mut dot = 0.0f32;
        for d in 0..head_dim {
            dot += q[d] * kv[d];
        }
        // Trap 2: the scale is applied here and nowhere else.
        //
        // Trap 5: the mask is **additive**, and `MASKED` is so much larger
        // than any real score that the sum rounds back to exactly `MASKED`.
        *score = dot * scale + mask_row[k];
    }

    // Trap 7: `softmax_in_place` subtracts the row maximum and accumulates in
    // f64. A global layer sums over the whole sequence, and f32 accumulation
    // there drifts far enough to miss the 1e-4 tolerance against PyTorch.
    softmax_in_place(scores);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real config, with `hidden_size` and `num_attention_heads`
    /// overridden so the tests can work at a size that fits on a page.
    ///
    /// Built through `Config::from_json` rather than by filling the struct,
    /// so the window predicates under test are the ones the real loader
    /// produces, validation included.
    fn config_with(hidden_size: usize, num_attention_heads: usize) -> Config {
        let json = serde_json::json!({
            "num_hidden_layers": 28,
            "hidden_size": hidden_size,
            "intermediate_size": 2624,
            "num_attention_heads": num_attention_heads,
            "vocab_size": 50368,
            "max_position_embeddings": 8192,
            "global_attn_every_n_layers": 3,
            "local_attention": 128,
            "global_rope_theta": 160000.0,
            "local_rope_theta": 10000.0,
            "norm_eps": 1e-5,
            "norm_bias": false,
            "attention_bias": false,
            "mlp_bias": false,
            "decoder_bias": true,
        });
        Config::from_json(&json).expect("the synthetic config must parse")
    }

    /// Build an in-memory safetensors file and read it back, so the tests
    /// exercise the real loader and the real `require` error path without
    /// touching `weights/model.safetensors`. That file is 791 MB on disk and
    /// 1.58 GB as f32; see hard rule 8 in `CLAUDE.md`. Every weight here is
    /// written by hand instead.
    ///
    /// The values are stored as `F32` rather than the checkpoint's `BF16`,
    /// which the loader also accepts, so the test arithmetic is not rounded
    /// to 8 mantissa bits before it starts.
    fn checkpoint_of(tensors: &[(String, Vec<usize>, Vec<f32>)]) -> Checkpoint {
        let mut entries: Vec<String> = Vec::new();
        let mut payload: Vec<u8> = Vec::new();

        for (name, shape, data) in tensors {
            let start = payload.len();
            for value in data {
                payload.extend_from_slice(&value.to_le_bytes());
            }
            let dims: Vec<String> = shape.iter().map(usize::to_string).collect();
            entries.push(format!(
                "{:?}:{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
                name,
                dims.join(","),
                start,
                payload.len()
            ));
        }

        let header = format!("{{{}}}", entries.join(","));
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&payload);

        Checkpoint::from_bytes(&bytes).expect("the synthetic checkpoint must parse")
    }

    /// A one-layer checkpoint holding just `Wqkv` and `Wo` for layer 0, with
    /// the shapes given rather than inferred, so a test can deliberately
    /// store a wrong one.
    fn layer_zero(wqkv: (Vec<usize>, Vec<f32>), wo: (Vec<usize>, Vec<f32>)) -> Checkpoint {
        checkpoint_of(&[
            (
                "model.layers.0.attn.Wqkv.weight".to_string(),
                wqkv.0,
                wqkv.1,
            ),
            ("model.layers.0.attn.Wo.weight".to_string(), wo.0, wo.1),
        ])
    }

    /// Load an [`Attention`] from hand-written `Wqkv` values and an identity
    /// `Wo`, so the output *is* the merged heads and nothing is hidden behind
    /// one more matmul.
    fn attention_with(wqkv: Vec<f32>, hidden: usize, heads: usize) -> Attention {
        let mut checkpoint = layer_zero(
            (vec![3 * hidden, hidden], wqkv),
            (vec![hidden, hidden], identity(hidden)),
        );
        Attention::load(&mut checkpoint, 0, hidden, heads).expect("the shapes are right")
    }

    fn identity(n: usize) -> Vec<f32> {
        let mut values = vec![0.0f32; n * n];
        for i in 0..n {
            values[i * n + i] = 1.0;
        }
        values
    }

    /// Deterministic values in roughly `[-0.5, 0.5]`, for the tests that care
    /// about structure rather than exact arithmetic. A linear congruential
    /// generator, so the sequence is the same on every run and every machine.
    fn spread(count: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                let unit = (state >> 8) as f32 / 16_777_216.0;
                unit - 0.5
            })
            .collect()
    }

    /// The rows of `x` whose position carries a real token, as a one-hot
    /// basis: row `t` is `e_t`, so `Wqkv`'s column `t` is token `t`'s fused
    /// projection. Several tests below are written as "column `t` holds token
    /// `t`'s q, k and v", and this is what makes that true.
    fn one_hot_rows(seq_len: usize, hidden: usize) -> Matrix {
        let mut x = Matrix::zeros(seq_len, hidden);
        for t in 0..seq_len {
            x.set(t, t, 1.0);
        }
        x
    }

    /// Which keys a mask row leaves visible.
    fn visible_keys(mask_row: &[f32]) -> Vec<usize> {
        mask_row
            .iter()
            .enumerate()
            .filter(|(_, &m)| m != MASKED)
            .map(|(k, _)| k)
            .collect()
    }

    // ---------------------------------------------------------------- traps

    /// Trap 4, the window arithmetic, and the most valuable test in the file.
    ///
    /// `local_attention: 128` is the **total** window, so 64 each side, and
    /// the comparison is inclusive. A query in the middle of a long sequence
    /// therefore sees **129** keys: 64 behind, 64 ahead, and itself. Three
    /// separate off-by-ones live here:
    ///
    /// - reading 128 as per-side doubles the receptive field of 18 of the 28
    ///   layers, giving 257 keys;
    /// - a strict `<` instead of `<=` clips one key on each side, giving 127;
    /// - copying current transformers' `sliding_window` of
    ///   `local_attention / 2 + 1` gives 131, and that `+1` exists only to
    ///   cancel a `-1` inside the flash-attention wrapper.
    ///
    /// At the edges the window is truncated by the sequence, not wrapped, so
    /// query 0 sees 65 keys (itself and 64 ahead) and the last query sees 65
    /// (itself and 64 behind).
    #[test]
    fn a_local_query_sees_129_keys() {
        let config = config_with(1024, 16);
        assert_eq!(config.local_window_each_side(), 64);
        assert_eq!(config.local_window_keys(), 129);

        let seq_len = 300;
        let masks = Masks::new(&config, seq_len, None);

        let middle = visible_keys(masks.additive_row(150, false));
        assert_eq!(
            middle.len(),
            129,
            "a query at 150 of {seq_len} must see 64 + 1 + 64 keys, not {}",
            middle.len()
        );
        assert_eq!(middle[0], 150 - 64, "the window starts 64 behind the query");
        assert_eq!(
            middle[128],
            150 + 64,
            "the window ends 64 ahead of the query"
        );

        let first = visible_keys(masks.additive_row(0, false));
        assert_eq!(first.len(), 65, "query 0 has nothing behind it");
        assert_eq!(first[64], 64, "and reaches exactly as far as key 64");

        let last = visible_keys(masks.additive_row(seq_len - 1, false));
        assert_eq!(last.len(), 65, "the last query has nothing ahead of it");
        assert_eq!(
            last[0],
            seq_len - 1 - 64,
            "and reaches exactly as far back as key {}",
            seq_len - 1 - 64
        );

        // The same count after the softmax, which is where it has to survive.
        // With an all-zero query every visible score is 0 and every hidden
        // score is `MASKED`, so the probabilities come out uniform over
        // exactly the visible keys: 1/129 each, and an exact 0.0 elsewhere
        // because `exp(MASKED - 0)` underflows to zero.
        let head_dim = 64;
        let mut probabilities = vec![0.0f32; seq_len];
        attention_row(
            &vec![0.0f32; head_dim],
            &Matrix::zeros(seq_len, head_dim),
            0,
            head_dim,
            masks.additive_row(150, false),
            0.125,
            &mut probabilities,
        );

        let attended: Vec<usize> = probabilities
            .iter()
            .enumerate()
            .filter(|(_, &p)| p != 0.0)
            .map(|(k, _)| k)
            .collect();
        assert_eq!(attended, middle, "the softmax must keep exactly the window");
        for &k in &attended {
            assert!(
                (probabilities[k] - 1.0 / 129.0).abs() < 1e-7,
                "key {k} got {} rather than 1/129",
                probabilities[k]
            );
        }
    }

    /// Trap 4 again, from the other side: an implementation that quietly used
    /// the global mask in every layer passes almost every other test in this
    /// file, so here is the one it fails.
    ///
    /// The two mask kinds agree exactly when **every pair** of positions is
    /// within the window, which needs `seq_len - 1 <= 64`, so `seq_len <=
    /// 65`. Note that 129 is *not* the threshold: at `seq_len = 129` the
    /// query at 0 still cannot see the key at 128, because 129 is the number
    /// of keys one query reaches, not the length of sequence the window
    /// covers. That misreading is why this test checks 66 as well.
    #[test]
    fn global_and_local_agree_only_while_the_whole_sequence_fits() {
        let hidden = 4;
        let heads = 2;
        let config = config_with(hidden, heads);
        let rope = RopeTable::new(config.head_dim(), 512, 10000.0);
        let attention = attention_with(spread(3 * hidden * hidden, 11), hidden, heads);

        let run = |seq_len: usize| -> (Matrix, Matrix) {
            let x = Matrix::from_vec(seq_len, hidden, spread(seq_len * hidden, 12));
            let masks = Masks::new(&config, seq_len, None);
            (
                attention.forward(&x, &rope, &masks, true),
                attention.forward(&x, &rope, &masks, false),
            )
        };

        // 65 positions means a maximum distance of 64, which the inclusive
        // test keeps. Every mask entry is zero in both masks, so the two
        // results are not merely close, they are identical.
        let (global, local) = run(65);
        assert_eq!(
            global.max_abs_diff(&local),
            0.0,
            "at 65 positions every pair is inside the window"
        );

        // 66 positions puts the pair (0, 65) at distance 65, just outside.
        let (global, local) = run(66);
        assert!(
            global.max_abs_diff(&local) > 1e-6,
            "at 66 positions the pair (0, 65) is outside the window, so the two \
             masks must differ; got {}",
            global.max_abs_diff(&local)
        );

        // And at 300 the difference is large, not marginal.
        let (global, local) = run(300);
        assert!(
            global.max_abs_diff(&local) > 1e-4,
            "at 300 positions a local layer sees 129 of 300 keys, so the results \
             must differ substantially; got {}",
            global.max_abs_diff(&local)
        );
    }

    /// The probabilities are a distribution: one row per query per head, each
    /// summing to 1.
    ///
    /// Checked for both layer kinds at a length where the window bites, so a
    /// local row sums over 129 terms and a global row over 300.
    #[test]
    fn the_probabilities_sum_to_one_per_query_and_head() {
        let hidden = 8;
        let heads = 2;
        let head_dim = hidden / heads;
        let seq_len = 300;

        let config = config_with(hidden, heads);
        let masks = Masks::new(&config, seq_len, None);
        let query = Matrix::from_vec(seq_len, hidden, spread(seq_len * hidden, 21));
        let key = Matrix::from_vec(seq_len, hidden, spread(seq_len * hidden, 22));

        let mut probabilities = vec![0.0f32; seq_len];
        for is_global in [true, false] {
            for head in 0..heads {
                for q in 0..seq_len {
                    attention_row(
                        query.row(q),
                        &key,
                        head * head_dim,
                        head_dim,
                        masks.additive_row(q, is_global),
                        1.0 / (head_dim as f32).sqrt(),
                        &mut probabilities,
                    );
                    // Summed in f64 on purpose. Adding 300 f32 values in f32
                    // drifts by about 1e-6 all by itself, which would be the
                    // test measuring its own arithmetic rather than the
                    // softmax's.
                    let total: f64 = probabilities.iter().map(|&p| p as f64).sum();
                    assert!(
                        (total - 1.0).abs() < 1e-6,
                        "is_global {is_global}, head {head}, query {q}: the row sums \
                         to {total}"
                    );
                }
            }
        }
    }

    /// Trap 5: the mask hides padded **keys**. Changing what sits at a padded
    /// position must not move any output at an unpadded one.
    ///
    /// The equality asserted is exact, not approximate, and it can be:
    /// `exp(MASKED - max)` for a finite `max` underflows to 0.0, so a hidden
    /// key contributes `0.0 * v`, which is an exact zero, and the softmax's
    /// denominator never sees it either.
    #[test]
    fn a_padded_key_cannot_reach_a_real_query() {
        let hidden = 4;
        let heads = 2;
        let seq_len = 8;
        let real = 6;

        let config = config_with(hidden, heads);
        let rope = RopeTable::new(config.head_dim(), 64, 160000.0);
        let attention = attention_with(spread(3 * hidden * hidden, 31), hidden, heads);

        let padding: Vec<bool> = (0..seq_len).map(|t| t < real).collect();
        let masks = Masks::new(&config, seq_len, Some(&padding));

        let mut x = Matrix::from_vec(seq_len, hidden, spread(seq_len * hidden, 32));
        let before = attention.forward(&x, &rope, &masks, true);

        // Put wildly different values at the two padded positions.
        for t in real..seq_len {
            for c in 0..hidden {
                x.set(t, c, 7.5 * (c as f32 + 1.0));
            }
        }
        let after = attention.forward(&x, &rope, &masks, true);

        for t in 0..real {
            assert_eq!(
                before.row(t),
                after.row(t),
                "output row {t} moved when padded positions changed"
            );
        }

        // The same for a local layer, where the window mask has to carry the
        // padding mask with it. Trap 6: build the window alone and padding
        // leaks straight into all 18 local layers, and at this length every
        // position is inside every window so nothing else would hide it.
        let before = attention.forward(&x, &rope, &masks, false);
        for t in real..seq_len {
            for c in 0..hidden {
                x.set(t, c, -3.25 * (c as f32 + 1.0));
            }
        }
        let after = attention.forward(&x, &rope, &masks, false);
        for t in 0..real {
            assert_eq!(
                before.row(t),
                after.row(t),
                "local layer: output row {t} moved when padded positions changed"
            );
        }
    }

    /// Trap 5, the other half: padded **queries** are not masked.
    ///
    /// The reference's padding mask is `[bs, 1, 1, seq]`, so the query axis is
    /// a broadcast and no row of the attention matrix is ever suppressed. A
    /// pad row therefore attends normally to the real keys and emits finite
    /// junk. This test asserts the finiteness, because the alternative
    /// implementation — masking pad queries too, or using `-inf` — produces
    /// NaN that then poisons every later layer through the residual stream.
    ///
    /// **The consequence is that outputs at pad positions must be ignored
    /// downstream.** They are meaningless, not zero. Anything that reduces
    /// over the sequence (mean pooling, or `classifier_pooling: "mean"`) has
    /// to exclude them itself; this model's MLM path reads one mask position
    /// and never looks at the rest, so it is safe.
    #[test]
    fn a_padded_query_is_not_masked_and_stays_finite() {
        let hidden = 4;
        let heads = 2;
        let seq_len = 8;
        let real = 3;

        let config = config_with(hidden, heads);
        let rope = RopeTable::new(config.head_dim(), 64, 160000.0);
        let attention = attention_with(spread(3 * hidden * hidden, 41), hidden, heads);

        let padding: Vec<bool> = (0..seq_len).map(|t| t < real).collect();
        let masks = Masks::new(&config, seq_len, Some(&padding));
        let x = Matrix::from_vec(seq_len, hidden, spread(seq_len * hidden, 42));

        // The structural statement, which is what the reference's
        // `[bs, 1, 1, seq]` shape means: the padding mask has no query axis,
        // so a pad query's row is identical to a real query's row.
        for q in 1..seq_len {
            assert_eq!(
                masks.additive_row(q, true),
                masks.additive_row(0, true),
                "the padding mask must not depend on the query, but row {q} differs \
                 from row 0"
            );
        }
        // And in a local layer, where the mask does have a query axis, a pad
        // query still sees every real key inside its window. At 8 positions
        // every key is inside every window, so all `real` of them are open.
        for q in 0..seq_len {
            let row = masks.additive_row(q, false);
            for (k, &m) in row.iter().take(real).enumerate() {
                assert_eq!(
                    m, 0.0,
                    "query {q} is padded but must still see real key {k}"
                );
            }
        }

        for is_global in [true, false] {
            let y = attention.forward(&x, &rope, &masks, is_global);
            assert!(
                y.data.iter().all(|v| v.is_finite()),
                "is_global {is_global}: every output must be finite, including the \
                 rows at pad positions; got {y:?}"
            );
        }

        // And the extreme case the finite `MASKED` exists for: a local query
        // more than 64 positions from every real token, whose whole row is
        // masked. PyTorch's softmax over a row of `finfo.min` returns a
        // uniform distribution rather than NaN, and so does this one.
        let seq_len = 300;
        let rope = RopeTable::new(config.head_dim(), seq_len, 160000.0);
        let padding: Vec<bool> = (0..seq_len).map(|t| t < 4).collect();
        let masks = Masks::new(&config, seq_len, Some(&padding));
        let x = Matrix::from_vec(seq_len, hidden, spread(seq_len * hidden, 43));
        let y = attention.forward(&x, &rope, &masks, false);
        assert!(
            y.data.iter().all(|v| v.is_finite()),
            "a fully masked local row must stay finite, not become NaN"
        );

        // And what that row actually contains, which is the assertion that
        // separates [`MASKED`] from `-inf`. Every score in it is `MASKED`, so
        // the maximum is finite, the subtraction gives zeros, and the softmax
        // returns a **uniform** distribution over all `seq_len` keys. That is
        // bit-for-bit what PyTorch produces from `finfo(float32).min`. With
        // `-inf` the row's maximum is not finite, `softmax_in_place` returns
        // all zeros instead, and the output quietly becomes zero rather than
        // junk — a different answer from the reference, and one no finiteness
        // check can see.
        let head_dim = config.head_dim();
        let mut probabilities = vec![0.0f32; seq_len];
        attention_row(
            &vec![0.5f32; hidden],
            &Matrix::from_vec(seq_len, hidden, spread(seq_len * hidden, 44)),
            0,
            head_dim,
            masks.additive_row(200, false),
            1.0 / (head_dim as f32).sqrt(),
            &mut probabilities,
        );
        let uniform = 1.0 / seq_len as f32;
        for (k, &p) in probabilities.iter().enumerate() {
            assert!(
                (p - uniform).abs() < 1e-9,
                "a fully masked row must come out uniform at 1/{seq_len}, as PyTorch \
                 does from finfo.min; key {k} got {p}"
            );
        }
    }

    /// Trap 1: `Wqkv` is three contiguous blocks `[Q | K | V]`, not per-head
    /// interleaved `[q0 k0 v0 q1 k1 v1 ...]`.
    ///
    /// Three parts. `hidden_size = 4` with 2 heads of 2, which is the
    /// smallest size where the two layouts are not the same permutation, and
    /// `Wo` is the identity so the output *is* the merged heads.
    ///
    /// **V is not where the interleaved layout puts it.** Zero the third
    /// block. Under the contiguous reading V is then identically zero and the
    /// output is exactly zero whatever Q and K do. Under the interleaved
    /// reading head 0's V would come from rows `2..4`, which still hold
    /// nonzero values, so the output would not be zero.
    ///
    /// **V is the third block.** Zero everything *except* the third block.
    /// Under the contiguous reading Q and K vanish, so every score is 0, the
    /// probabilities are uniform, and every query gets the mean of V, which
    /// is nonzero in both heads. Under the interleaved reading head 0's V
    /// would come from rows `4..6`, now zero, so head 0's output block would
    /// be exactly zero.
    ///
    /// **The three blocks are in the order Q, K, V.** Interleaving is not the
    /// only way to mis-split a fused projection; reading `[K | Q | V]` is the
    /// other. So give token 0 a nonzero `q` and no `k`, and token 1 a nonzero
    /// `k` and no `q`. Then query 0's scores are `[q0.k0, q0.k1] = [0,
    /// nonzero]`, which is not uniform, while query 1's are `[0, 0]`, which
    /// is. So row 1 comes out as the mean of V and row 0 does not. Swap the
    /// two blocks and the signature reverses: row 0 becomes the mean and row
    /// 1 stops being it.
    #[test]
    fn the_qkv_split_is_three_contiguous_blocks() {
        let hidden = 4;
        let heads = 2;
        let head_dim = hidden / heads;
        let seq_len = 3;

        let config = config_with(hidden, heads);
        let rope = RopeTable::new(config.head_dim(), 64, 160000.0);
        let x = Matrix::from_vec(seq_len, hidden, spread(seq_len * hidden, 51));
        let masks = Masks::new(&config, seq_len, None);

        // Reading weight `w` under the interleaved convention gives exactly
        // what reading `interleaved_as_contiguous(w)` under the contiguous
        // convention gives, so the wrong answer can be computed with the same
        // `Attention` rather than a second implementation.
        //
        //   contiguous slot (block b, head h, dim d) -> row b * hidden + h * head_dim + d
        //   interleaved slot (head h, block b, dim d) -> row h * 3 * head_dim + b * head_dim + d
        let interleaved_as_contiguous = |w: &[f32]| -> Vec<f32> {
            let mut out = vec![0.0f32; w.len()];
            for b in 0..3 {
                for h in 0..heads {
                    for d in 0..head_dim {
                        let contiguous = b * hidden + h * head_dim + d;
                        let interleaved = h * 3 * head_dim + b * head_dim + d;
                        for c in 0..hidden {
                            out[contiguous * hidden + c] = w[interleaved * hidden + c];
                        }
                    }
                }
            }
            out
        };

        // --- V zeroed: rows 2 * hidden .. 3 * hidden are all zero.
        let mut wqkv = spread(3 * hidden * hidden, 52);
        for row in (2 * hidden)..(3 * hidden) {
            for c in 0..hidden {
                wqkv[row * hidden + c] = 0.0;
            }
        }
        let contiguous =
            attention_with(wqkv.clone(), hidden, heads).forward(&x, &rope, &masks, true);
        let interleaved = attention_with(interleaved_as_contiguous(&wqkv), hidden, heads)
            .forward(&x, &rope, &masks, true);

        let interleaved_scale = interleaved.data.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(
            contiguous.data.iter().all(|&v| v == 0.0),
            "with the third block zeroed, V is zero and the output must be exactly \
             zero; got {contiguous:?}"
        );
        assert!(
            interleaved_scale > 1e-3,
            "this weight must separate the two layouts: the interleaved reading \
             would have given values up to {interleaved_scale}, which is too small \
             to tell them apart"
        );

        // --- Only V nonzero: rows 0 .. 2 * hidden are all zero.
        let mut wqkv = spread(3 * hidden * hidden, 53);
        for row in 0..(2 * hidden) {
            for c in 0..hidden {
                wqkv[row * hidden + c] = 0.0;
            }
        }
        let contiguous =
            attention_with(wqkv.clone(), hidden, heads).forward(&x, &rope, &masks, true);
        let interleaved = attention_with(interleaved_as_contiguous(&wqkv), hidden, heads)
            .forward(&x, &rope, &masks, true);

        let head_zero = |y: &Matrix| -> f32 {
            (0..seq_len)
                .flat_map(|t| (0..head_dim).map(move |d| (t, d)))
                .map(|(t, d)| y.at(t, d).abs())
                .fold(0.0f32, f32::max)
        };
        assert!(
            head_zero(&contiguous) > 1e-3,
            "V is the third block, so head 0 must produce something; got {contiguous:?}"
        );
        assert_eq!(
            head_zero(&interleaved),
            0.0,
            "this weight must separate the two layouts: the interleaved reading puts \
             head 0's V in the zeroed region and must give an exactly zero block"
        );

        // Every query sees q = k = 0, so the probabilities are uniform and
        // every row is the mean of V. That is the same statement as "nothing
        // outside the third block reached the scores".
        for t in 1..seq_len {
            for c in 0..hidden {
                assert!(
                    (contiguous.at(t, c) - contiguous.at(0, c)).abs() < 1e-6,
                    "with Q and K zeroed every query must get the same answer; row \
                     {t} col {c} is {} against {}",
                    contiguous.at(t, c),
                    contiguous.at(0, c)
                );
            }
        }

        // --- The block order is Q, K, V and not K, Q, V.
        let seq_len = 2;
        let masks = Masks::new(&config, seq_len, None);
        let x = one_hot_rows(seq_len, hidden);

        let mut wqkv = vec![0.0f32; 3 * hidden * hidden];
        let mut put = |row: usize, col: usize, value: f32| wqkv[row * hidden + col] = value;
        // Q block, rows 0..4: token 0 only, so column 0.
        for (d, value) in [1.0f32, 0.5, -0.5, 0.25].iter().enumerate() {
            put(d, 0, *value);
        }
        // K block, rows 4..8: token 1 only, so column 1.
        for (d, value) in [0.5f32, -1.0, 0.75, 0.5].iter().enumerate() {
            put(hidden + d, 1, *value);
        }
        // V block, rows 8..12: both tokens, and different.
        let v0 = [1.0f32, 2.0, 3.0, 4.0];
        let v1 = [-1.0f32, 0.5, 2.0, -3.0];
        for d in 0..hidden {
            put(2 * hidden + d, 0, v0[d]);
            put(2 * hidden + d, 1, v1[d]);
        }

        let y = attention_with(wqkv, hidden, heads).forward(&x, &rope, &masks, true);

        for d in 0..hidden {
            let mean = 0.5 * (v0[d] + v1[d]);
            assert!(
                (y.at(1, d) - mean).abs() < 1e-6,
                "token 1 has no q, so query 1's scores are both zero and its row \
                 must be the mean of V; dim {d} is {} rather than {mean}. Reading \
                 the blocks as [K | Q | V] gives this row a non-uniform \
                 distribution instead.",
                y.at(1, d)
            );
            assert!(
                (y.at(0, d) - mean).abs() > 1e-4,
                "query 0 has a nonzero q and sees a nonzero k only at token 1, so \
                 its distribution must not be uniform; dim {d} is the mean {mean}, \
                 which is what [K | Q | V] would have given"
            );
        }
    }

    /// Trap 2: the scale is `head_dim ** -0.5`, applied exactly once.
    ///
    /// One head of `head_dim = 4`, so the scale is `1 / sqrt(4) = 0.5`, which
    /// is far enough from the alternatives (`1.0` for no scale, `0.25` for
    /// `1 / head_dim`, `2.0` for a reciprocal mix-up) to separate them.
    ///
    /// `hidden_size = 4` and `x` is the one-hot basis, so `Wqkv`'s column `t`
    /// is token `t`'s fused projection. Two tokens, and everything is chosen
    /// so the arithmetic closes:
    ///
    /// ```text
    /// token 0:  q = [0,0,0,0]   k = [1,0,0,0]   v = [ 1,0,0,0]
    /// token 1:  q = [4,0,0,0]   k = [0,0,0,0]   v = [10,0,0,0]
    /// ```
    ///
    /// Look at query 1. RoPE's `inv_freq[0]` is `theta^0 = 1` for every
    /// theta, so the angle at position 1 for the first dimension pair is
    /// exactly 1 radian, and with only dim 0 populated:
    ///
    /// ```text
    /// rope(q at 1) = [4*cos(1), 0, 4*sin(1), 0]
    /// rope(k at 0) = [1, 0, 0, 0]              (position 0: cos = 1, sin = 0)
    /// rope(k at 1) = [0, 0, 0, 0]              (zero rotates to zero)
    ///
    /// q.k for key 0 = 4*cos(1) = 2.161209
    /// q.k for key 1 = 0
    ///
    /// scores = [0.5 * 2.161209, 0.5 * 0] = [1.080605, 0]
    /// p      = softmax(scores) = [0.746606, 0.253394]
    /// out    = p0 * 1 + p1 * 10 = 3.280345        (Wo is the identity)
    /// ```
    ///
    /// With no scale the scores would be `[2.161209, 0]`, giving
    /// `p = [0.896932, 0.103068]` and `out = 1.824549`. With `1 / head_dim`
    /// they would be `[0.540302, 0]`, giving `out = 4.302287`. All three are
    /// recomputed in f64 below rather than trusted as transcribed constants.
    #[test]
    fn the_scale_is_one_over_the_square_root_of_head_dim() {
        let hidden = 4;
        let heads = 1;
        let config = config_with(hidden, heads);
        assert_eq!(config.head_dim(), 4);

        let mut wqkv = vec![0.0f32; 3 * hidden * hidden];
        let mut put = |row: usize, col: usize, value: f32| wqkv[row * hidden + col] = value;
        // Q block, rows 0..4. Token 1's q is column 1 of row 0.
        put(0, 1, 4.0);
        // K block, rows 4..8. Token 0's k is column 0 of row 4.
        put(4, 0, 1.0);
        // V block, rows 8..12. Token 0's v, then token 1's v.
        put(8, 0, 1.0);
        put(8, 1, 10.0);

        let attention = attention_with(wqkv, hidden, heads);
        assert_eq!(attention.scale(), 0.5);
        // The real model's number, while the constant is in view.
        assert_eq!(1.0 / (64.0f32).sqrt(), 0.125);

        let rope = RopeTable::new(config.head_dim(), 64, 160000.0);
        let masks = Masks::new(&config, 2, None);
        let y = attention.forward(&one_hot_rows(2, hidden), &rope, &masks, true);

        let expected = |scale: f64| -> f64 {
            let logit = scale * 4.0 * 1.0f64.cos();
            let p0 = logit.exp() / (logit.exp() + 1.0);
            p0 * 1.0 + (1.0 - p0) * 10.0
        };
        let correct = expected(0.5);
        let unscaled = expected(1.0);
        let over_head_dim = expected(0.25);

        assert!(
            (y.at(1, 0) as f64 - correct).abs() < 1e-5,
            "query 1 gave {}, expected {correct}",
            y.at(1, 0)
        );
        assert!(
            (correct - unscaled).abs() > 1.0 && (correct - over_head_dim).abs() > 1.0,
            "this case must separate the candidate scales: 0.5 gives {correct}, \
             1.0 gives {unscaled}, 0.25 gives {over_head_dim}"
        );
        assert!(
            (y.at(1, 0) as f64 - unscaled).abs() > 1.0,
            "the output matches the unscaled logits, so no scale is being applied"
        );
    }

    /// A wrong `Wqkv` shape is an error that names the tensor, not a panic
    /// inside `Matrix::from_vec` and not a silently reshaped weight.
    #[test]
    fn a_wrong_shaped_weight_names_the_tensor() {
        // `Wqkv` must have 3 * hidden_size rows. Two blocks instead of three
        // is the mistake someone makes by forgetting that V shares the
        // projection.
        let mut checkpoint = layer_zero((vec![8, 4], spread(32, 61)), (vec![4, 4], identity(4)));

        let message = match Attention::load(&mut checkpoint, 0, 4, 2) {
            Err(error) => error.to_string(),
            Ok(_) => panic!("a Wqkv with 8 rows must not load"),
        };
        assert!(
            message.contains("model.layers.0.attn.Wqkv.weight"),
            "the message must name the tensor, got {message:?}"
        );
        assert!(
            message.contains("[8, 4]") && message.contains("[12, 4]"),
            "the message must give both shapes, got {message:?}"
        );

        // And the same for `Wo`, which is the squarer and therefore easier to
        // get wrong without noticing.
        let mut checkpoint =
            layer_zero((vec![12, 4], spread(48, 62)), (vec![4, 8], spread(32, 63)));
        let message = match Attention::load(&mut checkpoint, 0, 4, 2) {
            Err(error) => error.to_string(),
            Ok(_) => panic!("a Wo of [4, 8] must not load"),
        };
        assert!(
            message.contains("model.layers.0.attn.Wo.weight"),
            "the message must name the tensor, got {message:?}"
        );

        // A missing tensor names the layer it was looking for, so a loader
        // pointed at the wrong layer index says so rather than reporting a
        // shape problem.
        let message = match Attention::load(&mut checkpoint, 3, 4, 2) {
            Err(error) => error.to_string(),
            Ok(_) => panic!("layer 3 is not in this checkpoint"),
        };
        assert!(
            message.contains("model.layers.3.attn.Wqkv.weight"),
            "the message must name the tensor it wanted, got {message:?}"
        );
    }

    /// Trap 3: RoPE rotates Q and K, and never V.
    ///
    /// Two assertions, and between them they rule out all four wrong
    /// variants. `hidden_size = 4`, one head of 4, three tokens, and `x` is
    /// the one-hot basis, so `Wqkv`'s column `t` is token `t`'s projection and
    /// a value can be held constant across positions by giving a block three
    /// identical columns.
    ///
    /// **V is not rotated.** Give V three identical columns, so every token's
    /// `v` is the same vector. The probabilities sum to 1, so an unrotated V
    /// makes every output row exactly that vector, whatever the attention
    /// weights are. A rotated V would instead be `v` turned by a different
    /// angle at each position, and the weighted blend of those would be
    /// shorter than `v` and would have picked up a component in the second
    /// half of the head. Note that a *scale* test cannot see this: RoPE is
    /// linear, so scaling V scales a rotated V by the same factor.
    ///
    /// **RoPE reaches Q and K, and reaches both.** Hold Q and K constant
    /// across positions too, and let V vary. Then:
    ///
    /// - with RoPE on Q and K, the score is `(R_i q).(R_j k) = q.(R_{j-i} k)`,
    ///   which depends on `j - i`, so each query gets a different
    ///   distribution and the output rows differ;
    /// - with RoPE on neither, every score is `q.k`, the probabilities are
    ///   uniform, and every row is the mean of V;
    /// - with RoPE on Q only, the score is `(R_i q).k`, which does not depend
    ///   on `j`, so the probabilities are again uniform and every row is the
    ///   mean of V;
    /// - with RoPE on K only, the score is `q.(R_j k)`, which does not depend
    ///   on `i`, so every query gets the *same* distribution and every row is
    ///   identical again.
    ///
    /// So "the output rows differ from each other" is exactly the statement
    /// that the relative-position effect is present, which needs the rotation
    /// on both sides. The test also checks that the rows are not the mean of
    /// V, which is the specific wrong answer the two one-sided variants give.
    #[test]
    fn rope_reaches_q_and_k_but_not_v() {
        let hidden = 4;
        let heads = 1;
        let seq_len = 3;
        let config = config_with(hidden, heads);
        let rope = RopeTable::new(config.head_dim(), 64, 10000.0);
        let masks = Masks::new(&config, seq_len, None);
        let x = one_hot_rows(seq_len, hidden);

        // A block with three identical columns, so the projected value is the
        // same for tokens 0, 1 and 2. Column 3 is never read, because `x` has
        // only three rows.
        let constant_block = |v: [f32; 4]| -> Vec<f32> {
            let mut rows = vec![0.0f32; hidden * hidden];
            for (d, value) in v.iter().enumerate() {
                for t in 0..seq_len {
                    rows[d * hidden + t] = *value;
                }
            }
            rows
        };

        // --- V constant across positions.
        let v = [0.75f32, -0.25, 0.5, 1.25];
        let mut wqkv = Vec::new();
        wqkv.extend(constant_block([1.0, 0.5, -0.5, 0.25])); // Q
        wqkv.extend(constant_block([0.5, -1.0, 0.75, 0.5])); // K
        wqkv.extend(constant_block(v)); // V
        let y = attention_with(wqkv, hidden, heads).forward(&x, &rope, &masks, true);

        for t in 0..seq_len {
            for (d, &expected) in v.iter().enumerate() {
                assert!(
                    (y.at(t, d) - expected).abs() < 1e-6,
                    "every v is the same vector and the probabilities sum to 1, so \
                     row {t} must be v exactly; dim {d} is {} rather than {expected}. \
                     A rotated V would blend three different angles here.",
                    y.at(t, d)
                );
            }
        }

        // --- Q and K constant, V varying.
        let mut wqkv = Vec::new();
        wqkv.extend(constant_block([1.0, 0.5, -0.5, 0.25])); // Q
        wqkv.extend(constant_block([0.5, -1.0, 0.75, 0.5])); // K
        wqkv.extend(spread(hidden * hidden, 71)); // V, different per token
        let attention = attention_with(wqkv, hidden, heads);
        let y = attention.forward(&x, &rope, &masks, true);

        let row_spread = (0..hidden)
            .map(|d| (y.at(1, d) - y.at(0, d)).abs())
            .fold(0.0f32, f32::max);
        assert!(
            row_spread > 1e-4,
            "q and k are the same at every position, so the only thing that can make \
             two query rows differ is the relative rotation; rows 0 and 1 differ by \
             {row_spread}, which means RoPE is missing from Q, from K, or from both"
        );

        // The specific wrong answer the one-sided variants give: uniform
        // probabilities, so every row is the mean of V.
        let qkv = linear(&x, &attention.wqkv);
        let (_, _, value) = attention.split_qkv(&qkv);
        for d in 0..hidden {
            let mean: f32 = (0..seq_len).map(|t| value.at(t, d)).sum::<f32>() / seq_len as f32;
            assert!(
                (y.at(0, d) - mean).abs() > 1e-5,
                "row 0 dim {d} is the mean of V ({mean}), which is what uniform \
                 probabilities give and what a one-sided RoPE produces"
            );
        }

        // And the table itself matters: a different theta is a different
        // rotation, so the same weights must give a different answer.
        let other = RopeTable::new(config.head_dim(), 64, 160000.0);
        let z = attention.forward(&x, &rope, &masks, true);
        let w = attention.forward(&x, &other, &masks, true);
        assert!(
            z.max_abs_diff(&w) > 1e-5,
            "theta 10000 and theta 160000 are different rotations and must give \
             different outputs; they differ by {}",
            z.max_abs_diff(&w)
        );
    }

    /// The real shapes, once: `[seq, 1024]` in and out, 16 heads of 64,
    /// through a `[3072, 1024]` and a `[1024, 1024]`.
    ///
    /// The weights are synthetic, which is 16 MB of f32 rather than the
    /// 1.58 GB checkpoint. The sequence is deliberately only 8 tokens: this
    /// test is about the geometry, and `linear` is the naive triple loop of
    /// rung 6, so every extra token costs another 3 million multiplies per
    /// call in a debug build. The window behaviour is checked at 300 tokens
    /// by `a_local_query_sees_129_keys` and
    /// `global_and_local_agree_only_while_the_whole_sequence_fits`, where the
    /// hidden size is small enough for it to be free.
    #[test]
    fn the_hidden_size_survives_the_round_trip() {
        let hidden = 1024;
        let heads = 16;
        let seq_len = 8;

        let config = config_with(hidden, heads);
        assert_eq!(config.head_dim(), 64);

        let attention = attention_with(spread(3 * hidden * hidden, 81), hidden, heads);
        let x = Matrix::from_vec(seq_len, hidden, spread(seq_len * hidden, 82));
        let masks = Masks::new(&config, seq_len, None);

        // Layer 0 is global, layer 1 is local. `is_global_layer` owns that;
        // this just confirms the pairing the forward pass will use.
        assert!(config.is_global_layer(0));
        assert!(!config.is_global_layer(1));

        for layer in [0usize, 1] {
            let is_global = config.is_global_layer(layer);
            let rope = RopeTable::new(config.head_dim(), 256, config.rope_theta(layer));
            let y = attention.forward(&x, &rope, &masks, is_global);
            assert_eq!(y.shape(), (seq_len, hidden));
            assert!(
                y.data.iter().all(|v| v.is_finite()),
                "layer {layer}: every output value must be finite"
            );
        }
    }
}
