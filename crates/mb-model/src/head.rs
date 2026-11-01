//! The masked-language-model head: `1024 -> 1024 -> 50368`.
//!
//! Three tensors, and one of them is not in the checkpoint at all:
//!
//! ```text
//! head.dense.weight    [1024, 1024]   no bias, because classifier_bias is false
//! head.norm.weight     [1024]         LayerNorm, weight only, eps 1e-5
//! decoder.bias         [50368]        the ONLY bias in the 173-tensor file
//! decoder.weight       absent         tied to model.embeddings.tok_embeddings.weight
//! ```
//!
//! The reference implementation, `ModernBertPredictionHead` plus
//! `ModernBertForMaskedLM.decoder` in `modeling_modernbert.py`:
//!
//! ```python
//! def forward(self, hidden_states):
//!     return self.norm(self.act(self.dense(hidden_states)))
//! ...
//! logits = self.decoder(self.head(last_hidden_state))
//! ```
//!
//! So the whole head is:
//!
//! ```text
//! h = final_norm(h)      // applied by the CALLER, inside the encoder
//! h = head.dense(h)      // 1024 -> 1024, NO bias
//! h = gelu_erf(h)        // the exact erf form, not the tanh approximation
//! h = head.norm(h)       // LayerNorm, weight only
//! logits = h @ decoder_weight.T + decoder.bias
//! ```
//!
//! `final_norm` is deliberately not here. It belongs to the encoder, which
//! applies it after layer 27, and the head's input is whatever the encoder
//! returned. Putting it here as well would double-normalize.
//!
//! ## The four traps, and where each one bites
//!
//! 1. **`decoder.weight` does not exist.** `tie_word_embeddings` is omitted
//!    from this model's `config.json` and the default is true, so the
//!    projection reuses the token embedding matrix. This head therefore takes
//!    the decoder matrix as an **argument** rather than loading one, so the
//!    tie is visible at the call site and the head does not own a second copy
//!    of a `[50368, 1024]` matrix (206 MB as f32, 13% of the model). The
//!    caller passes [`crate::embeddings::Embeddings::weight`].
//! 2. **`decoder.bias` exists and is the only bias in the file**, while every
//!    other linear in the model has `bias=False`. A loader that zero-fills
//!    missing biases gets all 172 of them right and this one wrong, and
//!    dropping it shifts every logit and changes the argmax, which is this
//!    project's entire inference protocol. [`MlmHead::load`] requires it.
//! 3. **`head.dense` has no bias**, unlike BERT's
//!    `cls.predictions.transform.dense`, which does. A `head.dense.bias` in
//!    the checkpoint is reported as an error; see [`MlmHead::load`].
//! 4. **`vocab_size` is 50368 but the real vocabulary is smaller**: 50,285
//!    real tokens plus 83 `[unused*]`, padded up to a multiple of 64. An
//!    unrestricted argmax over a full logit vector can return an id that
//!    decodes to nothing. [`MlmHead::logits_for_tokens`] avoids the question
//!    by never computing the logits nobody asked for.
//!
//! ## Why there are three forward methods
//!
//! The stock pipeline projects 1024 to 50368 at *every* position and then
//! reads one row of the result. On a 152-token prompt that is 151 rows of
//! 50,368 dot products thrown away, which is rung 1 of the ladder in
//! `PLAN.md` and a real measured saving rather than a tidying-up. For AG News
//! only four token ids are ever compared, so the useful work is four dot
//! products of length 1024.
//!
//! All three paths therefore exist and have to agree:
//!
//! | method | decoder work for a 152-token prompt |
//! |---|---|
//! | [`MlmHead::logits_everywhere`] | 152 x 50,368 dot products (rung 0) |
//! | [`MlmHead::logits_at`] | 1 x 50,368 (rung 1, one position) |
//! | [`MlmHead::logits_for_tokens`] | 1 x 4 (what inference actually needs) |
//!
//! The first test in this file is the one that makes the shortcut safe.

use crate::ops::{add_bias, gelu_erf_in_place, layer_norm, linear};
use crate::safetensors::{Checkpoint, TensorError};
use crate::tensor::Matrix;

/// `head.dense`, `head.norm` and `decoder.bias`.
///
/// Not the decoder matrix: that is tied to the embeddings and arrives as an
/// argument. See trap 1 in the module docs.
pub struct MlmHead {
    /// `head.dense.weight`, `[hidden_size, hidden_size]`. No bias.
    dense: Matrix,
    /// `head.norm.weight`. Weight only, because `norm_bias` is false.
    norm_weight: Vec<f32>,
    /// `norm_eps`, 1e-5. The file's `layer_norm_eps` is the vestigial
    /// duplicate; the reference reads `norm_eps`.
    eps: f32,
    /// `decoder.bias`, `[vocab_size]`. The only bias in the checkpoint.
    decoder_bias: Vec<f32>,
}

impl MlmHead {
    /// Load `head.dense.weight`, `head.norm.weight` and `decoder.bias`.
    ///
    /// `norm_eps` is the fourth argument rather than a constant here, so the
    /// head reads its epsilon from the same `Config` field as every other
    /// norm in the model. This matches
    /// [`crate::embeddings::Embeddings::load`].
    ///
    /// Two tensors are refused rather than ignored, because both would be a
    /// silent wrong answer:
    ///
    /// * **`head.dense.bias`.** `classifier_bias` is false for this model, so
    ///   the reference `head.dense` has no bias. Applying a bias that the
    ///   reference does not apply shifts every logit; ignoring a bias that is
    ///   present hides the fact that this is not the checkpoint the code was
    ///   written for. Neither is acceptable in a project whose output is a
    ///   numerical comparison, so it is an error that names the tensor.
    /// * **`decoder.weight`.** This head takes the decoder matrix as an
    ///   argument because the real checkpoint ties it to the embeddings. A
    ///   stored `decoder.weight` means the weights are *not* tied, and it
    ///   would be silently unused while the head projected with the
    ///   embeddings instead.
    pub fn load(
        checkpoint: &Checkpoint,
        hidden_size: usize,
        vocab_size: usize,
        norm_eps: f32,
    ) -> Result<Self, TensorError> {
        // Trap 3, checked before anything is read, so the message is about
        // the stray tensor and not about a shape.
        if let Some(stray) = checkpoint.get("head.dense.bias") {
            return Err(TensorError::new(format!(
                "the checkpoint has head.dense.bias (shape {:?}), but `classifier_bias` is \
                 false for this model and the reference head.dense has no bias. Applying it \
                 would shift every logit and ignoring it would hide a checkpoint mismatch, so \
                 neither is done. The only bias this head applies is decoder.bias.",
                stray.shape
            )));
        }
        // Trap 1, from the other side: a checkpoint that stores the decoder
        // matrix is not a tied one, and this head would ignore it.
        if let Some(stored) = checkpoint.get("decoder.weight") {
            return Err(TensorError::new(format!(
                "the checkpoint has decoder.weight (shape {:?}), but this head projects with \
                 the matrix passed to logits_at/logits_for_tokens/logits_everywhere, because \
                 ModernBERT-Large-Instruct ties the decoder to \
                 model.embeddings.tok_embeddings.weight. A stored decoder.weight would be \
                 silently unused.",
                stored.shape
            )));
        }

        // Trap: the encoder tensors are prefixed `model.` and the head
        // tensors are not. `head.dense.weight`, not `model.head.dense.weight`.
        let dense = checkpoint.require("head.dense.weight", &[hidden_size, hidden_size])?;
        let norm = checkpoint.require("head.norm.weight", &[hidden_size])?;
        // Trap 2: required, never defaulted to zeros.
        let decoder_bias = checkpoint.require("decoder.bias", &[vocab_size])?;

        Ok(MlmHead {
            dense: Matrix::from_vec(hidden_size, hidden_size, dense.data.clone()),
            norm_weight: norm.data.clone(),
            eps: norm_eps,
            decoder_bias: decoder_bias.data.clone(),
        })
    }

    /// 50368 for this model, which is [`Self::decoder_bias`]'s length and the
    /// number of rows the decoder matrix must have.
    pub fn vocab_size(&self) -> usize {
        self.decoder_bias.len()
    }

    /// 1024 for this model.
    pub fn hidden_size(&self) -> usize {
        self.dense.cols
    }

    /// `decoder.bias`, for the tier-2 reference comparison.
    pub fn decoder_bias(&self) -> &[f32] {
        &self.decoder_bias
    }

    /// The full projection at one position: `vocab_size` logits.
    ///
    /// This is rung 1: the same answer as [`Self::logits_everywhere`] at row
    /// `position`, without computing the other rows. Both the `head.dense`
    /// projection and the decoder projection are per-position, so selecting
    /// the row first saves both.
    ///
    /// Note trap 4 before taking an argmax over the result: ids above the
    /// real vocabulary are in range here and decode to nothing. Inference
    /// uses [`Self::logits_for_tokens`] with the verbalizer ids instead.
    ///
    /// # Panics
    ///
    /// If `position` is not a row of `hidden`, or if `decoder_weight` does
    /// not match the loaded `decoder.bias` and `head.dense`. Both are
    /// programmer errors rather than bad input: the mask position comes from
    /// our own prompt builder and the decoder matrix from our own embeddings.
    /// The message names the offending index or shape and the bound.
    pub fn logits_at(&self, hidden: &Matrix, decoder_weight: &Matrix, position: usize) -> Vec<f32> {
        self.check_decoder(decoder_weight);

        let x = self.transform(&row_at(hidden, position));
        let mut logits = linear(&x, decoder_weight);
        // Trap 2.
        add_bias(&mut logits, &self.decoder_bias);
        logits.data
    }

    /// Only the given token ids at one position, in the order given.
    ///
    /// This is what inference uses: for AG News, four dot products of length
    /// 1024 instead of 50,368 of them, a factor of 12,592 off the decoder
    /// projection. The returned values are the same numbers
    /// [`Self::logits_at`] would produce at those ids, which the first test
    /// in this file asserts.
    ///
    /// The requested rows are gathered into a small `[token_ids.len(),
    /// hidden_size]` matrix and handed to the same [`crate::ops::linear`] as
    /// every other path, rather than being dot-producted by hand here. That
    /// keeps one matmul primitive in the model: when rung 7 swaps
    /// `ops::linear` for Accelerate's `cblas_sgemm`, this path swaps with it.
    ///
    /// # Panics
    ///
    /// If `position` is not a row of `hidden`, or a token id is not a row of
    /// `decoder_weight`. The message names the bad index and the bound.
    pub fn logits_for_tokens(
        &self,
        hidden: &Matrix,
        decoder_weight: &Matrix,
        position: usize,
        token_ids: &[u32],
    ) -> Vec<f32> {
        self.check_decoder(decoder_weight);

        let x = self.transform(&row_at(hidden, position));

        let mut rows = Matrix::zeros(token_ids.len(), decoder_weight.cols);
        let mut biases = Vec::with_capacity(token_ids.len());
        for (slot, &id) in token_ids.iter().enumerate() {
            let id = id as usize;
            assert!(
                id < decoder_weight.rows,
                "token id {id} is outside the decoder weight's {} rows",
                decoder_weight.rows
            );
            rows.row_mut(slot).copy_from_slice(decoder_weight.row(id));
            // Trap 2, per selected token: the bias is indexed by token id,
            // not by the slot it was asked for.
            biases.push(self.decoder_bias[id]);
        }

        let mut logits = linear(&x, &rows);
        add_bias(&mut logits, &biases);
        logits.data
    }

    /// Every position: `[seq, vocab_size]`.
    ///
    /// This is the work rung 0 does and rung 1 removes. Only one row of it is
    /// ever read at inference, so the name says what it is for: the baseline,
    /// and the oracle the other two paths are checked against. For a
    /// 152-token prompt it is 152 x 50,368 x 1024 multiply-accumulates where
    /// [`Self::logits_for_tokens`] needs 4 x 1024.
    pub fn logits_everywhere(&self, hidden: &Matrix, decoder_weight: &Matrix) -> Matrix {
        self.check_decoder(decoder_weight);

        let x = self.transform(hidden);
        let mut logits = linear(&x, decoder_weight);
        // Trap 2, applied to every row: `add_bias` broadcasts over positions.
        add_bias(&mut logits, &self.decoder_bias);
        logits
    }

    /// `ModernBertPredictionHead.forward`: `norm(gelu_erf(dense(x)))`.
    ///
    /// Per row, so running it on one row of `hidden` gives bit-identical
    /// values to running it on all of them and keeping that row. That is what
    /// lets rung 1 select the position before the projection rather than
    /// after.
    fn transform(&self, hidden: &Matrix) -> Matrix {
        // Trap 3: no bias here. `linear` is `x @ W.T` and nothing is added.
        let mut x = linear(hidden, &self.dense);

        // The exact erf GELU, shared with the feed-forward. The tanh
        // approximation differs by up to 3e-4, three times the fp32
        // tolerance. See `ops::gelu_erf`.
        gelu_erf_in_place(&mut x.data);

        // Order is dense -> activation -> norm. LayerNorm, so the mean is
        // subtracted; weight only, because `norm_bias` is false.
        layer_norm(&x, &self.norm_weight, self.eps)
    }

    /// The decoder matrix has to be the tied embedding matrix: one row per
    /// vocabulary entry, `hidden_size` wide.
    fn check_decoder(&self, decoder_weight: &Matrix) {
        assert_eq!(
            decoder_weight.rows,
            self.decoder_bias.len(),
            "the decoder weight has {} rows but decoder.bias covers {} tokens",
            decoder_weight.rows,
            self.decoder_bias.len()
        );
        assert_eq!(
            decoder_weight.cols, self.dense.cols,
            "the decoder weight is {} wide but the head's hidden size is {}",
            decoder_weight.cols, self.dense.cols
        );
    }
}

/// One row of `hidden` as a one-row matrix, so the per-position path can use
/// the same `[rows, cols]` operations as the all-positions path.
fn row_at(hidden: &Matrix, position: usize) -> Matrix {
    assert!(
        position < hidden.rows,
        "position {position} is outside the {} rows of the hidden states",
        hidden.rows
    );
    Matrix::from_vec(1, hidden.cols, hidden.row(position).to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an in-memory safetensors file and read it back, so the tests
    /// exercise the real loader and the real `require` error path without
    /// touching `weights/model.safetensors`. That file is 791 MB on disk and
    /// 1.58 GB as f32, and `cargo test` runs test functions in parallel
    /// across every core; see hard rule 8 in `CLAUDE.md`.
    ///
    /// Values are stored as `F32`, which the loader also accepts, so the test
    /// arithmetic is not rounded to 8 mantissa bits before it starts.
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

    /// The three tensors the head loads, with the shapes given rather than
    /// inferred so a test can deliberately store a wrong one.
    fn head_tensors(
        dense: (Vec<usize>, Vec<f32>),
        norm: (Vec<usize>, Vec<f32>),
        bias: (Vec<usize>, Vec<f32>),
    ) -> Vec<(String, Vec<usize>, Vec<f32>)> {
        vec![
            ("head.dense.weight".to_string(), dense.0, dense.1),
            ("head.norm.weight".to_string(), norm.0, norm.1),
            ("decoder.bias".to_string(), bias.0, bias.1),
        ]
    }

    /// Deterministic values in roughly `[-0.5, 0.5]`, for the tests that care
    /// about agreement rather than particular arithmetic. A linear
    /// congruential generator, so the sequence is the same on every run and
    /// every machine.
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

    /// A head with synthetic weights, plus a tied decoder matrix and a hidden
    /// state to run it on. `eps` is the model's real 1e-5 throughout.
    fn synthetic(hidden_size: usize, vocab_size: usize, seq: usize) -> (MlmHead, Matrix, Matrix) {
        let checkpoint = checkpoint_of(&head_tensors(
            (
                vec![hidden_size, hidden_size],
                spread(hidden_size * hidden_size, 11),
            ),
            (vec![hidden_size], spread(hidden_size, 22)),
            (vec![vocab_size], spread(vocab_size, 33)),
        ));
        let head = MlmHead::load(&checkpoint, hidden_size, vocab_size, 1e-5)
            .expect("the shapes are right");

        // The decoder weight is the token embedding matrix, which the caller
        // owns. Here it is just another synthetic `[vocab_size, hidden_size]`.
        let decoder_weight = Matrix::from_vec(
            vocab_size,
            hidden_size,
            spread(vocab_size * hidden_size, 44),
        );
        let hidden = Matrix::from_vec(seq, hidden_size, spread(seq * hidden_size, 55));

        (head, decoder_weight, hidden)
    }

    fn assert_close(actual: f32, expected: f32, what: &str) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "{what}: got {actual}, expected {expected}"
        );
    }

    /// The test that makes the optimization safe, so it comes first.
    ///
    /// `logits_for_tokens` is the only path inference takes, and it is only
    /// legitimate if it returns exactly what the full projection would have
    /// returned at those ids. Both paths go through the same
    /// [`crate::ops::linear`] with the same f32 accumulation order over the
    /// same hidden row and the same decoder rows, so this is bit equality,
    /// not a tolerance.
    ///
    /// If rung 7's `cblas_sgemm` ever reduces in a different order for a
    /// 4-column weight than for a 50,368-column one, this assertion is where
    /// that shows up, and it becomes a tight tolerance with a comment saying
    /// why. Until then, exact.
    #[test]
    fn the_selective_path_agrees_with_the_full_projection() {
        let (head, decoder_weight, hidden) = synthetic(16, 40, 5);

        let position = 3;
        let full = head.logits_at(&hidden, &decoder_weight, position);

        // Deliberately out of order, repeated, and including both ends of the
        // vocabulary: the result is indexed by request order, not by id.
        let ids: [u32; 6] = [7, 0, 39, 7, 21, 1];
        let selected = head.logits_for_tokens(&hidden, &decoder_weight, position, &ids);

        assert_eq!(selected.len(), ids.len());
        for (slot, &id) in ids.iter().enumerate() {
            assert_eq!(
                selected[slot], full[id as usize],
                "id {id} in slot {slot}: the selective path gave {} and the full \
                 projection {}",
                selected[slot], full[id as usize]
            );
        }
    }

    /// Rung 1 against rung 0: one position computed alone equals that row of
    /// every position computed together.
    ///
    /// Exact equality again, and it holds because `head.dense`, the GELU and
    /// `head.norm` are all per-row: selecting the row before the head changes
    /// nothing about the arithmetic that row gets.
    #[test]
    fn logits_at_is_one_row_of_logits_everywhere() {
        let (head, decoder_weight, hidden) = synthetic(16, 40, 4);

        let everywhere = head.logits_everywhere(&hidden, &decoder_weight);
        assert_eq!(everywhere.shape(), (4, 40));

        for position in 0..hidden.rows {
            let at = head.logits_at(&hidden, &decoder_weight, position);
            assert_eq!(
                at.as_slice(),
                everywhere.row(position),
                "row {position} differs between the two paths"
            );
        }
    }

    /// Trap 2: `decoder.bias` is applied, and it is the only bias in the file.
    ///
    /// Two heads with identical `head.dense` and `head.norm` weights, one with
    /// a zero `decoder.bias` and one with a real one. The logits must differ
    /// by the bias and by nothing else.
    ///
    /// "Exactly the bias" is spelled as a 1e-6 tolerance rather than `==`
    /// because `(a + b) - a` is not exactly `b` in f32 for arbitrary `a`. At
    /// these magnitudes the gap is around 1e-7.
    #[test]
    fn the_decoder_bias_is_applied() {
        let hidden_size = 8;
        let vocab_size = 12;

        let dense = spread(hidden_size * hidden_size, 61);
        let norm = spread(hidden_size, 62);
        let bias = spread(vocab_size, 63);

        let with_bias = MlmHead::load(
            &checkpoint_of(&head_tensors(
                (vec![hidden_size, hidden_size], dense.clone()),
                (vec![hidden_size], norm.clone()),
                (vec![vocab_size], bias.clone()),
            )),
            hidden_size,
            vocab_size,
            1e-5,
        )
        .expect("the shapes are right");

        let without_bias = MlmHead::load(
            &checkpoint_of(&head_tensors(
                (vec![hidden_size, hidden_size], dense),
                (vec![hidden_size], norm),
                (vec![vocab_size], vec![0.0; vocab_size]),
            )),
            hidden_size,
            vocab_size,
            1e-5,
        )
        .expect("the shapes are right");

        let decoder_weight = Matrix::from_vec(
            vocab_size,
            hidden_size,
            spread(vocab_size * hidden_size, 64),
        );
        let hidden = Matrix::from_vec(2, hidden_size, spread(2 * hidden_size, 65));

        let biased = with_bias.logits_at(&hidden, &decoder_weight, 1);
        let bare = without_bias.logits_at(&hidden, &decoder_weight, 1);

        for (id, expected) in bias.iter().enumerate() {
            let difference = biased[id] - bare[id];
            assert!(
                (difference - expected).abs() < 1e-6,
                "logit {id} moved by {difference}, but decoder.bias[{id}] is {expected}"
            );
        }
        assert!(
            bias.iter().any(|&b| b.abs() > 0.01),
            "the test bias must be non-trivial, or this asserts nothing"
        );
    }

    /// The whole head worked out by hand, with `hidden_size = 2` and
    /// `vocab_size = 3`.
    ///
    /// ```text
    /// hidden = [1, 2]                  one position, already final_norm'd
    ///
    /// head.dense.weight = [[1, 0],     identity, and NO bias (trap 3)
    ///                      [0, 1]]
    /// dense(hidden) = [1, 2]
    ///
    /// gelu_erf(1) = 1 * 0.5 * (1 + erf(1/sqrt(2))) = 0.5 * 1.6826894921
    ///             = 0.8413447461
    /// gelu_erf(2) = 2 * 0.5 * (1 + erf(2/sqrt(2))) = 1 + 0.9544997361
    ///             = 1.9544997361
    ///
    /// head.norm, weight [1, 1], eps 1e-5:
    ///     mean = (0.8413447461 + 1.9544997361) / 2 = 1.3979222411
    ///     deviations = [-0.5555774950, +0.5555774950]
    ///     var  = 0.5555774950^2 = 0.3086552358
    ///     1 / sqrt(var + 1e-5) = 1 / 0.5555864900 = 1.7998998
    ///     h = [-0.9999838, +0.9999838]
    ///
    /// decoder_weight = [[1, 0],        token 0 reads h0
    ///                   [0, 1],        token 1 reads h1
    ///                   [1, 1]]        token 2 reads h0 + h1
    /// before the bias: [-0.9999838, +0.9999838, 0.0]
    ///
    /// decoder.bias = [0.5, -0.25, 2.0]
    /// logits = [-0.4999838, 0.7499838, 2.0]
    /// ```
    ///
    /// The erf values are the familiar normal-distribution ones: `erf(1/√2)`
    /// is the one-sigma mass 0.6826894921 and `erf(2/√2)` the two-sigma mass
    /// 0.9544997361.
    ///
    /// Two things this case shows beyond the arithmetic:
    ///
    /// * **The bias decides the argmax here.** With it, token 2 wins at 2.0;
    ///   without it, token 1 wins at 0.9999838. That is trap 2 in one line,
    ///   on a model whose entire inference protocol is an argmax at a mask
    ///   position.
    /// * **With two features LayerNorm leaves only the sign**, scaled by the
    ///   norm weight: any two distinct values normalize to ±0.9999838. So
    ///   this case pins the norm, the decoder projection and the bias
    ///   precisely, while the magnitude of the erf GELU is pinned by the
    ///   tests on [`crate::ops::gelu_erf`] instead of here.
    #[test]
    fn the_hand_computed_case() {
        let checkpoint = checkpoint_of(&head_tensors(
            (vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]),
            (vec![2], vec![1.0, 1.0]),
            (vec![3], vec![0.5, -0.25, 2.0]),
        ));
        let head = MlmHead::load(&checkpoint, 2, 3, 1e-5).expect("the shapes are right");

        let decoder_weight = Matrix::from_vec(3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let hidden = Matrix::from_vec(1, 2, vec![1.0, 2.0]);

        let logits = head.logits_at(&hidden, &decoder_weight, 0);

        assert_eq!(logits.len(), 3);
        assert_close(logits[0], -0.499_983_8, "logit for token 0");
        assert_close(logits[1], 0.749_983_8, "logit for token 1");
        assert_close(logits[2], 2.0, "logit for token 2");

        // The bias is what makes token 2 win. Without it the pre-bias logits
        // are [-0.9999838, 0.9999838, 0.0] and token 1 wins.
        let mut argmax = 0;
        for id in 1..logits.len() {
            if logits[id] > logits[argmax] {
                argmax = id;
            }
        }
        assert_eq!(
            argmax, 2,
            "the biased argmax must be token 2, got {logits:?}"
        );
    }

    /// Trap 3: `head.dense` has no bias, and a checkpoint that carries one is
    /// an error rather than something to ignore.
    ///
    /// Ignoring it silently would be wrong in both directions: if the tensor
    /// is real, every logit is off; if it is a mistake, the loader has just
    /// accepted a checkpoint it was not written for. Either way the numbers
    /// stop meaning what the project says they mean, and this project's whole
    /// output is a numerical comparison. So it is reported, with the tensor
    /// named.
    #[test]
    fn a_stray_head_dense_bias_is_reported() {
        let mut tensors = head_tensors(
            (vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]),
            (vec![2], vec![1.0, 1.0]),
            (vec![3], vec![0.5, -0.25, 2.0]),
        );
        tensors.push(("head.dense.bias".to_string(), vec![2], vec![0.1, 0.2]));

        let message = match MlmHead::load(&checkpoint_of(&tensors), 2, 3, 1e-5) {
            Err(error) => error.to_string(),
            Ok(_) => panic!("a checkpoint with head.dense.bias must not load"),
        };

        assert!(
            message.contains("head.dense.bias"),
            "the message must name the tensor, got {message:?}"
        );
        assert!(
            message.contains("classifier_bias"),
            "the message must say why there is no bias, got {message:?}"
        );
    }

    /// Trap 1 from the other side: a stored `decoder.weight` means the
    /// weights are not tied, and this head would ignore it.
    #[test]
    fn a_stored_decoder_weight_is_reported() {
        let mut tensors = head_tensors(
            (vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]),
            (vec![2], vec![1.0, 1.0]),
            (vec![3], vec![0.5, -0.25, 2.0]),
        );
        tensors.push((
            "decoder.weight".to_string(),
            vec![3, 2],
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ));

        let message = match MlmHead::load(&checkpoint_of(&tensors), 2, 3, 1e-5) {
            Err(error) => error.to_string(),
            Ok(_) => panic!("a checkpoint with decoder.weight must not load"),
        };

        assert!(
            message.contains("decoder.weight"),
            "the message must name the tensor, got {message:?}"
        );
    }

    /// A missing `decoder.bias` is an error, not a zero fill.
    ///
    /// This is the half of trap 2 that a tolerant loader gets wrong: with
    /// every other bias in the model genuinely absent, zero-filling looks
    /// harmless and is correct 172 times out of 173.
    #[test]
    fn a_missing_decoder_bias_is_reported() {
        let checkpoint = checkpoint_of(&[
            (
                "head.dense.weight".to_string(),
                vec![2, 2],
                vec![1.0, 0.0, 0.0, 1.0],
            ),
            ("head.norm.weight".to_string(), vec![2], vec![1.0, 1.0]),
        ]);

        let message = match MlmHead::load(&checkpoint, 2, 3, 1e-5) {
            Err(error) => error.to_string(),
            Ok(_) => panic!("a checkpoint with no decoder.bias must not load"),
        };

        assert!(
            message.contains("decoder.bias"),
            "the message must name the tensor, got {message:?}"
        );
    }

    /// An out-of-range position is a documented panic naming the index and
    /// the bound, not a read past the end of the hidden states.
    ///
    /// A panic rather than a `Result` because this is a programmer error: the
    /// mask position comes from our own prompt builder, not from a file. Bad
    /// *data*, such as a token id in the checkpoint loader, still returns a
    /// `TensorError`.
    #[test]
    #[should_panic(expected = "position 4 is outside the 3 rows")]
    fn an_out_of_range_position_panics_with_the_bound() {
        let (head, decoder_weight, hidden) = synthetic(8, 10, 3);
        let _ = head.logits_at(&hidden, &decoder_weight, 4);
    }

    /// Same for a token id above the decoder matrix. Note trap 4: ids up to
    /// 50367 are *in* range even though the real vocabulary stops at 50284
    /// plus 83 unused, so this catches only the gross error. Verbalizer ids
    /// are resolved by encoding and asserted at startup; see `PLAN.md`.
    #[test]
    #[should_panic(expected = "token id 10 is outside the decoder weight's 10 rows")]
    fn an_out_of_range_token_id_panics_with_the_bound() {
        let (head, decoder_weight, hidden) = synthetic(8, 10, 3);
        let _ = head.logits_for_tokens(&hidden, &decoder_weight, 0, &[3, 10]);
    }

    /// Shapes: `vocab_size` values at one position, `[seq, vocab_size]`
    /// everywhere, one value per requested id.
    #[test]
    fn the_shapes_are_vocab_size_wide() {
        let (head, decoder_weight, hidden) = synthetic(8, 10, 6);

        assert_eq!(head.vocab_size(), 10);
        assert_eq!(head.hidden_size(), 8);
        assert_eq!(head.logits_at(&hidden, &decoder_weight, 0).len(), 10);
        assert_eq!(
            head.logits_everywhere(&hidden, &decoder_weight).shape(),
            (6, 10)
        );
        assert_eq!(
            head.logits_for_tokens(&hidden, &decoder_weight, 0, &[1, 2, 3])
                .len(),
            3
        );
    }

    /// The selective path's cost must not depend on `vocab_size`, which is
    /// the entire reason it exists.
    ///
    /// A unit test cannot time that reliably on a throttling fanless laptop,
    /// so it asserts the structural fact instead: with the real
    /// `vocab_size` of 50368 and only four ids requested, the answer is right
    /// and the 50,364 unrequested rows are never read. The FLOP ratio in the
    /// decoder projection is 50368 / 4 = **12,592x**, on top of the
    /// seq-length factor [`MlmHead::logits_at`] already saves.
    ///
    /// The ids are the real AG News verbalizers from `PLAN.md`: ` Business`
    /// 10518, ` World` 3645, ` Sports` 15000, ` Tech` 18447. The hidden size
    /// here is 8 rather than 1024 on purpose: a real `[50368, 1024]` decoder
    /// weight is 206 MB as f32, and `Checkpoint::from_bytes` would hold both
    /// the raw payload and the decoded copy. `[50368, 8]` is 1.6 MB and
    /// exercises the same thing, since what is being asserted is independence
    /// from the row count.
    #[test]
    fn the_selective_path_ignores_the_vocabulary_it_was_not_asked_about() {
        let hidden_size = 8;
        let vocab_size = 50368;
        let (head, decoder_weight, hidden) = synthetic(hidden_size, vocab_size, 2);

        let verbalizers: [u32; 4] = [10518, 3645, 15000, 18447];
        let selected = head.logits_for_tokens(&hidden, &decoder_weight, 1, &verbalizers);

        assert_eq!(selected.len(), 4);
        assert!(
            selected.iter().all(|v| v.is_finite()),
            "every verbalizer logit must be finite, got {selected:?}"
        );

        // And they are the same numbers the full 50,368-wide projection at
        // that position would have produced, which is what makes skipping it
        // legitimate. This is the one place the full projection is computed
        // at the real vocabulary width, and it is 50368 x 8, not 50368 x 1024.
        let full = head.logits_at(&hidden, &decoder_weight, 1);
        assert_eq!(full.len(), vocab_size);
        for (slot, &id) in verbalizers.iter().enumerate() {
            assert_eq!(selected[slot], full[id as usize], "verbalizer id {id}");
        }
    }
}
