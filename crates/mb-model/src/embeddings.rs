//! The embedding lookup, and the norm that follows it.
//!
//! Two tensors: `model.embeddings.tok_embeddings.weight` with shape
//! `[50368, 1024]`, and `model.embeddings.norm.weight` with shape `[1024]`.
//! There are **no position embeddings**. `config.json` carries
//! `position_embedding_type: "absolute"`, but it is vestigial, no such tensor
//! exists, and position enters the model through RoPE alone.
//!
//! The embedding matrix is also the MLM head's decoder weight, because
//! `decoder.weight` is absent from the checkpoint and tied to this one. See
//! [`Embeddings::weight`].

use crate::ops::layer_norm;
use crate::safetensors::{Checkpoint, TensorError};
use crate::tensor::Matrix;

pub struct Embeddings {
    /// `[vocab_size, hidden_size]`. One row per token.
    weight: Matrix,
    /// The LayerNorm applied after the lookup. Weight only, no bias.
    norm_weight: Vec<f32>,
    eps: f32,
}

impl Embeddings {
    pub fn load(
        checkpoint: &Checkpoint,
        vocab_size: usize,
        hidden_size: usize,
        eps: f32,
    ) -> Result<Self, TensorError> {
        let table = checkpoint.require(
            "model.embeddings.tok_embeddings.weight",
            &[vocab_size, hidden_size],
        )?;
        let norm = checkpoint.require("model.embeddings.norm.weight", &[hidden_size])?;

        Ok(Embeddings {
            weight: Matrix::from_vec(vocab_size, hidden_size, table.data.clone()),
            norm_weight: norm.data.clone(),
            eps,
        })
    }

    /// Look up `ids` and normalize, giving `[ids.len(), hidden_size]`.
    pub fn forward(&self, ids: &[u32]) -> Result<Matrix, TensorError> {
        let mut out = Matrix::zeros(ids.len(), self.weight.cols);
        for (row, &id) in ids.iter().enumerate() {
            let id = id as usize;
            if id >= self.weight.rows {
                return Err(TensorError::new(format!(
                    "token id {id} at position {row} is outside the vocabulary of {}",
                    self.weight.rows
                )));
            }
            out.row_mut(row).copy_from_slice(self.weight.row(id));
        }
        Ok(layer_norm(&out, &self.norm_weight, self.eps))
    }

    /// The raw lookup with no norm, for the tier-2 `embeddings.tok`
    /// reference checkpoint.
    pub fn lookup_only(&self, ids: &[u32]) -> Result<Matrix, TensorError> {
        let mut out = Matrix::zeros(ids.len(), self.weight.cols);
        for (row, &id) in ids.iter().enumerate() {
            let id = id as usize;
            if id >= self.weight.rows {
                return Err(TensorError::new(format!(
                    "token id {id} at position {row} is outside the vocabulary of {}",
                    self.weight.rows
                )));
            }
            out.row_mut(row).copy_from_slice(self.weight.row(id));
        }
        Ok(out)
    }

    /// The embedding matrix, which is also the MLM head's decoder weight.
    ///
    /// `decoder.weight` is not in the checkpoint: `tie_word_embeddings` is
    /// absent from `config.json` and defaults to true, so the head projects
    /// with this same matrix. The head's `decoder.bias` is separate, and is
    /// the only bias in the file.
    pub fn weight(&self) -> &Matrix {
        &self.weight
    }

    /// One note for anyone tempted to zero a row here: the embedding has
    /// `padding_idx = pad_token_id` in the reference, but PyTorch's
    /// `padding_idx` only zeroes *gradients*. The row holds trained values at
    /// inference, so it must be loaded as it is.
    pub fn hidden_size(&self) -> usize {
        self.weight.cols
    }

    pub fn vocab_size(&self) -> usize {
        self.weight.rows
    }
}
