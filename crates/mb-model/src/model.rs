//! The whole forward pass: embeddings, 28 encoder layers, the final norm and
//! the MLM head.
//!
//! ```text
//! h = embeddings.norm(tok_embeddings[ids])
//! for each layer:
//!     h = h + Attn(attn_norm(h))      // attn_norm absent for layer 0
//!     h = h + MLP(mlp_norm(h))
//! h = final_norm(h)
//! logits = decoder(head.norm(gelu_erf(head.dense(h))))   // mask position only
//! ```
//!
//! Pre-norm: each residual adds to the **un-normed** `h`, and there is no
//! norm on the attention output.

use std::path::Path;

use crate::attention::{Attention, Masks};
use crate::config::{Config, ConfigError};
use crate::embeddings::Embeddings;
use crate::head::MlmHead;
use crate::mlp::Mlp;
use crate::ops::{add_in_place, layer_norm};
use crate::rope::RopeTable;
use crate::safetensors::{Checkpoint, TensorError};
use crate::tensor::Matrix;

/// A problem loading the model: either file.
#[derive(Debug)]
pub enum LoadError {
    Config(ConfigError),
    Tensor(TensorError),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Config(e) => write!(f, "config.json: {e}"),
            LoadError::Tensor(e) => write!(f, "model.safetensors: {e}"),
        }
    }
}

impl std::error::Error for LoadError {}

impl From<ConfigError> for LoadError {
    fn from(e: ConfigError) -> Self {
        LoadError::Config(e)
    }
}

impl From<TensorError> for LoadError {
    fn from(e: TensorError) -> Self {
        LoadError::Tensor(e)
    }
}

/// One encoder layer.
pub(crate) struct Layer {
    /// `None` for layer 0. The reference sets it to `nn.Identity`, and the
    /// tensor is absent from the checkpoint. Defaulting a missing norm
    /// weight to ones would normalize twice.
    pub(crate) attn_norm: Option<Vec<f32>>,
    pub(crate) attn: Attention,
    pub(crate) mlp_norm: Vec<f32>,
    pub(crate) mlp: Mlp,
    /// Layer 0 is global. `Config::is_global_layer` decides.
    pub(crate) is_global: bool,
}

pub struct ModernBert {
    pub(crate) config: Config,
    pub(crate) embeddings: Embeddings,
    pub(crate) layers: Vec<Layer>,
    pub(crate) final_norm: Vec<f32>,
    pub(crate) head: MlmHead,
    /// Global layers use base 160000 and local layers 10000. Two tables
    /// serve all 28 layers.
    pub(crate) rope_global: RopeTable,
    pub(crate) rope_local: RopeTable,
}

impl ModernBert {
    /// Load from a directory holding `config.json` and `model.safetensors`.
    ///
    /// Before rung 13, peak memory is about 3.2 GB: the checkpoint's 1.58 GB
    /// of f32 plus the copies the components take, until the checkpoint drops
    /// at the end of this function, and macOS's allocator keeps about 0.9 GB
    /// of the freed copy afterwards. Rung 13 (`owned-weights`) moves each
    /// tensor instead, so the peak is the 1.58 GB of weights plus one
    /// tensor's raw bytes while it decodes.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, LoadError> {
        let dir = dir.as_ref();
        let config = Config::from_file(dir.join("config.json"))?;
        let mut checkpoint = Checkpoint::from_file(dir.join("model.safetensors"))?;
        #[allow(unused_mut)]
        let mut model = Self::from_checkpoint(config, &mut checkpoint)?;
        // Rung 17: every linear layer's weight gets its bf16 copy, so the
        // bf16 matmul runs them. Rung 20 takes the same five for int8. The
        // verbalizer rows of the decoder stay f32: four rows, no measurable
        // time.
        #[cfg(feature = "bf16")]
        {
            for layer in &mut model.layers {
                keep_precision(&mut layer.attn.wqkv, "wqkv");
                keep_precision(&mut layer.attn.wo, "attn_wo");
                keep_precision(&mut layer.mlp.wi, "wi");
                keep_precision(&mut layer.mlp.wo, "mlp_wo");
            }
            keep_precision(&mut model.head.dense, "head");
        }
        // Rung 13 moves every tensor out, so anything left was never read:
        // a checkpoint with a tensor this model does not know about.
        if cfg!(feature = "owned-weights") && !checkpoint.is_empty() {
            let mut left: Vec<&str> = checkpoint.names().collect();
            left.sort();
            return Err(TensorError::new(format!(
                "the model did not read {} tensor(s) from the checkpoint: {left:?}",
                left.len()
            ))
            .into());
        }
        Ok(model)
    }

    pub fn from_checkpoint(config: Config, checkpoint: &mut Checkpoint) -> Result<Self, LoadError> {
        let c = &config;
        let embeddings = Embeddings::load(checkpoint, c.vocab_size, c.hidden_size, c.norm_eps)?;

        let mut layers = Vec::with_capacity(c.num_hidden_layers);
        for i in 0..c.num_hidden_layers {
            let attn_norm = if i == 0 {
                // Absent by design. Refuse a checkpoint that has one, since
                // this loader would otherwise ignore it silently.
                if checkpoint.get("model.layers.0.attn_norm.weight").is_some() {
                    return Err(TensorError::new(
                        "model.layers.0.attn_norm.weight is present, but layer 0 has no \
                         attention norm in this architecture",
                    )
                    .into());
                }
                None
            } else {
                let name = format!("model.layers.{i}.attn_norm.weight");
                Some(checkpoint.take(&name, &[c.hidden_size])?)
            };
            let mlp_norm_name = format!("model.layers.{i}.mlp_norm.weight");
            layers.push(Layer {
                attn_norm,
                attn: Attention::load(checkpoint, i, c.hidden_size, c.num_attention_heads)?,
                mlp_norm: checkpoint.take(&mlp_norm_name, &[c.hidden_size])?,
                mlp: Mlp::load(checkpoint, i, c.hidden_size, c.intermediate_size)?,
                is_global: c.is_global_layer(i),
            });
        }

        let final_norm = checkpoint.take("model.final_norm.weight", &[c.hidden_size])?;
        let head = MlmHead::load(checkpoint, c.hidden_size, c.vocab_size, c.norm_eps)?;

        let head_dim = c.head_dim();
        let positions = c.max_position_embeddings;
        Ok(ModernBert {
            rope_global: RopeTable::new(head_dim, positions, c.global_rope_theta),
            rope_local: RopeTable::new(head_dim, positions, c.local_rope_theta),
            embeddings,
            layers,
            final_norm,
            head,
            config,
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Run the encoder and return the `final_norm` output, `[seq, hidden]`.
    ///
    /// `observe` is called with the name of each checkpoint the reference
    /// dump uses (`embeddings.norm`, `layer.0.out` to `layer.27.out`,
    /// `final_norm`) and the activation at that point. The tier-2 tests use
    /// it to compare every layer against PyTorch without storing 28 copies.
    /// Pass `|_, _| {}` when nothing needs to look.
    pub fn encode(
        &self,
        ids: &[u32],
        mut observe: impl FnMut(&str, &Matrix),
    ) -> Result<Matrix, TensorError> {
        let c = &self.config;
        let mut h = self.embeddings.forward(ids)?;
        observe("embeddings.norm", &h);

        // Built once and shared by every layer. No padding: one sequence.
        let masks = Masks::new(c, ids.len(), None);

        for i in 0..self.layers.len() {
            self.layer(i, &mut h, &masks);
            observe(&format!("layer.{i}.out"), &h);
        }

        let h = layer_norm(&h, &self.final_norm, c.norm_eps);
        observe("final_norm", &h);
        Ok(h)
    }

    /// Run encoder layer `index` on `h` in place.
    ///
    /// Public so a test can feed one layer PyTorch's own input and compare
    /// only that layer's output. Then an error from an earlier layer cannot
    /// accumulate into a later one, and a gap that appears is that layer's.
    pub fn layer(&self, index: usize, h: &mut Matrix, masks: &Masks) {
        let c = &self.config;
        let layer = &self.layers[index];
        let rope = if layer.is_global {
            &self.rope_global
        } else {
            &self.rope_local
        };

        // Residual adds go onto the un-normed h.
        let attn_in = match &layer.attn_norm {
            Some(w) => layer_norm(h, w, c.norm_eps),
            None => h.clone(),
        };
        let attn_out = layer.attn.forward(&attn_in, rope, masks, layer.is_global);
        add_in_place(h, &attn_out);

        let mlp_out = layer
            .mlp
            .forward(&layer_norm(h, &layer.mlp_norm, c.norm_eps));
        add_in_place(h, &mlp_out);
    }

    /// The logits for `token_ids` at `position`, which for AG News is the
    /// four verbalizers at the mask. The full 50,368-wide projection is
    /// never computed.
    pub fn logits_for_tokens(
        &self,
        ids: &[u32],
        position: usize,
        token_ids: &[u32],
    ) -> Result<Vec<f32>, TensorError> {
        let h = self.encode(ids, |_, _| {})?;
        Ok(self
            .head
            .logits_for_tokens(&h, self.embeddings.weight(), position, token_ids))
    }

    /// The head and the tied decoder weight, for callers that already hold
    /// the `final_norm` output.
    pub fn head(&self) -> (&MlmHead, &Matrix) {
        (&self.head, self.embeddings.weight())
    }
}

/// Rung 17's bf16 copy of a weight, or rung 20's s8 one.
#[cfg(feature = "bf16")]
fn keep_precision(weight: &mut crate::tensor::Matrix, kind: &str) {
    #[cfg(feature = "int8")]
    {
        // Analysis only, never set in a benchmark: `MB_INT8_ONLY=wi,head`
        // keeps just those kinds in s8 and the rest in bf16, to find which
        // layer's rounding moves the logits most. Unset, every kind is s8.
        let only = std::env::var("MB_INT8_ONLY").ok();
        if only.is_none_or(|list| list.split(',').any(|k| k == kind)) {
            weight.keep_int8();
        } else {
            weight.keep_bf16();
        }
    }
    #[cfg(not(feature = "int8"))]
    {
        let _ = kind;
        weight.keep_bf16();
    }
}
