//! The model's shape, read from `config.json`.
//!
//! Only the fields the forward pass actually uses are here. Three fields in
//! the file look relevant and are not, and they are named below so nobody
//! wires them up by mistake.

use std::path::Path;

use serde_json::Value;

/// A problem with `config.json`.
#[derive(Debug)]
pub struct ConfigError(String);

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ConfigError {}

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,

    /// A layer is global when `layer_id % global_attn_every_n_layers == 0`,
    /// so **layer 0 is global**, not local. For 28 layers and a value of 3
    /// the global layers are 0, 3, 6, 9, 12, 15, 18, 21, 24 and 27, which is
    /// 10 global and 18 local, and both the first and the last are global.
    pub global_attn_every_n_layers: usize,

    /// The **total** width of the local attention window, not the width on
    /// each side. See [`Config::local_window_each_side`].
    pub local_attention: usize,

    /// RoPE base for the global layers. Larger than the local one, which is
    /// the opposite of what the window sizes suggest.
    pub global_rope_theta: f32,
    /// RoPE base for the local layers.
    pub local_rope_theta: f32,

    /// LayerNorm epsilon. The file also carries `layer_norm_eps` with the
    /// same value, but the reference implementation reads `norm_eps`.
    pub norm_eps: f32,

    /// False for this model: the LayerNorms have a weight and no bias. This
    /// does not make them RMSNorm; the mean is still subtracted.
    pub norm_bias: bool,
    /// False: no bias on the fused QKV projection or on the output
    /// projection.
    pub attention_bias: bool,
    /// False: no bias in the feed-forward.
    pub mlp_bias: bool,
    /// **True**, and it is the only bias in the whole checkpoint. Dropping it
    /// shifts every MLM logit and changes the argmax, which is the entire
    /// inference protocol. Note that `head.dense` has no bias even so.
    pub decoder_bias: bool,
}

impl Config {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| ConfigError(format!("cannot read {}: {e}", path.display())))?;
        let json: Value = serde_json::from_str(&text)
            .map_err(|e| ConfigError(format!("{} is not valid JSON: {e}", path.display())))?;
        Self::from_json(&json)
    }

    pub fn from_json(json: &Value) -> Result<Self, ConfigError> {
        let integer = |key: &str| -> Result<usize, ConfigError> {
            json.get(key)
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .ok_or_else(|| ConfigError(format!("config.json has no integer {key:?}")))
        };
        let real = |key: &str| -> Result<f32, ConfigError> {
            json.get(key)
                .and_then(Value::as_f64)
                .map(|n| n as f32)
                .ok_or_else(|| ConfigError(format!("config.json has no number {key:?}")))
        };
        let flag = |key: &str| -> Result<bool, ConfigError> {
            json.get(key)
                .and_then(Value::as_bool)
                .ok_or_else(|| ConfigError(format!("config.json has no boolean {key:?}")))
        };

        let config = Config {
            num_hidden_layers: integer("num_hidden_layers")?,
            hidden_size: integer("hidden_size")?,
            intermediate_size: integer("intermediate_size")?,
            num_attention_heads: integer("num_attention_heads")?,
            vocab_size: integer("vocab_size")?,
            max_position_embeddings: integer("max_position_embeddings")?,
            global_attn_every_n_layers: integer("global_attn_every_n_layers")?,
            local_attention: integer("local_attention")?,
            global_rope_theta: real("global_rope_theta")?,
            local_rope_theta: real("local_rope_theta")?,
            norm_eps: real("norm_eps")?,
            norm_bias: flag("norm_bias")?,
            attention_bias: flag("attention_bias")?,
            mlp_bias: flag("mlp_bias")?,
            decoder_bias: flag("decoder_bias")?,
        };

        if config.hidden_size % config.num_attention_heads != 0 {
            return Err(ConfigError(format!(
                "hidden_size {} does not divide into {} heads",
                config.hidden_size, config.num_attention_heads
            )));
        }
        if config.local_attention % 2 != 0 {
            return Err(ConfigError(format!(
                "local_attention {} is the total window width and must be even",
                config.local_attention
            )));
        }
        Ok(config)
    }

    /// 64 for this model: `hidden_size / num_attention_heads`.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Whether layer `index` uses global attention.
    ///
    /// Layer 0 is global. Reading "every third layer" as 2, 5, 8 is the
    /// mistake this method exists to prevent.
    pub fn is_global_layer(&self, index: usize) -> bool {
        index % self.global_attn_every_n_layers == 0
    }

    /// The RoPE base for layer `index`. Global layers get the large base.
    pub fn rope_theta(&self, index: usize) -> f32 {
        if self.is_global_layer(index) {
            self.global_rope_theta
        } else {
            self.local_rope_theta
        }
    }

    /// How far a local-attention query reaches on each side: 64 here.
    ///
    /// `local_attention` is the **total** window, so this is half of it. The
    /// reference implementation's predicate is `distance <= local_attention /
    /// 2`, which is inclusive, so a query attends to
    /// [`Self::local_window_keys`] keys rather than `local_attention` of them.
    pub fn local_window_each_side(&self) -> usize {
        self.local_attention / 2
    }

    /// How many keys a local-attention query attends to: **129**, not 128.
    ///
    /// 64 on each side plus the query's own position, because the distance
    /// test is inclusive. Current transformers stores
    /// `sliding_window = local_attention / 2 + 1` and its flash-attention
    /// path then subtracts the 1 straight back out, so the `+1` cancels and
    /// must not be copied here.
    pub fn local_window_keys(&self) -> usize {
        2 * self.local_window_each_side() + 1
    }

    /// Whether key `key` is visible to query `query` in a local layer.
    pub fn in_local_window(&self, query: usize, key: usize) -> bool {
        query.abs_diff(key) <= self.local_window_each_side()
    }
}

// Fields in `config.json` that the forward pass must ignore:
//
//   position_embedding_type: "absolute"
//       Vestigial. No such tensor exists and the reference implementation
//       never reads it. Position comes from RoPE alone.
//   layer_norm_eps
//       Vestigial duplicate of `norm_eps`. The reference reads `norm_eps`.
//   hidden_activation: "gelu"
//       Means the exact erf GELU, not the tanh approximation. The name alone
//       does not say which, so it is resolved in the feed-forward code.
//   sparse_prediction, reference_compile, deterministic_flash_attn,
//   repad_logits_with_grad, gradient_checkpointing, every *_dropout
//       Training or backend switches with no effect at inference.
//   classifier_pooling, classifier_activation, classifier_bias,
//   classifier_dropout
//       For the sequence-classification head, which this project does not use.
